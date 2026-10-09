// Offline smoke test: streams a fake Claude response through the proxy and checks the
// OpenAI chunks, the usage numbers, and the request record. No network or Redis needed.
// Run: npx tsx scripts/obs-smoke-test.ts
import assert from 'node:assert/strict'

const records: any[] = []
const realLog = console.log
console.log = (line: string, ...rest: any[]) => {
  if (typeof line === 'string' && line.startsWith('{')) records.push(JSON.parse(line))
  else realLog(line, ...rest)
}
console.error = console.log

const sse = [
  { type: 'message_start', message: { id: 'msg_1', model: 'claude-opus-5-5', usage: { input_tokens: 4, cache_read_input_tokens: 9000, cache_creation_input_tokens: 300, output_tokens: 1 } } },
  { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
  { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'hello' } },
  { type: 'content_block_stop', index: 0 },
  { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { input_tokens: 4, cache_read_input_tokens: 9000, cache_creation_input_tokens: 300, output_tokens: 42 } },
  { type: 'message_stop' },
].map((e) => `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`).join('')

let sentBody: any
const realFetch = globalThis.fetch
globalThis.fetch = (async (url: any, init: any) => {
  const u = String(url)
  if (u.includes('api.anthropic.com/v1/messages')) {
    sentBody = JSON.parse(init.body)
    // Split mid-line to exercise the carry logic.
    const half = Math.floor(sse.length / 2)
    const body = new ReadableStream({
      start(c) {
        c.enqueue(new TextEncoder().encode(sse.slice(0, half)))
        c.enqueue(new TextEncoder().encode(sse.slice(half)))
        c.close()
      },
    })
    return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
  }
  if (u.includes('upstash')) {
    // Upstash REST: commands arrive as JSON arrays; pipelines as arrays of arrays.
    const cmds = JSON.parse(init.body)
    const answer = (cmd: string[]) =>
      cmd[0].toLowerCase() === 'get' && cmd[1] === 'auth:anthropic'
        ? { result: JSON.stringify({ type: 'oauth', access: 'tok', refresh: 'r', expires: Date.now() + 3600_000 }) }
        : { result: 'OK' }
    const out = Array.isArray(cmds[0]) ? cmds.map(answer) : answer(cmds)
    return new Response(JSON.stringify(out), { status: 200 })
  }
  return realFetch(url, init)
}) as typeof fetch

process.env.UPSTASH_REDIS_REST_URL = 'https://fake.upstash.io'
process.env.UPSTASH_REDIS_REST_TOKEN = 'x'
delete process.env.API_KEY

async function main() {
  const { default: app } = await import('../src/server')

const res = await app.request('/v1/chat/completions', {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({
    model: 'proxy-opus',
    stream: true,
    messages: [
      { role: 'system', content: 'be brief' },
      { role: 'user', content: 'hi' },
    ],
    tools: [{ type: 'function', function: { name: 'read_file', parameters: { type: 'object', properties: {} } } }],
  }),
})
const text = await res.text()
await new Promise((r) => setTimeout(r, 50))

assert.equal(res.status, 200)
assert.match(res.headers.get('x-proxy-request-id') || '', /^req_/)
assert.ok(text.includes('"content":"hello"'), 'text streamed')
assert.ok(text.trim().endsWith('data: [DONE]'), 'stream terminated')
const usage = text.split('\n').filter((l) => l.includes('"usage"')).map((l) => JSON.parse(l.slice(6)).usage)[0]
assert.equal(usage.prompt_tokens, 9304)
assert.equal(usage.prompt_tokens_details.cached_tokens, 9000)
assert.equal(usage.completion_tokens, 42)
assert.equal(sentBody.tools[0].cache_control?.type, 'ephemeral', 'cache breakpoint sent upstream')

const rec = records.find((r) => r.evt === 'request')
assert.ok(rec, 'request logged')
assert.equal(rec.model, 'claude-opus-5-5')
assert.equal(rec.cache_read, 9000)
assert.equal(rec.in, 9304)
assert.equal(rec.out, 42)
assert.ok(rec.tools_hash && rec.system_hash)
realLog('obs smoke test passed')
}

main().catch((e) => {
  console.log = realLog
  realLog(e)
  process.exit(1)
})
