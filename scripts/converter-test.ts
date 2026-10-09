// Offline checks for the OpenAI -> Anthropic message converter.
// Run: npx tsx scripts/converter-test.ts
import assert from 'node:assert/strict'
import { convertMessages, sanitizeToolId } from '../src/utils/openai-to-anthropic-converter'

const ID = /^[a-zA-Z0-9_-]+$/

function checkValid(out: any[]) {
  for (let i = 0; i < out.length; i++) {
    const m = out[i]
    if (i > 0) assert.notEqual(m.role, out[i - 1].role, 'roles must alternate')
    for (const b of m.content) {
      if (b.type === 'tool_use') assert.match(b.id, ID)
      if (b.type === 'tool_result') assert.match(b.tool_use_id, ID)
    }
    const uses = m.content.filter((b: any) => b.type === 'tool_use').map((b: any) => b.id)
    if (uses.length) {
      const next = out[i + 1]
      assert.equal(next?.role, 'user', 'tool_use must be followed by a user turn')
      const results = next.content.filter((b: any) => b.type === 'tool_result').map((b: any) => b.tool_use_id)
      assert.deepEqual([...results].sort(), [...uses].sort(), 'every tool_use answered exactly once')
    }
  }
}

const call = (id: any, name = 'read_file') => ({
  id,
  type: 'function',
  function: { name, arguments: '{"path":"a.ts"}' },
})

// 1. Invalid characters in IDs (the reported error)
{
  const weird = 'call_abc|fc_123.4:5'
  const out = convertMessages([
    { role: 'user', content: 'hi' },
    { role: 'assistant', content: null, tool_calls: [call(weird)] },
    { role: 'tool', tool_call_id: weird, content: 'file' },
  ])
  checkValid(out)
  assert.equal(out[1].content[0].id, out[2].content[0].tool_use_id)
}

// 2. Valid IDs pass through untouched
assert.equal(sanitizeToolId('toolu_01ABC-xyz'), 'toolu_01ABC-xyz')
assert.notEqual(sanitizeToolId('a.b'), sanitizeToolId('a:b'), 'no collisions')
assert.ok(sanitizeToolId('x'.repeat(200)).length <= 64)
assert.match(sanitizeToolId(''), ID)

// 3. Cancelled run: tool_use with no tool_result, followed by a new user message
{
  const out = convertMessages([
    { role: 'user', content: 'go' },
    { role: 'assistant', content: 'ok', tool_calls: [call('call_1'), call('call_2')] },
    { role: 'tool', tool_call_id: 'call_1', content: 'done' },
    { role: 'user', content: 'stop, do something else' },
  ])
  checkValid(out)
}

// 4. Trailing tool_use with nothing after it
checkValid(
  convertMessages([
    { role: 'user', content: 'go' },
    { role: 'assistant', content: null, tool_calls: [call('call_1')] },
  ]),
)

// 5. Orphan tool_result, duplicate IDs, missing IDs
checkValid(
  convertMessages([
    { role: 'user', content: 'go' },
    { role: 'tool', tool_call_id: 'ghost', content: 'x' },
    { role: 'assistant', content: null, tool_calls: [call('dup'), call('dup'), call(undefined)] },
    { role: 'tool', tool_call_id: 'dup', content: 'x' },
    { role: 'tool', tool_call_id: 'dup', content: 'x again' },
  ]),
)

console.log('converter tests passed')
