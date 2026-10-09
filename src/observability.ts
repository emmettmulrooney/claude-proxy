// Observability for the proxy: structured logs, a rolling request history in Redis,
// aggregate stats, and a plain-English summary written by Haiku.
//
// Logs are one JSON object per line ({"evt": ..., ...}) so `railway logs | grep '"evt":"request"'`
// or any log tool can filter them. Every request gets an id, returned as x-proxy-request-id,
// so a Cursor error can be matched to its log line.
import { Redis } from '@upstash/redis'

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL || '',
  token: process.env.UPSTASH_REDIS_REST_TOKEN || '',
})

const HISTORY_KEY = 'obs:requests'
const HISTORY_MAX = 2000
const SUMMARY_KEY = 'obs:summary'
const SUMMARY_TTL_SECONDS = 600

export function log(evt: string, fields: Record<string, unknown> = {}): void {
  const line = JSON.stringify({ t: new Date().toISOString(), evt, ...fields })
  if (evt.endsWith('error')) console.error(line)
  else console.log(line)
}

export function newRequestId(): string {
  return 'req_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

export interface RequestRecord {
  id: string
  t: number // epoch ms at start
  ms: number // total duration, including streaming
  path: string
  model: string
  stream: boolean
  status: number
  in: number // all input tokens = uncached + cache_read + cache_write
  cache_read: number
  cache_write: number
  uncached: number
  out: number
  messages: number
  tools_hash?: string
  system_hash?: string
  error?: string
}

// Never let observability break a request.
export async function recordRequest(r: RequestRecord): Promise<void> {
  log(r.error || r.status >= 400 ? 'request_error' : 'request', { ...r })
  try {
    await redis.lpush(HISTORY_KEY, JSON.stringify(r))
    await redis.ltrim(HISTORY_KEY, 0, HISTORY_MAX - 1)
  } catch (error) {
    console.error(JSON.stringify({ evt: 'obs_store_error', message: (error as Error).message }))
  }
}

async function history(): Promise<RequestRecord[]> {
  const raw = await redis.lrange<string | RequestRecord>(HISTORY_KEY, 0, HISTORY_MAX - 1)
  return raw.map((x) => (typeof x === 'string' ? JSON.parse(x) : x))
}

export interface WindowStats {
  requests: number
  errors: number
  input_tokens: number
  cache_read: number
  cache_write: number
  uncached: number
  output_tokens: number
  cache_hit_rate: number // share of input tokens served from cache, 0–1
  avg_ms: number
  p95_ms: number
  models: Record<string, number>
  // Requests whose tools/system hash differed from the previous request: each one forces a
  // cache rewrite. A high count means the prompt prefix is unstable.
  prefix_changes: number
}

function summarize(records: RequestRecord[]): WindowStats {
  const s: WindowStats = {
    requests: records.length,
    errors: 0,
    input_tokens: 0,
    cache_read: 0,
    cache_write: 0,
    uncached: 0,
    output_tokens: 0,
    cache_hit_rate: 0,
    avg_ms: 0,
    p95_ms: 0,
    models: {},
    prefix_changes: 0,
  }
  const durations: number[] = []
  let prevPrefix: string | null = null
  // Records are newest-first; walk oldest-first to count prefix changes in order.
  for (const r of [...records].reverse()) {
    if (r.error || r.status >= 400) s.errors++
    s.input_tokens += r.in
    s.cache_read += r.cache_read
    s.cache_write += r.cache_write
    s.uncached += r.uncached
    s.output_tokens += r.out
    s.models[r.model] = (s.models[r.model] || 0) + 1
    durations.push(r.ms)
    if (r.tools_hash || r.system_hash) {
      const prefix = `${r.tools_hash}/${r.system_hash}`
      if (prevPrefix && prefix !== prevPrefix) s.prefix_changes++
      prevPrefix = prefix
    }
  }
  if (s.input_tokens) s.cache_hit_rate = s.cache_read / s.input_tokens
  if (durations.length) {
    durations.sort((a, b) => a - b)
    s.avg_ms = Math.round(durations.reduce((a, b) => a + b, 0) / durations.length)
    s.p95_ms = durations[Math.min(durations.length - 1, Math.floor(durations.length * 0.95))]
  }
  return s
}

export async function getStats() {
  const all = await history()
  const now = Date.now()
  const within = (ms: number) => all.filter((r) => now - r.t <= ms)
  return {
    generated_at: new Date(now).toISOString(),
    last_hour: summarize(within(3600_000)),
    last_24h: summarize(within(86_400_000)),
    recent: all.slice(0, 20),
    recent_errors: all.filter((r) => r.error || r.status >= 400).slice(0, 10),
  }
}

const HAIKU_MODEL = 'claude-haiku-5-5'

// Plain-English read of the stats, cached so the dashboard costs at most one Haiku call
// per 10 minutes. `fresh` skips the cache.
export async function getSummary(
  callClaude: (body: Record<string, unknown>) => Promise<Response>,
  fresh = false,
): Promise<{ summary: string; generated_at: string; cached: boolean }> {
  if (!fresh) {
    const cached = await redis.get<{ summary: string; generated_at: string }>(SUMMARY_KEY)
    if (cached) return { ...cached, cached: true }
  }
  const stats = await getStats()
  const compact = {
    last_hour: stats.last_hour,
    last_24h: stats.last_24h,
    recent_errors: stats.recent_errors.map((r) => ({ t: new Date(r.t).toISOString(), status: r.status, error: r.error })),
  }
  const response = await callClaude({
    model: HAIKU_MODEL,
    max_tokens: 400,
    system: [
      { type: 'text', text: "You are Claude Code, Anthropic's official CLI for Claude." },
      {
        type: 'text',
        text:
          'You write a short status report for the owner of an LLM proxy that forwards Cursor requests to Claude. ' +
          'Given JSON stats, write 3–5 plain-English sentences, no headings or bullet points: is prompt caching working ' +
          '(cache_hit_rate above 0.8 is healthy after the first request of a session; frequent prefix_changes mean the prompt prefix ' +
          'keeps changing and the cache is being rewritten), how busy it is, whether there are errors and what they look like, ' +
          'and one concrete suggestion only if something looks wrong. Use round numbers. Don’t invent data.',
      },
    ],
    messages: [{ role: 'user', content: JSON.stringify(compact) }],
  })
  if (!response.ok) throw new Error(`Haiku summary failed (${response.status}): ${await response.text()}`)
  const data = (await response.json()) as { content?: Array<{ type: string; text?: string }> }
  const summary = (data.content || []).filter((b) => b.type === 'text').map((b) => b.text).join('').trim()
  const result = { summary, generated_at: new Date().toISOString() }
  try {
    await redis.set(SUMMARY_KEY, result, { ex: SUMMARY_TTL_SECONDS })
  } catch {}
  log('summary', { chars: summary.length })
  return { ...result, cached: false }
}
