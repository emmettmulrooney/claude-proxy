// Translates the request side of OpenAI chat completions (what Cursor sends) into
// Anthropic Messages format: tool definitions, tool_choice, assistant tool calls,
// tool results, and message content parts.

type AnyRecord = Record<string, any>

export function textOf(content: unknown): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .map((part: any) => (typeof part === 'string' ? part : part?.text ?? ''))
      .join('')
  }
  return content == null ? '' : String(content)
}

export function isSystemRole(role: unknown): boolean {
  return role === 'system' || role === 'developer'
}

export function convertTools(tools: unknown): AnyRecord[] | undefined {
  if (!Array.isArray(tools)) return undefined
  const converted: AnyRecord[] = []
  for (const tool of tools) {
    if (tool?.type === 'function' && tool.function?.name) {
      converted.push({
        name: tool.function.name,
        description: tool.function.description || '',
        input_schema: tool.function.parameters || { type: 'object', properties: {} },
      })
    } else if (tool?.name && tool?.input_schema) {
      converted.push(tool)
    } else if (tool?.type === 'custom' && tool.custom?.name) {
      // Freeform tools take raw text; expose them to Claude as a single string argument.
      converted.push({
        name: tool.custom.name,
        description: tool.custom.description || '',
        input_schema: {
          type: 'object',
          properties: { input: { type: 'string' } },
          required: ['input'],
        },
      })
    }
  }
  return converted.length ? converted : undefined
}

export function convertToolChoice(
  toolChoice: unknown,
  parallelToolCalls: unknown,
): AnyRecord | undefined {
  let choice: AnyRecord | undefined
  if (toolChoice === 'auto') choice = { type: 'auto' }
  else if (toolChoice === 'none') choice = { type: 'none' }
  else if (toolChoice === 'required') choice = { type: 'any' }
  else if (toolChoice && typeof toolChoice === 'object') {
    const tc = toolChoice as AnyRecord
    if (tc.type === 'function' && tc.function?.name) choice = { type: 'tool', name: tc.function.name }
    else if (['auto', 'any', 'tool', 'none'].includes(tc.type)) choice = tc
  }
  if (parallelToolCalls === false) {
    choice = { ...(choice || { type: 'auto' }), disable_parallel_tool_use: true }
  }
  return choice
}

function convertContentPart(part: any): AnyRecord | null {
  if (typeof part === 'string') return { type: 'text', text: part }
  if (!part || typeof part !== 'object') return null
  if (part.type === 'text') return part.text ? { type: 'text', text: part.text } : null
  if (part.type === 'image_url') {
    const url: string = part.image_url?.url ?? part.image_url ?? ''
    const match = /^data:([^;]+);base64,(.*)$/s.exec(url)
    return match
      ? { type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } }
      : { type: 'image', source: { type: 'url', url } }
  }
  // Already Anthropic-shaped (image, document, tool_use, tool_result, ...)
  if (part.type === 'tool_use') return { ...part, id: sanitizeToolId(part.id) }
  if (part.type === 'tool_result') return { ...part, tool_use_id: sanitizeToolId(part.tool_use_id) }
  return part
}

function convertContent(content: unknown): AnyRecord[] {
  if (content == null) return []
  if (typeof content === 'string') return content ? [{ type: 'text', text: content }] : []
  if (Array.isArray(content)) {
    return content.map(convertContentPart).filter((p): p is AnyRecord => p !== null)
  }
  return []
}

function parseArguments(args: unknown): unknown {
  if (typeof args !== 'string') return args ?? {}
  try {
    return JSON.parse(args || '{}')
  } catch {
    return { input: args }
  }
}

const TOOL_ID_PATTERN = /^[a-zA-Z0-9_-]+$/
const MAX_TOOL_ID_LENGTH = 64

// 32-bit FNV-1a, enough to keep two sanitized IDs from colliding.
function shortHash(s: string): string {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return (h >>> 0).toString(36)
}

// Anthropic requires tool_use.id / tool_result.tool_use_id to match ^[a-zA-Z0-9_-]+$.
// Cursor's IDs don't always, so map them deterministically: the same input always
// yields the same output, keeping tool_use and tool_result paired.
export function sanitizeToolId(id: unknown): string {
  const raw = id == null ? '' : String(id)
  if (raw && TOOL_ID_PATTERN.test(raw) && raw.length <= MAX_TOOL_ID_LENGTH) return raw
  const cleaned = raw.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, MAX_TOOL_ID_LENGTH - 10)
  return `${cleaned || 'toolu'}_${shortHash(raw)}`
}

export function convertMessages(messages: AnyRecord[]): AnyRecord[] {
  const out: AnyRecord[] = []
  let missingIdCount = 0

  const push = (role: 'user' | 'assistant', blocks: AnyRecord[]) => {
    if (!blocks.length) return
    const last = out[out.length - 1]
    if (last && last.role === role) {
      last.content.push(...blocks)
    } else {
      out.push({ role, content: blocks })
    }
  }

  for (const msg of messages) {
    if (isSystemRole(msg.role)) continue

    if (msg.role === 'tool') {
      push('user', [
        {
          type: 'tool_result',
          tool_use_id: sanitizeToolId(msg.tool_call_id),
          content: textOf(msg.content) || '(no output)',
        },
      ])
      continue
    }

    if (msg.role === 'assistant') {
      const blocks = convertContent(msg.content)
      for (const call of msg.tool_calls || []) {
        const fn = call.function || call.custom || {}
        if (!fn.name) continue
        blocks.push({
          type: 'tool_use',
          id: call.id ? sanitizeToolId(call.id) : `toolu_missing_${++missingIdCount}`,
          name: fn.name,
          input: call.custom ? { input: fn.input ?? '' } : parseArguments(fn.arguments),
        })
      }
      push('assistant', blocks)
      continue
    }

    push('user', convertContent(msg.content))
  }

  return repairToolPairing(out)
}

// Anthropic rejects a conversation unless every tool_use is answered by a tool_result
// in the very next user turn, and every tool_result answers a tool_use in the turn
// right before it. Cancelled or interrupted Cursor runs break that, so fix it up:
// add placeholder results for unanswered calls and drop results with no call.
function repairToolPairing(messages: AnyRecord[]): AnyRecord[] {
  const out: AnyRecord[] = []
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]

    if (msg.role === 'assistant') {
      // Duplicate tool_use ids are also rejected; keep the first.
      const seen = new Set<string>()
      msg.content = msg.content.filter((b: AnyRecord) => {
        if (b.type !== 'tool_use') return true
        if (seen.has(b.id)) return false
        seen.add(b.id)
        return true
      })
      if (!msg.content.length) continue
      out.push(msg)

      if (!seen.size) continue
      let next = messages[i + 1]
      if (!next || next.role !== 'user') {
        next = { role: 'user', content: [] }
        messages.splice(i + 1, 0, next)
      }
      const answered = new Set(
        next.content.filter((b: AnyRecord) => b.type === 'tool_result').map((b: AnyRecord) => b.tool_use_id),
      )
      for (const id of seen) {
        if (!answered.has(id)) {
          next.content.push({
            type: 'tool_result',
            tool_use_id: id,
            content: '(tool call was cancelled before it returned a result)',
            is_error: true,
          })
        }
      }
      continue
    }

    // User turn: keep only results that answer the preceding assistant turn.
    const prev = out[out.length - 1]
    const validIds = new Set(
      prev?.role === 'assistant'
        ? prev.content.filter((b: AnyRecord) => b.type === 'tool_use').map((b: AnyRecord) => b.id)
        : [],
    )
    const usedIds = new Set<string>()
    msg.content = msg.content.filter((b: AnyRecord) => {
      if (b.type !== 'tool_result') return true
      if (!validIds.has(b.tool_use_id) || usedIds.has(b.tool_use_id)) return false
      usedIds.add(b.tool_use_id)
      return true
    })
    // Tool results must come first in their user turn.
    msg.content.sort(
      (a: AnyRecord, b: AnyRecord) =>
        Number(b.type === 'tool_result') - Number(a.type === 'tool_result'),
    )
    if (!msg.content.length) continue

    const last = out[out.length - 1]
    if (last?.role === 'user') last.content.push(...msg.content)
    else out.push(msg)
  }
  return out
}
