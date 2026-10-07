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

export function convertMessages(messages: AnyRecord[]): AnyRecord[] {
  const out: AnyRecord[] = []

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
          tool_use_id: msg.tool_call_id,
          content: textOf(msg.content) || '(no output)',
        },
      ])
      continue
    }

    if (msg.role === 'assistant') {
      const blocks = convertContent(msg.content)
      for (const call of msg.tool_calls || []) {
        const fn = call.function || call.custom || {}
        blocks.push({
          type: 'tool_use',
          id: call.id,
          name: fn.name,
          input: call.custom ? { input: fn.input ?? '' } : parseArguments(fn.arguments),
        })
      }
      push('assistant', blocks)
      continue
    }

    push('user', convertContent(msg.content))
  }

  // Tool results must come first in their user turn.
  for (const m of out) {
    if (m.role === 'user' && m.content.some((b: AnyRecord) => b.type === 'tool_result')) {
      m.content.sort(
        (a: AnyRecord, b: AnyRecord) =>
          Number(b.type === 'tool_result') - Number(a.type === 'tool_result'),
      )
    }
  }
  return out
}
