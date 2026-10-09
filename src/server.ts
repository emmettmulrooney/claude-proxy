import { Hono, Context, Next } from 'hono'
import { logger } from 'hono/logger'
import { stream } from 'hono/streaming'
import { createHash, timingSafeEqual } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { getAccessToken } from './auth/oauth-manager'
import { getClaudeQuota } from './auth/usage'
import {
  getStats,
  getSummary,
  log,
  newRequestId,
  recordRequest,
  type RequestRecord,
} from './observability'
import {
  login as oauthLogin,
  logout as oauthLogout,
  generateAuthSession,
  handleOAuthCallback,
} from './auth/oauth-flow'
import {
  createConverterState,
  processChunk,
  convertNonStreamingResponse,
} from './utils/anthropic-to-openai-converter'
import {
  addCacheBreakpoints,
  convertMessages,
  convertToolChoice,
  convertTools,
  isSystemRole,
  textOf,
} from './utils/openai-to-anthropic-converter'
import { corsPreflightHandler, corsMiddleware } from './utils/cors-bypass'
import {
  isCursorKeyCheck,
  createCursorBypassResponse,
} from './utils/cursor-byok-bypass'
import type {
  AnthropicRequestBody,
  AnthropicResponse,
  ErrorResponse,
  SuccessResponse,
  ModelsListResponse,
  ModelInfo,
} from './types'

// Static files are served by Vercel, not needed here

const app = new Hono()

app.use('*', logger())

// Handle CORS preflight requests for all routes
app.options('*', corsPreflightHandler)

// Also add CORS headers to all responses
app.use('*', corsMiddleware)

// Hashing both sides first makes the lengths equal, which timingSafeEqual requires.
const keysMatch = (given: string, expected: string) =>
  timingSafeEqual(
    createHash('sha256').update(given).digest(),
    createHash('sha256').update(expected).digest(),
  )

const requireApiKey = async (c: Context, next: Next) => {
  const expected = process.env.API_KEY
  if (!expected) return next()

  const header = c.req.header('authorization') || c.req.header('x-api-key') || ''
  const given = header.replace(/^Bearer\s+/i, '')
  if (!given || !keysMatch(given, expected)) {
    return c.json<ErrorResponse>(
      { error: 'Unauthorized', message: 'Invalid API key' },
      401,
    )
  }
  return next()
}

app.use('/v1/*', requireApiKey)
app.use('/auth/oauth/*', requireApiKey)
app.use('/auth/login/*', requireApiKey)
app.use('/auth/logout', requireApiKey)
app.use('/obs/*', requireApiKey)

const indexHtmlPath = join(process.cwd(), 'public', 'index.html')
let cachedIndexHtml: string | null = null

const getIndexHtml = async () => {
  if (!cachedIndexHtml) {
    cachedIndexHtml = await readFile(indexHtmlPath, 'utf-8')
  }
  return cachedIndexHtml
}

// Root route is handled by serving public/index.html directly
app.get('/', async (c) => {
  const html = await getIndexHtml()
  return c.html(html)
})

app.get('/index.html', async (c) => {
  const html = await getIndexHtml()
  return c.html(html)
})

// New OAuth start endpoint for UI
app.post('/auth/oauth/start', async (c: Context) => {
  try {
    const { authUrl, sessionId } = await generateAuthSession()

    return c.json({
      success: true,
      authUrl,
      sessionId,
    })
  } catch (error) {
    return c.json<ErrorResponse>(
      {
        error: 'Failed to start OAuth flow',
        message: (error as Error).message,
      },
      500,
    )
  }
})

// New OAuth callback endpoint for UI
app.post('/auth/oauth/callback', async (c: Context) => {
  try {
    const body = await c.req.json()
    const { code } = body

    if (!code) {
      return c.json<ErrorResponse>(
        {
          error: 'Missing OAuth code',
          message: 'OAuth code is required',
        },
        400,
      )
    }

    // Extract verifier from code if it contains #
    const splits = code.split('#')
    const verifier = splits[1] || ''

    await handleOAuthCallback(code, verifier)

    return c.json<SuccessResponse>({
      success: true,
      message: 'OAuth authentication successful',
    })
  } catch (error) {
    return c.json<ErrorResponse>(
      {
        error: 'OAuth callback failed',
        message: (error as Error).message,
      },
      500,
    )
  }
})

app.post('/auth/login/start', async (c: Context) => {
  try {
    console.log('\n Starting OAuth authentication flow...')
    const result = await oauthLogin()
    if (result) {
      return c.json<SuccessResponse>({
        success: true,
        message: 'OAuth authentication successful',
      })
    } else {
      return c.json<SuccessResponse>(
        { success: false, message: 'OAuth authentication failed' },
        401,
      )
    }
  } catch (error) {
    return c.json<SuccessResponse>(
      { success: false, message: (error as Error).message },
      500,
    )
  }
})

app.get('/auth/logout', async (c: Context) => {
  try {
    await oauthLogout()
    return c.json<SuccessResponse>({
      success: true,
      message: 'Logged out successfully',
    })
  } catch (error) {
    return c.json<SuccessResponse>(
      { success: false, message: (error as Error).message },
      500,
    )
  }
})

app.get('/auth/status', async (c: Context) => {
  try {
    const token = await getAccessToken()
    return c.json({ authenticated: !!token })
  } catch (error) {
    return c.json({ authenticated: false })
  }
})

app.get('/auth/usage', async (c: Context) => {
  try {
    const quota = await getClaudeQuota()
    if (!quota) {
      return c.json<ErrorResponse>(
        {
          error: 'Authentication required',
          message: 'Connect Claude before checking quota.',
        },
        401,
      )
    }
    return c.json(quota)
  } catch (error) {
    return c.json<ErrorResponse>(
      { error: 'Usage unavailable', message: (error as Error).message },
      502,
    )
  }
})

app.get('/v1/models', async (c: Context) => {
  try {
    // Fetch models from models.dev
    const response = await fetch('https://models.dev/api.json', {
      method: 'GET',
      headers: {
        accept: 'application/json',
        'user-agent': '@anthropic-ai/sdk 1.2.12 node/22.13.1',
      },
    })

    if (!response.ok) {
      const error = await response.text()
      log('models_error', { status: response.status, message: error.slice(0, 300) })
      return new Response(error, {
        status: response.status,
        headers: { 'Content-Type': 'text/plain' },
      })
    }

    const modelsData = (await response.json()) as any

    // Extract Anthropic models and format them like OpenAI's API would
    const anthropicProvider = modelsData.anthropic
    if (!anthropicProvider || !anthropicProvider.models) {
      return c.json<ModelsListResponse>({
        object: 'list',
        data: [],
      })
    }

    // Convert models to OpenAI's format
    const models: ModelInfo[] = Object.entries(anthropicProvider.models).map(
      ([modelId, modelData]: [string, any]) => {
        // Convert release date to Unix timestamp
        const releaseDate = modelData.release_date || '1970-01-01'
        const created = Math.floor(new Date(releaseDate).getTime() / 1000)

        return {
          id: modelId,
          object: 'model' as const,
          created: created,
          owned_by: 'anthropic',
        }
      },
    )

    // Sort models by created timestamp (newest first)
    models.sort((a, b) => b.created - a.created)

    for (const id of Object.keys(MODEL_ALIASES)) {
      models.unshift({ id, object: 'model', created: 0, owned_by: 'anthropic' })
    }

    const response_data: ModelsListResponse = {
      object: 'list',
      data: models,
    }

    return c.json(response_data)
  } catch (error) {
    log('models_error', { message: (error as Error).message })
    return c.json<ErrorResponse>(
      { error: 'Proxy error', details: (error as Error).message },
      500,
    )
  }
})

// Short names for Cursor's custom-model list. Cursor sends its own built-in Claude
// models to Anthropic directly, so a name that doesn't start with "claude" is what
// makes it route through the proxy.
const MODEL_ALIASES: Record<string, { model: string; effort: string }> = {
  'proxy-opus': { model: 'claude-opus-5-5', effort: 'medium' },
}

const ANTHROPIC_FIELDS = new Set([
  'model',
  'messages',
  'system',
  'max_tokens',
  'metadata',
  'stop_sequences',
  'stream',
  'temperature',
  'top_p',
  'top_k',
  'tools',
  'tool_choice',
  'thinking',
  'output_config',
])

type MetricsUsage = {
  input_tokens: number
  output_tokens: number
  cache_creation_input_tokens: number
  cache_read_input_tokens: number
}

// Direct call to Claude with the stored OAuth token, for the proxy's own needs (the Haiku summary).
const callClaude = async (body: Record<string, unknown>): Promise<Response> => {
  const token = await getAccessToken()
  if (!token) throw new Error('Not authenticated with Claude')
  return fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${token}`,
      'anthropic-beta': 'oauth-2025-04-20',
      'anthropic-version': '2023-06-01',
      'user-agent': '@anthropic-ai/sdk 1.2.12 node/22.13.1',
    },
    body: JSON.stringify(body),
  })
}

// Observability: raw stats for agents/scripts, and a Haiku-written summary for people.
app.get('/obs/stats', async (c) => {
  try {
    return c.json(await getStats())
  } catch (error) {
    log('obs_error', { route: 'stats', message: (error as Error).message })
    return c.json<ErrorResponse>({ error: 'Stats unavailable', message: (error as Error).message }, 502)
  }
})

app.get('/obs/summary', async (c) => {
  try {
    return c.json(await getSummary(callClaude, c.req.query('fresh') === '1'))
  } catch (error) {
    log('obs_error', { route: 'summary', message: (error as Error).message })
    return c.json<ErrorResponse>({ error: 'Summary unavailable', message: (error as Error).message }, 502)
  }
})

const prefixHash = (v: unknown) =>
  v == null ? undefined : createHash('sha256').update(JSON.stringify(v)).digest('hex').slice(0, 8)

const messagesFn = async (c: Context) => {
  let headers: Record<string, string> = c.req.header() as Record<string, string>
  headers.host = 'api.anthropic.com'
  const body: AnthropicRequestBody = await c.req.json()
  const isStreaming = body.stream === true

  const rec: RequestRecord = {
    id: newRequestId(),
    t: Date.now(),
    ms: 0,
    path: c.req.path,
    model: String(body.model),
    stream: isStreaming,
    status: 0,
    in: 0,
    cache_read: 0,
    cache_write: 0,
    uncached: 0,
    out: 0,
    messages: Array.isArray(body.messages) ? body.messages.length : 0,
  }
  c.header('x-proxy-request-id', rec.id)
  const finish = (status: number, usage?: Partial<MetricsUsage>, error?: string) => {
    rec.status = status
    rec.ms = Date.now() - rec.t
    if (usage) {
      rec.uncached = usage.input_tokens || 0
      rec.cache_read = usage.cache_read_input_tokens || 0
      rec.cache_write = usage.cache_creation_input_tokens || 0
      rec.out = usage.output_tokens || 0
      rec.in = rec.uncached + rec.cache_read + rec.cache_write
    }
    if (error) rec.error = error.slice(0, 500)
    void recordRequest(rec)
  }

  const alias = MODEL_ALIASES[body.model]
  if (alias) {
    body.model = alias.model
    body.output_config = {
      ...((body.output_config as object) || {}),
      effort: alias.effort,
    }
    // Opus 5.5 always thinks, and thinking requires the default sampling settings.
    delete body.temperature
    delete body.top_p
    delete body.top_k
  }

  // Bypass cursor enable openai key check
  if (isCursorKeyCheck(body)) {
    return c.json(createCursorBypassResponse())
  }

  try {
    let transformToOpenAIFormat = false

    if (
      !body.system?.[0]?.text?.includes(
        "You are Claude Code, Anthropic's official CLI for Claude.",
      ) && body.messages
    ) {
      const systemMessages = body.messages.filter((msg: any) => isSystemRole(msg.role))
      body.messages = convertMessages(body.messages)
      const tools = convertTools(body.tools)
      if (tools) body.tools = tools
      else delete body.tools
      const toolChoice = tools
        ? convertToolChoice(body.tool_choice, body.parallel_tool_calls)
        : undefined
      if (toolChoice) body.tool_choice = toolChoice
      else delete body.tool_choice
      transformToOpenAIFormat = true // not claude-code, need to transform to openai format
      if (!body.system) {
        body.system = []
      }
      body.system.unshift({
        type: 'text',
        text: "You are Claude Code, Anthropic's official CLI for Claude.",
      })

      for (const sysMsg of systemMessages) {
        if (!textOf(sysMsg.content)) continue
        body.system.push({
          type: 'text',
          text: textOf(sysMsg.content),
        })
      }

      if (body.stop && !body.stop_sequences) {
        body.stop_sequences = Array.isArray(body.stop) ? body.stop : [body.stop]
      }
      if (body.max_completion_tokens && !body.max_tokens) {
        body.max_tokens = body.max_completion_tokens
      }
      // Anthropic rejects unknown fields, and OpenAI clients send many (stream_options, n, user, ...).
      for (const key of Object.keys(body)) {
        if (!ANTHROPIC_FIELDS.has(key)) delete body[key]
      }

      if (body.model.includes('opus')) {
        body.max_tokens = 32_000
      }
      if (body.model.includes('sonnet')) {
        body.max_tokens = 64_000
      }
      if (!body.max_tokens) {
        body.max_tokens = 8_192
      }
      addCacheBreakpoints(body)
    }
    // If these change between steps of one agent run, the cache can't hit.
    rec.model = String(body.model)
    rec.tools_hash = prefixHash(body.tools)
    rec.system_hash = prefixHash(body.system)

    const oauthToken = await getAccessToken()

    if (!oauthToken) {
      finish(401, undefined, 'no OAuth token')
      return c.json<ErrorResponse>(
        {
          error: 'Authentication required',
          message:
            'Please authenticate using OAuth first. Visit /auth/login for instructions.',
        },
        401,
      )
    }

    headers = {
      'content-type': 'application/json',
      authorization: `Bearer ${oauthToken}`,
      'anthropic-beta':
        'oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14',
      'anthropic-version': '2023-06-01',
      'user-agent': '@anthropic-ai/sdk 1.2.12 node/22.13.1',
      accept: isStreaming ? 'text/event-stream' : 'application/json',
      'accept-encoding': 'gzip, deflate',
    }

    if (transformToOpenAIFormat) {
      if (!body.metadata) {
        body.metadata = {}
      }

      if (!body.system) {
        body.system = []
      }
    }

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const error = await response.text()
      finish(response.status, undefined, error)

      if (response.status === 401) {
        return c.json<ErrorResponse>(
          {
            error: 'Authentication failed',
            message:
              'OAuth token may be expired. Please re-authenticate using /auth/login/start',
            details: error,
          },
          401,
        )
      }
      return new Response(error, {
        status: response.status,
        headers: { 'Content-Type': 'text/plain', 'x-proxy-request-id': rec.id },
      })
    }

    if (isStreaming) {
      response.headers.forEach((value, key) => {
        if (
          key.toLowerCase() !== 'content-encoding' &&
          key.toLowerCase() !== 'content-length' &&
          key.toLowerCase() !== 'transfer-encoding'
        ) {
          c.header(key, value)
        }
      })

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()

      return stream(c, async (stream) => {
        const converterState = createConverterState()
        // An SSE line can be split across network reads; hold back the unfinished tail.
        let carry = ''
        let streamError: string | undefined

        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            const raw = decoder.decode(value, { stream: true })
            const text = carry + raw
            const cut = text.lastIndexOf('\n')
            const complete = cut === -1 ? '' : text.slice(0, cut + 1)
            carry = cut === -1 ? text : text.slice(cut + 1)

            if (!transformToOpenAIFormat) {
              // Claude-format clients get the bytes untouched; parse only to read usage.
              await stream.write(raw)
              if (complete) processChunk(converterState, complete)
              continue
            }
            if (!complete) continue

            for (const result of processChunk(converterState, complete)) {
              if (result.type === 'chunk') {
                await stream.write(`data: ${JSON.stringify(result.data)}\n\n`)
              } else if (result.type === 'done') {
                await stream.write('data: [DONE]\n\n')
              }
            }
          }
        } catch (error) {
          streamError = `stream: ${(error as Error).message}`
        } finally {
          reader.releaseLock()
          finish(200, converterState.metricsData, streamError)
        }
      })
    } else {
      const responseData = (await response.json()) as AnthropicResponse
      finish(200, responseData.usage)

      response.headers.forEach((value, key) => {
        if (key.toLowerCase() !== 'content-encoding') {
          c.header(key, value)
        }
      })

      if (transformToOpenAIFormat) {
        return c.json(convertNonStreamingResponse(responseData))
      }
      return c.json(responseData)
    }
  } catch (error) {
    finish(500, undefined, `proxy: ${(error as Error).message}`)
    return c.json<ErrorResponse>(
      { error: 'Proxy error', details: (error as Error).message },
      500,
    )
  }
}

app.post('/v1/chat/completions', messagesFn)
app.post('/v1/messages', messagesFn)

export default app
