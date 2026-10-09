import { getAccessToken } from './oauth-manager'

export interface UsageWindow {
  utilization: number
  resets_at: string | null
}

export interface ClaudeQuota {
  five_hour: UsageWindow | null
  seven_day: UsageWindow | null
}

const USAGE_URL = 'https://api.anthropic.com/api/oauth/usage'
const CACHE_MS = 60_000

let cached: { at: number; quota: ClaudeQuota } | null = null
let pending: Promise<ClaudeQuota> | null = null

const windowOf = (
  raw: { utilization?: number; resets_at?: string | null } | null | undefined,
): UsageWindow | null => {
  if (!raw || typeof raw.utilization !== 'number') return null
  return { utilization: raw.utilization, resets_at: raw.resets_at ?? null }
}

const fetchQuota = async (token: string): Promise<ClaudeQuota> => {
  // A claude-code user agent is what this endpoint rate-limits generously.
  const response = await fetch(USAGE_URL, {
    headers: {
      authorization: `Bearer ${token}`,
      'anthropic-beta': 'oauth-2025-04-20',
      'user-agent': 'claude-code/2.1.72',
      accept: 'application/json',
    },
  })

  if (response.status === 429 && cached) return cached.quota

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      error?: { message?: string }
    } | null
    const message =
      response.status === 429
        ? 'Claude is rate-limiting usage checks. Try again in a minute.'
        : body?.error?.message || `Usage request failed (${response.status})`
    throw new Error(message)
  }

  const data = (await response.json()) as {
    five_hour?: { utilization?: number; resets_at?: string | null } | null
    seven_day?: { utilization?: number; resets_at?: string | null } | null
  }

  const quota: ClaudeQuota = {
    five_hour: windowOf(data.five_hour),
    seven_day: windowOf(data.seven_day),
  }
  cached = { at: Date.now(), quota }
  return quota
}

export const getClaudeQuota = async (): Promise<ClaudeQuota | null> => {
  const token = await getAccessToken()
  if (!token) return null

  if (cached && Date.now() - cached.at < CACHE_MS) return cached.quota
  if (!pending) {
    pending = fetchQuota(token).finally(() => {
      pending = null
    })
  }
  return pending
}
