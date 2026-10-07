import { serve } from '@hono/node-server'
import app from './server'

const port = Number(process.env.PORT) || 9095

serve({ fetch: app.fetch, port }, (info) => {
  console.log(`Claude proxy listening on http://localhost:${info.port}`)
})
