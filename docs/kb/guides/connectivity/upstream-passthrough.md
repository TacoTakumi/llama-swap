---
title: Upstream API passthrough
summary: Control which /upstream model paths are ignored instead of triggering a model swap or counting as model activity.
category: guides
tags: [upstream, proxy, api, ttl, polling]
config_keys: [upstream, upstream.ignorePaths]
updated: 2026-10-28
---

# Upstream API passthrough

`upstream` controls the `/upstream/<model>/<path>` passthrough endpoint. Its
`ignorePaths` expressions prevent static or UI paths from triggering a model
swap; it does not configure a fallback OpenAI service for unmatched requests.

Use RE2-compatible expressions and keep the default static-asset patterns
unless a client needs different paths. Check logs to distinguish an ignored
path from a model-routing request.

## What an ignored path does

A request whose path matches an `ignorePaths` expression is kept outside the
model's lifecycle, the same way `compat.ignoreWebsockets` keeps a websocket
out of it:

| model state | result |
| --- | --- |
| not loaded (stopped, starting, stopping) | `409 Conflict`, `path matches upstream.ignorePaths`; nothing starts |
| loaded | forwarded straight to the process: no concurrency slot, not counted in flight, no `ttl` reset |

So a client that polls an ignored path (a UI tab, a metrics scraper, a health
probe, a control call the engine must answer itself) cannot keep the model
loaded past its `ttl`, and is not refused with a `429` while the model's
`concurrencyLimit` is in use. Every other path through `/upstream/` is an
ordinary request: it starts the model and counts as activity.

`ignorePaths` matches every method. A `POST` to a matched path is ignored the
same way as a `GET`.

```yaml
upstream:
  ignorePaths:
    # listing any pattern replaces the default, so repeat it
    - '.*\.(js|json|css|png|gif|jpg|jpeg|ico|txt)$'
    # a scraper or probe that must not keep the model loaded
    - ^/metrics$
    - ^/health$

models:
  qwen-coder:
    ttl: 300
    cmd: llama-server --port ${PORT} -m /models/qwen.gguf --metrics
```

With this, `GET /upstream/qwen-coder/metrics` every 15 s answers `409` while
the model is unloaded and real metrics while it is loaded, and the model still
unloads 300 s after its last inference request.

## What goes wrong

- **A model never unloads.** A client is polling a path that is not in
  `ignorePaths`. Every such request resets the `ttl` timer. Add the path.
- **`409` on a path that should start the model.** The path matches a pattern,
  often the default static-asset one (`/index.html` does not match it,
  `/app.js` does). Narrow the pattern or send a request to a path that is not
  ignored first.
- **A polled path does not match.** `ignorePaths` is matched against the path
  after `/upstream/<model>` is stripped, so write `^/metrics$`, not
  `^/upstream/.*/metrics$`.

## Related

- `guides/upstreams/comfyui` - the `/comfyui/` endpoint applies a built-in list
- `guides/model-runtime/ttl-and-unloading` - what resets the `ttl` timer
