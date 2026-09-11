# opencode client configuration

[中文](opencode-config-zh.md)

Point opencode at your gateway. The provider id, `baseURL` and the login command all
reference `llm.example.com` — replace it with your own gateway domain, then save the
result as `~/.config/opencode/opencode.jsonc`.

## Config

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llm.example.com": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "New API",
      "options": {
        "baseURL": "https://llm.example.com/v1"
      },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen3.8 27B",
          "tool_call": true,
          "reasoning": true,
          "attachment": true,
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          },
          "limit": {
            "context": 262144,
            "output": 20000
          }
        }
      }
    }
  },
  "attachment": {
    "image": {
      "auto_resize": true,
      "max_width": 1448,
      "max_height": 1448,
      "max_base64_bytes": 10485760
    }
  },
  "compaction": {
    "reserved": 16000,
    "prune": true
  },
  "share": "disabled"
}
```

## Steps

1. Install opencode and launch it.
2. **Authenticate.** A custom provider id is not in the `/connect` models.dev list, so
   use either:
   - **CLI (recommended):** `opencode auth login -p llm.example.com` (with *your*
     domain), then paste a key created on the gateway's **Tokens** page (prefix `sk-`).
     Credentials are stored in `~/.local/share/opencode/auth.json`.
   - **Config:** add `"apiKey": "sk-..."` inside `options` (plaintext, discouraged), or
     `"apiKey": "{env:NEWAPI_API_KEY}"` with `export NEWAPI_API_KEY=sk-...`.
3. Select **Qwen3.8 27B** (`qwen3.8-27b`) in `/models`.
4. Restart opencode — the config is not hot-reloaded.

## Limits

| Item | Value |
| --- | --- |
| Model id | `qwen3.8-27b` |
| Context | 262144 tokens |
| Max output | 20000 tokens |
| Tools / reasoning / vision | supported |
| Images | effectively capped by the context window (~2050 tokens per 2 Mpx image, i.e. ~127 images); server count guard is 128/request; 2 Mpx area cap |
| Video | not supported |

`attachment.image.max_width/height = 1448` approximates the server's 2 Mpx area cap
(`--mm-process-config` `longest_edge=2097152`, which is a pixel **area**, not an edge
length). The client resizes first so the server does not downscale again.
