# opencode 客户端配置

[English](opencode-config.md)

让 opencode 接入你自己的网关。provider id 是 `llm-example-com`、`baseURL` 是
`https://llm.example.com/v1`,**两者都要换成你自己的**(opencode 的 provider id 不能含
`.`),然后保存为 `~/.config/opencode/opencode.jsonc`。

## 配置

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llm-example-com": {
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

## 步骤

1. 安装 opencode 并启动。
2. **认证。** 自定义 provider 不在 `/connect` 的 models.dev 列表里,用下面任一方式:
   - **CLI(推荐):** `opencode auth login -p llm-example-com`(换成*你的* id),粘贴
     网关「令牌」页创建的 key(带 `sk-` 前缀)。凭据存于
     `~/.local/share/opencode/auth.json`。
   - **写进配置:** 在 `options` 内加 `"apiKey": "sk-..."`(明文,慎用),或
     `"apiKey": "{env:NEWAPI_API_KEY}"` 并 `export NEWAPI_API_KEY=sk-...`。
3. 在 `/models` 里选 **Qwen3.8 27B**(`qwen3.8-27b`)。
4. 重启 opencode —— 配置不热重载。

## 限制

| 项 | 值 |
| --- | --- |
| 模型 id | `qwen3.8-27b` |
| 上下文 | 262144 token |
| 单次最大输出 | 20000 token |
| 工具 / 思考 / 视觉 | 支持 |
| 图片 | 实际上限由上下文窗口决定(2 Mpx 每张 ≈2050 token,约 127 张封顶;服务端计数护栏 128);2 Mpx 面积上限 |
| 视频 | 不支持 |

`attachment.image.max_width/height = 1448` 近似服务端的 2 Mpx 面积上限
(`--mm-process-config` 的 `longest_edge=2097152` 是像素**面积**,不是边长)。客户端先
缩好,服务端便不再二次降采样。
