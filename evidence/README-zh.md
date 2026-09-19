# 证据

[English](README.md)

支撑顶层 [README](../README-zh.md) 中数字的实测结果。所有测量均在目标卡上进行
(单张 32 GB Blackwell sm120、NVFP4 权重、262144 token 池)。

## 吞吐 vs 上下文

`tools/benchmark.py`,开启思考,单流。

| 上下文 | 2K | 4K | 64K | 128K | 240K |
| --- | --- | --- | --- | --- | --- |
| prefill(tok/s) | 5716 | 5737 | 3300 | 2203 | 1386 |
| decode(tok/s) | 42.0 | 41.8 | 39.3 | 37.0 | 33.2 |
| TTFT(s) | 0.38 | 0.74 | 19.2 | 57.5 | 171.3 |

prompt token 数:2160 / 4233 / 63432 / 126778 / 237477。2K/4K 的 decode 由 16 个生成
token 测得;更长上下文用 128-256 个。

## 并发与稳定性(NVFP4,mrr 4)

`tools/concurrency-load.py`。

| 场景 | 结果 |
| --- | --- |
| 4 x 1K,2048 gen | batch-4 decode 中位 152.3 tok/s,eager 0 |
| 2 x 48K | batch-2 中位 72.6 tok/s,retract 0 |
| 4 x 48K | batch-4 中位 128.2 tok/s,wall 114.9 s,retract 0 |
| 2 x 80K + 2 x 40K(~240K) | wall 143.2 s,retract/assert/fatal 0 |
| 单流 1K / 64K | 42.4 / 39.6 tok/s |
| cached TTFT | 0.06 s(cached 2176 / 2189) |
| soak 600 s | 106 请求,0 错误,0 重启 |

在加上 `--mamba-max-states-per-path 2` 之前,soak 会在 1-2 分钟内崩溃
(`Can not alloc mamba cache`);见顶层 README「已知坑」。

## 调度器假闩锁修复

打补丁前,`mrr=4` 只能真正准入 3 路。挂载 `inference/patches/sched-latch-fix` 后:

| 场景 | 结果 |
| --- | --- |
| 4 x 1K | 全部准入,batch-4 中位 153-158 tok/s |
| 6 x 1K(超发) | 3 运行 + 3 排队,零崩溃 |
| 4 x 48K | wall 108-115 s,retract 0 |
| soak 600 s | 106 请求,0 错误 |

钩子在受影响的 pass 上打印 `suppressed false latch`。

## GSM8K

`tools/gsm8k.py`,开启思考,temp 1.0 / top_p 0.95 / top_k 20,前 200 题。

| 指标 | 值 |
| --- | --- |
| 正确 | 195 / 200 |
| 准确率 | **97.5%** |
| wall | 718.7 s |
| 量化方参考(全 1319 题) | 97.27% |

## 质量 spot-check

`tools/quality-spotcheck.py`,确定性(temperature 0)。

| 用例 | 结果 |
| --- | --- |
| 12 个短 prompt(数学、逻辑、翻译、JSON、代码、格式) | 12/12 正确 |
| ~57.6K prompt token 的长上下文 needle | 正确 |

## RULER 长上下文

completion 协议(`tools/ruler-niah.py`、`ruler-niah-multi.py`、`ruler-fwe.py`)。

### 单针 NIAH(needle 深度 90%)

| 长度 | 得分 |
| --- | --- |
| 32K | 2/2 |
| 64K | 2/2 |
| 128K | 2/2 |
| 240K | 2/2 |

### 多针 / multikey / multivalue @ 240K

| 任务 | 得分 |
| --- | --- |
| niah_single_1 | 100% |
| niah_single_2 | 100% |
| niah_single_3 | 100% |
| niah_multikey_1 | 100% |
| niah_multivalue | 100% |

### freq_words_extraction(最难聚合)

| 长度 | 得分 |
| --- | --- |
| 32K | 2/2 |
| 240K | 2/2 |

## NVFP4 vs FP8

同一部署,仅 KV 缓存 dtype(与视觉)不同。

| 场景 | FP8 | NVFP4 |
| --- | --- | --- |
| 单流 decode @1K | 42.6 | 42.4 tok/s |
| 单流 decode @64K | 38.5 | 39.6 tok/s |
| cached TTFT | 0.05 s | 0.06 s |
| batch 4 @1K | 153.4 | 152.3 tok/s |
| batch 4 @48K | 118.6 | 128.2 tok/s |
| soak 600 s | 106 req / 0 err | 106 req / 0 err |

## FP8 KV + 视觉 @32 GB(当前默认档,2026-09-11)

启动账:KV 池 262144 @ 8.0 GB,mamba 池 8 @ 0.65 GB,decode 图捕获后余 ~1.91 GB。
prefill 图 关 vs 开:0 ~ +1%(1K/14K/70K;TTFT 完全一致)。

| 场景 | 结果 |
| --- | --- |
| 166K 冷 prefill | 87.3 s(1905 tok/s),retract 0 |
| 单流 decode @1K(mrr 2) | 41~43 tok/s |
| mamba 槽,双流进行中 | used 5 / evictable 2 / available 1(池 8) |
| 图片 1..96 张(2 Mpx/张) | 全过,GPU 峰值恒定(~31168/32623 MiB) |
| 图片 128 张 | HTTP 400,来自 262144 上下文检查(262453 token) |
| 质量 spot check(含 104K 单针) | 通过 |

## 乱码调查工具链(NVFP4 时代,2026-09-11)

在一次 mrr 4 被离线打分流量打满、mamba 池钉在 `available = 0` 的窗口里,一个交互会话
间歇性产出 token-soup 式乱码 reasoning(最严重一次失控到 20000 token)。两轮受控饱和
(`tools/mamba-saturation-probe.py`:RM 式压力 worker + 长/短多轮受害者 + 在线乱码检测器)
保持了 26 分钟 `usage 0.75 / available 0`,且驱逐/换入确凿被反复触发(受害者轮间
`cached_tokens` 塌陷),**29 轮 0 命中**。

结论:仅靠槽饱和不足以复现;当前主解释是 FP4-KV 的稀有采样脱轨,被客户端把乱码写回
历史后自喂放大。默认档已切换为 FP8 KV,整类问题(连同 page 64 / trtllm_mha)一并移除。

## mamba stash 崩溃与 E10 修复(2026-09-15)

v0.5.19 切换后 13h 的生产崩溃:`stash_chunked_request` 期间 `_alloc_mamba_slot` 断言
(池 8;`extra_buffer_lazy` 准入系数 2 < 每请求峰值 3 槽——上游 FIXME 与单测把此钉为
fail-loud 设计)。修复 = E10:`--mamba-radix-cache-strategy extra_buffer` +
`--max-mamba-cache-size 10`(上游自动定容 ratio 5 × mrr 2)。KV 池维持 262144 token 不变
(受 flag 卡而非内存卡;+0.16GB 被静态预算余量吸收)。孪生验证、回归数据与新验收件
`verify-mamba-stash.py`(T3,镜像升级门禁):[`mamba-stash-T3-0915/`](mamba-stash-T3-0915/)。

## HiCache 混合 Mamba 修复与 host 池定容(2026-09-17)

v0.5.19 的 HiCache 对混合 GDN(Mamba)模型基本没有真正复用 host 层:190K 续传仍要
~109 s,且 host-hit 指标虚高(设备常驻 token 被记到了 host 层)。根因是三个上游缺口:
chunked prefill 不入写通备份、mamba 锚点池按上游判据 `262144/cps` 定容不足、分配饥饿
可断言打崩调度器。打上 `hicache-mamba-fix` 补丁集 + `cps 6144` +
`SGLANG_HICACHE_MAMBA_SIZE_GB=7.0` 后:被逐出的 36.6K 会话以 **0.26 s** 从 host RAM
回来(全量重算 8.76 s,34×),分支场景 25.1 s → 3.09 s,2×68.5K 并发 prefill 无异常。
回归门禁(`verify-hicache-thrash.py`)现在要求真实的 host 装载
(`sglang:load_back_tokens_total{pool="kv"}` 增量),而非只是响应快。
证据:[`hicache-mamba-fix-0917/`](hicache-mamba-fix-0917/)。

## SGLang v0.5.20 迁移与 HRRN 观察(2026-09-19)

补丁构建重锚到 v0.5.20(`llm-infer:hicache-06e4f2ed`,树 `94602c9`):
false-latch 锚点 scheduler.py:3661 → 3887(门控结构未变,上游仍未修);
P1/P3/C 继续携带(#36647/#36770 仍 open;P3 跟随上游
`mamba_component.py` → `components/mamba.py` 改名);**P2 退役** —— v0.5.20
已在上游吸收 honest host-hit 语义(`host_loaded_length` +
`materialized_host_hit_len()`)。LPM waitfix 故意不再烘焙:v0.5.20 自带
`--schedule-policy hrrn`(aging 策略),以观察代替补丁。

实机验收:T3 stash 门禁 `--expect safe` 2/2 PASS(0 崩溃行、0 重启);
hicache thrash PASS(244K 冷 124 s → host 重载 35 s、`load_back` +136K →
续传 0.4 s);latch T1 行为学 PASS。饥饿场景中确认 HRRN aging 生效
(冷请求约 1 个回合即被准入,抢在一路热续传之前)。
证据:[`mamba-stash-T3-0919/`](mamba-stash-T3-0919/)。
