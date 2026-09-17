# 基于单个 Blackwell GPU 的 LLM 推理 + 计费一体机

[English](README.md)

一套可复现的自托管 LLM 栈,针对**单张 Blackwell(sm120) GPU、24 GB 起(推荐 32 GB)**调优:
OpenAI 兼容的 SGLang 服务运行 **Qwen3.8-27B**,262144 token 上下文、支持视觉;
默认 2 路并发(FP8 KV 档),NVFP4 档可到 4 路;
前面架 New API 网关,提供多用户 key、按 token 计费与故障转移。

本仓库是长期调优后的精炼结果。compose 里的注释刻意写得很短,「为什么」都放在
[调优要点](#调优要点)与[已知坑](#已知坑)。

## 特性

- 单张 32 GB 卡**常驻完整 262144 token 上下文**。
- 并发流近线性扩展、零 retraction:默认 2 路(FP8 KV 档),NVFP4 档 4 路。
- **FP8 KV + 视觉**为 32 GB 主线默认档(满 262144 上下文、mrr 2、关 prefill CUDA graph;
  v0.5.19 树 + E10 mamba 定容);**NVFP4 KV + 视觉**用于 32 GB 上需要 4 路并发的场景
  (legacy 档,旧钉定树;已知偶发 reasoning 乱码不稳定);**FP8 KV 纯文本**用于高并发
  (4 路)纯文本场景(legacy 档)。
- **分层 KV 缓存**(主机内存 L2),驱逐后重载极快。
- **网关**:每用户 key、三档缓存感知计费、主通道故障自动切换到备通道。
- **可复现**:在线一键安装,离线打包还原。

## 环境要求

| | |
| --- | --- |
| GPU | 1x NVIDIA Blackwell(sm120),**推荐 32 GB**(24 GB 仅可用 NVFP4 档)——见[显卡适配](#显卡适配) |
| 驱动 | 较新的 NVIDIA 驱动 + CUDA;`nvidia-container-toolkit` |
| 主机内存 | 最低 32 GB,推荐 64 GB(主机 KV 池约 17 GB) |
| 磁盘 | 约 90 GB(SGLang 镜像 ~48 GB + 模型 ~22 GB + 网关) |
| 系统 | 现代 Linux(在 Ubuntu 上开发) |
| 软件 | Docker Engine、Docker Compose v2、Python 3(仅工具需要) |
| 模型 | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)(Apache-2.0) |

### 显卡适配

本栈在 RTX PRO 4500(32 GB)上构建与实测;其他 Blackwell(sm120)卡跑同一镜像,
按显存选择配置版本:

| 显卡 | 显存 | 推荐版本 |
| --- | --- | --- |
| RTX 5090D V2 / RTX PRO 4000 Blackwell | 24 GB | `kv-nvfp4-text-image`(唯一有意义的选择;预计需下调 `--max-total-tokens`) |
| RTX 5090 | 32 GB | `kv-fp8-text-image`(默认);要 4 路并发用 `kv-nvfp4-text-image` |
| RTX PRO 4500 Blackwell(本仓库) | 32 GB | `kv-fp8-text-image`(默认,已实测) |
| RTX PRO 5000 Blackwell | 48 GB | `kv-fp8-text-image`,可选上调 mrr 至 4(见[调优要点](#调优要点)) |
| RTX PRO 6000 Blackwell | 96 GB | `kv-fp8-text-image`,并可上调并发(见[调优要点](#调优要点)) |

为什么 32 GB 上 FP8 KV 现在是默认:KV 池预算 = `free − fraction slack − mm 预留 − mamba 池`,
而 **CUDA graph 不在这个公式里**(图在池之后捕获,花的是 slack)。关掉 prefill CUDA graph
(1.19 GB,实测 prefill 影响 0~1%)并把并发改为 mrr 2 后,FP8 KV + 视觉 +
满 262144 上下文能在 32 GB 内落地,还剩 ~1.8 GB 余量——彻底不需要 FP4 KV,且每流 decode
带宽约翻倍。主线 mamba 池为 10(E10 定容 = 上游 ratio 5 × mrr,见
[mamba 槽计价](#mamba-槽计价为什么小的离线请求也能打满整卡));KV 池被
`--max-total-tokens` 钉住而非被内存钉住,因此 +2 槽不损失任何上下文。需要 4 路并发的
32 GB 卡仍可选 `kv-nvfp4-text-image`(KV 8→5 GB,FP4 注意事项见下);不需要图片且要高并发时用
`kv-fp8-text-only`。两者都跑在 legacy 钉定树上。

## 架构

```
客户端(opencode / curl / 任意 OpenAI SDK)
        │  Authorization: Bearer sk-...
        ▼
New API 网关  :8088          key、额度、三档计费、故障转移
        │  http://host.docker.internal:8080
        ▼
SGLang 服务   :8080          qwen3.8-27b,262144 ctx,2-4 路并发
        │
        ▼
NVIDIA GPU(sm120,32 GB)     NVFP4 权重 + NVFP4/FP8 KV
```

网关是可选的;只用 SGLang 时,`:8080` 本身就是 OpenAI API。

## 快速开始

### 在线(有网机器)

```bash
git clone https://github.com/mikecovlee/qwen3.8-27b-blackwell-recipe.git
cd qwen3.8-27b-blackwell-recipe
make online          # 默认 VARIANT=fp8v(FP8 KV + 视觉,mrr 2,32 GB)
# 其他档位:          make online VARIANT=nvfp4  (32 GB,FP4 KV,4 路)
#                     make online VARIANT=fp8    (32 GB,纯文本,4 路)
```

`make online` 调用 `scripts/online/setup.sh`,依次:检查 docker / GPU / compose →
按变体准备 SGLang 镜像——`fp8v`(默认):按 digest 拉取钉定的 v0.5.19 基座并**构建烘焙补丁的
派生镜像** `llm-infer:hicache-d6e72886`(来自 `inference/patches/`:调度器假闩锁 + LPM +
HiCache 混合 Mamba 补丁);
legacy 变体:拉取旧钉定镜像(digest,失败回退 tag)——并拉取网关镜像 → 下载模型到
`$MODELS_DIR` → 生成 `.env`(随机 `SESSION_SECRET`,`SGLANG_IMAGE` 与变体匹配)→
启动服务并等待 `/health`。

然后打开网关 `http://localhost:8088`,创建用户与令牌,把客户端指向它——见
[`gateway/opencode-config.md`](gateway/opencode-config.md)。

<details>
<summary>手动在线安装</summary>

```bash
cp .env.example .env          # 修改 MODELS_DIR 与 SESSION_SECRET
pip install -U "huggingface_hub[cli]"
hf download nvidia/Qwen3.8-27B-NVFP4 --local-dir "$MODELS_DIR/Qwen3.8-27B-NVFP4"
# 主线(fp8v):钉定 v0.5.19 基座 + 烘焙调度器 + HiCache 补丁
docker pull lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
docker build -f inference/patches/hicache-mamba-fix/img-d6e72886/Dockerfile \
  -t llm-infer:hicache-d6e72886 inference/patches
# legacy 变体(nvfp4 / fp8 纯文本)改为拉取旧钉定树:
#   docker pull lmsysorg/sglang@sha256:b91d664a8e4825afc16ab831c6035a6c88ac20ef8bd26da4fe2b9813a9f44376
docker pull calciumion/new-api:v1.0.0-rc.36
docker compose --env-file .env -f inference/kv-fp8-text-image.yml up -d
docker compose --env-file .env -f gateway/docker-compose.yml up -d
```
</details>

### 离线(无网机器)

先在**有网**机器上:

```bash
make export          # 生成 bundle/:两个镜像(fp8v 为烘焙补丁的派生镜像)、模型、
                     # 校验和(约 70 GB);VARIANT= 开关同 online
```

把整个 `bundle/` 目录与本仓库拷到离线机器,然后:

```bash
make offline BUNDLE=/path/to/bundle
# 或:scripts/offline/setup.sh /path/to/bundle
```

`setup.sh` 会校验 checksum、`docker load` 镜像、解压模型、启动服务并等待健康。

## 配置版本

`inference/` 下有三个各自完整可读的 compose 文件,除下表所列外参数完全一致
(`make check` 会强制校验这一点)。

| 参数 | `kv-fp8-text-image.yml`(**主线默认**,32 GB,v0.5.19 树) | `kv-nvfp4-text-image.yml`(legacy,32 GB,4 路) | `kv-fp8-text-only.yml`(legacy,32 GB,4 路) |
| --- | --- | --- | --- |
| 镜像(默认) | `llm-infer:hicache-d6e72886`(派生;调度器 + HiCache 补丁均已烘焙) | `lmsysorg/sglang@sha256:b91d664a…`(旧钉定树,`/patches` 挂载) | 同 NVFP4 |
| `--kv-cache-dtype` | `fp8_e4m3` | `nvfp4` | `fp8_e4m3` |
| attention 后端 | `--attention-backend flashinfer` | `--prefill-attention-backend flashinfer` + `--decode-attention-backend trtllm_mha` | `--attention-backend flashinfer` |
| 视觉 | 开 | 开 | 关(`language_model_only`) |
| 图像参数 | `--mm-process-config`、`--image-processor-backend pil`、护栏 `image:128` | 同左 | 无 |
| mrr / mamba 池 / graph bs | **2 / 10 / 2** + `--disable-prefill-cuda-graph` | 4 / 16 / 4 | 4 / 16 / 4 |
| `--chunked-prefill-size` | `6144`(HiCache 锚点判据,见[调优要点](#调优要点)) | 2048(默认) | 2048(默认) |
| `SGLANG_HICACHE_MAMBA_SIZE_GB` | `7.0`(88 个 host mamba 锚点) | 未设 | 未设 |
| `--mamba-radix-cache-strategy` | `extra_buffer` | `extra_buffer_lazy` | `extra_buffer_lazy` |
| `--schedule-policy` | `lpm` + LPM 超时钉顶 env(20 s / 1) | 默认(`fcfs`) | 默认(`fcfs`) |
| `--mem-fraction-static` | `0.94` | `0.90` | `0.92` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | `expandable_segments:True` | 无 |
| 启动后显存余量(32 GB) | ~1.78 GB | ~2.65 GB | ~0.58 GB |

**怎么选?** 先用 **FP8 KV + 视觉**(主线默认):32 GB 上满上下文 + 视觉、零 FP4 KV 顾虑,
代价是 2 路并发(每流 decode 带宽约翻倍),跑在 v0.5.19 树 + 烘焙调度器/HiCache 补丁上。同一张卡需要
4 路并发、且接受其**偶发 reasoning 乱码不稳定**([NVFP4 注意事项](#nvfp4-注意事项))时才选
**NVFP4 KV + 视觉**(legacy);24 GB 卡上它是唯一有意义的选择;从不发图片且要高并发选
**FP8 纯文本**(legacy)。48 GB+ 可把主线档上调为 mrr 4 / 池 20 / graph 4(池 = 5 × mrr,
见[调优要点](#调优要点))并重开 prefill 图(fraction 回调 0.90~0.92),然后重跑
`tools/concurrency-load.py soak`。

```bash
make up                   # 默认:FP8 KV + 视觉,mrr 2(32 GB)
make up VARIANT=nvfp4     # 32 GB,FP4 KV,4 路
make up VARIANT=fp8       # 32 GB,纯文本,4 路
```

安装脚本认同一个开关(`make online|offline VARIANT=...`);`fp8v` 预检门槛已降为 32 GB。

## 调优要点

本仓库最有价值的部分是「为什么这样配」。以下均在该目标卡上实测。

### 并发是三个联动的旋钮

`--max-running-requests`(mrr)、`--max-mamba-cache-size`、`--cuda-graph-max-bs-decode`
必须一起改:

- mamba 状态池钳制并发。主线(v0.5.19 树,`extra_buffer` + overlap 调度):上游自动定容为
  **5 × mrr**(ratio 5 = 基础 3 + overlap 2),即池 10 → mrr 2、池 20 → mrr 4。legacy 档
  (旧钉定树,`extra_buffer_lazy`)按每请求 4 槽预算(`池 // 4`):池 16 → mrr 4。
- CUDA graph 的 decode batch 必须覆盖 mrr,否则大 batch 会静默回退 eager(慢约 60%)。
- 默认档:`kv-fp8-text-image`(主线)= `262144 / mrr 2 / 池 10 / graph 2 / extra_buffer`
  (并关 prefill 图);legacy 的 NVFP4 与纯文本档 = `262144 / 4 / 16 / 4 / extra_buffer_lazy`
  (旧钉定树)。
- 96 GB 卡可以三个旋钮同步上调(主线 ratio:mrr 6 / 池 30 / graph 8,保持 graph ≥ mrr)。
  闩锁修复已在 v0.5.19 线的 mrr 2 上生产验证(24h+),legacy 树在 4 流上验证过;再上调务必用
  `tools/concurrency-load.py`(`t1`、`soak`)复测后再信任。

### mamba 槽计价(为什么"小"的离线请求也能打满整卡)

状态槽**按请求、按 radix 路径计价,不按 token**:一条运行中的请求占 1 active +
路径上至多 2 个 checkpoint(track 间隔 256,受 `--mamba-max-states-per-path 2` 封顶)
= 2~3 槽。legacy 树上准入按每请求 4 槽预算(实测:池 8 上双流进行中 used 5 /
evictable 2 / available 1)。

v0.5.19 主线的 unified radix cache 多了一个消费者:**chunked prefill 请求的首次 stash
会向树捐赠一个额外槽**——每请求峰值 = own + locked + 1 donated(上游 sizing 公式,由其
单测 `test_mamba_donated_alloc_ratio.py` 钉死)。`extra_buffer_lazy` 下分配器只按每请求
2 槽预算,池吃紧时会撞上 `assert slot is not None`("Can not alloc mamba cache")把调度器
打崩——生产实际发生过一次(2026-09-15:双流占 6/8 槽,150K chunked prefill 在释放出的
2 槽上被准入,其 stash 需要第 9 槽)。**E10 修复**(此后为主线):`extra_buffer`
(预算 3,准入拒绝排队而非超额承诺)+ 池 **10** = 上游 ratio 5 × mrr 2。主线实测:双流
峰值用 8/10,恰好留下 2 槽 stash 余量。回归门禁:
`inference/tools/acceptance/verify-mamba-stash.py`(证据:`evidence/mamba-stash-T3-0915/`)。

后果(不变):一个离线"RM 打分"循环——prompt 60~1300 token、生成顶满 700、
30~50 req/min——按 Little 定律需要 10~15 路并发,足以把任何交互服务钉死在满 mrr,
并让路径状态持续处于"驱逐→换入"扰动中。**短请求在 KV 上便宜,在状态槽上昂贵。**
运维规则:离线/批量任务用**独立令牌 + 客户端并发 ≤2**(或错峰)。

### 调度器假闩锁热修

两棵钉定的 SGLang 树都有同一个假闩锁 bug(legacy `b91d664a` 在 `scheduler.py:3355`;
v0.5.19 `d6e72886` 在 `scheduler.py:3661`):chunked prefill 的续传(已持有请求行、
不申请新行)被计入 `can_run`,却与按空闲行算出的额度比较,于是双重计数、提前置位
`batch_is_full`,把实际并发压到 `mrr - 1`。

补丁位于 `inference/patches/sched-latch-fix/`,按钉定镜像一构建一目录(`img-<digest>/`,
升级流程见其 README)。主线 `kv-fp8-text-image` 档钉定**派生镜像
`llm-infer:hicache-d6e72886`**(一体构建自 `inference/patches/`:本补丁族 + 下节的
HiCache 混合 Mamba 补丁):调度器补丁经 `.pth` import 烘焙——
v0.5.19 基座自带的系统 `sitecustomize.py` 会静默遮蔽旧的 `PYTHONPATH=/patches` 挂载法——
启动时**自检锚点、漂移即响亮报错**,并含 LPM 超时钉顶伴生模块,主线档默认启用
(`--schedule-policy lpm` + `SGLANG_LPM_WAIT_BOOST_SECONDS=20` / `SGLANG_LPM_WAIT_BOOST_MAX=1`:
等待超过 20 s 的冷请求一次性跳到 LPM 队首——只改顺序、不加容量;env 设 0 即禁用)。
legacy 档仍钉定 `b91d664a`,经 `/patches` 挂载加载 `img-b91d664a/sitecustomize.py`
(**按行号 3355 + 函数名匹配**——镜像升级后必须复核锚点,否则钩子静默失效;安全:退化为
`mrr - 1`)。

两个构建共享同一条抑制规则:只在「本 pass 真正的新准入数 < pass 起始空闲行数」时抑制
假闩锁——因此绝不会过度准入,任何 `mrr` 下都安全。

假闩锁需要存在"正在进行的分块 prefill"——即 prompt 超过 `chunked_prefill_size`
(2048 legacy / 6144 主线)被切成多 pass 续传。只有此时,第二路请求**在该 prefill 尚未结束
时到达**,才会被压到第一路 prefill 结束(瞬时 `mrr - 1`),停滞时长等于该 prefill 的时长。
同一 pass 内同时到达的两路都会进;≤`chunked_prefill_size` 的 prompt 根本不会分块,均不受影响。

### 上下文池预算与 prefill CUDA graph

池按 `pool_bytes = free_after_weights − fraction slack − mm 预留 − mamba 池` 测得,
而 **CUDA graph 捕获不在公式里**——图在池分配之后捕获,花的是 slack。因此:当池被
`--max-total-tokens` 钉住时,改 `--mem-fraction-static` 只是重新划分 slack,池不变
(0.90 与 0.92 实测完全相同);但当**测得值(profiled)低于用户上限**时,fraction 与
图开销就直接决定池大小。2026-09-11 实测:FP8 KV + 视觉 + mrr 2 在 0.90 + prefill 图开启
时 profiled 只有 242337;关掉该图(1.19 GB,实测 prefill 影响 0~1%——chunked prefill
是计算受限而非启动受限)并把 fraction 提到 0.94,池恢复满额 262144,
还剩 ~1.9 GB 余量(166K 冷 prefill 87s、零 retraction)。E10 复验(2026-09-15,mamba
池 8→10 = +0.16 GB):KV 池被 **flag 钉住而非内存钉住**——约 0.7 GB 静态 slack 吸收了
新增槽位,KV 仍分配满额 262144,图捕获后余量 ~1.78 GB,166K 冷 prefill 复测 87.1 s。
池与冷 prefill 余量仍是 1:1 交换(下表为 NVFP4 KV / mrr 4 时期的测量,规律相同):

| `max-total-tokens` | 显存余量 | 冷 prefill | 4 并发 |
| --- | --- | --- | --- |
| **262144(本栈)** | 2.65 GB | ~252K | 4 x 65K |
| 300000 | ~1.9 GB | ~180K | 4 x 75K |
| 320000 | 1.54 GB | ~140K | 4 x 80K |

### 分层缓存

`--enable-hierarchical-cache --hicache-ratio 2` 在主机内存保留 L2 KV 缓存,把 190K
token 的重载从约 109 s 降到 <1 s,decode/TTFT 无回退。`ratio 2` 约占 17 GB 主机内存;
想要更大 L2 就调高(代价是内存)。

[2026-09-17] 在这款混合 GDN(Mamba)模型上,host 层真正可用是打了 `hicache-mamba-fix`
补丁 + `--chunked-prefill-size 6144` + `SGLANG_HICACHE_MAMBA_SIZE_GB=7.0` 之后:被逐出
的 36.6K 会话现在 **0.26 s** 从 host 内存回来(此前全量重算 8.76 s——约 34 倍),分支
重入 25.1 s → 3.09 s,2×68.5K 并发 prefill 全程干净。细节见下一节。

### HiCache 混合 Mamba 修复

原版 v0.5.19 的 HiCache 在这类混合 GDN(Mamba)模型上并没有真正复用 host 层:chunked
prefill 从不写通备份(chunked 节点被按操作计的命中计数跳过)、mamba 锚点池远低于上游
判据 `kv_pool_tokens × hicache_ratio / chunked_prefill_size`(旧 cps 2048 时需要约 128
个,实际只有 10 设备 + 20 host)、host-hit 计数是幻影(device 常驻 token 记到了 host
层),且 mamba 分配饥饿会把调度器断言打崩。`inference/patches/hicache-mamba-fix/` 一体
解决这四点(chunked 写通回移 #36647;诚实的 `loaded_host_hit_length` 分层 #26976;mamba
耗尽时跳过而非断言 #36770;`SGLANG_HICACHE_MAMBA_SIZE_GB` host 池旋钮)——锚点、上游
状态与退役表见该目录 README。主线跑 `--chunked-prefill-size 6144`(8192 会把池打
OOM)+ `SGLANG_HICACHE_MAMBA_SIZE_GB=7.0` = 约 88 个 host 锚点,满足判据 262144/6144
≈ 43 ≤ 10 + 88。回归门禁 `verify-hicache-thrash.py` 现在要求出现真实 host 装载
(`sglang:load_back_tokens_total{pool="kv"}`),而不只是"答得快";测量见
`evidence/hicache-mamba-fix-0917/`。

### NVFP4 注意事项

> **已知偶发不稳定(生产环境实际遇到)。** 本档会偶发 reasoning 乱码——见首条。除非需要 32 GB
> 上的 4 路并发,或使用 **24 GB** 卡(NVFP4 KV 更小的占用是唯一有意义的选择),否则请用 FP8 默认档。

- **多会话扰动下的偶发 reasoning 乱码(未定案,生产环境实际遇到)**:在一次 40 分钟的窗口里(mrr 4 打满 +
  mamba 池钉在 `available = 0`,路径状态持续被驱逐/主机换入),一个长 agent 会话间歇性
  产出 token-soup 式乱码 reasoning,一次失控到 20000 token。26 分钟受控合成饱和复现
  0 命中;当前主解释是 FP4-KV 的稀有采样脱轨,再被客户端把乱码写回历史而自喂放大。
  两说都未证实——这也是 FP8 KV 成为默认档的原因之一。复现工具:`tools/mamba-saturation-probe.py`。
- **KV 主机池对 NVFP4 超额 2 倍**:主机池按 32 KB/token 计算,仿佛 KV 未打包,于是占用
  与 FP8 相同的内存。这是上游 sizing bug;打补丁会破坏 host→GPU 重载 kernel,故保留现状。
- checkpoint 未带校准的 KV scale,NVFP4 使用全局 scale 1.0 + per-block 兜底,实测质量无影响。
- **MTP / 投机解码在此不可行**:draft 模型需约 5.5 GB 会把 KV 池挤垮,且 NVFP4 verify 撞到
  未支持的路径。

## 已知坑

- **NVFP4 KV 档偶发 reasoning 乱码**(间歇性,生产环境实际遇到;见 [NVFP4 注意事项](#nvfp4-注意事项))。
  仅在需要 32 GB 上 4 路并发、或 24 GB 卡上(此时是唯一有意义的选择)时使用。
- **`trtllm_mha` 会把 `page_size` 改成 64**,此后默认的 `--mamba-max-states-per-path -1`
  会让 mamba 状态沿 radix 路径无限囤积、占满 16 槽池并锁死新分配(容器重启)。三档都设
  `--mamba-max-states-per-path 2`(零显存成本的行为限制,与 extra-buffer 乒乓槽位设计吻合;
  主线 `extra_buffer` 下同样保留 2——注意它只封顶**树上** checkpoint,不管在飞请求的活槽)。
  **每次改动 KV 配方后务必跑 soak。**
- **v0.5.19 + `extra_buffer_lazy` + 池 8 可能把调度器断言打崩**("Can not alloc mamba
  cache"):chunked prefill 的 stash 在池吃紧且无可驱逐牺牲者时需要一个捐赠槽(生产实际
  发生一次,2026-09-15)。已由 E10 修复:主线跑 `extra_buffer`(分配器按每请求 3 槽预算,
  准入拒绝排队而非超额承诺)+ 池 10(上游 ratio 5 × mrr 2)。断言本身是上游 fail-loud 设计
  (main 分支今天仍在);主线补丁构建另含上游「跳过而非断言」回移(#36770,经
  `radix_cache_aux_alloc_failed_total` 计数)。回归门禁:
  `inference/tools/acceptance/verify-mamba-stash.py`,证据见 `evidence/mamba-stash-T3-0915/`
  与 `evidence/hicache-mamba-fix-0917/`。
- **`--mm-process-config` 用的是像素「面积」而非边长**。本版处理器忽略 `image.max_pixels`,
  必须用 `image.size.longest_edge`(2097152 = 2 Mpx 面积)。
- **图片需要 `--image-processor-backend pil`**。GPU 处理器会一次性把所有图 resize 成 fp32
  张量把 tokenizer 进程打 OOM;`pil` 把预处理挪到 CPU。计数护栏
  `--limit-mm-data-per-request '{"image":128,"video":0}'` 存在的意义是它在预处理**前**拦截,
  而上下文长度检查在**后**。128 恰好落在自然天花板外一格:每张 2 Mpx 图 ≈2050 token,
  ~127 张就已耗尽 262144 窗口——显存从不成为瓶颈(实测 4~96 张 GPU 峰值恒定)。
- **New API 按 `abilities` 表路由**,不是 `channels`。直改 channels(优先级/模型列表)不生效,
  必须同步 abilities。
- **防火墙导致的回退延迟**:ufw 对无监听端口是 DROP(30 s SYN 超时),所以「端口被墙」类
  故障约 30 s 才回退;进程级挂掉(RST)则亚秒级回退。

## 基准

在目标卡(RTX PRO 4500,32 GB,sm120)实测。完整结果见 [`evidence/README.md`](evidence/README.md)。

### NVFP4 vs FP8(本部署)

| 场景 | FP8 | NVFP4 |
| --- | --- | --- |
| 单流 decode @1K | 42.6 | 42.4 tok/s |
| 单流 decode @64K | 38.5 | 39.6 tok/s |
| cached TTFT | 0.05 s | 0.06 s |
| batch 4 @1K | 153.4 | 152.3 tok/s |
| batch 4 @48K | 118.6 | 128.2 tok/s |
| soak 600 s | 106 req / 0 err | 106 req / 0 err |

### FP8 KV + 视觉 @32 GB(池 8 基线,2026-09-11)

| 检查项 | 结果 |
| --- | --- |
| 启动账 | KV 8.0 GB + mamba 0.65 GB,图捕获后余 ~1.91 GB |
| prefill 图 关 vs 开(同栈) | 0 ~ +1%(1K/14K/70K;TTFT 完全一致) |
| 166K 冷 prefill | 87.3 s(1905 tok/s),零 retraction |
| 单流 decode @1K | 41~43 tok/s(mrr 2) |
| mamba 槽,双流进行中 | used 5 / evictable 2 / available 1(池 8) |
| 图片(2 Mpx/张) | 1~96 全过、GPU 峰值恒定;128 被上下文窗口拒绝 |
| 质量 spot check | 全过(含 104K 单针) |

### E10 复验(主线:v0.5.19、`extra_buffer`、池 10,2026-09-15)

| 检查项 | 结果 |
| --- | --- |
| 启动账 | KV 8.0 GB / 262144 token **不变**(flag 钉住)+ mamba 0.80 GB;池分配后余 2.43 GB,图捕获后 ~1.78 GB |
| T3 stash 崩溃矩阵(`verify-mamba-stash.py --expect safe`) | 153K chunked prefill 全程 avail ≥ 2/10、零强制驱逐、零重启 |
| 单流 decode | 43.1 tok/s @1K;38.8 @64K(TTFT 18.7 s) |
| 双流 decode(引擎侧,bs-2 graph) | 中位 69.7 tok/s,零 retraction |
| 166K 冷 prefill | 87.1 s(池 8 时为 87.3 s) |
| 缓存命中计费(第二枪) | 99.4% 命中,TTFT 0.13 s |
| 150K × 2 挤兑 | 零 retraction/断言(第二个 150K 排队属设计行为:300K > 262144 池) |
| T1/T2 调度补丁 | PASS(闩锁交织 + wait-boost 排序) |
| 9.5 h 生产 soak | 双流 mamba 峰值 8/10,零断言、零重启 |

### 吞吐 vs 上下文

| 上下文 | 2K | 4K | 64K | 128K | 240K |
| --- | --- | --- | --- | --- | --- |
| prefill(tok/s) | 5716 | 5737 | 3300 | 2203 | 1386 |
| decode(tok/s) | 42.0 | 41.8 | 39.3 | 37.0 | 33.2 |
| TTFT(s) | 0.38 | 0.74 | 19.2 | 57.5 | 171.3 |

2K/4K 的 decode 由 16 个生成 token 测得;更长上下文用 128-256 个。

### 质量

- **GSM8K**:量化方报告 **97.27%**(1283/1319,4x GB300);本单卡部署同协议实测
  **97.5%**(195/200)。
- **Terminal-Bench 2.1**:量化方 73.81%,上游 Qwen 卡 73.0。
- **RULER 长上下文**:单针 NIAH 在 32K/64K/128K/**234K** 全 100%;多针 NIAH 与更难的
  `freq_words_extraction` 在 234K 也全 100% —— 262144 窗口内有效上下文长度至少 234K。
- 质量 spot-check:13/13(含 104K needle)。

结论:在准确率基准上,NVFP4 权重 + NVFP4 KV 未引入可测退化(与上文的 NVFP4 稳定性问题无关)。

## 网关(New API)

见 [`gateway/`](gateway/)。要点:

- **计费**:由上游价格推导出三档——输入、缓存命中输入、输出。
  `quota = round((未命中输入 + 命中输入 * CacheRatio + 输出 * CompletionRatio) * ModelRatio)`。
  依赖 SGLang 的 `--enable-cache-report`(否则命中按全价计)。
- **故障转移**:通道 1 = SGLang(`priority 10`),通道 2 = 备上游(`priority 0`)。需设
  `RetryTimes=2`;默认 `0` **不会**重试。路由真源是 `abilities` 表。
- **备份**:`docker cp llm-gateway:/data/one-api.db ./backup.db`(连同 `-wal` 一起拷)。
  SQLite 里 token key 是明文,`data/` 属凭据级。
- **公网暴露前**:必须加 TLS 反代(网关是明文 HTTP);另注意上游 SGLang 默认无 API key。

## 工具

所有工具读取 `SGLANG_BASE`、`MODEL_NAME`(网关用例另需 `GW_BASE`/`GW_KEY`,引擎指标用例另需
`C4_CONTAINER`/`LLM_SERVICE_ROOT`)。

| 工具 | 用途 |
| --- | --- |
| `tools/benchmark.py` | prompt-eval / decode 吞吐 vs 上下文长度 |
| `tools/concurrency-load.py` | 并发门禁 + soak 驱动(`b2\|t1\|t2\|t3\|soak`) |
| `tools/gsm8k.py` | GSM8K 准确率(需 GSM8K test JSONL,见 `GSM8K_DATA`) |
| `tools/ruler-niah.py` | RULER 单针 NIAH(completion 协议) |
| `tools/ruler-niah-multi.py` | RULER 多针 / multikey / multivalue NIAH |
| `tools/ruler-fwe.py` | RULER `freq_words_extraction`(最难聚合) |
| `tools/quality-spotcheck.py` | 固定确定性 prompt,可对 FP8 vs FP4 做 diff |
| `tools/test-cache-report.py` | 验证 `cached_tokens` 上报(计费依赖) |
| `tools/image-limit-ladder.py` | 图片张数阶梯 + 显存峰值采样(找真实天花板) |
| `tools/mamba-saturation-probe.py` | 合成 RM 式饱和 + 乱码检测器(调查工具) |

```bash
make bench     # 吞吐
make eval      # 打印如何跑质量 / 长上下文评测
make check     # compose 一致性 + 机密扫描
```

工具需要 Python 3 与 `requests`;RULER 工具另需一个 haystack 文本(`RULER_HAYSTACK`,默认
`haystack.txt`)以及 Python 包 `tiktoken`、`wonderwords`。

## 目录结构

```
.
├── README.md / README-zh.md
├── LICENSE                     Apache-2.0
├── Makefile  .env.example
├── inference/
│   ├── kv-fp8-text-image.yml       主线默认:FP8 KV + 视觉(32 GB,mrr 2,
│   │                               v0.5.19 树,E10 + hicache 修复:cps 6144、host mamba 7 GB)
│   ├── kv-nvfp4-text-image.yml     legacy:NVFP4 KV + 视觉(32 GB,mrr 4,旧钉定树)
│   ├── kv-fp8-text-only.yml        legacy:FP8 KV,纯文本,4 路(旧钉定树)
│   ├── patches/sched-latch-fix/    调度器补丁(假闩锁 / LPM 超时钉顶)
│   ├── patches/hicache-mamba-fix/  HiCache 补丁(写通、诚实指标、host 池定容)
│   └── tools/acceptance/           验收套件(T1/T2 调度、T3 mamba stash、T4 hicache)
├── gateway/
│   ├── docker-compose.yml
│   └── opencode-config.md / opencode-config-zh.md
├── tools/                      基准 / 评测 / 运维脚本
├── evidence/                   基准与评测结果
└── scripts/
    ├── online/setup.sh
    ├── offline/export.sh  offline/setup.sh
    └── check.sh
```

## 许可

Apache-2.0(见 [LICENSE](LICENSE))。

本项目只是配置与封装第三方软件,不重分发它们:
[SGLang](https://github.com/sgl-project/sglang)(Apache-2.0)、
[New API](https://github.com/Calcium-Ion/new-api)(AGPLv3)、
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 与
[nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)(Apache-2.0)。
若把网关对外提供服务,需自行履行合规义务(许可、内容安全、日志留存等)。
