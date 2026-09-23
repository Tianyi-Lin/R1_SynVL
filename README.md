# R1-SyntheticVL：CADS 数据合成开源实现

对照版本：[arXiv 2602.03300v2，ICML 2026 Camera Ready](https://arxiv.org/html/2602.03300v2)。论文流程核查日期 2026-09-16，接口和仓库复查日期 2026-09-22。
**`cads/` 是作者正在整理、准备开源的 CADS 数据合成模块，以论文方法为主线，并保留作者确认的工程补充。** 它不是所有历史代码版本的合集，当前范围也不包含完整训练和评测代码。R1 的视觉转述、可配置的候选数量 N 和多协议 API 适配属于开源版实现的一部分，不应因论文未展开这些接口细节而删除。

开源整理区分三层内容：论文算法流程、开源版工程补充、原实验配置与数据记录。当前可运行接口配置与历史实验配置的差异需要明确记录；新增补充不能反向描述为历史实验已经采用的设置。[项目仓库](https://github.com/jingyi0000/R1-SyntheticVL)。

## 目录内容

| 文件 | 用途 |
| --- | --- |
| `pipeline.py` | 仅负责命令行参数、选择输入和启动流程 |
| `workflow.py` | N 的分配、出题/绘图提示词、批次调度、反馈及结果导出 |
| `configuration.py` | YAML/JSON 配置读取、参数检查、连接设置及密钥脱敏 |
| `backend.py` | 模型调用、R1 视觉转述、图片落盘和请求缓存 |
| `model_api.py` | OpenAI 兼容参数构造、Gemini 原生和 Anthropic 原生协议适配 |
| `image_api.py` | OpenRouter Images API 请求、重试和安全响应解析 |
| `evaluation.py` | 独立求解、答案判等和共识分数 C |
| `seed_data.py` | EasyR1 格式的 Hugging Face/本地数据读取、旧 JSON 兼容、图片落盘 |
| `seed_memory.py` | seed 记忆、反思快照、跨任务复用和运行 manifest |
| `storage.py` | JSON 读写、内容指纹及项目内路径处理 |
| `config.yaml` | 主配置；模型列表、独立密钥引用和生图设置 |
| `config.example.json` | 同一套配置的 JSON 写法，不是另一版本的合成算法 |
| `.env.example` | 服务地址与密钥的空白环境变量模板，不绑定服务商 |
| `check_connectivity.py` | 发送 hello 检查文本模型连通性；超时、失败及非零退出码 |
| `config.connectivity.yaml` | 独立的三个 Gemini 名称的连通性测试配置，不是论文实验配置 |
| `audit_legacy.py`、`legacy_audit.json` | 旧数据的离线审计脚本及结果，不是新生成数据 |
| `pyproject.toml`、`uv.lock`、`.python-version` | 依赖声明、精确依赖锁定和默认 Python 版本 |
| `setup.sh` | 一键安装锁定依赖；可选下载数据并执行无模型调用的输入检查 |
| `README.md` | 环境、流程、论文差异和使用限制 |

工作区其他目录中的旧合成脚本及数据仍保留，不由 `cads/` 替换或自动迁移。这里是面向开源整理的新入口；原实验参数、模型版本和最终数据清单仍需与历史记录核对归档。

## 环境与复现：统一使用 uv

`cads/` 是独立的 uv 项目。环境安装在 `cads/.venv/`；依赖以 `pyproject.toml` 声明、`uv.lock` 精确锁定，不再维护第二份手写 `requirements.txt`。默认 Python 固定为 `.python-version` 中的 `3.13.7`。`uv sync --locked` 会按锁文件同步环境，如果声明与锁不一致则报错而不是偷偷升级，参见 [uv 官方说明](https://docs.astral.sh/uv/concepts/projects/sync/)。

### 一键安装（推荐）

适用于 macOS/Linux 的 Bash；Windows 可在 WSL 中执行。需要联网，以及已经安装的 uv，或可运行 `python3 -m pip` 的 Python 3。脚本优先复用本地/系统 uv；找不到时，用 pip 将 `uv==0.12.17` 安装到 `cads/.tools/uv/`，不会要求 sudo、修改系统 Python 或修改 shell 启动配置。uv 负责准备项目指定的 Python 版本。

从工作区根目录执行，以下两种方式任选其一：

```bash
bash cads/setup.sh
bash cads/setup.sh --download-dataset
```

第一条只安装依赖，**不下载数据集**。第二条还会读取 `config.yaml`，下载/缓存默认 `hiyouga/geometry3k`，检查前 2 条图文记录；两者都不调用模型 API、不需要模型密钥，也不启动合成。已安装和已缓存的内容可以复用，重复执行不会主动升级锁定依赖。

需要自定义数据集或检查数量时：

```bash
bash cads/setup.sh --download-dataset --dataset hiyouga/geometry3k --split train --limit 8
bash cads/setup.sh --download-dataset --config cads/config.example.json
bash cads/setup.sh --help
```

`--limit` 只限制检查和导出图片的记录数，**不是下载大小限制**；非流式数据加载可能先下载整个划分，甚至数据集其他划分所需文件。默认只将 `train` 作为输入，不把验证集/测试集合并用于合成。脚本可从任意目录调用；相对 `--config` 和本地 `--dataset` 路径按调用目录解析。下载中断后可以重跑。私有/受限数据集需自行提供 `HF_TOKEN` 并遵守其访问许可。

安装后无需手动激活环境，也不要求将项目内 uv 加入全局 PATH：

```bash
cd cads
.venv/bin/python pipeline.py --help
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python pipeline.py --limit 2 --dry-run
```

第二条要求已完成对应数据集版本/划分的下载；离线缓存缺失会报错，不自动调用模型或补造数据。

去掉 `--limit 2` 即可检查所选划分的全部记录和图片；这仍然只是输入检查，不会执行数据合成。

### 安装哪些依赖？

直接依赖以 `pyproject.toml` 为准，当前锁定版本如下；间接依赖由 uv 自动安装，不需要读者逐个手装：

| 包 | 当前锁定版本 | 用途 |
| --- | --- | --- |
| `datasets` | `5.0.1` | Hugging Face 与本地 Parquet/JSON/JSONL 数据读取 |
| `Pillow` | `12.3.0` | 多模态图片解码、检查和保存 |
| `PyYAML` | `6.0.3` | YAML 配置读取 |
| `openai` | `2.54.0` | GPT-4o、DeepSeek-R1 的 OpenAI 兼容接口 |
| `google-genai` | `2.24.0` | Nano Banana Pro 官方生图接口 |

例如 `huggingface-hub`、`pyarrow`、`httpx` 等间接依赖也记录在 `uv.lock`。当前 Gemini 文本接口和 Claude 接口采用原生 HTTP 适配，不额外依赖 Anthropic SDK。此目录是 API 驱动的数据合成模块，环境不包含完整训练栈，不需要为输入检查安装 PyTorch/CUDA，也不代表已经包含论文训练环境。

### 手动安装与维护

若希望手动管理 uv，可按[官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)先安装 uv，再从工作区根目录执行：

```bash
cd cads
mkdir -p .cache .tmp
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export HF_HOME="$PWD/dataset/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export TMPDIR="$PWD/.tmp"
uv sync --locked
uv run --locked python pipeline.py --help
```

`uv sync` 只构建环境，不运行数据集下载或数据合成；缺少指定 Python 时 uv 可能下载解释器到 `.python/`。查看 `--help` 同样不会访问模型或数据集。依赖调整时由维护者执行 `uv lock` 并提交更新后的 `uv.lock`，读者复现实验使用 `--locked`。环境锁定不等于云端模型快照、数据集版本和生成结果完全确定，后者仍需配置与运行记录。

源码修改、环境配置及本项目新增文件都放在 `cads/` 下。默认运行目录也收拢在这里：

```text
cads/
  .venv/                 uv 项目环境
  .cache/                uv 与 pip 缓存
  .tools/                可选的本地 uv 工具
  .tmp/                  本地验证/安装的临时文件
  dataset/
    huggingface/hub/      下载的 Hub 源数据与版本缓存
    huggingface/datasets/ datasets 处理后的 Arrow 缓存
    huggingface/xet/      可选的 Hub 传输缓存
    seed_images/         输入检查/合成时导出的 PNG 图片
  seed_memory/           持久化 seed 记忆
  outputs/default/       默认合成输出
```

配置中的相对 `seed_memory_dir`、`image_cache_dir` 以及命令行的相对 `--output` 都以 `cads/` 为基准，不会因从工作区根目录启动而散落到外面；显式绝对路径仍可用于已有存储位置。输入文件路径及 `--images-dir` 按调用时工作目录解析。环境、缓存和生成产物已列入 `.gitignore`，开源时保留代码与锁文件，不提交 `.venv` 或密钥。

### 数据下载与 Git 提交边界

下载的数据放在 `cads/dataset/`，保持 Hugging Face 原生缓存结构，而不是另复制一套数据集。`HF_HOME` 管理默认缓存根目录，Hub 源文件与 datasets 处理缓存是两层，参见 [Hugging Face 缓存说明](https://huggingface.co/docs/datasets/en/cache)。安装脚本统一设置这些位置；直接运行 Python 时默认使用相同位置，但显式设置的 `HF_HOME`/`HF_HUB_CACHE`/`HF_DATASETS_CACHE` 等环境变量仍优先。旧版 `.cache/huggingface/` 和 `seed_images/` 不自动删除或搬迁，默认新位置可能需要重新下载；旧输出需要原图片路径时应保留原目录。

应提交源码、`setup.sh`、`pyproject.toml`、`uv.lock`、`.python-version` 和不含密钥的示例配置。**不要提交 `dataset/`、`.venv/`、下载缓存、密钥或生成产物**：这些路径已写入 `cads/.gitignore`。克隆项目后由读者安装环境、下载数据，而不是把某台机器上的环境目录一起发布。

`.gitignore` 只防止未跟踪文件被常规添加，不会移除已经跟踪的文件，也挡不住 `git add -f`。若此前误提交了数据，可在仓库根目录执行 `git rm -r --cached -- cads/dataset` 停止跟踪（本地数据保留），提交前再检查 `git status --short`。本脚本不会执行 Git 提交或推送，也不会自动上传数据集。

## 当前代码与论文的差异

| 环节 | 旧脚本中的实现 | 新入口的对齐方向 |
| --- | --- | --- |
| 种子 | R1-ShareVL、MMR1-RL、MMK12 的图文题；生成入口缺图就跳过 | 支持图文种子及纯文本任务描述；论文未公开准确的种子组成，不能认定某个现有数据源就是论文种子集 |
| 生成模型 | R1-ShareVL 和 MMR1 主入口单独调用 Gemini-2.5-Flash；MMK12 配置 GPT-4o | 论文 §4.1 的 GPT-4o、Gemini-2.5-Flash、DeepSeek-R1、Claude-4-Sonnet 都参与生成和判题 |
| 知识分析 | 已有 domain/core concepts/分析，和后续生成合并在一次请求里 | 每个模型分析 seed；生成时只读取自己的 rationale，不共享各自分析 |
| 四种策略 | R1 并行脚本已有数值变换、逻辑反转、辅助构造、同构迁移，基本对应论文 | 保留四种 meta strategy，增加可审计的 seed-specific strategy_detail；无需推翻原策略 |
| 视觉生成 | 已使用 gemini-3-pro-image-preview，已有空间、标签、图表等绘图要求 | 沿用 Nano Banana Pro 路线；默认通过 OpenRouter 调用 `google/gemini-3-pro-image`，视觉提示词阶段独立保存 |
| 答题评审 | Adversarial_Filter.py 实际采集三个模型的独立图文回答，不直接筛选 | 所配模型独立答题；视觉模型看题图，R1 默认接收题目及另一视觉模型对实际图片的转述；均不看到参考答案、生成提示词或生成解答 |
| 保留标准 | Verify_Multi_Answer.py 的 strict_match 快速通过；其余使用语义判等，至少一个正确即 qualified | 保留 C>0；增加 C、K 和每个模型判定。原逻辑并非要求所有模型答对 |
| 分歧样本 | 缓存部分保留 gpt_judgement，但没有统一的 adversarial 字段和后续消费入口 | 将 0<C<K 的样本放入 adversarial_results.json；C=K 仍是合格训练样本 |
| 反馈闭环 | Analyze_Failed_Items_GPT52.py 分析失败原因，生成器不读取该报告 | 仅对分歧样本执行 Reflect→Optimize，将新上下文注入下一轮生成 |
| 迭代 | 生成器按 seed ID 写入一份结果；MAX_RETRIES 是请求重试 | 每个 seed 最多 10 批；样本 ID 包含 seed、候选序号、生成模型，另存轮数；网络重试不增加轮数 |
| 训练输出 | 有 filtered、mixed、expanded 多条导出路径 | 新入口只导出合格的合成图文 QA；不默认加入原始数据或 SCT 改写数据 |

本地证据位置：

- `R1-ShareVL-52K/generate_new_question_R1-ShareVL_parallel.py:190`：现有 prompt；`:244` 起四策略；`:568` 单样本流程。
- 同文件 `:345` 附近要求 MCQ 仅输出字母，但 `:436` 的 JSON 字段说明又要求字母和内容，存在冲突。新入口统一为字母且校验。
- `R1-ShareVL-52K/Adversarial_Filter.py:52`：三个求解模型的配置。
- `R1-ShareVL-52K/Verify_Multi_Answer.py:478`：strict_match；`:509`：仅全部判错才不合格。
- `R1-ShareVL-52K/Analyze_Failed_Items_GPT52.py:45`：失败归因任务，不能代替分歧样本的下一轮上下文优化。
- `R1-ShareVL-52K/expand_syn_data.py:4`：复用图片的语义等价文本扩充；与完整 CADS 再生成图文样本不同。

## 现有数据的离线审计结果

完整逐条结果在 `legacy_audit.json`，本次未调用任何模型重新判题。

| 指标 | 数量 |
| --- | ---: |
| 已生成记录 | 19,985 |
| 已有三模型回答的记录 | 19,837 |
| 使用 Gemini-3-Flash 的评审记录 | 5,635 |
| 使用 Gemini-2.5-Flash 的评审记录 | 14,202 |
| C=0 | 1,015 |
| C=1 | 2,623 |
| C=2 | 6,403 |
| C=3 | 9,786 |
| 可从缓存确认的合格记录 | 18,812 |
| 分歧样本 C=1 或 C=2 | 9,026 |
| 无法可靠恢复共识分数 | 10 |
| 已生成但缺少评审记录 | 148 |

旧 pass_ids 有 18,819 个 ID，比严格检查缓存后可确认的合格记录多 7 个。未知的 10 条包括：8 条判等结果未满足“每个模型一个布尔值”的完整性要求，1 条参考答案缺失或不匹配，1 条判等缺失/失败。不能将这些异常当成 C=0，也不能继续无条件沿用旧 pass_ids。

以上 C 都是 **K=3 的历史结果**；9,026 个分歧样本不是论文 K=4 的分歧样本数量。应先修复未知记录、统一评审模型版本、补齐第四模型，再选择是否重判或重新生成。

## 新入口实现的闭环

每个 seed：各模型分别分析 → 各模型只依据自己的分析生成变式 QA → 各自生成视觉 prompt → Nano Banana Pro 绘图 → 所配模型独立求解 → 参考答案判等 → 计算 C → 对 0<C<K 样本反思 → 更新后续批次的生成上下文。

## 配置 N 和 Collective 模型

默认配置为 `cads/config.yaml`。`questions_per_seed` 就是 N，表示本次任务中每个 seed 提出的候选总数；筛掉后不补齐，不是最终合格数，也不是每模型数量或每轮数量。所有候选保存在 `all_candidates.json`，只有评审合格的进入训练数据。同一输出目录的续跑不会重新增加 N；显式使用新输出目录开始新任务时，可继承该 seed 的记忆，再提出本次的 N 个候选。

Collective 模型数自动取 `models` 列表长度，增删条目即可调整；当前生成和评审使用同一组模型，K 也随之调整。删除模型时，需要同时调整引用它的 `reflection_model`、`equivalence_model` 或 `caption_model`。

按列表顺序轮询分配，任意两个模型的出题数量最多相差 1：

设 `K = len(models)`，`q, r = divmod(N, K)`。每个模型先分配 `q` 个候选，配置列表中的前 `r` 个模型各再分配 1 个，总数严格为 N；不要求 N 能被 K 整除，不丢弃余数，也不向上补成 K 的倍数。

| N | Collective 模型数 | 各模型出题数量 |
| ---: | ---: | --- |
| 8 | 4 | 2、2、2、2 |
| 10 | 4 | 3、3、2、2 |
| 11 | 4 | 3、3、3、2 |
| 3 | 4 | 1、1、1、0 |

例如 N=10、K=4、默认迭代上限时，三批依次由模型 1/2/3/4、模型 1/2/3/4、模型 1/2 出题。最后一批虽然只有两个生成模型，每个候选仍由全部 K 个模型评审；N<K 时未分到出题任务的模型也仍参与 seed 分析和评审。过滤掉的候选不补齐。

当前每个 seed 都从配置列表第一个模型开始分配，因而余数名额固定落在前 r 个模型；保证的是单个 seed 内的近似均分，不是多个 seed 累计后的完全均分。尚未实现跨 seed 轮转余数或随机分配。

同一道题在配置兼容、分析已成功保存的前提下，**每个模型只做一次初始分析**；不是四个模型合计只分析一次。后续批次或新任务复用各自的分析，避免重复请求，修改 seed ID 不会触发重新分析。各模型只接收自己的分析与自己之前提出的题目；所有模型的分析可以在日志中一同保存，但不会一同发送给生成器。评审分歧总结仍作为共享反馈上下文注入后续生成，这是反馈阶段的协作。若分析请求失败或尚未保存就更换任务，仍可能需要重试，不能将其理解为 API 严格只调用一次的保证。

默认 N=8、四模型时执行两批，每批每模型各提出一个候选。`max_iterations` 限制反馈批次数，不再决定扩展数量。如果 N 很大，代码将多个完整模型轮询合并到一批，确保候选数仍是 N，反馈批次不超过上限。例如 N=44、四模型、上限 10 时，每批最多 8 个候选，共 6 批。不同生成任务含独立 `sample_index`，不会因提示完全相同而被当作同一次请求。

`target_size: null` 表示遍历所有选中的 seed。可设置正整数作为全局合格数量的停止目标；只在完整处理一个 seed 后检查，不截断该 seed 的 N 个候选，因此允许超过全局目标。它不控制单个 seed 的 N，也不会为了达到目标而无限生成。

- `C=0`：不加入训练数据，不用于分歧反思。
- `0<C<K`：加入合格数据和分歧池，并产生后续生成指导。
- `C=K`：加入合格数据，不能误删为“太简单”。
- API 失败、缺答、非法 JSON、缺少布尔判定会中断运行，修复后重跑；这些工程失败不记为模型答错或 C=0。
- 每个成功请求独立缓存，完整轮次原子落盘；重跑复用成功阶段并恢复下一轮上下文。
- `manifest.json` 固定单次任务的输入、配置、代码指纹、依赖锁、图片哈希和服务地址；这些信息改变后必须换输出目录，避免混写同一次任务的结果。这与共享 seed 记忆的检索分开：只改代码或依赖锁，新任务仍可读取同 seed、同模型配置的历史。
- 当前按 seed 和轮次顺序执行；同一输出目录和共享 seed 记忆库都只允许一个写入进程。已有版本检查不能代替多进程锁，不支持并发写入。

## 按 seed 持久化与复用

反思不仅按批次保存在 `rounds/`，还会归入对应 seed 的持久化记忆。默认配置：

```yaml
seed_memory_dir: seed_memory
```

相对路径基于 `cads/` 解析，默认记忆库为 `cads/seed_memory/`。多个任务使用同一个记忆库目录，才能自动检索并复用历史；设为 `null` 时仍持久化到各自输出目录，但不跨任务共享。

每道题的记忆保存：各模型的私有分析、各模型已提出的题目文字（包括被筛掉的）、全部已提交反思、最新优化上下文、累计候选数量、遇到过的 seed ID 列表、来源任务及版本关联。seed ID 只用于追踪来源，不参与共享记忆的身份匹配。最后一批的反思也保存，新任务再次处理同一道题时可以使用。

```text
cads/seed_memory/
  <seed内容指纹>/<模型配置与服务地址指纹>/
    latest.json
    snapshots/<版本号>.json

<输出目录>/seed_states/<内容与来源ID组合指纹>/
  initial.json
  latest.json
```

共享库的 `latest.json` 指向最新快照；快照包含完整状态，旧快照不被新版本覆盖。输出目录的 `initial.json` 固定本次任务开始时读入的记忆，`latest.json` 保存本任务已完成批次的状态。共享库中的快照和输出目录中的记录都需作为运行产物保留，不应当作临时缓存随意删除。

共享记忆按内容定位，单次任务的进度按“内容 + 来源 ID”分开保存。因此，同一输入中相同题目以两个不同 ID 出现时，可以共享分析与反馈，又不会互相覆盖批次或快照。当前不会自动删除重复内容的输入行：每条记录仍各自产生 N 个候选，累计候选编号连续；生成结果保留本条输入的 seed ID。

### 为什么使用 SHA256？

SHA256 在这里是内容指纹，不是加密，也不是论文的生成策略；这些哈希值不会作为出题知识提供给模型。记忆检索分为 seed 身份和模型配置两层，两层均不包含源代码或依赖锁的哈希：

- 只按 `seed_id=123` 保存不够：更换图片或题目后，ID 仍可能是 123。现在只使用输入适配后的题目文字与有序图片内容哈希匹配记忆，seed ID 不参与；相同内容换 ID 仍复用，相同 ID 换内容则不复用。
- 把图片从一个文件夹搬到另一个文件夹，内容不变，指纹仍相同，不必依赖图片文件名识别。
- 模型配置或实际服务地址发生变化时，使用不同配置分组，避免把不同模型的私有分析直接混用；更改 N、超时、数据缓存位置不改变记忆分组。
- 状态 JSON 的 `seed_ids` 保存遇到过的来源 ID，`source_seed_id` 记录最新提交对应的 ID；题目、反思和上下文也直接保存，不是只留下无法理解的一串哈希。调用者无需自己计算指纹。

**修改代码、重构文件、调整日志或更新 `uv.lock`，不再导致 seed 记忆被隔离。** 代码和依赖锁的哈希仍记录在单次任务的 `manifest.json` 中用于追溯，也参与任务/快照标识，但不参与寻找该 seed 的共享记忆。比如今天积累了 seed A 的反思，明天仅重构代码，在新输出目录启动任务仍会读取 A 的分析、历史题目及反馈。若未来修改记忆文件的数据结构，需要单独实现格式迁移，不能把任意代码修改都等同于记忆不兼容。

- **同一个输出目录续跑**：使用固定的初始状态，重放已完成批次，恢复上下文和历史；仅执行未完成部分。已经完成 N 个候选时，不追加生成，也不重复写入历史。
- **新输出目录、相同 seed**：自动读取兼容的最新记忆；不重新请求已有初始分析，把旧上下文和该模型自己的历史题目带入新生成。本次仍提出 N 个候选，候选序号接续累计数量；本任务导出只含本次候选，不重复导出旧结果。
- **不同题目内容**：记忆相互隔离，不将前一道不同题目的反馈直接传给下一道题。这里是“同题跨 ID、跨任务复用”，不是相似题检索或全局经验池。
- **身份与配置校验**：题目文字和有序图片内容哈希共同确定身份；换 ID、数据行重排、图片搬家但内容不变，均不影响身份。模型配置或实际服务地址改变时使用独立分组，旧记录仍保留，但不自动混用；只修改代码、依赖锁、N 或批次数上限不会改变记忆分组。匹配是精确内容匹配，不自动判断改写题意、不同编码的图片是否等价；图片顺序改变也会视为不同内容。
- **恢复与过期任务保护**：每批先保存轮次，再写记忆快照并原子更新最新指针；若中途异常，续跑会重放已保存轮次补齐记忆。重新打开已完成的旧任务不会回滚共享记忆。如果旧任务尚未完成、但同 seed 的记忆已被其他任务推进，则在新的模型调用前报错，避免覆盖较新状态。

首次遇到且没有兼容记忆的 seed，指导仍从空字符串开始。没有分歧时沿用已有指导；有分歧时，将旧指导与本批分歧一起总结为新指导，所有旧反思仍保留在历史记录中。复用不会将其他模型的私有分析或历史题目直接放入当前模型的出题请求。

本功能不会自动导入旧脚本的数据，或搬迁早期包含代码指纹/seed ID 的记忆目录；当前内容索引的状态格式为 `schema_version: 2`，用于后续保存和复用。已有任务输出仍受 manifest 校验保护，代码、输入 ID 或依赖变化后需要新建输出目录，但共享记忆目录保持不变，就可以按新规则继续读取历史。N 的均分规则保持不变，不因累计候选序号延续而改变模型分配起点。

## 论文流程、开源补充与实验配置

论文主流程包括四模型、四类策略、共识公式、分歧反馈和最多 10 轮。开源入口保留这些环节，并按作者要求支持 N 个候选的轮询分配、各自分析不共享，以及 R1 视觉转述。完整 prompts、生成温度、模型端点版本、seed 配比和最终 20K 的精确采样方式等实验细节，应结合作者的历史记录核对，不以推测代替。

当前实现按 seed 维护反馈、由 GPT-4o 汇总反思，并将大 N 的生成任务合并到不超过 `max_iterations` 的批次。这些具体调度设置需要作为开源配置明确记录，不能仅因它们在代码中存在就认定与历史实验完全相同。配置支持增减 Collective 模型；改变数量后不再是论文四模型的实验设置。

当前默认模型按作者的试跑选择配置：GPT-4o、Gemini-2.5-Pro、DeepSeek-R1、Claude-Sonnet-4-6。其中 Gemini 和 Claude 不等同于论文所写的 Gemini-2.5-Flash、Claude-4-Sonnet；当前测试不要求模型版本与论文严格一致。模型 ID 原样发送给配置的端点，不能仅凭名称保证网关内部实现。`thinking`/`nothinking` 别名保留在独立的连通性测试配置，不再作为主配置默认值。R1 视觉转述和 Nano Banana Pro 当前稳定版也是显式适配，不能当作论文当时的原始配置。

语义判等沿用现有工程思路：先严格字符串匹配，未匹配时由配置的模型做语义判等。论文公式没有给出判等实现。这个辅助判等模型不额外计入 K。

## DeepSeek-R1 的视觉转述

R1 默认 `visual_mode: caption`，由 `caption_model: gemini-2.5-pro` 提供视觉信息。这是作者要求保留的开源版视觉输入补丁。根据作者说明，历史实验中 R1 只看文字；开源版新增转述与该历史设置分别记录。

实际图片 → Gemini 的结构化转述 → R1 的纯文本请求。转述包含 `description`、逐项 `visible_text` 和 `uncertainties`，覆盖对象、属性、空间关系、几何标记、图表坐标/单位/图例/数值等；不能辨认的内容明确标注，不补猜隐藏条件或求解。

- 转述器只接收图片，不接收题目、参考答案、生成解答或绘图提示词，因此描述来自实际图像而不是预期的绘图内容。
- 种子分析时转述种子图，独立答题时转述新生成图；按图像内容与转述请求缓存，重复使用同一图片不会重复转述。
- R1 接收原任务文字和转述 JSON，实际 API 消息是纯文本字符串，不含 `image_url`。生成阶段只使用 R1 自己的 rationale；答题阶段不接收任何生成分析。
- `model_answers` 记录 `input_mode: caption` 和转述模型名称，转述请求及结果保存在 `calls/`，转述器不额外计入 K。
- 转述有可能遗漏信息或产生错误，而且会让 R1 与转述模型的判断相关；不能把适配后的 R1 当作原生视觉模型。
- 若想复现历史文字输入方式，将 R1 改为 `visual_mode: text_only`；视觉模型使用 `native`。缺少必要证据时，提示 R1 返回 `INSUFFICIENT_INFORMATION`，正常经过判等，而 API 错误会中断并保留续跑状态。

默认文本模型 ID 保留作者指定的写法，但公开配置不指定服务商地址。读者需要填写自己的端点与密钥，并通过连通性脚本验证访问权限；同名模型在不同服务商的可用性与行为不一定相同。

## 使用

### 默认输入：EasyR1 推荐格式

默认输入数据集为 Hugging Face 上的 **`hiyouga/geometry3k`，使用 `train` split**。[EasyR1 官方说明](https://github.com/hiyouga/EasyR1#custom-dataset)将其作为图文数据集格式示例，[数据集卡片](https://huggingface.co/datasets/hiyouga/geometry3k/blob/main/README.md)定义了 `problem`、`answer`、`images` 三列。因此，本项目默认要求扩展数据集遵循同样的结构；该约定同时写在 `config.yaml` 和 `seed_data.py` 的注释中。

| 字段 | 默认要求 |
| --- | --- |
| `problem` | 非空问题字符串；每张图片对应一个 `<image>` 占位符 |
| `answer` | 非空答案字符串；源答案仅归档，不传给生成器或新题评审模型 |
| `images` | 有序图片列表；支持 Hugging Face 解码后的 PIL 图片、图片 bytes、`{bytes, path}` 字典及本地路径 |

本地路径形式的单条示例：

```json
{"problem": "<image>求图中角度 x。", "answer": "45", "images": ["diagram.png"]}
```

多图时按占位符顺序提供图片；纯文本记录可使用没有 `<image>` 的问题及 `images: []`。缺列、空问题/答案、图片无法读取或占位符数量不匹配都会报错，不静默跳过。这里对齐的是图文数据格式，不包含 EasyR1 的视频输入。

配置中的 `dataset.path` 默认是 `hiyouga/geometry3k`，可换为其他同格式 Hub 数据集 ID，或者一个本地 Parquet、JSON、JSONL 文件。`dataset.name` 可指定 Hub 子配置；`split` 选择划分；`revision` 指定 Hub 分支、标签或 commit。发布可复现实验时应固定 commit，不能把默认的 `main` 当成不可变版本。合成默认只用训练划分，不自动合并 validation/test。

字段名不同时，修改 `prompt_key`、`answer_key`、`image_key` 即可，默认名称与 [EasyR1 示例配置](https://github.com/hiyouga/EasyR1/blob/main/examples/config.yaml)一致。JSON/JSONL 标准入口要求记录数组或逐行记录；原先以 ID 为键的 JSON 字典继续使用 `--seeds` 入口。

`geometry3k` 没有单独的 ID 列，默认使用“数据集来源 + 子配置 + split + 行号”构造 seed ID；如果自有数据集有稳定 ID，可设置 `dataset.id_key`。重复 ID 会报错。ID 仅用于来源追踪和任务进度：数据重排或更换 ID 后，只要题目文字与有序图片内容相同，仍可复用记忆。固定数据集版本或提供稳定 ID 仍有助于实验追溯。

图片不再要求事先人工导出：读取时统一存为 PNG 到 `dataset.image_cache_dir`（默认 `cads/dataset/seed_images/`），不调整分辨率，并保留透明通道。文件名按转换后图片内容哈希确定，相同图片可复用；图片列表顺序保持不变。`--images-dir` 可用于解析数据中的相对路径。不同任务应复用并保留这个目录，以维持输出路径和恢复检查稳定。

### 输入预检查与兼容入口

完成 uv 环境构建后，在 `cads/` 内运行。以下默认数据集预检查只加载输入并展示 N 的分配，**不调用任何生成或评审模型，但首次加载 Hub 数据集可能联网下载数据并写入图片缓存**：

```bash
uv run --locked python pipeline.py --limit 1 --dry-run
uv run --locked python pipeline.py --dataset hiyouga/geometry3k --split train --limit 1 --dry-run
uv run --locked python pipeline.py --dataset /path/to/train.parquet --limit 1 --dry-run
```

使用 Hugging Face 输入需要 `datasets` 和 Pillow；下载鉴权沿用 Hugging Face 本地登录或环境变量配置，不使用模型服务的 API key。安装脚本使用项目内 `HF_HOME`，不会自动读取旧缓存目录中的登录凭据，私有数据建议使用 `HF_TOKEN`。`--dry-run` 输出的 `model_api_calls: 0` 不代表 Hub 下载请求为零。

旧 JSON 入口仍然保留，`--seeds` 会覆盖 YAML 中的数据集设置，不能与 `--dataset` 或 `--split` 同时使用。离线审计只需标准库；旧 JSON 的 YAML dry-run 只需 PyYAML，不需 `datasets` 或模型 SDK：

```bash
uv run --locked python audit_legacy.py
uv run --locked python pipeline.py --seeds ../R1-ShareVL-52K/original_dataset/text_data.json --images-dir ../R1-ShareVL-52K/original_dataset/images --limit 1 --dry-run
```

如历史图片路径来自服务器，`--images-dir` 优先按文件名映射到本地；声明存在的图片找不到时会报错，不会静默改成纯文本。纯文本种子可用 `[{"idx":"task-1","description":"Create visual geometry problems about similar triangles."}]`。

### 服务商无关的端点配置

每个模型有独立的 API 客户端，可使用不同网关、地址和密钥；转述、反思和语义判等复用所选模型的客户端。**`provider` 在这里表示 API 请求协议，不表示必须向该公司购买服务。** 只要服务端兼容相应协议，就可以配置自己的地址；不能仅改模型名而混用消息格式。公开 YAML/JSON 只保存环境变量名称，不内置某个网关地址或实际密钥：

| 模型 | provider | 端点环境变量 | 密钥环境变量 |
| --- | --- | --- | --- |
| `deepseek-r1` | `openai` | `DEEPSEEK_BASE_URL` | `DEEPSEEK_API_KEY` |
| `gpt-4o` | `openai` | `GPT4O_BASE_URL` | `GPT4O_API_KEY` |
| `gemini-2.5-pro`（OpenRouter ID：`google/gemini-2.5-pro`） | `openai` | `OPENROUTER_BASE_URL` | `OPENROUTER_API_KEY` |
| `claude-sonnet-4-6` | `anthropic` | `CLAUDE_BASE_URL` | `CLAUDE_API_KEY` |

填写 API 基础地址，不要填写完整操作路径，也不要把密钥放进 URL。默认 `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`，文本客户端会追加 `/chat/completions`；Anthropic 基础地址通常以 `/v1` 结尾并追加 `/messages`。其他 OpenAI 兼容端点通常也以 `/v1` 结尾，以服务商文档为准。

默认 Gemini 通过 OpenRouter 的 OpenAI 兼容协议接收 `image_url` 数据 URL，因此配置中的 `provider` 是 `openai`；模型身份仍由 `google/gemini-2.5-pro` 决定。代码仍保留 Gemini 原生协议适配，供其他配置使用。Claude 原生请求使用 base64 `image/source`。原生 Claude/Gemini 适配只提取最终文本并拒绝截断输出；Claude 的 1024 token 上限沿用作者提供的示例，长 JSON 可能需要后续提高。

在 `cads/` 中复制模板并自行编辑 `.env.local` 中的地址和密钥，再导入当前 shell；Python 不会自动加载该文件：

```bash
cp .env.example .env.local
```

编辑完成后执行：

```bash
set -a
source .env.local
set +a
```

只 source 你自己编辑且信任的文件。`.env.local` 已被 Git 忽略，`.env.example` 只给出 OpenRouter 公共基础地址，不包含密钥。默认配置中 Gemini 文本模型和 Nano Banana Pro 共用一个 `OPENROUTER_API_KEY`；GPT-4o、DeepSeek-R1、Claude 仍使用各自变量。不同服务商请分别配置，不能把一个服务商的密钥随意发给其他服务。

配置中的模型条目例如：

```yaml
questions_per_seed: 8
models:
  - name: deepseek-r1
    model: deepseek-r1
    provider: openai
    visual_mode: caption
    base_url_env: DEEPSEEK_BASE_URL
    api_key_env: DEEPSEEK_API_KEY
```

该片段只展示单个模型条目的写法，不是完整配置；其余模型和角色设置保留在完整 YAML 中。仍兼容私有配置中的 `base_url`/`api_key` 直接值，但非空直接值优先于环境变量，公开配置不应写入真实密钥。模型未声明某个连接字段时，兼容旧的顶层 `api_key_env` / `base_url_env`。日志、缓存、manifest 和 dry-run 不保存直接配置的 API key；配置文件本身由你保管。

### 文本模型连通性检查

完成环境安装并设置好四个文本模型的地址/密钥后，在 `cads/` 执行：

```bash
.venv/bin/python check_connectivity.py
.venv/bin/python check_connectivity.py --all --timeout 30
.venv/bin/python check_connectivity.py --model gemini-2.5-pro --max-tokens 2048
```

每个模型只发送一条纯文本 `hello`，不使用缓存、不要求 JSON、不传图片、不加载数据集、不触发生图或合成。默认单次请求超时 30 秒、输出上限 128 tokens、自动重试为 0；`--max-tokens` 可调整测试上限，不会修改正式生成配置。需要收到非空最终文本才通过；HTTP 错误、超时、空响应、截断或协议不匹配均报错，不把空 HTTP 200 当成功。

- 默认遇到第一个失败立即输出 `[FAIL]` 并退出，退出码为 1；配置缺失/错误退出码为 2。
- `--all` 仍即时报告每个错误，但继续检查剩余模型，最终只要有失败就返回 1；全通过返回 0。
- `--model` 可按配置中的名称选择单个模型，重复该参数可选择多个，避免排查失败模型时重复请求已通过的模型。
- 输出只包含模型名、耗时、响应字符数和安全错误类别，不输出密钥、原始响应正文或含密钥的异常 URL。
- 生图保持禁用也能检查这四个文本模型，不要求填写生图地址/密钥。

如果当前四个端点确实使用同一个 key，可先设置端点变量，再用终端隐藏输入临时密钥，不落盘，也不修改环境变量：

```bash
.venv/bin/python check_connectivity.py --all --prompt-api-key
```

此测试会消耗少量模型额度。它只验证文本请求与最终文本响应，不证明原生看图、R1 视觉转述、生图、结构化 JSON 或论文数据质量可用，也不能核实网关背后的真实模型。安装、数据 dry-run 和断点重放不会偷偷执行 hello 请求；需要在正式合成前检查时，用 `check_connectivity.py && ...` 串联命令，让检查失败阻止后续运行。

只检查 OpenRouter 上的 Gemini，不调用其他三个 Collective 模型时，设置 `OPENROUTER_API_KEY` 后运行：

```bash
.venv/bin/python check_connectivity.py --config config.connectivity.yaml --all --timeout 60 --max-tokens 2048
```

`config.connectivity.yaml` 只包含一个逻辑名为 `gemini-2.5-pro` 的模型，请求实际发送 OpenRouter ID `google/gemini-2.5-pro`。也可加 `--prompt-api-key` 临时输入密钥；这只验证文本响应，不触发生图。

```bash
.venv/bin/python check_connectivity.py --config config.connectivity.yaml --model gemini-2.5-pro --timeout 60 --max-tokens 2048
```

### 正式代码与本地实验的边界

单 seed 读题、生成前检查及缺图故障注入等实验脚本仅保存在 Git 忽略的 `.tmp/` 中，不属于正式开源入口，不随正式代码提交。实验输出保存在同样被忽略的 `outputs/` 中；本文的本机验证记录不代表发行包包含这些诊断工具。通用的文本连通性检查保留供读者验证自己的接口。

正式 `pipeline.py` / `evaluation.py` 不提供缺图绕过模式；缺图诊断结果不能当成论文共识评分或合格训练数据，其临时反馈也不得写入正式 seed 记忆。

## Gemini 与 Nano Banana Pro 的 OpenRouter 接口

默认配置将 Gemini 文本模型和 Nano Banana Pro 都放在 OpenRouter。OpenRouter 的文本模型 ID 是 [`google/gemini-2.5-pro`](https://openrouter.ai/google/gemini-2.5-pro)，Nano Banana Pro 的模型 ID 是 [`google/gemini-3-pro-image`](https://openrouter.ai/google/gemini-3-pro-image)。这里没有换成 Nano Banana 2。

截至 2026-09-23，OpenRouter 将 `google/gemini-2.5-pro` 标记为 2026-10-20 下线。当前配置仍固定该 ID，以便复现实验选择，不自动漂移到“latest”别名；如果正式运行时该模型已经不可用，需要明确更换模型 ID，并把变更记录到实验配置和结果说明中。

两者共用 `OPENROUTER_BASE_URL` 和 `OPENROUTER_API_KEY`。`provider: openai` 表示 Gemini 文本请求使用 OpenAI 兼容聊天协议；`image_generation.provider: openrouter` 表示生图使用 OpenRouter 专用 Images API：

```yaml
image_generation:
  enabled: false
  provider: openrouter
  model: google/gemini-3-pro-image
  base_url_env: OPENROUTER_BASE_URL
  api_key_env: OPENROUTER_API_KEY
  aspect_ratio: "1:1"
  resolution: "1K"
```

实现按 [OpenRouter Image Generation 文档](https://openrouter.ai/docs/guides/overview/multimodal/image-generation) 调用 `POST /api/v1/images`，请求包含 `model`、`prompt`、`n=1`、`aspect_ratio` 和 `resolution`。返回的 `b64_json` 会先做严格 base64 解码和 Pillow 图片校验，再统一保存为 PNG。Nano Banana Pro 当前支持 `1K`、`2K`、`4K`；默认使用最低档 `1K`，OpenRouter 配置省略分辨率时也补为 `1K`。这些画幅与分辨率不是论文参数；修改后应使用新的输出目录，避免与旧缓存混用。

`OPENROUTER_API_KEY` 必须具备两个模型的调用额度。启用生图后会检查地址和密钥均已填写；HTTP 429、常见 5xx 和远端连接中断按统一网络重试次数处理，错误信息不包含密钥或服务端响应正文。超时沿用 `request_timeout`（秒），也可在 `image_generation` 单独设置。

代码仍兼容旧的 Google GenAI 生图配置：省略 `image_generation.provider` 或设为 `google`，使用 `gemini-3-pro-image`、`api_version` 和 `image_size`。默认开源配置不走该分支，OpenRouter 配置应使用 `resolution`，不要同时填写 `resolution` 和 `image_size`。

**目前保持 `enabled: false`，没有执行真实生图。** 正式合成入口会在任何模型调用前拒绝运行；独立文本连通性检查不受此开关限制。等你准备好生图服务地址、key、配额并决定开始时，再手动设为 `true`。文本 hello 测试不代表生图接口已经可用。

## 后续实际运行

旧 JSON 配置格式仍可在 `cads/` 内通过 `--config config.example.json` 使用，并与 YAML 保持相同的 OpenRouter 默认项。正式生成前构建 uv 环境、配置所有地址与密钥并手动启用生图；以下命令先检查文本模型，全部通过才读取 YAML 中的默认 Hugging Face 数据集并启动合成：

```bash
.venv/bin/python check_connectivity.py --config config.yaml && \
  .venv/bin/python pipeline.py --config config.yaml --limit 1 --output outputs/pilot
```

示例配置单 seed 提出 8 个候选，每个模型各提出 2 个；每个候选由全部 4 个模型求解，另有分析、策略、绘图、转述、判等和反馈调用。先使用独立输出目录小规模验证真实 API；运行入口会实际消耗配额。目前没有运行真实候选题合成或向 Hub 上传数据；已授权的单 seed 分析会将选定题目及图片提交给模型服务。

输出：`all_candidates.json` 保存全部候选（包括筛掉的）；`generated_results.json` 为合格样本及全部追踪字段；`adversarial_results.json` 为所有分歧样本；`training_data.json` 使用现有 `idx/images/problem/answer` 字段，图片为本地路径，尚未打包为 Parquet；`rounds/` 保存每批轨迹；`calls/` 保存成功请求与响应。

标准数据入口还会将源答案、数据集来源、split、请求的 revision、行号及数据集 fingerprint 写入 seed 元数据，随 manifest 归档。源答案不会作为新题参考答案；新题仍由生成模型给出参考答案，再交给独立求解器评审。当前支持从 Hub 读取，不自动上传生成数据到 Hub。

候选数 N 不保证语义互不重复；生成提示会提供当前模型的历史题目，但尚未增加语义去重器。修改配置后需要新输出目录；不要将 `training_data.json` 中本地图片路径误当成已嵌入图片的 Hugging Face 数据集。

现有数据可先用 `audit_legacy.py` 恢复三模型的 C 分布和分歧池索引。它不会把旧数据升级为四模型已验证样本，也不会重判已有语义判等缓存。需要补跑缺失模型、统一 Gemini 版本并重新核算 C 后，才能用于四模型实验。旧数据没有闭环 provenance，不能事后宣称由反馈流程生成。

## 本次验证范围

已通过 Python 语法检查、真实本地 seed 的零请求 dry-run，以及临时模拟后端的两轮闭环检查：C=0/部分正确/全正确、只向评审提供题图、仅分歧样本进入反思、上下文进入第二轮、合格数据导出、断点恢复不重调后端、配置变化拒用旧输出、缺图和不完整布尔判定拒绝处理。

本次扩展还验证了 YAML/旧 JSON 读取，1/3/4 个模型及多种 N 的均衡分配，N=3/8/10/44 的完整生成与筛选后不补齐，私有分析和历史题目隔离，断点续跑，完整 seed 边界停止，独立模型地址/密钥路由，转述器仅接收图片，R1 收到纯文本及不确定性说明，转述缓存复用和密钥脱敏。全部使用离线模拟后端。

2026-09-23 将默认 Gemini 与 Nano Banana Pro 接入改为 OpenRouter：YAML/JSON 模型 ID 和环境变量一致性已通过；离线替身验证了 `POST /api/v1/images` 的 Bearer 鉴权、请求参数、base64 解码、PNG 校验和同请求缓存。随后使用作者授权的临时 key 跑通单 seed、N=4 的真实全流程：Gemini hello、四模型各出一题、4 张 1K 生图、四模型评审、语义判等、分歧反思、状态持久化和零请求恢复均完成。首次生图暴露的远端主动断连已加入安全重试。程序按 C>0 保留了 4/4，但人工复核发现错误参考答案、图文冲突、标签歧义和视觉非必要等问题，因此该 smoke 只证明接口链路可用，不证明训练数据质量合格；真实密钥未写入文件。

2026-09-22 的早期接口检查使用禁用 socket 连接的临时模拟客户端：YAML/JSON 配置一致、Gemini/Claude 原生请求路径/消息/图片/鉴权、响应解析、截断报错、重试及异常密钥脱敏均通过；同时验证了 Gemini 图片转述到 R1 的纯文本路由和缓存。生图部分验证 Google SDK 参数、毫秒超时、跳过 thought 图片、分辨率参与缓存、无最终图片时报错，以及默认禁用时不进入模型调用。该阶段使用模拟 SDK；参数检查不等于真实服务集成测试。

seed 记忆扩展已用断网模拟后端验证：同任务零调用恢复、新任务加载同 seed 的最终反馈/私有分析/历史、N 改变后的复用、累计候选 ID、无分歧时沿用指导、题目/模型变化隔离、关闭跨任务共享、生成中断恢复、轮次已落盘但记忆提交失败后的恢复、旧任务不回滚共享状态、过期未完成任务在请求前停止，以及持久化文件不含配置密钥。

记忆匹配规则修正后，再次用禁止联网的模拟后端验证：改变源码文件名/内容哈希或依赖锁哈希，仍定位同一份 seed 记忆；新任务直接使用已有私有分析、历史题目及反思，累计候选序号继续递增。题目、图片或模型变化仍隔离；同输出目录零调用恢复、不同 manifest 禁止混写同一次任务的保护保持不变。

内容索引扩展已用断网模拟后端验证：同题跨 ID、跨数据集来源、图片路径变化后的复用；四模型在多个批次与任务中各仅请求一次初始分析；同一任务内不同 ID 的同题共享历史、反馈与连续编号，同时独立保存来源进度和快照。中断恢复、旧任务重放不回滚最新记忆、题目/图片/图片顺序变化隔离也通过检查。以上没有调用真实模型。

模块拆分后，在 `cads/.venv` 中再次通过离线回归：真实 `datasets`/Pillow 的本地 Parquet、JSONL、PIL 图片及原始字节读取，字段映射、多图、透明度与尺寸保留、格式错误检查；Hub 默认参数通过模拟加载边界验证，未下载远端数据集。另验证了 N 不能整除模型数量时的均衡分配、独立分析与历史、同模型绘图提示词、筛选不补齐、共享反馈、同任务零请求恢复及跨任务 seed 记忆复用。`uv lock --check --offline` 和锁定环境下的 CLI 帮助检查也已通过。临时输入和输出仅在 `cads/.tmp/` 创建。

锁定环境中的实际 Google SDK 类型、图片响应解析与落盘也通过离线检查：服务调用被模拟替换，覆盖毫秒超时、图片参数、thought 图片跳过、无最终图片时报错和图片缓存；R1 的纯文本转述路由与缓存再次通过检查。整个检查过程禁止 socket 连接，没有请求真实模型服务。

一键安装与真实数据下载补充验证（2026-09-22）：`setup.sh` 的语法、参数报错、仅安装模式和已有环境的锁定同步均通过；本地 uv、系统 uv、自动安装 uv 三条分支用隔离模拟程序验证，包括带空格的路径及参数转发。随后实际执行 `bash cads/setup.sh --download-dataset --limit 2`，下载 `hiyouga/geometry3k` 到 `cads/dataset/`，解析到版本 `fd21e533e1e50d0662a2bf7b223e60511bd5f8b7`。下载器准备了 train/validation/test 缓存，但仅选择 train 进行输入检查。断网后全部 2,101 条训练记录与 2,101 张图片通过格式、解码和缓存路径检查；再次离线执行安装/数据检查命令也通过。此次网络访问仅用于下载数据，没有调用模型 API；本机验证报告保存在 Git 忽略的 `dataset/validation_report.json`。

连通性脚本补充验证（2026-09-22）：离线覆盖服务商无关配置、协议请求构造、零自动重试、首次失败停止/全部检查、缺少凭据、非法地址、空响应、超时、错误脱敏及非零退出码。随后按作者授权，对临时指定的服务端点发送纯文本 `hello`：GPT-4o、DeepSeek-R1、Claude-Sonnet-4-6 返回非空最终文本；Gemini-2.5-Pro-nothinking 首次有效响应检查失败，单独提高输出上限到 512 tokens 复测返回 HTTP 500，因此未通过。该结果只反映当次账号/端点状态，不代表模型普遍不可用。真实密钥通过终端隐藏输入使用，未写入文件；实际网关地址没有写入公开配置。

随后按作者要求，用独立的 `config.connectivity.yaml` 比较两个 Gemini 别名（同一临时端点/密钥，超时 60 秒，输出上限 2048 tokens，各发一次 `hello`，无自动重试）：`gemini-2.5-pro-nothinking` 在 10.13 秒后返回 38 字符的非空最终文本，通过；`gemini-2.5-pro-thinking` 在 6.62 秒后返回 HTTP 500，未通过，脚本退出码为 1。nothinking 前后结果存在波动，不能据此确定先前失败的根因是 token 上限；thinking 的单次 HTTP 500 也不能证明该型号永久不支持。该测试不要求模型版本与论文一致，没有改动主配置的四个 Collective 模型。

追加 `gemini-2.5-pro` 到同一测试配置后，仅对该名称发送一次 `hello`（同一临时端点/密钥，超时 60 秒，输出上限 2048 tokens，无自动重试）：18.37 秒后返回 32 字符的非空最终文本，通过，脚本退出码为 0。未重复请求其他模型，也未将主配置的 Gemini 型号自动替换为该名称。

随后按作者要求，将主配置及 R1 转述模型切换为 `gemini-2.5-pro`，独立 Gemini 对比配置仍保留三个别名。再次检查四个模型的 hello（60 秒超时、输出上限 2048 tokens 且不超过模型配置上限、无重试）：GPT-4o、DeepSeek-R1、Claude-Sonnet-4-6 通过，Gemini-2.5-Pro 返回 HTTP 500。此前 Pro 单次通过不代表该端点持续稳定，当前不能宣称四模型全部就绪。

单 seed 分析入口先通过禁止 socket 连接的模拟检查：四次独立分析加一次图片转述、转述器只看图、R1 只收文本及不确定性、源答案和他人分析隔离、零重试、失败后继续及非零退出码、成功缓存重放、配置变化拒绝混写，以及不调用生图/评审/反思、不更新 seed 记忆。随后按作者授权真实检查训练集第 1 题 `Find x.`（每次请求 120 秒超时、各模型输出上限 4096 tokens、无重试）：GPT-4o 和 Claude 返回有效分析 JSON，分别用时 19.18 秒和 18.41 秒；GPT 识别相交弦定理，Claude 读出同弦分段 4、6 与 x、8，给出 `4 × 6 = 8 × x`、`x = 3`。Gemini 自身分析返回 HTTP 500（7.33 秒），为 R1 提供图片转述的 Gemini 请求也返回 HTTP 500（6.05 秒），因此尚未调用 R1 的分析接口。总体 2/4 通过，退出码为 1；报告和两份成功响应保存在 Git 忽略的 `outputs/seed_analysis_pro_20260922/` 中。没有发送源答案，也没有生成新题或图片。

独立 Gemini 诊断（2026-09-22）：同一临时网关上的 `gemini-2.5-pro` 两种 hello 请求均成功（4096 tokens 上限 / 不设 generationConfig）；相同 seed 的正式分析和只看图的转述请求均返回 HTTP 500，脱敏错误类型为 `new_api_error`、错误码为 `do_request_failed`。最简 hello 加图片请求成功并读出 `x = 3`，但 OpenAI 兼容接口上的正式分析仍返回同类 500。因此不能将失败简单归因于不支持图片，也未证明换协议或修改生成参数能够修复。报告保存在本机 `outputs/gemini_diagnostic_20260922/`，没有修改公开服务地址或自动切换模型。

Google 官方临时检查：独立密钥可列出官方模型，目录中包含 `gemini-3-pro-image`（Nano Banana Pro）；但 `gemini-2.5-pro` 和 `gemini-2.5-flash` 的实际 hello 请求返回 404，提示不再向新用户开放。接口推荐的 `gemini-3.1-pro-preview` 返回 429，错误明确列出免费层配额上限为 0。模型列表成功不能等同于有实际生成额度，更不能证明 Nano Banana Pro 已经完成生图。随后按作者要求停止官方调用，恢复原网关的测试；官方临时 YAML 仅保存在 `.tmp/`，没有更改正式 Collective 配置，没有保存真实密钥，也没有发出生图请求。

生图前检查补充验证：新增入口通过禁止联网的 N=4/8/7 分配、私有分析/历史隔离、R1 转述、零重试、失败继续、不补齐、阶段中断恢复及 CLI 退出码检查；生图、评审、反思和 seed 记忆写入均设陷阱确认未触发。恢复时仅依赖完整请求缓存，早期失败恢复后会使后续受历史变化影响的请求重新执行，未受影响的模型继续复用缓存。

按作者确认的 N=4，使用原临时网关实际检查同一 seed：四份初始分析和 Gemini 到 R1 的纯文本转述全部完成，其中 GPT/Claude 两份相同分析从上次缓存复用。首轮 GPT/Claude 完成出题及绘图提示词；Gemini 出题返回 HTTP 500，R1 输出缺少顶层 domain。保留首轮报告后，只补测失败项一次：R1 完成出题及绘图提示词，Gemini 返回了题目内容，但 core_concepts 为数组而非约定的字符串，未通过校验。最终分析 4/4、合格结构的新题 3/4、绘图提示词 3/4，退出码 1；报告位于本机 `outputs/preimage_gateway_n4_20260922/`，首轮记录保存在其 `attempt1/`。未生图、未评审，候选尚不能作为已验证训练数据。

针对模型误添 all_strings 字段或将必填字段放入嵌套对象的问题，已将出题提示改为明确的顶层字段和字符串类型要求；没有放宽校验或静默改写模型答案。修改后离线回归通过，随后按作者要求进行了真实复测。

修订字段说明后的 N=4 复测：四份初始分析和图片转述复用原有成功缓存，四个候选均使用新提示实际请求。GPT、R1、Claude 完成新题和绘图提示词；Gemini 出题首轮返回 HTTP 500（5.40 秒），只对失败项补测一次仍为 HTTP 500（5.48 秒）。其他三个模型的完成结果没有重复调用。最终仍为分析 4/4、新题 3/4、绘图提示词 3/4，退出码 1；没有获得 Gemini 在新提示下的题目响应，不能判断其字段类型问题是否解决。独立报告位于本机 `outputs/preimage_gateway_n4_schema_20260922/`，首轮记录保存在其 `attempt1/`。

这次内容抽查也发现结构检查之外的问题：GPT 和 Claude 的题目文字已包含全部必要条件，图片不是求解必需；R1 重复了原 seed 的数值、关系和答案，数值变换策略没有真正落实。Claude 还输出了额外的 strategy_selected_note 字段，当前校验只检查必填字段，未拒绝额外字段；其绘图提示词的点数量描述也有小矛盾。具体记录见输出目录的 `content_review.md`。这些发现没有被计为正式评审、共识评分或反馈，也未写入 seed 记忆；现有候选不能直接视为合格训练数据。

早期临时依赖检查不是开源环境的复现方式，当前统一使用 `cads/.venv` 和 `uv.lock`。真实模型调用已覆盖 hello、四模型 seed 分析、R1 图片转述及部分生图前候选流程；四模型全部出题及绘图提示词尚未同时通过，更未执行真实生图、评审反馈闭环、训练或向 Hub 上传数据。这些检查不代表已复现论文数据质量或指标。
