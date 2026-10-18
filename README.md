# BizMind ML

**面向业务任务、通过本地验证持续自迭代的机器学习工程 Agent。**

提供业务目标、评估指标、约束、带标签训练数据和待预测数据，BizMind ML 自动建立基线、提出改进假设、生成训练代码、执行实验，并根据真实验证结果继续迭代。

## 自迭代流程

```text
业务要求 + 本地训练数据 + 待预测数据
                ↓
       建立固定验证划分和本地评分脚本
                ↓
       提出假设 → 生成方案 → 执行实验
                ↓
       独立评分 → 保存最佳预测与实验记录
                ↓
       分析效果和失败原因 → 下一轮改进
```

- **业务输入**：支持 YAML、Markdown、文本任务说明，可定义目标、数据列、验证规则和输出格式。
- **本地实验**：Prompt 要求只使用提供的数据和已完成的本地实验，不下载外部数据、模型或预测文件。
- **结果驱动**：第一轮建立基线；后续复用有效实验的代码和产物，同时保留失败经验。
- **探索与改进**：单代理交替改进当前最佳方案与探索新方法；多代理同时安排不同方案。策略位于 [strategy.py](bizmind/strategy.py)。
- **固定评估**：整个运行共用一次验证划分和同一评分脚本，验证标签由评估器单独保存。
- **设备适配**：自动检测 CUDA、Mac MPS 或 CPU，将实际设备和预算传给规划与代码执行代理。

## 快速开始

独立仓库：[AlanSong2077/BizMind](https://github.com/AlanSong2077/BizMind)。

需要 Python 3.11 或以上版本。获取代码并创建环境：

```bash
git clone https://github.com/AlanSong2077/BizMind.git
cd BizMind
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

模型通过 LiteLLM 调用。根据 `bizmind.yaml` 中的 `llm.model` 配置对应服务凭据，例如：

```bash
export OPENAI_API_KEY="your-api-key"
```

依赖包含多种训练库，安装时需要匹配系统、Python 和设备；完整依赖安装尚未在所有平台验证。模型服务调用会产生对应费用，生成的训练代码在本机执行。

编辑 [examples/business_task.yaml](examples/business_task.yaml)，提供真实本地数据和业务要求：

```yaml
background: 客户流失风险分析，支持运营确定联系优先级。
objective: 预测客户未来 30 天是否流失。
data:
  train: ../data/train.csv
  predict: ../data/test.csv
  target_column: churn
  id_column: customer_id
  time_column: snapshot_date
evaluation:
  primary_metric: roc_auc
  direction: maximize
  validation: 按时间划分，较早的 80% 用于训练，最近的 20% 用于验证。
constraints:
  - 只使用提供的本地数据。
  - 禁止使用预测时点后的信息，预处理只在训练部分拟合。
output:
  prediction_file: predictions.csv
  columns: [customer_id, churn]
  format: 每个预测样本一行，churn 为流失概率，保持原始顺序。
  artifacts: 保存模型、预处理流程和推理说明。
```

任务 YAML 中的数据相对路径以**该任务文件所在目录**为基准；命令行和运行配置中的相对路径以**当前工作目录**为基准。训练和预测数据必须同时提供，可以是文件或目录，不能相同或互相包含。示例不会自动生成业务数据。

启动实验：

```bash
python main.py
# 或使用示例运行配置
python main.py --config examples/local_training.yaml
```

也可以使用 [Markdown 任务模板](examples/business_task.md)，单独指定数据：

```bash
python main.py --task-id customer-churn --task-file ./my_task.md \
  --train-data ./data/train.csv --predict-data ./data/test.csv --device cpu
```

CLI 参数优先于运行配置；数据路径优先级为 **CLI → 运行配置 → 业务 YAML**。也支持 `--data-dir ./data`，此时需在任务文件中说明目录结构；已提供训练、预测数据对时优先使用该数据对。

## 运行配置

默认读取 [bizmind.yaml](bizmind.yaml)：

```yaml
task:
  id: business-task
  input_file: ./examples/business_task.yaml
  train_data: null
  predict_data: null
  input_dir: null
llm:
  model: gpt-5-mini
  max_retries: 3
  timeout: 600
agent:
  num_code_agents: 1
  num_iterations: 3
  num_iterations_code_agents: 20
  time_limit_code_agents: 18000
  workspace_dir: workspace
  prediction_file_name: predictions.csv
execution:
  device: auto
  max_gpu_count: 1
  max_cpu_cores: 4
  timeout: 3600
  inspect_interval: 600
```

| 配置 | 含义 |
| --- | --- |
| `task.id` | 本地任务名称，用于隔离运行目录 |
| `agent.num_iterations` | 外层实验规划、执行和反馈的轮数 |
| `agent.num_code_agents` | 每轮并发实验数量 |
| `agent.num_iterations_code_agents` | 每个代码代理的执行次数上限，也用于评估器准备阶段的代码生成次数上限 |
| `agent.time_limit_code_agents` | 每个代码代理的时间预算，秒 |
| `agent.prediction_file_name` | 业务预测输出文件名，应与任务输出要求一致 |
| `execution.timeout` | Notebook 单个代码单元的超时，秒 |
| `execution.inspect_interval` | 长时间执行中的进度检查间隔，秒 |
| `execution.max_cpu_cores` | 每个 Notebook 会话的 CPU 核心及线程上限配置 |
| `execution.max_gpu_count` | 每个会话的加速设备数量上限，`0` 强制 CPU |

配置仅接受 `task`、`llm`、`agent`、`execution` 四个部分，未知字段会报错。旧版比赛配置、社区开关和复现参数已移除，请使用新的配置模板。

`execution.device: auto` 按 **CUDA → Mac MPS → CPU** 选择可用设备，也可手动指定。MPS 使用共享内存；不支持所选加速器的库使用 CPU。建议先用单代理跑通，再增加并发；设备检测不提供独占资源调度。

## 输出与实验记录

结果保存在 `workspace/<task.id>/<时间戳>/`：

| 路径 | 内容 |
| --- | --- |
| `task_data/` | 本次数据映射，显式训练与预测数据对通过符号链接接入 |
| `evaluation/public/` | 训练划分、无标签验证输入和待预测数据 |
| `evaluation/private/` | 验证标签及评分报告 |
| `evaluation/split_dataset.py`、`evaluation/evaluate.py` | 固定划分与评分脚本 |
| `round-*/working/` | 每个实验的 Notebook、模型、执行日志及其他产物 |
| `round-*/working/best_prediction/` | 对应实验的最佳业务预测快照 |
| `round-*/working/best_code.py` | 最佳实验代码 |
| `round-*/working/experiment.json` | 假设、父实验、分数、经验与改进建议 |
| `predictions/` | 全局最佳预测及历史改进快照 |
| `best.json` | 全局最佳验证分数、优化方向和预测路径 |
| `experiments.json` | 已完成实验摘要和执行失败记录 |

后续轮次只选择已完成、已评分且具备代码的本地实验作为起点。实验记录保存在当前运行目录中，目前不会自动跨运行恢复。模型与预处理文件的具体格式由业务要求及生成代码决定，尚无统一模型部署接口。

## 验证与使用边界

```bash
python -m unittest discover -s tests -v
python main.py --help
```

安装运行依赖后，可执行不调用模型服务的合成数据冒烟验证：

```bash
python tests/smoke_local_loop.py
```

该检查使用固定模型回复，实际启动 Notebook、执行两轮训练与独立评分，验证实验复用、最佳预测和结果记录。

测试覆盖本地输入、配置、硬件选择、迭代反馈、产物复用及评分行为。使用模型和 Notebook 替身的测试不等同于真实模型服务与业务数据的端到端训练。

业务约束通过 Prompt 传递。验证划分和评分脚本仍需按任务核对；`private/` 是流程隔离，当前执行环境没有独立文件系统沙箱，生成代码具备当前进程的文件访问权限。反复使用同一验证集可能造成过拟合，重要任务建议另留最终测试集。

## 项目结构

```text
main.py                  # 命令行入口与本地输入组装
bizmind.yaml             # 默认运行配置
bizmind/
  agent.py               # 实验规划、自迭代与全局最佳结果
  coder.py               # Notebook 编码、执行与验证反馈
  evaluate.py            # 固定划分与独立本地评分
  strategy.py            # 基线、改进与探索策略
  task_input.py          # 业务任务读取与本地数据映射
  experiments/           # 方案草稿与本地实验报告
  config/                # 配置模型与读写
  utils/                 # 模型调用、硬件检测及 Notebook 执行
examples/                # 业务任务和运行配置模板
tests/                   # 回归测试
```

## 许可证

采用 [MIT License](LICENSE)。基础代码结构的来源及保留声明见 [NOTICE](NOTICE)。BizMind ML 独立维护。
