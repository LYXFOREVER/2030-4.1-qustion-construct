# Dynamic Science Prompt Construction

本仓库用于从真实的开放式科学问答 benchmark 中动态抽取同一科学子领域的示例，并构建用于生成新科学问题的 Prompt。

当前版本只负责构建和打印 Prompt，不调用任何 LLM API，也不执行问题生成、幻觉检测、事实核验、去重或修复。

## 使用的 Benchmark

当前使用 [NVIDIA Nemotron-RL-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Science-v1) 的 `so_openq` 数据：

- 150,644 条英文开放式科学问答；
- 覆盖 Physics、Chemistry 和 Biology；
- 包含 39 个 `(topic, subtopic)` 组合；
- 核心字段为 `problem`、`expected_answer`、`metadata.topic` 和 `metadata.subtopic`；
- 数据许可证为 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。

原始数据约 266 MB，不纳入 Git 仓库。每条数据的具体来源和许可证信息以原始数据集中的 `metadata.QuestionLink` 和 `license` 字段为准。

## 当前状态

当前已经实现：

- 直接逐行读取 Nemotron 原始 JSONL；
- 在内存中仅保留构建 Prompt 所需字段；
- 随机选择一个 seed 样本；
- 按相同 `(topic, subtopic)` 寻找候选；
- 默认使用 seed 加最多两条同组问题，共最多三个 exemplar；
- 构建并打印要求模型生成开放式科学问答的完整 Prompt；
- 支持通过随机种子复现抽样结果。

当前尚未实现：

- 学科均衡抽样；
- 复杂的数据质量过滤；
- 模型调用与生成结果保存；
- 科学正确性检查、幻觉筛选和去重。

目前 seed 从全部数据中等概率抽取，因此输出会遵循原始数据分布，Physics 出现的概率明显高于 Chemistry 和 Biology。

## 项目结构

```text
.
├── benchmark/
│   └── raw/
│       └── so_openq.jsonl  # 本地下载，不纳入 Git
├── prompt.py               # 动态抽样并构建 Prompt
└── README.md
```

`logs/`、`benchmark/raw/` 和本地计划书均通过 `.gitignore` 排除。

## 下载数据

在仓库根目录执行：

```bash
mkdir -p benchmark/raw
curl --fail --location \
  'https://huggingface.co/datasets/nvidia/Nemotron-RL-Science-v1/resolve/main/so_openq.jsonl?download=true' \
  --output benchmark/raw/so_openq.jsonl
```

程序默认从以下位置读取数据：

```text
benchmark/raw/so_openq.jsonl
```

也可以通过 `--samples-path` 指定其他位置。

## 运行

项目只依赖 Python 标准库，不需要安装额外 Python 包。

直接随机构建一个 Prompt：

```bash
python3 prompt.py
```

使用固定随机种子复现抽样结果：

```bash
python3 prompt.py --seed 2030
```

调整 exemplar 数量和要求生成的问题数量：

```bash
python3 prompt.py --max-examples 3 --num-questions 5
```

指定数据文件：

```bash
python3 prompt.py --samples-path /path/to/so_openq.jsonl
```

## 抽样逻辑

1. 从全部样本中随机选择一个 seed；
2. 读取 seed 的 `topic` 和 `subtopic`；
3. 查找相同 `(topic, subtopic)` 的其他样本；
4. 将 seed 作为第一个 exemplar，再随机补充同组样本；
5. 将 exemplar 的问题和参考答案拼接到固定模板中；
6. 将完整 Prompt 输出到标准输出。

程序只检查输入 JSONL 是否包含必需字段，不改写原始问题和答案内容。

