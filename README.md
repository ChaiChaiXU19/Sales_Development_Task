# Sales Development Task

一个基于 FastAPI + CrewAI 的“多智能体销售头脑风暴系统”。

系统输入一个销售项目的 Markdown 情报，结合“项目六要素”规则库进行多轮分析，最终输出《销售当前处境风险及下一步行动规划》普通文本。当前版本已经支持第三阶段 RAG，能够将产品知识和真实销售经验注入到策略、挑战、收口 3 个 Agent 中。

## 1. 功能概览

- CLI 运行：读取单个 Markdown 文件，输出最终《销售当前处境风险及下一步行动规划》普通文本
- API 服务：提供健康检查和 `/api/v1/brainstorm` 头脑风暴接口
- 六要素规则库：从 `data/six_elements_rules.xlsx` 加载销售分析标准
- RAG 检索：使用 PostgreSQL + `pgvector` 作为向量数据库
- 知识入库：支持 `md`、`txt`、`pdf`、`json` 四类知识文件

六要素固定为：

- `需求`
- `技术认可`
- `决策链`
- `竞争对手`
- `合作伙伴`
- `流程`

## 2. 项目结构

```text
.
├── app/
│   ├── api/                 # FastAPI 路由
│   ├── core/                # 配置和异常
│   ├── rag/                 # RAG 能力层
│   ├── schemas/             # API 请求/响应模型
│   └── services/            # 业务逻辑、CrewAI 编排
├── data/
│   └── six_elements_rules.xlsx
├── docker/
│   └── postgres/initdb/     # pgvector 初始化 SQL
├── knowledge/
│   ├── product/             # 产品知识
│   └── sales/               # 销售经验
├── scripts/
│   ├── ingest_knowledge.py  # 增量入库
│   └── rebuild_knowledge.py # 全量重建
├── tests/                   # RAG 单元测试
├── cli.py                   # CLI 入口
├── main.py                  # CLI 启动入口
├── docker-compose.yml       # PostgreSQL + pgvector
└── project_input_template.md
```

## 3. 环境准备

建议使用 Python 3.12。

### 3.1 创建虚拟环境

```bash
cd /Users/chaichaixu/Downloads/Sales_Development_Task
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2 配置环境变量

```bash
cp .env.example .env
```

然后编辑 `.env`，至少补齐：

- 聊天模型配置
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`
  - `MODEL_NAME`
- RAG 数据库配置
  - `RAG_PG_HOST`
  - `RAG_PG_PORT`
  - `RAG_PG_DB`
  - `RAG_PG_USER`
  - `RAG_PG_PASSWORD`
- Embedding 配置
  - `EMBEDDING_API_KEY`
  - `EMBEDDING_API_BASE`
  - `EMBEDDING_MODEL`
  - `EMBEDDING_DIMENSION`

说明：

- 聊天模型配置用于 CrewAI 任务执行
- 当前默认聊天模型为 MiniMax `MiniMax-M2.7-highspeed`
- 聊天模型通过 OpenAI-compatible 接口接入，推荐默认值：
  - `OPENAI_BASE_URL=https://api.minimaxi.com/v1`
  - `MODEL_NAME=MiniMax-M2.7-highspeed`
- Embedding 配置用于知识向量化
- 这两套配置是独立的，互不强耦合
- 当前推荐的 embedding provider 是阿里百炼千问，走 OpenAI-compatible 接口
- 推荐默认值：
  - `EMBEDDING_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1`
  - `EMBEDDING_MODEL=text-embedding-v4`
  - `EMBEDDING_DIMENSION=1024`
- `text-embedding-v4` 支持多档维度；首版默认使用 `1024`
- 按阿里云 `text-embedding-v4` 当前限制，项目会自动按最多 `10` 条文本一批调用 Embedding API
- 如果不配置 RAG，系统会自动降级为原来的纯 prompt 流程

## 4. 启动 PostgreSQL + pgvector

项目内已提供 `docker-compose.yml`，可直接拉起向量数据库：

```bash
docker compose up -d
```

查看状态：

```bash
docker compose ps
```

停止服务：

```bash
docker compose down
```

默认数据库参数与 `.env.example` 一致：

- Host: `127.0.0.1`
- Port: `5432`
- DB: `sales_brainstorm`
- User: `postgres`
- Password: `postgres`

初始化时会自动执行：

- 启用 `vector` 扩展
- 后续由 Python 代码在首次入库时创建业务表和索引

## 5. 准备知识库

知识目录固定为：

```text
knowledge/
├── product/
└── sales/
```

建议放入以下类型内容：

- `knowledge/product/`
  - 产品能力说明
  - 方案手册
  - 交付边界
  - 常见实施约束
- `knowledge/sales/`
  - 销售复盘
  - 竞对打法总结
  - 成功案例
  - 失败教训
  - 常用推进话术

注意：

- `knowledge/product/` 和 `knowledge/sales/` 中的真实知识文件只用于本地入库，不应提交到 GitHub 仓库
- 仓库仅保留目录占位文件 `.gitkeep`

支持文件类型：

- `.md`
- `.txt`
- `.pdf`
- `.json`

首版处理策略：

- Markdown：按标题层级优先切分，再做长度分片
- TXT：按空行和长度分片
- PDF：提取文本后按段落和长度分片
- JSON：按常见字段抽取文本块

## 6. 知识入库

### 6.1 增量入库

适用于日常新增或更新知识文件：

```bash
python scripts/ingest_knowledge.py
```

只处理某个命名空间：

```bash
python scripts/ingest_knowledge.py --namespace product
python scripts/ingest_knowledge.py --namespace sales
```

指定其他知识目录：

```bash
python scripts/ingest_knowledge.py --knowledge-dir /absolute/path/to/knowledge
```

### 6.2 全量重建

适用于 embedding 模型变更、维度变更、切片策略变更等场景：

```bash
python scripts/rebuild_knowledge.py
```

只重建某个命名空间：

```bash
python scripts/rebuild_knowledge.py --namespace product
```

如果你之前已经按旧的 embedding 维度建过 pgvector 表，这次切到千问 `text-embedding-v4` 后建议执行一次全量重建：

```bash
python scripts/rebuild_knowledge.py
```

### 6.3 入库行为说明

- 使用 `source_path + sha256` 判断文档是否变化
- 未变化文件会跳过，不重复写入
- 已变化文件会更新文档记录并重建其 chunks
- 使用 `text-embedding-v4` 时，入库脚本会自动将待向量化文本按最多 `10` 条一批发送到兼容接口
- 数据库 schema 和索引会在脚本启动时自动补齐

## 7. 运行 CLI

输入文件格式参考 [project_input_template.md](/Users/chaichaixu/Downloads/Sales_Development_Task/project_input_template.md)。

基础命令：

```bash
python main.py /absolute/path/to/your_project.md
```

调试模式：

```bash
python main.py /absolute/path/to/your_project.md --debug
```

输出内容：

- 最终《销售当前处境风险及下一步行动规划》普通文本步骤
- 如果加了 `--debug`，还会输出诊断、策略、挑战等中间结果

## 8. 运行 API

启动 FastAPI 服务：

```bash
uvicorn app.main:app --reload
```

默认地址：

- 健康检查：`GET http://127.0.0.1:8000/health`
- 头脑风暴：`POST http://127.0.0.1:8000/api/v1/brainstorm`

请求体示例：

```json
{
  "project_name": "某制造业AI质检项目",
  "markdown_content": "# 项目名称：某制造业AI质检项目\n\n## 需求\n...\n\n## 技术认可\n...\n\n## 决策链\n...\n\n## 竞争对手\n...\n\n## 合作伙伴\n...\n\n## 流程\n...",
  "debug": true
}
```

## 9. RAG 如何接入到 Agent

当前检索工具只挂载到以下 3 个 Agent：

- Strategist
- Challenger
- Closer

Diagnostician 不使用知识库，避免诊断被外部经验污染。

运行逻辑：

1. 先解析项目 Markdown 输入
2. 加载六要素规则库
3. 如果 RAG 配置完整且数据库可连通，则创建检索工具
4. Strategist / Challenger / Closer 在任务执行前可调用知识检索
5. 如果 RAG 不可用，系统自动降级为纯 prompt 分析

当前 4 个 Agent 的输出约束补充如下：

- Diagnostician 在诊断阶段除已知事实、关键缺口、风险等级、高危卡点外，还会显式输出 `核心盲区` 与 `核心不足`
- Strategist 的动作设计要优先回应诊断中暴露出的高危卡点、核心盲区和核心不足
- Challenger 在压力测试阶段会显式输出 `策略盲区揭示` 与 `执行资源不足`
- Closer 若识别到前序存在核心盲区，最终文本中必须至少包含 1 个探雷/验证信息动作
- Closer 只输出 3-5 个普通文本步骤，每个步骤包含 `当前处境风险 / 下一步动作 / 时间期限 / 对接对象 / 核心目的`
- API 只返回最终展示字段 `closer_result`；该字段直接使用 Closer 原始输出，不做后处理修复、校验或兜底改写

## 10. 测试

当前已包含 RAG 相关单元测试：

```bash
python -m unittest discover -s tests
```

这些测试覆盖：

- 文档加载
- 文本切片
- 增量入库逻辑

语法编译检查：

```bash
python -m compileall app scripts tests
```

## 11. 常见问题

### 11.1 `缺少 API Key`

说明聊天模型配置未提供。补齐 `.env` 中的：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `MODEL_NAME`

### 11.2 `RAG 配置不完整`

说明数据库或 embedding 配置缺失。补齐：

- `RAG_PG_*`
- `EMBEDDING_*`

### 11.3 PostgreSQL 能启动，但无法检索

优先检查：

- 是否已执行 `python scripts/ingest_knowledge.py`
- `EMBEDDING_DIMENSION` 是否和实际 embedding 模型维度一致
- 如果刚从旧模型切到 `text-embedding-v4`，是否已执行 `python scripts/rebuild_knowledge.py`
- `knowledge/product` 或 `knowledge/sales` 中是否真的有文件

### 11.4 没配 RAG 会不会影响 API/CLI

不会。当前实现里，RAG 是可选增强能力，不可用时会自动降级。

## 12. 推荐启动顺序

如果你是第一次接手这个项目，建议按这个顺序跑：

```bash
cd /Users/chaichaixu/Downloads/Sales_Development_Task
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
python scripts/ingest_knowledge.py
uvicorn app.main:app --reload
```

如果只是先验证基础能力，不想配 RAG，也可以：

```bash
python main.py /absolute/path/to/your_project.md
```

## 13. 下一步建议

当前版本已经完成基础 RAG 闭环。后续可以继续扩展：

- 给 API `debug` 模式返回检索命中摘要
- 增加 BM25 / hybrid search
- 增加 reranker
- 增加知识权限隔离或多租户支持
- 对接 CRM、Notion、飞书文档等外部知识源
