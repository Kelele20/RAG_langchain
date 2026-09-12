# 🚀 RAG Studio 2.0 快速上手与操作指南

本手册提供针对 **RAG Studio 2.0**（`RAG_langchain` 项目）从零到一的快速搭建、大模型密钥对接、核心功能实操及常见问题排错的详尽指南。

---

## 目录
- [1. 系统要求与环境准备](#1-系统要求与环境准备)
- [2. 一键安装与环境搭建](#2-一键安装与环境搭建)
- [3. 大模型供应商对接与密钥配置](#3-大模型供应商对接与密钥配置)
- [4. Web UI 四大功能模块实操教程](#4-web-ui-四大功能模块实操教程)
- [5. 数据分析 Agent (`chat_csv.py`) 使用指南](#5-数据分析-agent-chat_csvpy-使用指南)
- [6. 常见问题排查与 FAQ](#6-常见问题排查与-faq)

---

## 1. 系统要求与环境准备

| 项目 | 最低配置 | 推荐配置 | 备注说明 |
|:---|:---|:---|:---|
| **操作系统** | Windows 10/11, macOS, Linux | Windows 11 / Ubuntu 22.04 LTS | 跨平台兼容 |
| **Python 版本** | Python 3.10 | Python 3.10 或 3.11 | 推荐使用虚拟环境 |
| **包管理工具** | `pip` | [`uv`](https://github.com/astral-sh/uv) | 推荐使用 `uv`，安装依赖速度提升 10 倍以上 |
| **内存 (RAM)** | 4 GB | 8 GB 及以上 | 本地运行 Embedding 模型与 Chroma 向量数据库 |
| **网络** | 访问对应大模型服务商 API | 国内直连（DeepSeek/阿里/智谱） | 本地 Ollama 模式无需任何外网 |

---

## 2. 一键安装与环境搭建

### 步骤一：克隆代码仓库到本地
```bash
git clone https://github.com/blackinkkkxi/RAG_langchain.git
cd RAG_langchain
```

### 步骤二：创建并激活虚拟环境

#### 推荐方式：使用极速包管理器 `uv`
```bash
# 如果尚未安装 uv，可通过官方命令安装:
# Windows (PowerShell): powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境
uv venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 安装全量依赖
uv pip install -r requirements.txt
```

#### 传统方式：使用 Python 内置 `venv` 与 `pip`
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 安装全量依赖
pip install -r requirements.txt
```

---

## 3. 大模型供应商对接与密钥配置

本项目采用业界标准的「凭据与界面偏好分立解耦」机制，彻底杜绝配置重合与安全隐患：
* **`.env` 文件（密钥凭据唯一中心）**：专门管理各大模型 API 密钥。您可以直接根据 `.env.example` 创建并编辑 `.env`，亦可在 Web UI 界面中直接填入 Key，系统会自动安全同步持久化至本地 `.env`，**刷新页面不丢失**。
* **`.kb_config.json` 文件（UI 偏好中心）**：仅用于记忆您在界面上选中的供应商、自定义模型名称、接口 Base URL、采样温度与 Top-K 等控件状态，**绝不存储任何 API Key**。

### 常用服务商获取与配置说明

#### 1. DeepSeek 官方（强烈推荐：性价比极高、推理能力出色）
- **获取地址**：[https://platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)
- **Base URL**：`https://api.deepseek.com/v1`
- **支持模型**：
  - `deepseek-chat`（通用对话与快速 RAG 检索生成，极力推荐）
  - `deepseek-reasoner`（R1 深度思维链推理模型）

#### 2. 阿里云百炼 / 通义千问 (DashScope)
- **获取地址**：[阿里云百炼控制台](https://dashscope.console.aliyun.com)
- **Base URL**：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- **支持模型**：`qwen-plus`, `qwen-turbo`, `qwen-max`, `qwen-vl-max`（多模态）

#### 3. 智谱 AI (GLM)
- **获取地址**：[智谱 AI 开放平台](https://open.bigmodel.cn)
- **Base URL**：`https://open.bigmodel.cn/api/paas/v4`
- **支持模型**：`glm-4-flash`（免费/极速）、`glm-4-plus`

#### 4. 本地私有化部署 (Ollama 零成本方案)
- **特点**：**数据不出本地局域网，无需消耗 API Token 额度，支持离线运行**。
- **配置步骤**：
  1. 下载安装 [Ollama](https://ollama.com/)；
  2. 命令行下载模型：
     ```bash
     ollama run llama3.1
     # 或中文较好的千问模型
     ollama run qwen2.5:7b
     ```
  3. 在 Web UI 供应商选择 `Ollama (本地私有部署)`，Base URL 保持 `http://localhost:11434/v1`，模型名称填写 `qwen2.5:7b`，无需输入 API Key 即可使用！

---

## 4. Web UI 四大功能模块实操教程

在终端中执行以下命令启动知识库工作台：
```bash
uv run streamlit run kb_app.py
```
终端将输出访问链接并自动调起浏览器访问：`http://localhost:8501`。

---

### 功能模块 1：💬 知识库智能问答

1. **左侧侧边栏配置**：
   - 选择大模型供应商（例如 `DeepSeek`）。
   - 填入对应供应商的 API Key。
   - 确认或自定义要调用的模型名称（如 `deepseek-chat`）。
   - 根据需要微调 **Temperature (采样温度)** 和 **检索切片 Top-K**（默认 4 个切片）。
2. **提问与交互**：
   - 在底部的聊天输入框输入针对您知识库文档的问题。
   - 观察回答上方的**响应类型徽标**：
     - 🟢 **`[L1 精确匹配缓存]`**：曾提问过完全一样的问题，0ms 秒级极速回放，0 Token 消耗。
     - 🔵 **`[L2 语义相似度命中 (xx%)]`**：提问虽然用词不同，但语义余弦相似度命中历史问答，毫秒级召回。
     - 🟡 **`[Chroma 向量检索]`**：首次提问或缓存未命中，实时检索向量切片召回并由大模型流式总结生成。
3. **查阅溯源卡片**：
   - 展开回答下方的「🔍 查看知识库切片溯源引用」，即可看到模型生成回答所参考的具体文件、页码/工作表、相似度打分及原文内容。

---

### 功能模块 2：📚 文件与增量管理

1. **上传文件构建知识库**：
   - 在「上传新文档至知识库」区域拖拽上传文档（支持 `.pdf`, `.docx`, `.xlsx`, `.csv`, `.json`, `.txt`, `.md`）。
   - 系统将自动提取内容、递归切分文本块，并存储至持久化 Chroma 向量库。
2. **本地目录一键同步 (/schedule)**：
   - 如果您有一个存放大量文档的本地文件夹（例如项目中的 `data/` 目录），无需逐个手动上传；
   - 填入目录路径 `data`，点击「执行全量增量同步」；
   - **秒级跳过 (`SKIPPED`)**：未改动的文件直接跳过，零 Token 消耗；
   - **智能更新 (`UPDATED`)**：修改过的文件会自动剔除旧切片并重建新切片；
   - **防误删保护**：仅清理同步目录中被物理删除的文件，通过 Web 手工上传的独立文件受保护不被误删。
3. **知识库文档看板与单个删除**：
   - 下方表格实时列出当前知识库已索引的所有文档及其切片数、更新时间。
   - 选中指定文档即可点击「从知识库中移除该文档」执行一键下架清理。

---

### 功能模块 3：👁️ OCR + VLM 图像解析

1. **上传图片**：
   - 支持上传 `.png`, `.jpg`, `.jpeg`, `.webp` 格式的架构设计图、流程图、数据报表或扫描文件。
2. **执行解析**：
   - 右侧可高清预览上传的图片；
   - 支持自定义解析 Prompt 指令（默认会自动把表格转录为 Markdown 表格，把架构图提炼为实体关系与时序流程）；
   - 点击「开始多模态解析」，模型输出格式清晰的 Markdown 文本。
3. **一键存入知识库**：
   - 解析完成后，点击「一键将解析内容保存至知识库」，即可将图片提取出的 Markdown 自动向量化存库，随后即可在问答 Tab 中针对该图片内容发起提问！

---

### 功能模块 4：⚡ 分层缓存与性能监控

1. **核心看板指标**：
   - **总查询次数**：统计自部署以来的累计提问数。
   - **L1 命中次数 / L2 命中次数**：分别统计精确匹配与语义匹配的次数。
   - **总命中率**：缓存命中数占总查询数的百分比。
   - **预估节省 Token 数量**：按每次命中节省的 Prompt 与 Completion Token 精准估算。
2. **动态语义阈值调节**：
   - 滑块调节 L2 语义余弦相似度阈值（建议 `0.88 ~ 0.95`）。
   - 阈值越高，对语义相似度要求越严苛；阈值越低，复用历史问答的宽容度越大。
3. **缓存维护**：
   - 查阅最近被缓存的问答历史列表；
   - 点击「一键清空全部缓存」可彻底释放缓存池。

---

## 5. 数据分析 Agent (`chat_csv.py`) 使用指南

如果您需要对结构化表格进行复杂的统计、计算、交叉分析，或者对超长文档进行快速单文件阅读，可启动轻量级数据助手：

```bash
uv run streamlit run agent/chat_csv.py
```

* **表格分析模式 (CSV / Excel / JSON)**：
  - 上传表格后，系统自动展示行数、列数与前 5 行预览。
  - 用户用自然语言提问（如：“统计各部门的平均薪资并找出前 3 名”、“计算各季度销售额的环比增长率”）。
  - 内置 Pandas DataFrame Agent 会自动生成 Python 代码并执行计算，输出最精准的数学运算结果。
* **文档分析模式 (PDF / Word / TXT / MD)**：
  - 上传后自动预览文本总字数。
  - 内置超长文档安全截断机制（上限 10,000 字符），防止发送给大模型时超出 Token 限制。

---

## 6. 常见问题排查与 FAQ

### Q1: 启动时报错 `ModuleNotFoundError: No module named 'xxx'`？
- **排查方法**：请确保虚拟环境已激活（终端前缀显示 `(.venv)`），并重新执行：
  ```bash
  uv pip install -r requirements.txt
  ```

### Q2: Streamlit 默认端口 8501 被占用怎么办？
- **排查方法**：可以显式指定其他端口启动，例如：
  ```bash
  uv run streamlit run kb_app.py --server.port 8502
  ```

### Q3: 为什么提问时提示 `Chroma 知识库为空`？
- **排查方法**：首次运行知识库时尚未建立索引。请先切换到第二个 Tab **「📚 文件与增量管理」**，上传 1~2 个文档或指定 `data` 目录点击「执行全量增量同步」，等待索引完成后即可正常提问。

### Q4: 提示 `OpenAI API error: Incorrect API key provided`？
- **排查方法**：
  1. 检查左侧侧边栏填入的 API Key 是否正确（注意前后不要有空格）。
  2. 检查选择的供应商是否与 API Key 相匹配（例如 DeepSeek 的 Key 请选择 DeepSeek 供应商，不能选 OpenAI 官方）。

### Q5: 本地运行速度较慢或显存不足？
- **排查方法**：本项目默认采用 CPU 极速运算的本地轻量级 Embedding 模型（`all-MiniLM-L6-v2`），无需显卡即可流畅运行；分层缓存已全面采用 NumPy 矩阵内积加速，常规消费级电脑（8GB 内存）均可平稳运行。
