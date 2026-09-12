import os
import sys

# 将项目根目录加入 sys.path 以规范引用 rag_core 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

# 兼容新老版本 LangChain 导入
try:
    from langchain.agents import AgentType
except ImportError:
    from langchain_classic.agents import AgentType

from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import pypdf
import docx
import io
import json

try:
    from langchain_community.callbacks import StreamlitCallbackHandler
except ImportError:
    from langchain.callbacks import StreamlitCallbackHandler

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

import streamlit as st
from rag_core.providers import PROVIDERS


# 页面配置
st.set_page_config(page_title="多功能数据与文档智能助手", page_icon="📊", layout="wide")
st.title("📊 多功能数据与文档智能分析助手")
st.caption("🚀 支持表格统计计算（CSV、Excel、JSON）与文档知识库问答（PDF、Word、TXT、Markdown）")


# 辅助函数：根据文件格式提取数据或文本
def load_file_content(file):
    filename = file.name.lower()
    
    # 1. 表格数据 (CSV)
    if filename.endswith(".csv"):
        df = pd.read_csv(file)
        return "dataframe", df
    
    # 2. 表格数据 (Excel)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
        return "dataframe", df
        
    # 3. JSON 格式
    elif filename.endswith(".json"):
        try:
            df = pd.read_json(file)
            return "dataframe", df
        except Exception:
            file.seek(0)
            text = file.read().decode("utf-8", errors="ignore")
            return "document", text
            
    # 4. PDF 文档
    elif filename.endswith(".pdf"):
        reader = pypdf.PdfReader(file)
        pages_text = []
        for idx, page in enumerate(reader.pages):
            page_str = page.extract_text()
            if page_str:
                pages_text.append(f"【第 {idx+1} 页】\n{page_str}")
        return "document", "\n\n".join(pages_text)
        
    # 5. Word 文档 (.docx)
    elif filename.endswith(".docx"):
        doc = docx.Document(file)
        lines = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_str = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                if row_str:
                    lines.append(row_str)
        return "document", "\n\n".join(lines)
        
    # 6. 纯文本 / Markdown
    elif filename.endswith((".txt", ".md")):
        text = file.read().decode("utf-8", errors="ignore")
        return "document", text
        
    else:
        return "unknown", None


# 上传文件组件
uploaded_file = st.file_uploader(
    "📁 请上传需要分析的文件（支持 CSV, Excel, PDF, Word, TXT, Markdown, JSON 等）",
    type=["csv", "xlsx", "xls", "pdf", "docx", "txt", "md", "json"],
    help="上传后即可针对表格进行数据统计，或针对文档进行智能问答与总结"
)
    
# 侧边栏配置：模型供应商与参数
st.sidebar.header("⚙️ 模型配置")
st.sidebar.warning("🛡️ 安全提示：Pandas Agent 允许直接执行生成的 Python 运算代码。请仅在个人受信任环境运行，切勿部署到不可信公网环境。")

provider = st.sidebar.selectbox("选择模型供应商", list(PROVIDERS.keys()))

# 动态展示对应供应商的 Base URL 与模型选项
if provider == "自定义 (Custom)":
    base_url = st.sidebar.text_input("Base URL (API 接口地址)", value="https://api.openai.com/v1")
    model_name = st.sidebar.text_input("模型名称 (Model Name)", value="gpt-4o-mini")
else:
    preset = PROVIDERS[provider]
    base_url = st.sidebar.text_input("Base URL (接口地址)", value=preset["base_url"])
    model_options = preset["models"] + ["手动输入其他模型..."]
    selected_option = st.sidebar.selectbox("选择模型", model_options)
    default_m = preset.get("default_model") or preset["models"][0]
    if selected_option == "手动输入其他模型...":
        model_name = st.sidebar.text_input("输入模型名称", value=default_m)
    else:
        model_name = selected_option

# API Key 输入（优先从环境变量/.env 读取，本地 Ollama 自动预设占位符）
if "Ollama" in provider:
    default_api_key = "ollama"
else:
    env_var = PROVIDERS.get(provider, {}).get("env_key", "OPENAI_API_KEY")
    default_api_key = os.getenv(env_var, os.getenv("OPENAI_API_KEY", ""))

api_key = st.sidebar.text_input(f"{provider} API Key", value=default_api_key, type="password")

# 若用户在前端修改或填入了新 Key，自动同步持久化到本地 .env 文件
if "Ollama" not in provider and api_key and api_key != os.getenv(PROVIDERS.get(provider, {}).get("env_key", "OPENAI_API_KEY"), ""):
    env_var = PROVIDERS.get(provider, {}).get("env_key", "OPENAI_API_KEY")
    dotenv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    try:
        from dotenv import set_key
        set_key(dotenv_file, env_var, api_key)
        os.environ[env_var] = api_key
    except Exception:
        pass

# 采样温度调节（部分推理模型如 o1/o3-mini/reasoner 默认固定温度或不支持自定义）
is_reasoning_model = any(tag in model_name.lower() for tag in ["o1", "o3", "reasoner"])
if is_reasoning_model:
    st.sidebar.caption("ℹ️ 当前为深度推理模型，温度参数已自动适配为官方默认。")
    temperature = 1.0
else:
    temperature = st.sidebar.slider("Temperature (随机性)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

# 主区域：文件解析与预览展示
if not uploaded_file:
    st.info("👆 请先在上方上传一个文件（CSV / Excel / PDF / Word / TXT / Markdown / JSON）。")
    file_type, file_data = None, None
else:
    file_type, file_data = load_file_content(uploaded_file)
    
    # 动态提示当前加载的文件
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("current_file_id") != current_file_id:
        st.session_state["current_file_id"] = current_file_id
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": f"已成功加载文件 `{uploaded_file.name}`！您可以向我提问关于该文件的任何数据分析、摘要或细节问题。"
        }]

    if file_type == "dataframe":
        col1, col2 = st.columns([1, 4])
        with col1:
            st.metric("📊 数据总行数", file_data.shape[0])
            st.metric("📋 数据总列数", file_data.shape[1])
        with col2:
            with st.expander("👀 点击展开/收起数据预览（前 5 行）", expanded=True):
                st.dataframe(file_data.head(5), use_container_width=True)
                
    elif file_type == "document":
        char_count = len(file_data)
        with st.expander(f"📄 查看文档内容预览（总计约 {char_count:,} 字）", expanded=True):
            preview_len = min(600, char_count)
            st.text(file_data[:preview_len] + ("\n\n...（后续内容已载入上下文）" if char_count > preview_len else ""))
            
    else:
        st.error("无法解析该文件，请确认格式是否正确。")

# 侧边栏清空历史记录按钮
if st.sidebar.button("🗑️ 清空对话历史"):
    st.session_state['messages'] = [{"role": "assistant", "content": "对话已重置，有什么可以帮您？"}]

if "messages" not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "您好！请先在上方上传文件，并在左侧配置 API Key。"}]
    
# 渲染对话气泡
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
# 接收用户输入
if query := st.chat_input(placeholder="向 AI 提问该文件的数据、统计、要点总结..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    
    if not uploaded_file or file_data is None:
        st.warning("请先上传需要分析的文件！")
        st.stop()
        
    if not api_key:
        st.info(f"请在侧边栏添加您的 {provider} API Key")
        st.stop()
    
    # 构造大语言模型初始化参数
    llm_kwargs = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url if base_url else None,
    }
    # 非推理模型传入用户设定的温度
    if not is_reasoning_model:
        llm_kwargs["temperature"] = temperature

    # 实例化大语言模型
    llm = ChatOpenAI(**llm_kwargs)   
    
    with st.chat_message("assistant"):
        if file_type == "dataframe":
            # 表格模式：调用 Pandas Agent 执行 Python 代码运算
            st_cb = StreamlitCallbackHandler(st.container())
            agent = create_pandas_dataframe_agent(
                llm, 
                file_data,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                allow_dangerous_code=True
            )
            response = agent.run(query, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
            
        elif file_type == "document":
            # 文档模式：防范大文档超长导致 Token 爆炸，实施智能截断保护 (最多保留 10000 字符)
            MAX_DOC_CHARS = 10000
            raw_len = len(file_data)
            if raw_len > MAX_DOC_CHARS:
                st.caption(f"📄 当前文档较长（共 {raw_len:,} 字符），为防止超出模型上下文限制，已截取前 {MAX_DOC_CHARS:,} 字符供分析。")
                doc_content = file_data[:MAX_DOC_CHARS] + f"\n\n... [文档内容过长，剩余 {raw_len - MAX_DOC_CHARS:,} 字符已省略] ..."
            else:
                doc_content = file_data

            # 构造提示词并流式生成回答
            rag_prompt = f"""你是一名专业的文档智能分析专家。请根据以下提供的文档内容，客观、严谨、详实地回答用户的问题。

【作答要求】：
1. 必须基于提供的文档内容回答；如果文档中未包含相关信息，请如实说明，严禁凭空捏造；
2. 尽可能引用文档中的关键数据、结论、条款或原文句子作为佐证；
3. 输出排版清晰，结构完整，重点突出。

【文档内容】：
{doc_content}

【用户提问】：
{query}
"""
            # 实时流式输出
            response = st.write_stream(llm.stream(rag_prompt))
            st.session_state.messages.append({"role": "assistant", "content": response})
