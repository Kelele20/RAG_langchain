# -*- coding: utf-8 -*-
"""
个人全功能本地知识库 Web UI (RAG Studio)
- 多供应商大模型支持 (DeepSeek, OpenAI, 阿里千问, 智谱GLM, Kimi, Ollama, OpenRouter)
- 4 大功能看板：
  1. 💬 知识库智能问答 (支持流式生成、分层缓存高亮、知识精确溯源)
  2. 📚 知识库文件与增量管理 (多格式解析、SHA-256增量去重、本地目录一键同步)
  3. 👁️ 多模态 OCR + VLM 图像解析 (高保真 Markdown 提取、一键存库)
  4. ⚡ 分层缓存与性能监控 (L1精确匹配、L2语义匹配、命中率与Token节省统计)
"""

import os
import sys
import time
import json
import base64
import streamlit as st
import pandas as pd
from dotenv import load_dotenv, set_key

# 优先加载根目录 .env 环境变量
load_dotenv()

from rag_core.cache import HierarchicalCache
from rag_core.indexer import IncrementalIndexer
from rag_core.engine import KnowledgeBaseEngine
from rag_core.vlm_ocr import parse_image_with_vlm, parse_document_file


# 页面基础配置
st.set_page_config(
    page_title="个人知识库系统 (RAG Studio)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

from rag_core.providers import PROVIDERS


def render_citations(sources: list, container=None):
    """提取的公共组件：渲染结构化参考来源与知识溯源卡片"""
    if not sources:
        return
    target = container.expander if container is not None else st.expander
    with target(f"📑 参考来源与溯源片段 (共 {len(sources)} 条)"):
        for idx, src in enumerate(sources, 1):
            loc = f"第 {src['page']} 页" if src.get("page") else (f"工作表: {src['sheet']}" if src.get("sheet") else f"切片 #{src.get('chunk_index', 0)}")
            st.markdown(
                f"**[{idx}] {src['source']}** `({loc})` "
                f"&nbsp;&nbsp;*相关度: {round(src['similarity']*100, 1)}%*"
            )
            st.code(src["content"][:400] + ("..." if len(src["content"]) > 400 else ""), language="text")


# 本地用户偏好配置持久化（仅保存 UI 控件状态，不存储任何 API Key，API Key 统一由 .env 管理）
CONFIG_FILE = ".kb_config.json"
DOTENV_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))


def load_persisted_config() -> dict:
    """加载本地保存的 UI 模型偏好配置，刷新页面不丢失"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_persisted_config(cfg: dict):
    """保存 UI 偏好配置至本地持久化文件（纯 UI 参数，绝不包含敏感 Key）"""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Config] 保存配置失败: {e}")


def save_api_key_to_env(env_key: str, api_key: str):
    """安全地将 API Key 保存/同步到本地 .env 文件中，不与 UI 配置文件重合"""
    if not env_key or not api_key:
        return
    try:
        if not os.path.exists(DOTENV_FILE):
            with open(DOTENV_FILE, "w", encoding="utf-8") as f:
                f.write("# ==========================================================\n# 本地真实环境变量配置 (.env)\n# ==========================================================\n")
        set_key(DOTENV_FILE, env_key, api_key)
        os.environ[env_key] = api_key
    except Exception as e:
        print(f"[Env] 同步密钥至 .env 失败: {e}")


@st.cache_resource
def get_indexer():
    """持久化加载增量索引器"""
    return IncrementalIndexer()


@st.cache_resource
def get_cache():
    """持久化加载分层缓存"""
    return HierarchicalCache()


# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "indexer" not in st.session_state:
    st.session_state["indexer"] = get_indexer()

if "cache" not in st.session_state:
    st.session_state["cache"] = get_cache()

if "engine" not in st.session_state:
    st.session_state["engine"] = KnowledgeBaseEngine(
        indexer=st.session_state["indexer"],
        cache=st.session_state["cache"]
    )

if "vlm_preview_md" not in st.session_state:
    st.session_state["vlm_preview_md"] = ""


# ==============================
# 侧边栏：供应商与大模型配置
# ==============================
with st.sidebar:
    st.title("🧠 知识库控制台")
    st.caption("RAG LangChain Studio v2.0")
    st.markdown("---")

    st.subheader("🤖 模型与供应商配置")
    persisted_cfg = load_persisted_config()
    saved_providers = persisted_cfg.get("providers", {})
    saved_selected_provider = persisted_cfg.get("selected_provider", "DeepSeek")

    provider_keys = list(PROVIDERS.keys())
    default_provider_idx = provider_keys.index(saved_selected_provider) if saved_selected_provider in provider_keys else 0

    provider_name = st.selectbox(
        "选择大模型服务商",
        provider_keys,
        index=default_provider_idx
    )
    selected_p = PROVIDERS[provider_name]
    curr_saved = saved_providers.get(provider_name, {})

    # API Key 处理：纯粹从系统环境变量或 .env 读取，.kb_config.json 绝不存储任何 Key
    default_env_key = selected_p["env_key"]
    existing_key = os.getenv(default_env_key, os.getenv("OPENAI_API_KEY", ""))
    
    api_key_input = st.text_input(
        "API Key",
        value=existing_key,
        type="password",
        key=f"api_key_{provider_name}",
        help=f"填入后自动安全同步保存至本地 .env 文件的 {default_env_key}，刷新不丢失"
    ).strip()

    # 若用户在前端修改或填入了新 Key，自动同步持久化到本地 .env 文件
    if api_key_input and api_key_input != os.getenv(default_env_key):
        save_api_key_to_env(default_env_key, api_key_input)

    # Base URL 处理（优先读取本地已保存配置）
    default_base_url = curr_saved.get("base_url") or selected_p["base_url"]
    base_url_input = st.text_input(
        "API Base URL",
        value=default_base_url,
        key=f"base_url_{provider_name}",
        help="OpenAI 兼容接口的 Base URL"
    )

    # 模型名称（纯自定义输入，优先读取本地已保存模型名称）
    preset_default_model = selected_p.get("default_model") or selected_p["models"][0]
    saved_model = curr_saved.get("model_name") or preset_default_model
    model_input = st.text_input(
        "模型名称 (Model Name)",
        value=saved_model,
        key=f"model_input_{provider_name}",
        help="直接填写所要调用的模型名称（如 deepseek-chat, gpt-4o, qwen3.8-max 等），刷新自动保留"
    ).strip()
    final_model = model_input if model_input else preset_default_model

    # 是否为推理模型
    is_reasoner = any(x in final_model.lower() for x in ["reasoner", "o1", "o3", "r1"])

    col_p1, col_p2 = st.columns(2)
    saved_temp = persisted_cfg.get("temperature", 0.2)
    saved_top_k = persisted_cfg.get("top_k", 4)
    with col_p1:
        temp_val = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(saved_temp),
            step=0.05,
            disabled=is_reasoner,
            help="推理模型（如 o1, deepseek-reasoner）通常锁定为 1.0 或固定温度"
        )
    with col_p2:
        top_k_val = st.slider("检索切片 Top-K", min_value=1, max_value=10, value=int(saved_top_k))

    # 自动持久化更新配置（防止页面刷新丢失，采用非明文混淆保护）
    needs_save = False
    if persisted_cfg.get("selected_provider") != provider_name:
        persisted_cfg["selected_provider"] = provider_name
        needs_save = True

    if provider_name not in saved_providers:
        saved_providers[provider_name] = {}

    # 彻底清除历史遗留的任何 key 字段，.kb_config.json 仅保存纯 UI 偏好
    saved_providers[provider_name].pop("api_key", None)
    saved_providers[provider_name].pop("api_key_enc", None)

    if (saved_providers[provider_name].get("base_url") != base_url_input or
        saved_providers[provider_name].get("model_name") != final_model):
        saved_providers[provider_name]["base_url"] = base_url_input
        saved_providers[provider_name]["model_name"] = final_model
        needs_save = True

    if (persisted_cfg.get("temperature") != temp_val or
        persisted_cfg.get("top_k") != top_k_val):
        persisted_cfg["temperature"] = temp_val
        persisted_cfg["top_k"] = top_k_val
        needs_save = True

    if needs_save:
        persisted_cfg["providers"] = saved_providers
        save_persisted_config(persisted_cfg)

    # 动态更新引擎配置
    st.session_state["engine"].update_config(
        api_key=api_key_input,
        base_url=base_url_input,
        model_name=final_model,
        temperature=temp_val,
        top_k=top_k_val
    )

    st.caption("💾 配置已自动持久化（刷新页面自动保留）")

    st.markdown("---")
    # 状态概览小组件
    st.subheader("📊 知识库实时状态")
    kb_stat = st.session_state["indexer"].get_status()
    cache_stat = st.session_state["cache"].get_stats()

    mcol1, mcol2 = st.columns(2)
    mcol1.metric("入库文档", f"{kb_stat['total_documents']} 篇")
    mcol2.metric("向量切片", f"{kb_stat['total_chunks']} 个")
    
    mcol3, mcol4 = st.columns(2)
    mcol3.metric("缓存命中率", f"{cache_stat['hit_rate_pct']}%")
    mcol4.metric("节省 Token", f"{cache_stat['estimated_saved_tokens']:,}")

    st.markdown("---")
    st.caption("提示：在左侧配置完成后，即可在右侧各功能选项卡中自由切换。")


# ==============================
# 主页面 4 大选项卡设计
# ==============================
tab_chat, tab_index, tab_vlm, tab_cache = st.tabs([
    "💬 知识库智能问答",
    "📚 文件与增量管理",
    "👁️ OCR + VLM 图像解析",
    "⚡ 分层缓存与性能监控"
])


# -------------------------------------------------------------
# TAB 1: 知识库智能问答
# -------------------------------------------------------------
with tab_chat:
    c_head1, c_head2 = st.columns([6, 2])
    with c_head1:
        st.markdown("### 💬 知识库多轮对话与流式问答")
        st.caption("优先检索 Chroma 本地向量库，结合分层缓存，支持精确溯源")
    with c_head2:
        skip_cache_toggle = st.checkbox("⚡ 跳过缓存强制检索", value=False)
        if st.button("🧹 清空对话记录", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    # 渲染历史对话
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            # 如果是助手回复且有缓存标识
            if msg["role"] == "assistant" and msg.get("cache_badge"):
                st.markdown(msg["cache_badge"], unsafe_allow_html=True)

            st.markdown(msg["content"])

            # 溯源信息卡片
            if msg.get("sources"):
                render_citations(msg["sources"])

    # 对话输入框
    user_prompt = st.chat_input("向你的个人知识库提问（如：总结某个文档的核心要点、查找数据等）...")

    if user_prompt:
        # 显示用户问题
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # 助手回复容器
        with st.chat_message("assistant"):
            badge_placeholder = st.empty()
            answer_placeholder = st.empty()
            citation_placeholder = st.empty()

            generator = st.session_state["engine"].stream_query(
                query=user_prompt,
                history=st.session_state["messages"][:-1],
                skip_cache=skip_cache_toggle
            )

            full_answer = ""
            event_sources = []
            badge_text = ""

            for event in generator:
                if event["type"] == "start":
                    event_sources = event.get("sources", [])
                    if event.get("cache_hit"):
                        cache_type = event.get("cache_type")
                        if cache_type == "L1_EXACT":
                            badge_text = "🟢 **L1 精确缓存命中** `(哈希无感直出 · 0 Token 消耗 · 0ms)`"
                        else:
                            sim_pct = round(event.get("cache_score", 0.0) * 100, 1)
                            badge_text = f"🔵 **L2 语义缓存命中** `(语义相似度: {sim_pct}% · 节省检索与生成)`"
                        badge_placeholder.markdown(badge_text)
                    else:
                        cnt = len(event_sources)
                        badge_text = f"🟡 **向量检索召回** `(Chroma 匹配到 {cnt} 个高相关度知识切片)`"
                        badge_placeholder.markdown(badge_text)

                elif event["type"] == "delta":
                    full_answer += event["content"]
                    answer_placeholder.markdown(full_answer + "▌")

                elif event["type"] == "done":
                    full_answer = event["answer"]
                    answer_placeholder.markdown(full_answer)

            # 渲染参考资料展开卡片
            if event_sources:
                render_citations(event_sources, container=citation_placeholder)

            # 保存到会话记录
            st.session_state["messages"].append({
                "role": "assistant",
                "content": full_answer,
                "cache_badge": badge_text,
                "sources": event_sources
            })


# -------------------------------------------------------------
# TAB 2: 知识库文件与增量管理
# -------------------------------------------------------------
with tab_index:
    st.markdown("### 📚 知识库文件与增量索引管理")
    st.caption("支持单文件/多文件上传入库与本地目录全自动增量同步（SHA-256 指纹去重）")

    col_up, col_sync = st.columns([1, 1])

    with col_up:
        st.markdown("#### 📤 上传新文件入库")
        uploaded_files = st.file_uploader(
            "支持 PDF、Word (.docx)、Excel (.xlsx)、CSV、JSON、TXT、Markdown、图片",
            type=["pdf", "docx", "xlsx", "xls", "csv", "json", "txt", "md", "png", "jpg", "jpeg"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("🚀 立即解析并增量索引", use_container_width=True):
            with st.spinner("正在解析文档并增量录入向量数据库..."):
                results = []
                for uf in uploaded_files:
                    file_bytes = uf.getvalue()
                    res = st.session_state["indexer"].index_document(
                        filename=uf.name,
                        file_bytes=file_bytes,
                        vlm_api_key=api_key_input,
                        vlm_base_url=base_url_input,
                        vlm_model=final_model
                    )
                    results.append(res)

                # 展示入库结果
                for r in results:
                    st_icon = {"NEW": "🆕", "SKIPPED": "⚡", "UPDATED": "🔄", "FAILED": "❌"}.get(r["status"], "ℹ️")
                    color = "success" if r["status"] in ["NEW", "UPDATED"] else ("info" if r["status"] == "SKIPPED" else "error")
                    msg = f"**{st_icon} [{r['status']}] {r['filename']}**: {r['message']}"
                    if color == "success":
                        st.success(msg)
                    elif color == "info":
                        st.info(msg)
                    else:
                        st.error(msg)
                st.rerun()

    with col_sync:
        st.markdown("#### 🔄 本地目录增量同步 (/schedule)")
        st.caption("监控指定本地文件夹，未变动文件自动跳过，新文件自动录入，已删除文件自动出库")
        sync_dir_path = st.text_input("本地扫描目录路径", value="data")
        
        sync_col1, sync_col2 = st.columns(2)
        with sync_col1:
            if st.button("🔄 立即同步指定目录", use_container_width=True):
                with st.spinner(f"正在扫描并同步 '{sync_dir_path}' 目录..."):
                    sync_res = st.session_state["indexer"].sync_directory(
                        directory_path=sync_dir_path,
                        vlm_api_key=api_key_input,
                        vlm_base_url=base_url_input,
                        vlm_model=final_model
                    )
                    if "error" in sync_res:
                        st.error(sync_res["error"])
                    else:
                        c = sync_res["counts"]
                        st.success(
                            f"✅ 同步完成！新增: {c['NEW']} 篇 | 跳过: {c['SKIPPED']} 篇 | "
                            f"更新: {c['UPDATED']} 篇 | 清理删除: {c['DELETED']} 篇"
                        )
                        st.rerun()

        with sync_col2:
            if st.button("⚠️ 清空整个知识库", type="secondary", use_container_width=True):
                st.session_state["indexer"].clear_all()
                st.session_state["cache"].clear()
                st.warning("已清空知识库与分层缓存！")
                st.rerun()

    st.markdown("---")
    st.markdown("#### 📋 知识库当前所有已索引文档")
    
    status_data = st.session_state["indexer"].get_status()
    docs = status_data.get("documents", [])

    if docs:
        df_docs = pd.DataFrame(docs)
        st.dataframe(df_docs, use_container_width=True)

        st.markdown("##### 🗑️ 单文档下架与删除")
        doc_to_delete = st.selectbox("选择要从知识库中移除的文档", [d["文件名"] for d in docs])
        if st.button(f"彻底删除 '{doc_to_delete}'", type="secondary"):
            if st.session_state["indexer"].delete_document(doc_to_delete):
                st.success(f"已成功移除文档 '{doc_to_delete}' 及其所有向量切片！")
                st.rerun()
            else:
                st.error("删除失败，请重试。")
    else:
        st.info("💡 当前知识库暂无文档。请在上方上传文件或执行目录同步！")


# -------------------------------------------------------------
# TAB 3: 多模态 OCR & VLM 图像解析
# -------------------------------------------------------------
with tab_vlm:
    st.markdown("### 👁️ 多模态 OCR 与 VLM 图像深度解析")
    st.caption("利用多模态视觉大模型将表格、架构图、手写笔记、扫描件转化为高精度的 Markdown 结构化内容")

    vcol_left, vcol_right = st.columns([1, 1])

    with vcol_left:
        vlm_file = st.file_uploader(
            "上传图像文件 (PNG, JPG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            key="vlm_img_uploader"
        )
        vlm_prompt = st.text_area(
            "自定义 VLM 指令 Prompt",
            value="请将图像内容完整、准确地转换为结构化的 Markdown 文本。图中的表格必须转换为标准 Markdown 表格，公式使用 LaTeX 格式，流程图详细描述逻辑和模块。",
            height=100
        )

        if vlm_file:
            st.image(vlm_file, caption=f"图像预览: {vlm_file.name}", width="stretch")

        run_vlm_btn = st.button("🚀 开始 VLM 深度解析", use_container_width=True, disabled=(vlm_file is None))

    with vcol_right:
        st.markdown("#### 📝 解析结果输出 (Markdown)")
        
        if run_vlm_btn and vlm_file:
            if not api_key_input:
                st.error("⚠️ 请先在左侧侧边栏填写有效的大模型 API Key！")
            else:
                with st.spinner("视觉大模型深度多模态解析中，请稍候..."):
                    img_bytes = vlm_file.getvalue()
                    parsed_text = parse_image_with_vlm(
                        image_bytes=img_bytes,
                        filename=vlm_file.name,
                        api_key=api_key_input,
                        base_url=base_url_input,
                        model_name=final_model,
                        custom_prompt=vlm_prompt
                    )
                    st.session_state["vlm_preview_md"] = parsed_text
                    st.session_state["vlm_last_file_name"] = vlm_file.name

        if st.session_state["vlm_preview_md"]:
            st.markdown(st.session_state["vlm_preview_md"])

            st.markdown("---")
            save_col1, save_col2 = st.columns(2)
            with save_col1:
                # 一键入库
                if st.button("📥 一键将此 Markdown 解析结果存入知识库", use_container_width=True):
                    doc_name = f"VLM_{st.session_state.get('vlm_last_file_name', 'image')}.md"
                    md_bytes = st.session_state["vlm_preview_md"].encode('utf-8')
                    res = st.session_state["indexer"].index_document(doc_name, md_bytes)
                    st.success(f"已成功将解析结果存入知识库！生成了 {res.get('chunk_count', 0)} 个向量切片。")
            with save_col2:
                st.download_button(
                    label="💾 下载 Markdown 文件",
                    data=st.session_state["vlm_preview_md"],
                    file_name="parsed_result.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        else:
            st.info("👈 上传图片并点击『开始 VLM 深度解析』，解析后的 Markdown 排版与数据将展示在此处。")


# -------------------------------------------------------------
# TAB 4: 分层缓存与性能监控
# -------------------------------------------------------------
with tab_cache:
    st.markdown("### ⚡ 分层缓存与性能监控看板")
    st.caption("L1 精确哈希缓存 + L2 语义余弦相似度缓存，大幅削减 Token 开销并消除常见重复问题的回答延迟")

    c_stats = st.session_state["cache"].get_stats()

    # 核心性能指标卡片
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    p_col1.metric("总查询次数", c_stats["total_queries"])
    p_col2.metric("L1 精确命中", c_stats["l1_hits"])
    p_col3.metric("L2 语义命中", c_stats["l2_hits"])
    p_col4.metric("综合命中率", f"{c_stats['hit_rate_pct']}%")
    p_col5.metric("预估节省 Token", f"{c_stats['estimated_saved_tokens']:,}")

    st.markdown("---")
    sc1, sc2 = st.columns([2, 1])

    with sc1:
        st.markdown("#### ⚙️ 缓存参数动态调优")
        new_threshold = st.slider(
            "L2 语义余弦相似度匹配阈值 (越高越严格，通常推荐 0.90 ~ 0.94)",
            min_value=0.75,
            max_value=0.99,
            value=float(st.session_state["cache"].semantic_threshold),
            step=0.01
        )
        st.session_state["cache"].semantic_threshold = new_threshold

    with sc2:
        st.markdown("#### 🗑️ 缓存管理")
        st.write("")
        st.write("")
        if st.button("清空所有 L1/L2 缓存与统计指标", type="secondary", use_container_width=True):
            st.session_state["cache"].clear()
            st.success("分层缓存已全部重置清空！")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 🕒 最近缓存的问答历史条目")
    recent = st.session_state["cache"].get_recent_entries(limit=15)
    if recent:
        table_rows = []
        for r in recent:
            table_rows.append({
                "查询 Query": r.get("query"),
                "命中次数": r.get("hit_count", 0),
                "最近访问时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.get("last_accessed", 0))),
                "缓存答案摘要": r.get("answer", "")[:80] + "..."
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    else:
        st.info("💡 暂无缓存记录。在智能问答中发起提问后，问答对将自动写入分层缓存。")
