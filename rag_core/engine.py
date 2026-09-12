# -*- coding: utf-8 -*-
"""
知识库问答核心引擎 (Knowledge Base Engine)
- 协同分层缓存 (L1/L2 Cache)、增量向量检索 (Chroma Vector DB) 与多供应商大语言模型 (LLM)。
- 支持多轮对话历史上下文、流式输出 (Streaming) 及知识溯源 (Citations)。
"""

import os
from typing import List, Dict, Any, Optional, Generator, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .cache import HierarchicalCache
from .indexer import IncrementalIndexer


DEFAULT_SYSTEM_PROMPT = """你是一个基于本地私有知识库的专业智能问答助手。
请严格依据下方提供的【参考资料】来回答用户的问题。

回答准则：
1. 优先使用【参考资料】中的客观事实与数据进行详尽、准确的解答；
2. 如果参考资料中未包含问题的直接答案，请诚实说明“参考资料中未提及相关信息”，切勿胡乱编造；
3. 输出请保持排版工整，多用 Markdown 小标题、列表或表格；
4. 语言使用简体中文。"""


class KnowledgeBaseEngine:
    def __init__(
        self,
        indexer: Optional[IncrementalIndexer] = None,
        cache: Optional[HierarchicalCache] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.3,
        top_k: int = 4,
        system_prompt: Optional[str] = None
    ):
        self.indexer = indexer or IncrementalIndexer()
        self.cache = cache or HierarchicalCache()
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", None)
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        
        self._llm = None
        self._init_llm()

    def update_config(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None
    ):
        """动态更新模型提供商与参数配置（仅当配置实际改变或未初始化时才重新实例化 LLM）"""
        llm_changed = False

        if api_key is not None and api_key != self.api_key:
            self.api_key = api_key
            llm_changed = True
        if base_url is not None and base_url != self.base_url:
            self.base_url = base_url
            llm_changed = True
        if model_name is not None and model_name != self.model_name:
            self.model_name = model_name
            llm_changed = True
        if temperature is not None and temperature != self.temperature:
            self.temperature = temperature
            llm_changed = True
        if top_k is not None:
            self.top_k = top_k
        if system_prompt is not None:
            self.system_prompt = system_prompt

        if llm_changed or self._llm is None:
            self._init_llm()

    def _init_llm(self):
        """初始化 LangChain ChatOpenAI 客户端"""
        if not self.api_key:
            self._llm = None
            return

        # 检查是否为推理模型（reasoning models 如 o1, o3, deepseek-reasoner 不支持自定义 temperature）
        is_reasoning_model = any(
            x in self.model_name.lower() 
            for x in ["reasoner", "o1", "o3", "r1"]
        )

        kwargs = {
            "model": self.model_name,
            "api_key": self.api_key,
            "streaming": True,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        
        if not is_reasoning_model:
            kwargs["temperature"] = self.temperature

        try:
            self._llm = ChatOpenAI(**kwargs)
        except Exception as e:
            print(f"[Engine] 初始化 LLM 失败: {e}")
            self._llm = None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        从 Chroma 向量数据库中检索相关文档切片
        """
        k = top_k or self.top_k
        count = self.indexer.collection.count()
        if count == 0:
            return []

        actual_k = min(k, count)
        try:
            results = self.indexer.collection.query(
                query_texts=[query],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"[Engine] 检索异常: {e}")
            return []

        sources = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                # Chroma cosine distance 范围通常在 0~2 之间，余弦相似度 = 1 - dist
                similarity = max(0.0, min(1.0, 1.0 - float(dist)))
                sources.append({
                    "content": doc,
                    "metadata": meta,
                    "source": meta.get("source", "未知来源"),
                    "page": meta.get("page", None),
                    "sheet": meta.get("sheet", None),
                    "chunk_index": meta.get("chunk_index", 0),
                    "similarity": round(similarity, 4)
                })

        # 按相似度从高到低排序
        sources.sort(key=lambda x: x["similarity"], reverse=True)
        return sources

    def _build_context_text(self, sources: List[Dict[str, Any]]) -> str:
        """构建注入 System Prompt 的参考知识上下文文本"""
        if not sources:
            return "【参考资料】：当前知识库中未检索到与问题高度相关的资料。"

        context_parts = ["【参考资料】：\n"]
        for idx, src in enumerate(sources, 1):
            info_label = f"来源: {src['source']}"
            if src.get('page'):
                info_label += f" (第 {src['page']} 页)"
            elif src.get('sheet'):
                info_label += f" (工作表: {src['sheet']})"
            
            sim_pct = f"{round(src['similarity'] * 100, 1)}%"
            context_parts.append(
                f"--- [资料 {idx}] ({info_label} | 相关度: {sim_pct}) ---\n"
                f"{src['content']}\n"
            )
        return "\n".join(context_parts)

    def _build_messages(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Any]:
        """组装完整的多轮对话消息序列"""
        context_text = self._build_context_text(sources)
        full_system = f"{self.system_prompt}\n\n{context_text}"
        
        messages = [SystemMessage(content=full_system)]

        # 拼接历史对话（最近 10 条消息，约 5 轮对话）
        if history:
            recent_hist = history[-10:]
            for msg in recent_hist:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        # 当前用户提问
        messages.append(HumanMessage(content=query))
        return messages

    def stream_query(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        skip_cache: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式问答生成器：
        1. 首先检查分层缓存（L1 精确匹配 / L2 语义匹配）；
        2. 若命中，立即派发缓存结果与溯源信息；
        3. 若未命中，执行 Chroma 向量检索，流式返回大模型 Token，并在完成后回写缓存。
        
        Yield 格式:
        - {"type": "start", "cache_hit": bool, "cache_type": str, "cache_score": float, "sources": list}
        - {"type": "delta", "content": str}
        - {"type": "done", "answer": str, "sources": list}
        """
        # 1. 查询分层缓存
        if not skip_cache:
            cache_hit = self.cache.get(query)
            if cache_hit:
                cached_answer, cached_sources, hit_type, hit_score = cache_hit
                yield {
                    "type": "start",
                    "cache_hit": True,
                    "cache_type": hit_type,
                    "cache_score": hit_score,
                    "sources": cached_sources
                }
                yield {
                    "type": "delta",
                    "content": cached_answer
                }
                yield {
                    "type": "done",
                    "answer": cached_answer,
                    "sources": cached_sources
                }
                return

        # 2. 向量库检索
        sources = self.retrieve(query)
        yield {
            "type": "start",
            "cache_hit": False,
            "cache_type": None,
            "cache_score": 0.0,
            "sources": sources
        }

        # 3. 校验 LLM 实例
        if not self._llm:
            err_msg = "⚠️ 未配置有效的大模型 API Key，请在左侧侧边栏填写 API Key 或配置 OPENAI_API_KEY 环境变量。"
            yield {"type": "delta", "content": err_msg}
            yield {"type": "done", "answer": err_msg, "sources": sources}
            return

        # 4. 组装 Prompt 并流式调用
        messages = self._build_messages(query, sources, history)
        answer_parts = []

        try:
            for chunk in self._llm.stream(messages):
                delta = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if delta:
                    answer_parts.append(delta)
                    yield {"type": "delta", "content": delta}
            
            full_answer = "".join(answer_parts)
            # 5. 回写分层缓存
            if full_answer.strip():
                self.cache.set(query, full_answer, sources)

            yield {
                "type": "done",
                "answer": full_answer,
                "sources": sources
            }

        except Exception as e:
            err_msg = f"\n\n❌ 模型生成发生异常: {str(e)}"
            yield {"type": "delta", "content": err_msg}
            yield {"type": "done", "answer": "".join(answer_parts) + err_msg, "sources": sources}

    def query(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        skip_cache: bool = False
    ) -> Dict[str, Any]:
        """
        同步非流式问答接口
        """
        start_event = None
        full_content = []
        done_event = None

        for event in self.stream_query(query, history=history, skip_cache=skip_cache):
            if event["type"] == "start":
                start_event = event
            elif event["type"] == "delta":
                full_content.append(event["content"])
            elif event["type"] == "done":
                done_event = event

        return {
            "answer": done_event["answer"] if done_event else "".join(full_content),
            "sources": start_event["sources"] if start_event else [],
            "cache_hit": start_event["cache_hit"] if start_event else False,
            "cache_type": start_event["cache_type"] if start_event else None,
            "cache_score": start_event["cache_score"] if start_event else 0.0
        }
