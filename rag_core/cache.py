# -*- coding: utf-8 -*-
"""
分层缓存模块 (Hierarchical Caching)
- L1 精确匹配缓存 (Exact Match Cache): 基于 Query 规范化哈希快速查询，零延迟与零 Token 开销。
- L2 语义向量缓存 (Semantic Cache): 基于 NumPy 矩阵并行余弦相似度进行意图匹配，相近问法直接复用高质量答案。
- 提供命中率统计、Token 节省估算、磁盘 IO 节流及持久化存储。
"""

import os
import re
import json
import time
import hashlib
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


def normalize_text(text: str) -> str:
    """
    文本规范化清洗：
    去除首尾空格、多余标点与统一大小写，保留英文单词间空格，同时消除中文字符间冗余空格，
    既保证中文问法归一，又避免英文单词粘连引发哈希冲突。
    """
    text = text.strip().lower()
    # 去除常见中英文标点符号
    text = re.sub(r'[\.,!\?，。！？、：:;；\(\)（）"\'`《》\n\r\t]+', '', text)
    # 去除中文字符与其它字符之间的多余空格（中文习惯无空格，英文间保留单个空格）
    text = re.sub(r'(?<=[\u4e00-\u9fa5])\s+', '', text)
    text = re.sub(r'\s+(?=[\u4e00-\u9fa5])', '', text)
    # 将英文单词间的多个连续空格压缩为单个空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def estimate_tokens(text: str) -> int:
    """更精确地估算中英文混合文本的 Token 消耗 (中文约 1.5 Token/字，英文单词约 1.3 Token/词)"""
    if not text:
        return 0
    cn_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    non_cn = re.sub(r'[\u4e00-\u9fa5]', ' ', text)
    en_words = len(non_cn.split())
    return max(1, int(cn_chars * 1.5 + en_words * 1.3))


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class HierarchicalCache:
    def __init__(
        self,
        cache_dir: str = ".kb_cache",
        semantic_threshold: float = 0.82,
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.cache_dir = cache_dir
        self.semantic_threshold = semantic_threshold
        self.embedding_model_name = embedding_model_name
        self._embedder = None
        self._stats_dirty_count = 0

        os.makedirs(self.cache_dir, exist_ok=True)
        self.l1_file = os.path.join(self.cache_dir, "l1_cache.json")
        self.l2_file = os.path.join(self.cache_dir, "l2_cache.json")
        self.stats_file = os.path.join(self.cache_dir, "cache_stats.json")

        # 内存缓存结构
        self.l1_cache: Dict[str, Dict[str, Any]] = {}
        self.l2_cache: List[Dict[str, Any]] = []

        # 监控统计数据
        self.stats = {
            "total_queries": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "estimated_saved_tokens": 0
        }

        self._load_cache()

    def _get_embedder(self):
        """延迟加载轻量级本地 Embedding 模型，避免启动阻塞"""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                print(f"[Cache] 无法加载本地 SentenceTransformer: {e}")
                self._embedder = None
        return self._embedder

    def compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """计算文本向量"""
        embedder = self._get_embedder()
        if embedder is not None:
            try:
                return embedder.encode(text, convert_to_numpy=True)
            except Exception as e:
                print(f"[Cache] 向量生成异常: {e}")
        return None

    def get(self, query: str) -> Optional[Tuple[str, List[Dict[str, Any]], str, float]]:
        """
        分层查询缓存
        返回: (answer, sources, hit_type, score)
        如果未命中返回 None
        """
        self.stats["total_queries"] += 1
        clean_q = normalize_text(query)
        if not clean_q:
            return None

        # 1. 尝试 L1 精确哈希匹配
        q_hash = hashlib.sha256(clean_q.encode('utf-8')).hexdigest()
        if q_hash in self.l1_cache:
            entry = self.l1_cache[q_hash]
            entry["hit_count"] = entry.get("hit_count", 0) + 1
            entry["last_accessed"] = time.time()
            self.stats["l1_hits"] += 1
            # 精确估算节省 Token (输入 + 输出)
            self.stats["estimated_saved_tokens"] += estimate_tokens(query) + estimate_tokens(entry["answer"])
            self._save_stats(force=False)
            return entry["answer"], entry.get("sources", []), "L1_EXACT", 1.0

        # 2. 尝试 L2 语义余弦相似度匹配（使用 NumPy 矩阵并行向量化内积）
        if self.l2_cache:
            query_vec = self.compute_embedding(query)
            if query_vec is not None:
                # 矩阵化加速计算
                vectors = np.array([e["vector"] for e in self.l2_cache], dtype=np.float32)
                q_norm = np.linalg.norm(query_vec)
                v_norms = np.linalg.norm(vectors, axis=1)

                valid_mask = (q_norm > 0) & (v_norms > 0)
                if np.any(valid_mask):
                    dot_prods = np.dot(vectors, query_vec)
                    sims = np.zeros(len(vectors), dtype=np.float32)
                    sims[valid_mask] = dot_prods[valid_mask] / (v_norms[valid_mask] * q_norm)

                    best_idx = int(np.argmax(sims))
                    best_score = float(sims[best_idx])
                    best_entry = self.l2_cache[best_idx]

                    if best_entry is not None and best_score >= self.semantic_threshold:
                        best_entry["hit_count"] = best_entry.get("hit_count", 0) + 1
                        best_entry["last_accessed"] = time.time()
                        self.stats["l2_hits"] += 1
                        self.stats["estimated_saved_tokens"] += estimate_tokens(query) + estimate_tokens(best_entry["answer"])
                        self._save_stats(force=False)
                        return best_entry["answer"], best_entry.get("sources", []), "L2_SEMANTIC", best_score

        # 3. 缓存未命中
        self.stats["misses"] += 1
        self._save_stats(force=False)
        return None

    def set(self, query: str, answer: str, sources: Optional[List[Dict[str, Any]]] = None):
        """写入缓存 (同时写入 L1 和 L2)"""
        clean_q = normalize_text(query)
        if not clean_q or not answer:
            return

        now = time.time()
        q_hash = hashlib.sha256(clean_q.encode('utf-8')).hexdigest()

        # 存入 L1 精确缓存
        self.l1_cache[q_hash] = {
            "query": query,
            "answer": answer,
            "sources": sources or [],
            "created_at": now,
            "hit_count": 0,
            "last_accessed": now
        }

        # 存入 L2 语义缓存
        query_vec = self.compute_embedding(query)
        if query_vec is not None:
            self.l2_cache.append({
                "query": query,
                "vector": query_vec.tolist(),
                "answer": answer,
                "sources": sources or [],
                "created_at": now,
                "hit_count": 0,
                "last_accessed": now
            })
            # 限制 L2 缓存上限，避免内存膨胀（最大保留 1000 条，淘汰最少访问的）
            if len(self.l2_cache) > 1000:
                self.l2_cache.sort(key=lambda x: x.get("hit_count", 0), reverse=True)
                self.l2_cache = self.l2_cache[:1000]

        self._save_cache()
        self._save_stats(force=True)

    def clear(self):
        """清空所有缓存与统计"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.stats = {
            "total_queries": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "estimated_saved_tokens": 0
        }
        self._save_cache()
        self._save_stats(force=True)

    def get_stats(self) -> Dict[str, Any]:
        """获取当前缓存监控统计指标"""
        total = self.stats["total_queries"]
        hits = self.stats["l1_hits"] + self.stats["l2_hits"]
        hit_rate = (hits / total * 100) if total > 0 else 0.0

        return {
            "total_queries": total,
            "l1_hits": self.stats["l1_hits"],
            "l2_hits": self.stats["l2_hits"],
            "total_hits": hits,
            "misses": self.stats["misses"],
            "hit_rate_pct": round(hit_rate, 1),
            "estimated_saved_tokens": self.stats["estimated_saved_tokens"],
            "l1_entries": len(self.l1_cache),
            "l2_entries": len(self.l2_cache),
            "semantic_threshold": self.semantic_threshold
        }

    def get_recent_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近缓存的问答列表"""
        entries = list(self.l1_cache.values())
        entries.sort(key=lambda x: x.get("last_accessed", 0), reverse=True)
        return entries[:limit]

    def _save_cache(self):
        """持久化保存缓存到文件"""
        try:
            with open(self.l1_file, 'w', encoding='utf-8') as f:
                json.dump(self.l1_cache, f, ensure_ascii=False, indent=2)
            with open(self.l2_file, 'w', encoding='utf-8') as f:
                json.dump(self.l2_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Cache] 缓存保存失败: {e}")

    def _save_stats(self, force: bool = False):
        """持久化保存统计数据（节流控制：每累计 5 次变动或强制保存时才落盘）"""
        self._stats_dirty_count += 1
        if force or self._stats_dirty_count >= 5:
            try:
                with open(self.stats_file, 'w', encoding='utf-8') as f:
                    json.dump(self.stats, f, ensure_ascii=False, indent=2)
                self._stats_dirty_count = 0
            except Exception:
                pass

    def _load_cache(self):
        """从文件恢复缓存"""
        if os.path.exists(self.l1_file):
            try:
                with open(self.l1_file, 'r', encoding='utf-8') as f:
                    self.l1_cache = json.load(f)
            except Exception:
                self.l1_cache = {}

        if os.path.exists(self.l2_file):
            try:
                with open(self.l2_file, 'r', encoding='utf-8') as f:
                    self.l2_cache = json.load(f)
            except Exception:
                self.l2_cache = []

        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            except Exception:
                pass
