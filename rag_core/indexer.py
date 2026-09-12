# -*- coding: utf-8 -*-
"""
增量索引管理器 (Incremental Indexer)
- 计算文件 SHA-256 哈希与元数据，维护本地持久化索引清单（Manifest）。
- 实现增量检测：未变更文件秒级跳过、已变更文件自动剔除旧切片并重编、已删除文件自动出库。
- 基于 Chroma 向量数据库进行持久化存储。
"""

import os
import io
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .vlm_ocr import parse_document_file


def calculate_sha256(data: bytes) -> str:
    """计算二进制数据的 SHA-256 哈希指纹"""
    return hashlib.sha256(data).hexdigest()


class IncrementalIndexer:
    def __init__(
        self,
        persist_dir: str = ".kb_chroma_db",
        metadata_dir: str = ".kb_metadata",
        collection_name: str = "personal_kb",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        embedding_function: Any = None
    ):
        self.persist_dir = persist_dir
        self.metadata_dir = metadata_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.persist_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.manifest_file = os.path.join(self.metadata_dir, "index_manifest.json")
        self.manifest: Dict[str, Dict[str, Any]] = self._load_manifest()

        # 初始化文本分块器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )

        # 初始化 Chroma 持久化客户端
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        
        # 配置嵌入模型（默认使用 Chroma 内置的轻量 SentenceTransformer）
        if embedding_function is None:
            from chromadb.utils import embedding_functions
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        else:
            self.embedding_fn = embedding_function

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        """加载索引文件清单"""
        if os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_manifest(self):
        """持久化保存清单文件"""
        try:
            with open(self.manifest_file, 'w', encoding='utf-8') as f:
                json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Indexer] 清单保存失败: {e}")

    def index_document(
        self,
        filename: str,
        file_bytes: bytes,
        vlm_api_key: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
        vlm_model: str = "gpt-4o",
        source_type: str = "upload",
        sync_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        核心方法：对单个文档进行增量索引
        返回操作状态: NEW / UPDATED / SKIPPED / FAILED
        """
        file_hash = calculate_sha256(file_bytes)
        file_size = len(file_bytes)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 检查是否已存在索引记录
        if filename in self.manifest:
            old_record = self.manifest[filename]
            if old_record.get("file_hash") == file_hash:
                # 文件未发生任何修改，直接跳过
                return {
                    "filename": filename,
                    "status": "SKIPPED",
                    "message": "文件内容无变动，已自动跳过向量化（0 Token 消耗）",
                    "chunk_count": old_record.get("chunk_count", 0),
                    "file_hash": file_hash
                }
            else:
                # 文件内容已被修改：先清理旧切片
                old_chunk_ids = old_record.get("chunk_ids", [])
                if old_chunk_ids:
                    try:
                        self.collection.delete(ids=old_chunk_ids)
                    except Exception as e:
                        print(f"[Indexer] 清理旧切片警告: {e}")
                status = "UPDATED"
        else:
            status = "NEW"

        # 开始解析文件内容
        try:
            raw_docs = parse_document_file(
                filename=filename,
                file_bytes=file_bytes,
                vlm_api_key=vlm_api_key,
                vlm_base_url=vlm_base_url,
                vlm_model=vlm_model
            )
            if not raw_docs:
                return {
                    "filename": filename,
                    "status": "FAILED",
                    "message": "未提取到有效文本内容",
                    "chunk_count": 0
                }

            # 进行文本分块
            chunked_docs = self.splitter.split_documents(raw_docs)
            if not chunked_docs:
                return {
                    "filename": filename,
                    "status": "FAILED",
                    "message": "分块结果为空",
                    "chunk_count": 0
                }

            # 生成唯一 ID 与组装元数据
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []

            for i, chunk in enumerate(chunked_docs):
                cid = f"{filename}_{file_hash[:8]}_{i}"
                chunk_ids.append(cid)
                chunk_texts.append(chunk.page_content)
                meta = chunk.metadata.copy()
                meta["source"] = filename
                meta["chunk_index"] = i
                meta["total_chunks"] = len(chunked_docs)
                meta["file_hash"] = file_hash
                meta["indexed_at"] = now_str
                # Chroma metadata 要求值不能为 None 或复杂字典
                clean_meta = {k: str(v) if not isinstance(v, (int, float, str, bool)) else v for k, v in meta.items()}
                chunk_metadatas.append(clean_meta)

            # 批量存入 Chroma 向量数据库
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )

            # 更新清单记录
            self.manifest[filename] = {
                "file_hash": file_hash,
                "file_size": file_size,
                "chunk_count": len(chunk_ids),
                "chunk_ids": chunk_ids,
                "indexed_at": now_str,
                "file_type": os.path.splitext(filename)[1].lower(),
                "source_type": source_type,
                "sync_dir": sync_dir
            }
            self._save_manifest()

            return {
                "filename": filename,
                "status": status,
                "message": f"成功索引 {len(chunk_ids)} 个切片",
                "chunk_count": len(chunk_ids),
                "file_hash": file_hash
            }

        except Exception as e:
            return {
                "filename": filename,
                "status": "FAILED",
                "message": f"索引过程异常: {str(e)}",
                "chunk_count": 0
            }

    def sync_directory(
        self,
        directory_path: str,
        vlm_api_key: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
        vlm_model: str = "gpt-4o"
    ) -> Dict[str, Any]:
        """
        同步整个目录中的文件（如 data/ 目录）
        支持新增、更新、与删除废弃文件同步
        """
        if not os.path.exists(directory_path):
            return {"error": f"目录不存在: {directory_path}"}

        norm_dir = os.path.abspath(directory_path).replace("\\", "/")
        supported_exts = {'.pdf', '.docx', '.xlsx', '.xls', '.csv', '.json', '.txt', '.md', '.png', '.jpg', '.jpeg'}
        current_disk_files = {}

        results = []
        counts = {"NEW": 0, "UPDATED": 0, "SKIPPED": 0, "FAILED": 0, "DELETED": 0}

        # 1. 扫描当前物理文件并增量索引
        for root, _, files in os.walk(directory_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_exts:
                    full_path = os.path.join(root, file)
                    rel_name = os.path.relpath(full_path, directory_path).replace("\\", "/")
                    current_disk_files[rel_name] = full_path

                    try:
                        with open(full_path, 'rb') as f:
                            data = f.read()
                        res = self.index_document(
                            filename=rel_name,
                            file_bytes=data,
                            vlm_api_key=vlm_api_key,
                            vlm_base_url=vlm_base_url,
                            vlm_model=vlm_model,
                            source_type="sync_dir",
                            sync_dir=norm_dir
                        )
                        results.append(res)
                        counts[res["status"]] += 1
                    except Exception as e:
                        results.append({"filename": rel_name, "status": "FAILED", "message": str(e)})
                        counts["FAILED"] += 1

        # 2. 检查已从硬盘中物理删除的文件，同步出库
        # 保护机制：仅针对该同步目录下的文件执行清理，绝不误删独立上传的文件或其它目录文件
        tracked_files = list(self.manifest.keys())
        for tracked in tracked_files:
            meta = self.manifest.get(tracked, {})
            if meta.get("source_type") == "sync_dir" and meta.get("sync_dir") == norm_dir:
                if tracked not in current_disk_files and not os.path.exists(os.path.join(directory_path, tracked)):
                    self.delete_document(tracked)
                    counts["DELETED"] += 1
                    results.append({"filename": tracked, "status": "DELETED", "message": "文件已在本地目录移除，已同步清理向量库"})

        return {
            "counts": counts,
            "total_processed": len(results),
            "details": results
        }

    def delete_document(self, filename: str) -> bool:
        """从知识库中彻底删除某文件及其所有向量切片"""
        if filename in self.manifest:
            chunk_ids = self.manifest[filename].get("chunk_ids", [])
            if chunk_ids:
                try:
                    self.collection.delete(ids=chunk_ids)
                except Exception as e:
                    print(f"[Indexer] 删除切片异常: {e}")
            del self.manifest[filename]
            self._save_manifest()
            return True
        return False

    def clear_all(self):
        """清空知识库与所有元数据"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self.manifest = {}
        self._save_manifest()

    def get_status(self) -> Dict[str, Any]:
        """获取知识库总状态（文档数、切片总数、文件列表）"""
        total_chunks = self.collection.count()
        file_list = []
        for name, meta in self.manifest.items():
            file_list.append({
                "文件名": name,
                "格式": meta.get("file_type", "-"),
                "切片数": meta.get("chunk_count", 0),
                "大小 (KB)": round(meta.get("file_size", 0) / 1024, 1),
                "索引时间": meta.get("indexed_at", "-")
            })
        return {
            "total_documents": len(self.manifest),
            "total_chunks": total_chunks,
            "documents": file_list
        }
