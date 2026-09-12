# -*- coding: utf-8 -*-
"""
知识库系统自动化单元测试
测试内容：
1. 分层缓存 (L1 精确匹配、L2 语义余弦相似度匹配、统计指标)
2. 增量索引 (NEW 新增、SKIPPED 无变动跳过、UPDATED 内容修改更新、DELETED 删除)
3. 多格式文档解析器 (TXT, Markdown, CSV, JSON 等)
4. KnowledgeBaseEngine 基础问答与检索集成
"""

import os
import shutil
import tempfile
import unittest
from rag_core.cache import HierarchicalCache, normalize_text
from rag_core.indexer import IncrementalIndexer
from rag_core.vlm_ocr import parse_document_file
from rag_core.engine import KnowledgeBaseEngine


class TestKnowledgeBaseSystem(unittest.TestCase):
    def setUp(self):
        # 使用独立的临时目录进行测试，避免污染真实知识库
        self.test_dir = tempfile.mkdtemp(prefix="test_kb_")
        self.cache_dir = os.path.join(self.test_dir, ".kb_cache")
        self.chroma_dir = os.path.join(self.test_dir, ".kb_chroma_db")
        self.meta_dir = os.path.join(self.test_dir, ".kb_metadata")

        self.cache = HierarchicalCache(
            cache_dir=self.cache_dir,
            semantic_threshold=0.78
        )
        self.indexer = IncrementalIndexer(
            persist_dir=self.chroma_dir,
            metadata_dir=self.meta_dir,
            collection_name="test_kb_collection",
            chunk_size=200,
            chunk_overlap=30
        )

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_normalize_text(self):
        """测试文本规范化清洗"""
        t1 = " 什么是 RAG 技术？？ "
        t2 = "什么是rag技术"
        self.assertEqual(normalize_text(t1), normalize_text(t2))

    def test_hierarchical_cache(self):
        """测试 L1 精确匹配与 L2 语义余弦相似度匹配"""
        query_orig = "什么是检索增强生成技术（RAG）？"
        answer = "RAG（Retrieval-Augmented Generation）是一种结合信息检索与生成式大模型的 AI 架构。"
        sources = [{"source": "rag_intro.md", "similarity": 0.95}]

        # 1. 初始查询应为未命中 (Miss)
        res_miss = self.cache.get(query_orig)
        self.assertIsNone(res_miss)

        # 2. 写入缓存
        self.cache.set(query_orig, answer, sources)

        # 3. L1 精确哈希匹配（标点和大小写轻微差异）
        res_l1 = self.cache.get("什么是检索增强生成技术(RAG)?")
        self.assertIsNotNone(res_l1)
        ans_l1, src_l1, hit_type_l1, score_l1 = res_l1
        self.assertEqual(ans_l1, answer)
        self.assertEqual(hit_type_l1, "L1_EXACT")
        self.assertEqual(score_l1, 1.0)

        # 4. L2 语义向量匹配（语义相近的问法）
        semantic_query = "介绍一下 RAG 检索增强生成技术"
        res_l2 = self.cache.get(semantic_query)
        self.assertIsNotNone(res_l2)
        ans_l2, src_l2, hit_type_l2, score_l2 = res_l2
        self.assertEqual(ans_l2, answer)
        self.assertEqual(hit_type_l2, "L2_SEMANTIC")
        self.assertGreaterEqual(score_l2, 0.75)

        # 5. 校验监控统计
        stats = self.cache.get_stats()
        self.assertEqual(stats["l1_hits"], 1)
        self.assertEqual(stats["l2_hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertGreater(stats["estimated_saved_tokens"], 0)

    def test_incremental_indexing(self):
        """测试增量索引的三种核心状态：NEW、SKIPPED、UPDATED"""
        doc_filename = "rag_guide.txt"
        content_v1 = "LangChain 是构建 LLM 应用的现代框架。RAG 技术大幅减少大模型幻觉。".encode('utf-8')
        content_v2 = "LangChain 是构建 LLM 应用的现代框架。RAG 技术大幅减少大模型幻觉。新增第二版特性。".encode('utf-8')

        # 1. 首次索引，状态应为 NEW
        res1 = self.indexer.index_document(doc_filename, content_v1)
        self.assertEqual(res1["status"], "NEW")
        self.assertGreater(res1["chunk_count"], 0)
        self.assertEqual(self.indexer.collection.count(), res1["chunk_count"])

        # 2. 相同内容再次索引，状态应为 SKIPPED，无需重复向量化
        res2 = self.indexer.index_document(doc_filename, content_v1)
        self.assertEqual(res2["status"], "SKIPPED")
        self.assertIn("0 Token", res2["message"])

        # 3. 内容发生修改，状态应为 UPDATED，旧切片自动剔除
        res3 = self.indexer.index_document(doc_filename, content_v2)
        self.assertEqual(res3["status"], "UPDATED")
        self.assertEqual(self.indexer.collection.count(), res3["chunk_count"])

        # 4. 删除文档，切片与元数据同步清空
        del_success = self.indexer.delete_document(doc_filename)
        self.assertTrue(del_success)
        self.assertEqual(self.indexer.collection.count(), 0)

    def test_document_parser(self):
        """测试多格式文档解析：CSV, JSON, Markdown"""
        # CSV
        csv_bytes = "name,role,level\nAlice,Engineer,Senior\nBob,PM,Lead\n".encode('utf-8')
        docs_csv = parse_document_file("staff.csv", csv_bytes)
        self.assertEqual(len(docs_csv), 1)
        self.assertIn("Alice", docs_csv[0].page_content)
        self.assertIn("Senior", docs_csv[0].page_content)

        # JSON
        json_bytes = '{"project": "RAG_langchain", "version": "2.0"}'.encode('utf-8')
        docs_json = parse_document_file("project.json", json_bytes)
        self.assertEqual(len(docs_json), 1)
        self.assertIn("RAG_langchain", docs_json[0].page_content)

        # Markdown
        md_bytes = "# 标题\n这是 Markdown 测试内容。".encode('utf-8')
        docs_md = parse_document_file("test.md", md_bytes)
        self.assertEqual(len(docs_md), 1)
        self.assertIn("Markdown 测试内容", docs_md[0].page_content)

    def test_engine_retrieval(self):
        """测试引擎检索与上下文召回"""
        doc_bytes = "Chroma 是一款开源的高性能嵌入式向量数据库，非常适合本地 RAG 部署。".encode('utf-8')
        self.indexer.index_document("chroma_intro.txt", doc_bytes)

        engine = KnowledgeBaseEngine(indexer=self.indexer, cache=self.cache)
        sources = engine.retrieve("什么是 Chroma 向量库？", top_k=2)

        self.assertGreater(len(sources), 0)
        self.assertEqual(sources[0]["source"], "chroma_intro.txt")
        self.assertIn("高性能嵌入式向量数据库", sources[0]["content"])
        self.assertGreater(sources[0]["similarity"], 0.0)


if __name__ == "__main__":
    unittest.main()
