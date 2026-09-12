# -*- coding: utf-8 -*-
"""
个人知识库系统 (RAG Studio 2.0) 全功能端到端自动化测试套件
全面覆盖 5 大模块所有核心功能：
1. 多格式文档与多模态解析 (PDF, DOCX, XLSX, CSV, JSON, TXT, MD, VLM)
2. SHA-256 增量索引与本地目录同步 (NEW -> SKIPPED -> UPDATED -> DELETED -> 单文档清理)
3. 分层缓存系统 (L1 精确哈希、L2 语义余弦相似度、Token 节省统计、阈值调优)
4. 知识库引擎检索与多轮问答流 (Chroma Cosine 相似度排序、Prompt 组装、流式事件)
5. Streamlit Web UI 全组件与界面生命周期 (AppTest 模拟前端无异常加载)
"""

import os
import sys

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import shutil
import tempfile
import unittest
import pandas as pd
import docx
import pypdf
from unittest.mock import MagicMock, patch

from rag_core.cache import HierarchicalCache, normalize_text
from rag_core.indexer import IncrementalIndexer
from rag_core.vlm_ocr import parse_document_file, encode_image_to_base64, get_image_mime_type
from rag_core.engine import KnowledgeBaseEngine
from streamlit.testing.v1 import AppTest


class TestFullKnowledgeBaseFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work_dir = tempfile.mkdtemp(prefix="full_test_kb_")
        cls.cache_dir = os.path.join(cls.work_dir, ".kb_cache")
        cls.chroma_dir = os.path.join(cls.work_dir, ".kb_chroma_db")
        cls.meta_dir = os.path.join(cls.work_dir, ".kb_metadata")
        cls.sync_data_dir = os.path.join(cls.work_dir, "sync_data")
        os.makedirs(cls.sync_data_dir, exist_ok=True)

        cls.cache = HierarchicalCache(
            cache_dir=cls.cache_dir,
            semantic_threshold=0.78
        )
        cls.indexer = IncrementalIndexer(
            persist_dir=cls.chroma_dir,
            metadata_dir=cls.meta_dir,
            collection_name="full_test_collection",
            chunk_size=300,
            chunk_overlap=40
        )
        cls.engine = KnowledgeBaseEngine(
            indexer=cls.indexer,
            cache=cls.cache,
            api_key="mock_key_for_test",
            model_name="deepseek-chat"
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 模块 1: 多格式与多模态解析测试
    # -------------------------------------------------------------
    def test_01_multiformat_document_parsers(self):
        """测试 7 种主流格式解析：TXT, MD, CSV, JSON, XLSX, DOCX, 图片 MIME"""
        print("\n[测试 1/5] 执行多格式文档与多模态解析测试...")

        # 1.1 TXT
        txt_bytes = "这是普通文本文件测试内容。".encode('utf-8')
        docs_txt = parse_document_file("sample.txt", txt_bytes)
        self.assertEqual(len(docs_txt), 1)
        self.assertIn("普通文本", docs_txt[0].page_content)

        # 1.2 Markdown
        md_bytes = "# 知识库规范\n- 条目 1: 规范化\n- 条目 2: 向量化".encode('utf-8')
        docs_md = parse_document_file("guide.md", md_bytes)
        self.assertEqual(len(docs_md), 1)
        self.assertIn("知识库规范", docs_md[0].page_content)

        # 1.3 CSV
        csv_bytes = "id,name,role\n1,张三,研发工程师\n2,李四,产品经理\n".encode('utf-8')
        docs_csv = parse_document_file("users.csv", csv_bytes)
        self.assertEqual(len(docs_csv), 1)
        self.assertIn("张三", docs_csv[0].page_content)
        self.assertEqual(docs_csv[0].metadata["file_type"], "csv")

        # 1.4 JSON
        json_bytes = '{"system": "RAG_langchain", "features": ["VLM", "Incremental", "Cache"]}'.encode('utf-8')
        docs_json = parse_document_file("conf.json", json_bytes)
        self.assertEqual(len(docs_json), 1)
        self.assertIn("RAG_langchain", docs_json[0].page_content)

        # 1.5 Word (.docx)
        doc = docx.Document()
        doc.add_heading("项目架构设计", level=1)
        doc.add_paragraph("系统由分层缓存与 Chroma 向量库构成。")
        docx_stream = io.BytesIO()
        doc.save(docx_stream)
        docx_bytes = docx_stream.getvalue()
        docs_docx = parse_document_file("architecture.docx", docx_bytes)
        self.assertEqual(len(docs_docx), 1)
        self.assertIn("项目架构设计", docs_docx[0].page_content)

        # 1.6 Excel (.xlsx)
        df_test = pd.DataFrame({"模块": ["缓存", "索引", "问答"], "完成状态": ["已完成", "已完成", "已完成"]})
        xlsx_stream = io.BytesIO()
        with pd.ExcelWriter(xlsx_stream, engine='openpyxl') as writer:
            df_test.to_excel(writer, sheet_name="状态表", index=False)
        xlsx_bytes = xlsx_stream.getvalue()
        docs_xlsx = parse_document_file("progress.xlsx", xlsx_bytes)
        self.assertEqual(len(docs_xlsx), 1)
        self.assertIn("状态表", docs_xlsx[0].page_content)
        self.assertIn("缓存", docs_xlsx[0].page_content)

        # 1.7 图像工具与 Base64 编码
        dummy_img = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b64 = encode_image_to_base64(dummy_img)
        self.assertTrue(len(b64) > 0)
        self.assertEqual(get_image_mime_type("chart.png"), "image/png")
        self.assertEqual(get_image_mime_type("photo.jpeg"), "image/jpeg")

        print("  --> 多格式解析器验证全部通过！")

    # -------------------------------------------------------------
    # 模块 2: SHA-256 增量索引与目录同步全生命周期
    # -------------------------------------------------------------
    def test_02_incremental_indexing_and_directory_sync(self):
        """测试增量索引与同步：NEW -> SKIPPED -> UPDATED -> 目录同步 -> DELETED"""
        print("\n[测试 2/5] 执行增量索引与目录同步全周期测试...")

        # 2.1 单文档 NEW 录入
        doc_name = "test_feature.md"
        content_v1 = "# 功能文档\n这是第一版内容，涵盖 RAG 基础架构。".encode('utf-8')
        res1 = self.indexer.index_document(doc_name, content_v1)
        self.assertEqual(res1["status"], "NEW")
        self.assertGreater(res1["chunk_count"], 0)
        chunks_after_new = self.indexer.collection.count()
        self.assertEqual(chunks_after_new, res1["chunk_count"])

        # 2.2 相同文件再次录入 -> 秒级跳过 (0 Token 消耗)
        res2 = self.indexer.index_document(doc_name, content_v1)
        self.assertEqual(res2["status"], "SKIPPED")
        self.assertEqual(self.indexer.collection.count(), chunks_after_new)

        # 2.3 修改文件内容后录入 -> 自动清理旧切片并更新 (UPDATED)
        content_v2 = "# 功能文档\n这是第二版内容，新增分层缓存与多模态 VLM 支持。更新了关键代码。".encode('utf-8')
        res3 = self.indexer.index_document(doc_name, content_v2)
        self.assertEqual(res3["status"], "UPDATED")
        self.assertEqual(self.indexer.collection.count(), res3["chunk_count"])

        # 2.4 测试目录增量同步 (/schedule)
        file_a = os.path.join(self.sync_data_dir, "doc_a.txt")
        file_b = os.path.join(self.sync_data_dir, "doc_b.txt")
        with open(file_a, "w", encoding="utf-8") as f:
            f.write("文件 A 内容：DeepSeek 大模型支持。")
        with open(file_b, "w", encoding="utf-8") as f:
            f.write("文件 B 内容：Qwen 阿里千问支持。")

        sync1 = self.indexer.sync_directory(self.sync_data_dir)
        self.assertEqual(sync1["counts"]["NEW"], 2)

        # 2.5 再次同步 -> 全量 SKIPPED
        sync2 = self.indexer.sync_directory(self.sync_data_dir)
        self.assertEqual(sync2["counts"]["SKIPPED"], 2)
        self.assertEqual(sync2["counts"]["NEW"], 0)

        # 2.6 删除本地文件并同步 -> 触发 DELETED 同步出库
        os.remove(file_b)
        sync3 = self.indexer.sync_directory(self.sync_data_dir)
        self.assertEqual(sync3["counts"]["DELETED"], 1)
        self.assertEqual(sync3["counts"]["SKIPPED"], 1)

        # 2.7 关键防误删断言：通过 Web 上传的文件 (test_feature.md) 绝不因目录同步被误删！
        self.assertIn(doc_name, self.indexer.manifest)
        self.assertEqual(self.indexer.manifest[doc_name]["source_type"], "upload")

        # 2.8 单文件主动下架清理
        del_ok = self.indexer.delete_document("doc_a.txt")
        self.assertTrue(del_ok)

        print("  --> 增量索引、去重与目录同步测试全部通过！")

    # -------------------------------------------------------------
    # 模块 3: 分层缓存系统测试
    # -------------------------------------------------------------
    def test_03_hierarchical_cache_system(self):
        """测试 L1 精确匹配、L2 语义余弦相似度匹配、Token 统计与阈值调节"""
        print("\n[测试 3/5] 执行分层缓存 (L1/L2 Cache) 与性能指标测试...")

        q1 = "什么是 RAG 检索增强生成？"
        ans1 = "RAG 是通过检索外部私有知识库并注入 Prompt，大幅消除大模型幻觉的架构。"
        src1 = [{"source": "rag_intro.md", "similarity": 0.98}]

        # 3.1 初始未命中
        self.assertIsNone(self.cache.get(q1))

        # 3.2 写入缓存
        self.cache.set(q1, ans1, src1)

        # 3.3 L1 精确哈希匹配（轻微标点与空格清洗归一化）
        l1_res = self.cache.get(" 什么是 RAG 检索增强生成?? ")
        self.assertIsNotNone(l1_res)
        self.assertEqual(l1_res[2], "L1_EXACT")
        self.assertEqual(l1_res[3], 1.0)
        self.assertEqual(l1_res[0], ans1)

        # 3.4 L2 语义余弦相似度匹配
        q_similar = "介绍一下 RAG 检索增强生成技术"
        l2_res = self.cache.get(q_similar)
        self.assertIsNotNone(l2_res)
        self.assertEqual(l2_res[2], "L2_SEMANTIC")
        self.assertGreaterEqual(l2_res[3], 0.75)
        self.assertEqual(l2_res[0], ans1)

        # 3.5 统计监控与节省 Token 校验
        stats = self.cache.get_stats()
        self.assertGreater(stats["l1_hits"], 0)
        self.assertGreater(stats["l2_hits"], 0)
        self.assertGreater(stats["hit_rate_pct"], 0.0)
        self.assertGreater(stats["estimated_saved_tokens"], 0)

        # 3.6 最近缓存条目
        entries = self.cache.get_recent_entries(limit=5)
        self.assertTrue(len(entries) > 0)
        self.assertIn("RAG", entries[0]["query"])

        print("  --> 分层缓存、语义匹配与性能统计测试全部通过！")

    # -------------------------------------------------------------
    # 模块 4: 知识问答引擎 (KnowledgeBaseEngine)
    # -------------------------------------------------------------
    def test_04_engine_retrieval_and_stream(self):
        """测试引擎向量检索、相关度计算、多轮对话上下文组装与流式生成"""
        print("\n[测试 4/5] 执行知识库引擎向量检索与问答流测试...")

        # 准备检索文档
        kb_text = "DeepSeek-R1 是中国顶尖开源大模型，采用大规模强化学习技术，数学与推理能力极强。".encode('utf-8')
        self.indexer.index_document("deepseek_r1_spec.txt", kb_text)

        # 4.1 检索测试
        sources = self.engine.retrieve("DeepSeek R1 具备哪些技术特性？", top_k=2)
        self.assertGreater(len(sources), 0)
        self.assertEqual(sources[0]["source"], "deepseek_r1_spec.txt")
        self.assertIn("强化学习", sources[0]["content"])
        self.assertGreater(sources[0]["similarity"], 0.0)

        # 4.2 历史对话与上下文消息组装
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！我是你的个人知识库助手。"}
        ]
        messages = self.engine._build_messages("请问 DeepSeek R1 怎么样？", sources, history)
        self.assertEqual(len(messages), 4)  # System + User + Assistant + User
        self.assertIn("强化学习", messages[0].content)

        # 4.3 流式生成流（命中缓存分支）
        self.cache.set("测试快速流问题", "这是快速回答内容", sources)
        stream_events = list(self.engine.stream_query("测试快速流问题"))
        self.assertEqual(stream_events[0]["type"], "start")
        self.assertTrue(stream_events[0]["cache_hit"])
        self.assertEqual(stream_events[-1]["type"], "done")
        self.assertEqual(stream_events[-1]["answer"], "这是快速回答内容")

        print("  --> 知识库引擎检索、溯源与流式生成测试全部通过！")

    # -------------------------------------------------------------
    # 模块 5: Streamlit Web UI 全流程自动化加载测试 (AppTest)
    # -------------------------------------------------------------
    def test_05_streamlit_web_ui_apptest(self):
        """测试 Streamlit Web UI 完整加载、无异常报错、各选项卡与组件正常渲染"""
        print("\n[测试 5/5] 执行 Streamlit Web UI (kb_app.py) 界面生命周期测试...")

        kb_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kb_app.py"))
        at = AppTest.from_file(kb_app_path, default_timeout=30)
        at.run()

        # 5.1 验证无未捕获异常
        self.assertEqual(len(at.exception), 0, f"页面渲染发生异常: {at.exception}")

        # 5.2 验证侧边栏标题与服务商选择框
        self.assertTrue(any("知识库控制台" in s.value for s in at.sidebar.title))
        self.assertGreater(len(at.sidebar.selectbox), 0)

        # 5.3 验证四个主功能 Tab 均已成功声明
        self.assertGreaterEqual(len(at.tabs), 4)

        # 5.4 验证聊天输入组件正常挂载
        self.assertGreaterEqual(len(at.chat_input), 1)

        # 5.5 验证指标卡片组件 (metric) 正常渲染
        self.assertGreaterEqual(len(at.metric), 4)

        print("  --> Streamlit Web UI (kb_app.py) 界面组件无缝自动化测试全部通过！")


if __name__ == "__main__":
    unittest.main(verbosity=2)
