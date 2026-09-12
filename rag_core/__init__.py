# -*- coding: utf-8 -*-
"""
RAG Core Package
包含分层缓存 (HierarchicalCache)、增量索引 (IncrementalIndexer)、
多模态 OCR/VLM 解析 (vlm_ocr) 以及集成问答引擎 (KnowledgeBaseEngine)。
"""

from .cache import HierarchicalCache
from .indexer import IncrementalIndexer
from .vlm_ocr import parse_document_file, parse_image_with_vlm
from .engine import KnowledgeBaseEngine
from .providers import PROVIDERS

__all__ = [
    "HierarchicalCache",
    "IncrementalIndexer",
    "parse_document_file",
    "parse_image_with_vlm",
    "KnowledgeBaseEngine",
    "PROVIDERS",
]
