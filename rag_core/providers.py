# -*- coding: utf-8 -*-
"""
大模型供应商与统一模型配置清单 (Model Providers Specification)
支持 DeepSeek, OpenAI, 阿里千问, 智谱清言, 月之暗面 Kimi, Ollama, OpenRouter
供 kb_app.py 与 agent/chat_csv.py 统一引用，避免配置重复与 URL 不一致。
"""

from typing import Dict, Any

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "description": "DeepSeek 官方 API，支持 V3.2 对话与 R1 推理模型"
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "o3-mini", "gpt-6-astra", "gpt-5.6-terra"],
        "default_model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "description": "OpenAI 官方通用与推理模型"
    },
    "阿里千问 (DashScope)": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen3.8-max", "qwen3.7-plus", "qwen-turbo", "qwen-vl-max"],
        "default_model": "qwen3.8-max",
        "env_key": "DASHSCOPE_API_KEY",
        "description": "阿里云通义千问兼容模式接口"
    },
    "智谱清言 (Zhipu/GLM)": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-5.3-flash", "glm-4-plus", "glm-4v-plus"],
        "default_model": "glm-5.3-flash",
        "env_key": "ZHIPUAI_API_KEY",
        "description": "智谱 AI GLM 大模型开放平台"
    },
    "月之暗面 (Kimi/Moonshot)": {
        "base_url": "https://api.moonshot.cn/v1",
        "models": ["kimi-k3", "moonshot-v1-32k", "moonshot-v1-8k"],
        "default_model": "kimi-k3",
        "env_key": "MOONSHOT_API_KEY",
        "description": "Moonshot Kimi 长上下文大模型"
    },
    "Ollama (本地私有部署)": {
        "base_url": "http://localhost:11434/v1",
        "models": ["deepseek-r1:8b", "llama3", "qwen2.5:7b", "llava"],
        "default_model": "deepseek-r1:8b",
        "env_key": "OLLAMA_API_KEY",
        "description": "本地私有化部署的大模型推理服务"
    },
    "OpenRouter (聚合网关)": {
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["openai/gpt-4o", "deepseek/deepseek-r1", "anthropic/claude-3.5-sonnet"],
        "default_model": "openai/gpt-4o",
        "env_key": "OPENROUTER_API_KEY",
        "description": "全球主流大模型聚合 API 网关"
    },
    "自定义 (Custom)": {
        "base_url": "https://api.openai.com/v1",
        "models": ["custom-model"],
        "default_model": "custom-model",
        "env_key": "OPENAI_API_KEY",
        "description": "任意兼容 OpenAI 协议的私有或代理服务端点"
    }
}
