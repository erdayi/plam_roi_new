# -*- coding: utf-8 -*-
"""网络环境配置: 为预训练权重下载设置实验室代理 (可被外部环境变量覆盖)."""
import os

DEFAULT_PROXY = "http://127.0.0.1:7897"


def apply_default_proxy(proxy: str = DEFAULT_PROXY):
    """设置 HTTP(S) 代理默认值. 已手动设置环境变量时不覆盖."""
    os.environ.setdefault("HTTP_PROXY", proxy)
    os.environ.setdefault("HTTPS_PROXY", proxy)
    os.environ.setdefault("http_proxy", proxy)
    os.environ.setdefault("https_proxy", proxy)
