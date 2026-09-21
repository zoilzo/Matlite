# -*- coding: utf-8 -*-
"""全局设置：读写 %APPDATA%\MatLite\config.json 里的应用偏好。

线程安全：所有写入操作受 threading.Lock 保护，支持多线程并发访问。
原子写入：save() 先写临时文件再 rename，避免写入中途崩溃导致配置损坏。
"""

import json
import os
import threading
import tempfile
import shutil

# 当前应用版本（版本号唯一来源，app/反馈/帮助中心都从这里读）
APP_VERSION = "1.3.0"

DEFAULTS = {
    "theme": "浅色",        # 浅色 / 深色
    "precision": 6,         # 结果小数位数（对应设置页「4 位/6 位」）
    "table_rows": 10,       # 数据预览默认行数
    "plot_dpi": 100,        # 绘图分辨率
    "font_scale": 1.0,      # 界面字体缩放
    "announce_on": True,    # 是否接收公告与更新提醒
    "announce_url": "https://raw.githubusercontent.com/zoilzo/MatLite/main/announce.json",  # 免费静态托管公告地址（留空=关闭）
    "release_repo": "zoilzo/MatLite",  # 版本检查的 GitHub 仓库
    # ---- AI 助手 ----
    "base_url": "http://localhost:11434",  # 本地/云端大模型服务地址
    "mode": "Ollama 原生接口（推荐）",        # Ollama 原生接口（推荐）/ OpenAi 兼容接口
    "model": "qwen3-vl:8b",               # 默认模型名（可在 AI 助手页手动改）
    "api_key": "",                        # API Key（本地服务一般留空）
    "temperature": 0.7,                   # 生成温度
    "ai_tools_on": True,                  # AI 助手是否自动调用内置功能模块
}

CONFIG_NAME = "config.json"
_lock = threading.Lock()


def cfg_path():
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "MatLite")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, CONFIG_NAME)


def load():
    """加载配置，失败返回空字典。"""
    try:
        with open(cfg_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save(cfg):
    """原子写入：先写临时文件，再 rename 覆盖目标，防崩溃。"""
    path = cfg_path()
    dir_name = os.path.dirname(path)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=dir_name, prefix="config_", suffix=".tmp", delete=False
        ) as tf:
            json.dump(cfg, tf, ensure_ascii=False, indent=2)
            tmp_path = tf.name
        shutil.move(tmp_path, path)
    except Exception:
        # 清理残留临时文件
        try:
            if 'tmp_path' in dir() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def get(key, default=None):
    return load().get(key, default)


def set_val(key, value):
    with _lock:
        c = load()
        c[key] = value
        save(c)
    return value


def update_dict(items):
    """批量更新多个键，一次性写入。"""
    with _lock:
        c = load()
        c.update(items)
        save(c)


def snapshot():
    """返回带默认值的完整设置副本。"""
    d = dict(DEFAULTS)
    d.update({k: v for k, v in load().items() if k in DEFAULTS})
    return d