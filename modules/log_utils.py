# -*- coding: utf-8 -*-
"""统一日志工具：用于替换项目中大量的 except Exception: pass。

用法：
  from modules.log_utils import logger
  logger.warning("消息")        # 写入日志
  logger.warning("消息", exc_info=True)  # 带异常堆栈
  logger.error("网络请求失败")   # error 级别

日志文件：%APPDATA%\MatLite\app.log，按天轮转，保留 7 天。
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler

_LOG_DIR = None  # 延迟初始化


def _log_path():
    global _LOG_DIR
    if _LOG_DIR is None:
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        _LOG_DIR = os.path.join(base, "MatLite")
        os.makedirs(_LOG_DIR, exist_ok=True)
    return os.path.join(_LOG_DIR, "app.log")


# 根日志器（单例）
_handler = TimedRotatingFileHandler(
    _log_path(), when="midnight", interval=1, backupCount=7, encoding="utf-8"
)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
_handler.setLevel(logging.DEBUG)

_logger = logging.getLogger("MatLite")
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_handler)

# 防止日志重复传播到根日志器
_logger.propagate = False

logger = _logger