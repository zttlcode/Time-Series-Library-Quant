"""
轻量级日志工具 —— 一行引入，到处使用。

用法：
    from SQTool.log_utils import get_logger
    log = get_logger("MyModule")       # 首次调用自动创建日志文件
    log.info("这是一条信息")
    log.warning("这是一条警告")
    log.error("这是一条错误")
"""

import os
import logging
import sys
from datetime import datetime

# 日志目录：放在调用方项目根目录下的 logs/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")

_initialized = False


def _init_root_logger():
    """全局初始化：只执行一次，配置根 logger"""
    global _initialized
    if _initialized:
        return
    _initialized = True

    os.makedirs(_LOG_DIR, exist_ok=True)

    log_file = os.path.join(_LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 避免重复添加 handler
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)


def get_logger(name: str = "") -> logging.Logger:
    """
    获取一个 logger 实例。

    参数:
        name: 模块名，如 "run_live"、"SignalLoader" 等。
              为空时返回根 logger。

    返回:
        logging.Logger 实例，支持 .info() .warning() .error() .debug()
    """
    _init_root_logger()
    return logging.getLogger(name)
