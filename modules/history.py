# -*- coding: utf-8 -*-
r"""历史记录系统。

每个账号独立保存，位于 %APPDATA%\MatLite\users\<用户名>\history\ 下：
    calc.jsonl       计算历史（公式）
    matrix.jsonl     矩阵运算历史
    analysis.jsonl   数据分析历史
    plots\*.png      绘图存档 + plots.jsonl 索引
    op_log.jsonl     全部操作日志

游客模式（未登录）下 HistoryStore.enabled 为 False，所有写入自动跳过。
"""

import json
import os
import shutil
import threading
from datetime import datetime

from modules.log_utils import logger

MAX_ENTRIES = 500          # 每个文件最多保留条数，超出自动裁掉最旧的
_lock = threading.Lock()   # 保护历史文件写入（append + trim 非原子，多线程并发会竞争）


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class HistoryStore:
    """绑定一个账号目录；游客模式下所有写入自动跳过。"""

    def __init__(self, account_manager):
        self.mgr = account_manager
        self.dir = None
        if account_manager is not None and not account_manager.is_guest():
            self.dir = os.path.join(account_manager.user_dir(), "history")
            os.makedirs(os.path.join(self.dir, "plots"), exist_ok=True)

    @property
    def enabled(self):
        return self.dir is not None

    # ---------------- 底层读写 ----------------
    def _path(self, filename):
        return os.path.join(self.dir, filename) if self.enabled else None

    def _append(self, filename, entry):
        if not self.enabled:
            return
        path = self._path(filename)
        entry = {"time": now_str(), **entry}
        try:
            with _lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._trim(path)
        except Exception:
            logger.warning("历史记录写入失败", exc_info=True)

    def _trim(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > MAX_ENTRIES:
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(lines[-MAX_ENTRIES:])
        except Exception:
            pass

    def _read(self, filename):
        path = self._path(filename)
        if not path or not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except Exception:
            return []

    def clear(self, filename):
        if not self.enabled:
            return
        path = self._path(filename)
        try:
            if os.path.isfile(path):
                os.remove(path)
            if filename == "plots.jsonl":
                shutil.rmtree(os.path.join(self.dir, "plots"), ignore_errors=True)
                os.makedirs(os.path.join(self.dir, "plots"), exist_ok=True)
        except Exception:
            pass

    # ---------------- 各类记录 ----------------
    def log_calc(self, mode, expr, result_text, latex_code=None):
        self._append("calc.jsonl", {"kind": "calc", "mode": mode,
                                    "expr": expr, "result": result_text,
                                    "latex": latex_code})

    def log_matrix(self, op, result_text, latex_code=None, matrix_data=None):
        entry = {"kind": "matrix", "op": op, "result": result_text,
                 "latex": latex_code}
        if matrix_data is not None:
            entry["matrix"] = matrix_data
        self._append("matrix.jsonl", entry)

    def log_analysis(self, atype, detail, result_text):
        self._append("analysis.jsonl", {"kind": "analysis", "type": atype,
                                        "detail": detail, "result": result_text})

    def save_plot(self, title, fig, extra=""):
        """保存一张图到绘图历史，并写入索引。"""
        if not self.enabled:
            return
        try:
            fname = now_str().replace(":", "-").replace(" ", "_") + ".png"
            full = os.path.join(self.dir, "plots", fname)
            fig.savefig(full, dpi=110, bbox_inches="tight")
            self._append("plots.jsonl", {"kind": "plot", "title": title,
                                         "file": os.path.join("plots", fname),
                                         "detail": extra})
        except Exception:
            pass

    def log_op(self, page, action, detail=""):
        self._append("op_log.jsonl", {"kind": "op", "page": page,
                                      "action": action, "detail": detail})

    # ---------------- 读取（新的在前） ----------------
    def calc_history(self):
        return list(reversed(self._read("calc.jsonl")))

    def matrix_history(self):
        return list(reversed(self._read("matrix.jsonl")))

    def analysis_history(self):
        return list(reversed(self._read("analysis.jsonl")))

    def plot_history(self):
        return list(reversed(self._read("plots.jsonl")))

    def op_log(self):
        return list(reversed(self._read("op_log.jsonl")))

    def export_all(self, filename):
        """把当前类型的所有记录导出成 txt 文件，返回文件路径。"""
        text = []
        for rec in self._read(filename):
            text.append("=" * 60)
            for k, v in rec.items():
                text.append(f"{k}: {v}")
            text.append("")
        out = os.path.join(self.dir, f"export_{filename.replace('.', '_')}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(text))
        return out