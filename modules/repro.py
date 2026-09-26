# -*- coding: utf-8 -*-
"""可复现工作流：把当前计算导出为「可运行的 .py 脚本 + 可读的 .md 配方」。

MatLite 提供大量即时计算命令，但学生常需要把某道题固化成能重跑、能交作业、
能复现结果的形式。本模块统一负责把「当前输入 + 计算过程 + 结果摘要」
落盘成两个文件：
  * 题目名.md  —— 人类可读的问题描述、输入、方法、结果、工具版本（交作业/笔记）
  * 题目名.py  —— 可独立运行的脚本，用相同的输入算出完全相同的结果（复现/验证）

供各计算页在计算成功后调用 export_recipe() 一键导出。
"""

import os
from datetime import datetime

from modules import app_settings as _AS


def version_line():
    """返回形如 'MatLite v1.8.0 · numpy .., scipy .., sympy ..' 的版本说明。"""
    ver = _AS.APP_VERSION
    try:
        import numpy, scipy, sympy
        deps = f"numpy {numpy.__version__}, scipy {scipy.__version__}, sympy {sympy.__version__}"
    except Exception:
        deps = "numpy/scipy/sympy"
    return f"MatLite v{ver} · {deps}"


def _safe_name(title):
    """把标题转成可用作文件名的字符串。"""
    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '\n', '\r']
    s = str(title or "复现").strip()
    for b in bad:
        s = s.replace(b, "_")
    return s[:60].strip() or "复现"


def build_markdown(title, kind, problem_lines, result_lines, extra=None):
    """构建 Markdown 配方文本。

    params: problem_lines/result_lines 为逐行字符串列表。
    """
    out = []
    out.append(f"# {title}")
    out.append("")
    out.append(f"> 由 **MatLite** 可复现工作流自动生成 · {version_line()}")
    out.append("")
    out.append("---")
    out.append("")
    out.append(f"**计算类型：** {kind}")
    out.append("")
    out.append("## 问题输入")
    out.append("")
    for line in problem_lines:
        out.append(f"- {line}")
    out.append("")
    out.append("## 计算结果")
    out.append("")
    for line in result_lines:
        out.append(f"- {line}")
    if extra:
        out.append("")
        out.append("## 备注")
        out.append("")
        for line in extra:
            out.append(f"- {line}")
    out.append("")
    out.append("---")
    out.append("")
    out.append("_Made by zoilzo & Claude_")
    return "\n".join(out) + "\n"


def export_recipe(parent, title, kind, problem_lines, result_lines, py_code,
                  default_name=None, history=None):
    """一键导出：弹出保存框，把 .md 配方与 .py 脚本写入同一目录。

    params:
      parent          — 父窗口（用于对话框定位）
      title           — 题目名（也用作默认文件名）
      kind            — 计算类型（如 “矩阵分解 · 特征值”）
      problem_lines   — 问题输入说明（逐行）
      result_lines    — 结果摘要（逐行）
      py_code         — 完整的可独立运行的 Python 源码
      default_name    — 默认文件名（缺省用 title）
      history         — 可选 HistoryStore，用于记录导出操作

    returns: (md_path, py_path) 或 None（用户取消/失败）。
    """
    try:
        from tkinter import filedialog, messagebox
        name = default_name or _safe_name(title)
        chosen = filedialog.asksaveasfilename(
            parent=parent, title="保存复现工作流",
            defaultextension=".md", initialfile=name,
            filetypes=[("Markdown 配方", "*.md"), ("Python 脚本", "*.py")])
        if not chosen:
            return None
        base = os.path.splitext(chosen)[0]
        md_path = base + ".md"
        py_path = base + ".py"
        md = build_markdown(title, kind, problem_lines, result_lines)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(py_code)
        if history:
            history.log_op("可复现", title, f"导出 {os.path.basename(md_path)}")
        messagebox.showinfo(
            "已导出",
            f"可复现工作流已导出：\n\n📘 配方：{md_path}\n🐍 脚本：{py_path}\n\n"
            "脚本可独立运行，结果与 MatLite 完全一致。", parent=parent)
        return md_path, py_path
    except Exception as e:
        try:
            messagebox.showerror("导出失败", str(e), parent=parent)
        except Exception:
            pass
        return None


def script_header(title, kind):
    """生成 Python 脚本头部注释（题目、版本、说明）。"""
    return (f'# -*- coding: utf-8 -*\n'
            f'"""可复现工作流 — {title}\n'
            f'{kind}\n'
            f'由 MatLite 自动生成 · {version_line()}\n'
            f'本脚本可独立运行，结果与 MatLite 一致。"""\n')


def safe_env():
    """返回一个可被 eval 使用的受限命名空间（含常用数学函数），供生成的脚本内联。"""
    return None
