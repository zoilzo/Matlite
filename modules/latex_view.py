# -*- coding: utf-8 -*-
"""LaTeX 公式渲染：把 sympy 结果渲染成图片，显示专业数学符号。

基于 matplotlib 的 mathtext（无需安装 TeX 环境）。
矩阵用 ax.table 自动排格：figure 尺寸取自行列数但设上限（宽≤8、高≤14 英寸，dpi=80），
文字用 transAxes 坐标居中在格内，避免数字与方括号或相邻数字重叠。

注意：本模块在函数内部按需导入 matplotlib，避免在模块级别修改全局后端。
"""

import io

import numpy as np
import PIL.Image
from sympy import latex, MatrixBase


def _render_fig(fig):
    """渲染 figure 到 PIL Image 并关闭 figure。"""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return PIL.Image.open(buf).convert("RGBA").copy()


def _center_text(ax, s, fontsize, color="black"):
    """在坐标轴中央写字（0.5, 0.5 的归一化坐标，不依赖像素值）。"""
    ax.text(0.5, 0.5, f"${s}$", ha="center", va="center",
            transform=ax.transAxes, fontsize=fontsize, color=color)


def _setup_rc():
    """设置数学字体（仅在需要时调用，不影响全局设置）。"""
    import matplotlib.pyplot as plt
    plt.rcParams["mathtext.fontset"] = "cm"


def expr_to_image(expr, fontsize=26):
    import matplotlib.pyplot as plt
    _setup_rc()
    s = latex(expr)
    w = min(12.0, max(4.0, len(s) / 14.0))
    fig, ax = plt.subplots(figsize=(w, 2.2), dpi=100)
    ax.axis("off")
    _center_text(ax, s, fontsize)
    return _render_fig(fig)


def matrix_to_image(mat, fontsize=16):
    """把 sympy 矩阵渲染成网格图（ax.table 自动排格，数字居中，避免与边框或相邻数字重叠）。"""
    import matplotlib.pyplot as plt
    _setup_rc()
    rows, cols = mat.shape
    w = min(cols, 8.0)   # 宽度上限，避免超大图
    h = min(rows, 14.0)  # 高度上限，避免超大图
    fig, ax = plt.subplots(figsize=(w, h), dpi=80)
    ax.axis("off")
    cell_text = [[f"${latex(mat[i, j])}$" for j in range(cols)] for i in range(rows)]
    tab = ax.table(cellText=cell_text, cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(fontsize)
    tab.scale(1.0, 1.0)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor("gray")
        cell.set_linewidth(0.9)
    return _render_fig(fig)


def sympy_to_image(expr, fontsize=26):
    if isinstance(expr, MatrixBase):
        return matrix_to_image(expr, fontsize=fontsize)
    return expr_to_image(expr, fontsize=fontsize)


def render_latex_text(latex_str, fontsize=12):
    import matplotlib.pyplot as plt
    _setup_rc()
    s = latex_str
    w = min(15.0, max(4.0, len(s) / 16.0))
    fig, ax = plt.subplots(figsize=(w, 1.1), dpi=100)
    ax.axis("off")
    _center_text(ax, s, fontsize)
    return _render_fig(fig)