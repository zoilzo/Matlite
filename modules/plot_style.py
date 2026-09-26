# -*- coding: utf-8 -*-
"""绘图主题/模板系统：为 MatLite 所有图表提供统一、可切换的出版级样式。

用法：
  from modules import plot_style as ps
  ps.activate_theme()          # 读配置并应用全局 rcParams
  colors = ps.palette(3)       # 当前主题色板前 3 个颜色
  ps.style_axes(ax)            # 对已有坐标轴套主题（spine/刻度颜色）

内置主题：
  * 出版默认 —— 白底、蓝橙绿紫经典配色（与历史版本外观一致，零回归）
  * 期刊      —— 灰度友好、无彩色，适合黑白印刷的论文图
  * 鲜艳      —— 高对比色板，适合课件 / 投屏演示
  * 深色      —— 深色背景，适合晚间 / 暗色界面使用
"""

import os
import shutil
import subprocess

import matplotlib
import matplotlib.pyplot as plt

from modules import app_settings as _AS

# 主题名（顺序即设置页下拉顺序）
THEME_NAMES = ["出版默认", "期刊", "鲜艳", "深色"]
DEFAULT_THEME = "出版默认"

# 中文字体：优先微软雅黑，其次黑体（所有主题统一）
_CN_FONTS = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

# 每个主题的完整定义（palette 为系列色板，其余为 rcParams 映射）
THEMES = {
    "出版默认": {
        "palette": ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                    "#8172B3", "#937860", "#CCB974", "#64B5CD"],
        "fig_face": "#ffffff", "ax_face": "#ffffff",
        "grid": True, "grid_style": "--", "grid_alpha": 0.35, "grid_color": "#cccccc",
        "spine": "#888888", "text": "#1f1f1f", "font_size": 11,
    },
    "期刊": {
        "palette": ["#111111", "#555555", "#999999", "#BBBBBB",
                    "#333333", "#777777", "#DDDDDD", "#222222"],
        "fig_face": "#ffffff", "ax_face": "#ffffff",
        "grid": False, "grid_style": "-", "grid_alpha": 0.25, "grid_color": "#aaaaaa",
        "spine": "#000000", "text": "#000000", "font_size": 10.5,
    },
    "鲜艳": {
        "palette": ["#E6194B", "#3CB44B", "#FFE119", "#4363D8",
                    "#F58231", "#911EB4", "#42D4F4", "#F032E6"],
        "fig_face": "#ffffff", "ax_face": "#fafafa",
        "grid": True, "grid_style": ":", "grid_alpha": 0.5, "grid_color": "#cccccc",
        "spine": "#666666", "text": "#111111", "font_size": 11.5,
    },
    "深色": {
        "palette": ["#7FB3FF", "#FFA94D", "#7EE787", "#FF8787",
                    "#B197FC", "#74C0FC", "#FFE066", "#63E6BE"],
        "fig_face": "#1e1e2e", "ax_face": "#181825",
        "grid": True, "grid_style": "--", "grid_alpha": 0.30, "grid_color": "#3f3f55",
        "spine": "#8888aa", "text": "#e8e8ee", "font_size": 11,
    },
}

_current = None  # 当前生效的主题 dict（None=尚未激活）


def current():
    """当前主题 dict（未激活时先激活）。"""
    global _current
    if _current is None:
        activate_theme()
    return _current


def theme_name():
    """当前主题名（未激活时读配置）。"""
    return current()["name"]


def activate_theme(name=None):
    """应用主题到 matplotlib 全局 rcParams。name 留空则读用户配置。"""
    global _current
    if not name:
        name = _AS.get("plot_theme", DEFAULT_THEME)
    if name not in THEMES:
        name = DEFAULT_THEME
    t = THEMES[name]
    t = dict(t)
    t["name"] = name
    _current = t

    plt.rcParams["font.sans-serif"] = _CN_FONTS
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = t["font_size"]
    plt.rcParams["figure.facecolor"] = t["fig_face"]
    plt.rcParams["savefig.facecolor"] = t["fig_face"]
    plt.rcParams["axes.facecolor"] = t["ax_face"]
    plt.rcParams["axes.edgecolor"] = t["spine"]
    plt.rcParams["axes.grid"] = t["grid"]
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["grid.color"] = t["grid_color"]
    plt.rcParams["grid.alpha"] = t["grid_alpha"]
    plt.rcParams["grid.linestyle"] = t["grid_style"]
    plt.rcParams["grid.linewidth"] = 0.8
    plt.rcParams["text.color"] = t["text"]
    plt.rcParams["axes.labelcolor"] = t["text"]
    plt.rcParams["xtick.color"] = t["text"]
    plt.rcParams["ytick.color"] = t["text"]
    return t


def palette(n):
    """返回当前主题色板前 n 个颜色（超长自动循环）。"""
    t = current()
    p = t["palette"]
    if n <= len(p):
        return p[:n]
    return [p[i % len(p)] for i in range(n)]


def color(i):
    """当前主题色板的第 i 个颜色（循环取）。"""
    p = current()["palette"]
    return p[i % len(p)]


def style_axes(ax):
    """对已有坐标轴套主题：spine / 刻度 / 网格颜色（新图由 rcParams 自动继承）。"""
    t = current()
    for s in ax.spines.values():
        s.set_color(t["spine"])
    ax.tick_params(colors=t["text"])
    ax.xaxis.label.set_color(t["text"])
    ax.yaxis.label.set_color(t["text"])
    if ax.title.get_text():
        ax.title.set_color(t["text"])
    if t["grid"]:
        ax.grid(True, ls=t["grid_style"], alpha=t["grid_alpha"], color=t["grid_color"])
    return ax


def style_figure(fig):
    """对已有 Figure 套主题背景色（图在创建后改主题时调用）。"""
    fig.patch.set_facecolor(current()["fig_face"])
    return fig


# ============================================================
# 统一导出：PNG / JPG / TIFF（位图）+ SVG / PDF（矢量）+ EMF
# ============================================================

# 保存对话框可选的扩展名（按优先级排序）
EXPORT_FORMATS = [("PNG 图片（位图）", "*.png"), ("SVG 矢量图", "*.svg"),
                  ("PDF 矢量图", "*.pdf"), ("EMF 矢量图（Word 可插入）", "*.emf"),
                  ("JPEG 图片", "*.jpg"), ("TIFF 高清图（论文）", "*.tif")]

export_path_filters = []
for _label, _pat in EXPORT_FORMATS:
    export_path_filters.append((_label, _pat))


def export_figure(fig, path, dpi=300):
    """按扩展名导出图表（海报 / 论文 / 插入 Word 均可）。

    - png / jpg / tif：位图，dpi 默认 300（论文高清）；
    - svg / pdf：matplotlib 原生矢量，Word 2016+ 可直接插入 SVG；
    - emf：需系统装有 LibreOffice（headless 将 SVG 转 EMF），否则抛出带替代方案的错误。

    返回实际保存路径；失败抛出 RuntimeError（含 user-friendly 提示）。
    """
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if not ext:
        ext = "png"
        path = path + ".png"
    if ext == "emf":
        return _export_emf(fig, path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _export_emf(fig, path):
    """EMF 导出：matplotlib 不支持原生 EMF，借 LibreOffice 把 SVG 转成 EMF。"""
    stem = os.path.splitext(os.path.abspath(path))[0]
    svg_path = stem + ".emf_tmp.svg"
    try:
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if not soffice:
            raise RuntimeError(
                "系统未安装 LibreOffice，无法导出 EMF。\n"
                "替代方案：① 选 SVG（Word 2016+ 可直接插入）；"
                "② 选 PDF 后再转换；③ 安装 LibreOffice 后重启再导出。")
        subprocess.run([soffice, "--headless", "--convert-to", "emf",
                        "--outdir", os.path.dirname(stem), svg_path],
                       check=True, timeout=180, capture_output=True)
        produced = os.path.join(os.path.dirname(stem),
                                os.path.splitext(os.path.basename(svg_path))[0] + ".emf")
        if os.path.exists(produced):
            if os.path.abspath(produced) != os.path.abspath(path):
                shutil.move(produced, path)
            return path
        raise RuntimeError("LibreOffice 转换未生成 EMF 文件。")
    finally:
        if os.path.exists(svg_path):
            try:
                os.remove(svg_path)
            except OSError:
                pass