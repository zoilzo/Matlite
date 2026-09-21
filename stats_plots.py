# -*- coding: utf-8 -*-
"""
stats_plots.py —— 画图逻辑
=====================================
包含：折线图 / 散点图（含拟合直线）/ 分布图（直方图+正态曲线、Q-Q 图）/
分组箱线图 / 回归残差图。

与 stats_utils.py 配合使用：regression_residual_plot 直接接收
stats_utils.simple_linear_regression 返回的结果字典（含 '残差'）。
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 中文字体设置：优先微软雅黑，其次黑体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_PALETTE = ["steelblue", "coral", "seagreen", "darkorange",
            "mediumpurple", "firebrick"]


def _as_numeric(s):
    return pd.to_numeric(s, errors="coerce").dropna()


def line_plot(df, columns, title=None, x=None):
    """
    折线图：把一个或多个数值变量按观测顺序连成折线。

    参数
    ----
    df      : DataFrame，数据源
    columns : 列名列表（至少 1 个）
    title   : 图表标题（可选）
    x       : 横轴列名（可选）。不传则用观测序号 0,1,2,...

    返回
    ----
    matplotlib Figure
    """
    cols = [c for c in columns if c in df.columns]
    if not cols:
        raise ValueError("所选列都不存在于数据中。")
    if x is not None and x in df.columns:
        xs = pd.to_numeric(df[x], errors="coerce").values
    else:
        xs = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, c in enumerate(cols):
        ys = _as_numeric(df[c])
        mask = ys.notna().values
        ax.plot(xs[mask], ys[mask].values, "-o", markersize=3,
                linewidth=1.6, color=_PALETTE[i % len(_PALETTE)], label=str(c))
    ax.set_xlabel(str(x) if x is not None else "观测序号")
    ax.set_ylabel("取值")
    ax.set_title(title or ("折线图：" + "、".join(str(c) for c in cols)))
    if len(cols) > 1:
        ax.legend()
    fig.tight_layout()
    return fig


def scatter_with_fit(df, x_col, y_col):
    """散点图 + 线性拟合直线。"""
    data = df[[x_col, y_col]].dropna()
    x = data[x_col].astype(float)
    y = data[y_col].astype(float)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(x, y, alpha=0.7, color="steelblue", edgecolor="white")

    if len(x) > 1:
        slope, intercept, r, p, se = stats.linregress(x.values, y.values)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, intercept + slope * xs, "r-", lw=2,
                label=f"y = {intercept:.3f} + {slope:.3f}·x")
        ax.legend()

    ax.set_title(f"{y_col} 与 {x_col} 的散点图")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    return fig


def hist_with_normal(series, col_name):
    """分布图：直方图 + 正态曲线。"""
    s = _as_numeric(series)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.hist(s, bins=20, density=True, alpha=0.7, color="steelblue",
            edgecolor="white")

    if len(s) > 1:
        mu, sigma = s.mean(), s.std(ddof=1)
        xs = np.linspace(s.min(), s.max(), 200)
        ax.plot(xs, stats.norm.pdf(xs, mu, sigma), "r-", lw=2, label="正态曲线")
        ax.legend()

    ax.set_title(f"{col_name} 直方图（分布图）")
    ax.set_xlabel(col_name)
    ax.set_ylabel("密度")
    fig.tight_layout()
    return fig


def qq_plot(series, col_name):
    """分布图：正态 Q-Q 图。"""
    s = _as_numeric(series)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    stats.probplot(s, dist="norm", plot=ax)
    ax.set_title(f"{col_name} 正态 Q-Q 图")
    ax.get_lines()[0].set_markerfacecolor("steelblue")
    ax.get_lines()[0].set_markeredgecolor("steelblue")
    ax.get_lines()[1].set_color("red")
    fig.tight_layout()
    return fig


def boxplot_by_group(df, value_col, group_col):
    """按分组变量画箱线图（分布对比）。"""
    data = df[[value_col, group_col]].dropna()
    groups = data[group_col].unique()
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.boxplot(
        [data.loc[data[group_col] == g, value_col].astype(float) for g in groups],
        labels=[str(g) for g in groups],
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.5),
    )
    ax.set_title(f"{value_col} 按 {group_col} 分组箱线图")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    fig.tight_layout()
    return fig


def regression_residual_plot(reg_result, x=None, xlabel="自变量 X"):
    """
    残差图：直接复用 stats_utils.simple_linear_regression 的返回结果。

    reg_result 里必须包含 '残差'。若传入 x（原始自变量的值），
    横轴用 X；否则用观测序号。
    """
    if "残差" not in reg_result:
        raise ValueError("回归结果缺少 '残差'，请先调用 stats_utils.simple_linear_regression。")
    resid = np.asarray(reg_result["残差"], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    if x is not None:
        xs = np.asarray(x, dtype=float)
        ax.scatter(xs, resid, alpha=0.7, color="coral", edgecolor="white")
        ax.set_xlabel(xlabel)
    else:
        ax.scatter(np.arange(len(resid)), resid, alpha=0.7,
                   color="coral", edgecolor="white")
        ax.set_xlabel("观测序号")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("残差")
    ax.set_title("回归残差图")
    fig.tight_layout()
    return fig


def chi_square_bar_plot(contingency, title=None):
    """卡方检验：各组合观测频数的分组条形图。"""
    fig, ax = plt.subplots(figsize=(6.6, 4))
    tbl = contingency.copy()
    tbl.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.set_title(title or "卡方检验：各组合观测频数")
    ax.set_ylabel("频数")
    ax.legend(title="", fontsize=9)
    fig.tight_layout()
    return fig