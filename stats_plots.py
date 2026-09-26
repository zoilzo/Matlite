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

# 主题系统：取色与坐标轴样式统一走 plot_style（默认主题与历史配色一致）
from modules import plot_style

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


def roc_curve_plot(y_true, prob_pred, title="ROC 曲线"):
    """逻辑回归 ROC 曲线：真阳性率 vs 假阳性率，AUC 越大判别力越强。

    y_true 为 0/1 真实标签，prob_pred 为模型预测的概率。
    """
    from sklearn.metrics import auc, roc_curve
    fpr, tpr, _ = roc_curve(y_true, prob_pred)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="随机猜测")
    ax.set_xlabel("假阳性率（1-特异度）")
    ax.set_ylabel("真阳性率（灵敏度）")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def proportion_bar_plot(counts, labels, title="二分类比例分布"):
    """比例检验：两个水平的频数柱状图，柱顶标注百分比。"""
    fig, ax = plt.subplots(figsize=(5.6, 4))
    total = sum(counts) if counts else 0
    fracs = [c / total for c in counts] if total > 0 else [0.0] * len(counts)
    bars = ax.bar([str(l) for l in labels], counts,
                  color=["steelblue", "coral"], alpha=0.75, edgecolor="white")
    for bar, f in zip(bars, fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{f:.1%}", ha="center", fontsize=10)
    ax.set_title(title)
    ax.set_ylabel("频数")
    fig.tight_layout()
    return fig


# ============================================================
# v1.5.0 新增：出版级图型（小提琴 / 热图 / 误差棒 / 森林）
# ============================================================


def violin_by_group(df, value_col, group_col):
    """小提琴图：按分组变量展示数值分布（含箱线中位数与四分位，比箱线图信息更全）。"""
    data = df[[value_col, group_col]].dropna()
    groups = [str(g) for g in data[group_col].unique()]
    vals = [pd.to_numeric(data.loc[data[group_col].astype(str) == g, value_col],
                          errors="coerce").dropna().astype(float) for g in groups]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    parts = ax.violinplot(vals, positions=range(len(groups)), showmedians=False,
                          showextrema=False, widths=0.85)
    colors = plot_style.palette(len(groups))
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i % len(colors)])
        body.set_alpha(0.55)
        body.set_edgecolor("white")
    # 内部画细箱线，标出中位数/四分位（白底深线，出版惯例）
    bp = ax.boxplot(vals, positions=range(len(groups)), widths=0.10,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.6))
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.85)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_title(f"{value_col} 按 {group_col} 分组小提琴图")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    fig.tight_layout()
    return fig


def correlation_heatmap(df, columns=None):
    """相关矩阵热图：数值变量两两 Pearson 相关系数的热图，格内标注 r 值。"""
    if columns is None:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        cols = [str(c) for c in columns if str(c) in df.columns
                and pd.api.types.is_numeric_dtype(df[str(c)])]
    if len(cols) < 2:
        raise ValueError("至少需要 2 个数值列才能画相关热图。")
    corr = df[cols].corr(numeric_only=True)
    n = len(cols)
    fig, ax = plt.subplots(figsize=(max(4.8, n * 0.9), max(4.2, n * 0.78)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(cols, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("相关矩阵热图（Pearson r）")
    fig.tight_layout()
    return fig


def errorbar_by_group(df, value_col, group_col, error="sd"):
    """误差棒图：各组均值 ± SD/SE 的点误差棒图（实验科学论文标配）。"""
    data = df[[value_col, group_col]].dropna()
    groups = [str(g) for g in data[group_col].unique()]
    use_se = str(error or "sd").lower().startswith("se")
    means, errs, kept = [], [], []
    for g in groups:
        s = pd.to_numeric(data.loc[data[group_col].astype(str) == g, value_col],
                          errors="coerce").dropna()
        if len(s) == 0:
            continue
        kept.append(g)
        means.append(float(s.mean()))
        sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        errs.append(sd / np.sqrt(len(s)) if use_se else sd)
    if not means:
        raise ValueError("没有可绘制的分组数据。")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = plot_style.palette(len(kept))
    ax.errorbar(range(len(kept)), means, yerr=errs, fmt="o-", capsize=4,
                lw=1.8, markersize=7, color=colors[0], ecolor=colors[1] if len(colors) > 1 else colors[0])
    ax.set_xticks(range(len(kept)))
    ax.set_xticklabels(kept)
    ax.set_title(f"{value_col} 按 {group_col} 分组均值 ± {'SE' if use_se else 'SD'}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    fig.tight_layout()
    return fig


def forest_plot(estimates, lower, upper, labels=None, title="森林图",
                xlabel="效应量 / 系数（95% CI）"):
    """森林图：一组点估计与 95% 置信区间的经典展示（回归系数 / OR / 均值差等）。

    estimates : 点估计列表；lower / upper：95% 置信下限 / 上限；labels：行标签。
    """
    est = np.asarray(estimates, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    n = len(est)
    if n == 0:
        raise ValueError("没有可绘制的估计值。")
    if labels is None:
        labels = [f"项 {i + 1}" for i in range(n)]
    ypos = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(7.4, max(2.6, 0.52 * n + 1.4)))
    colors = plot_style.palette(n)
    for i, (y, e, l, h) in enumerate(zip(ypos, est, lo, hi)):
        ax.plot([l, h], [y, y], color=colors[i], lw=3.2, zorder=2, solid_capstyle="round")
        ax.scatter([e], [y], s=54, color=colors[i], zorder=3, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="gray", ls="--", lw=1, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([str(x) for x in labels])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    # 行尾标注 数值 [下限, 上限]
    for y, e, l, h in zip(ypos, est, lo, hi):
        ax.text(h, y, f"  {e:.3g} [{l:.3g}, {h:.3g}]", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def regression_forest_plot(coef_df, title="回归系数森林图（95% CI）", xlabel="效应量 / 系数（95% CI）"):
    """从系数表（DataFrame）直接生成森林图，自动识别常见列名。

    兼容 stats_utils 多元/逻辑回归的系数表：
      变量；估计列（系数 B / 系数 / OR / t 值均可，默认选系数或 OR）；
      上下限列（95%CI低/95%CI高 / 95%下限/95%上限 / OR 95%CI低/OR 95%CI高）。
    """
    if "变量" not in coef_df.columns:
        if coef_df.index.name:
            coef_df = coef_df.rename_axis("变量").reset_index()
        elif "自变量" in coef_df.columns:
            coef_df = coef_df.rename(columns={"自变量": "变量"})
        else:
            coef_df = coef_df.reset_index().rename(columns={coef_df.index.name or "index": "变量"})

    def _pick(cands, default=None):
        for c in cands:
            if c in coef_df.columns:
                return c
        return default

    est_col = _pick(["系数 B", "系数", "OR", "z", "t", "Pearson r"])
    lo_col = _pick(["95%CI低", "95%下限", "OR 95%CI低", "95%CI 低", "置信下限"])
    hi_col = _pick(["95%CI高", "95%上限", "OR 95%CI高", "95%CI 高", "置信上限"])
    if not est_col:
        raise ValueError(f"系数表没有可用的估计列，实际列：{list(coef_df.columns)}")
    if not lo_col or not hi_col:
        raise ValueError(f"系数表缺少置信区间上下限列，实际列：{list(coef_df.columns)}")
    is_or = str(est_col).startswith("OR")
    return forest_plot(
        coef_df[est_col].astype(float),
        coef_df[lo_col].astype(float),
        coef_df[hi_col].astype(float),
        labels=coef_df["变量"],
        title=title,
        xlabel="OR（95% CI）" if is_or else xlabel,
    )