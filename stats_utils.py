# stats_utils.py
# -*- coding: utf-8 -*-
"""
所有统计计算都放在这里，方便本科生单独修改算法。
"""

import numpy as np
import pandas as pd
from scipy import stats


def format_p(p):
    """把 p 值格式化成好看的字符串。"""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def descriptive_stats_table(df, columns):
    """对指定的数值列做描述统计，返回 DataFrame。"""
    rows = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            rows.append({
                "变量": col, "样本量": 0, "均值": np.nan, "标准差": np.nan,
                "最小值": np.nan, "25%分位数": np.nan, "中位数": np.nan,
                "75%分位数": np.nan, "最大值": np.nan, "偏度": np.nan,
                "峰度": np.nan, "变异系数": np.nan,
            })
            continue
        rows.append({
            "变量": col,
            "样本量": len(s),
            "均值": s.mean(),
            "标准差": s.std(ddof=1),
            "最小值": s.min(),
            "25%分位数": s.quantile(0.25),
            "中位数": s.median(),
            "75%分位数": s.quantile(0.75),
            "最大值": s.max(),
            "偏度": s.skew(),
            "峰度": s.kurt(),          # pandas 的峰度是超额峰度，正态为 0
            "变异系数": s.std(ddof=1) / s.mean() if s.mean() != 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("变量")


def normality_test(data, alpha=0.05):
    """对一个序列做正态性检验，返回字典。"""
    s = pd.Series(data).dropna()
    n = len(s)
    if n < 3:
        return {"method": "样本量过小", "statistic": np.nan,
                "p": np.nan, "note": "无法检验", "n": n}

    if n <= 5000:
        stat, p = stats.shapiro(s)
        method = "Shapiro-Wilk"
    else:
        # 大样本用 D'Agostino K²，避免 Shapiro 对样本量敏感
        stat, p = stats.normaltest(s)
        method = "D'Agostino K²"

    note = "不拒绝正态假设" if p > alpha else "拒绝正态假设"
    return {"method": method, "statistic": stat, "p": p, "note": note, "n": n}


def normality_table(df, columns, alpha=0.05):
    """对多个变量做正态性检验，返回汇总表。"""
    rows = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        res = normality_test(s, alpha=alpha)
        rows.append({
            "变量": col,
            "样本量": len(s),
            "检验方法": res["method"],
            "统计量": res["statistic"],
            "p 值": res["p"],
            "结论": res["note"],
        })
    return pd.DataFrame(rows).set_index("变量")


def independent_t_test(df, value_col, group_col, alpha=0.05):
    """独立样本 t 检验（含 Levene 方差齐性检验、Cohen's d、Mann-Whitney）。"""
    data = df[[value_col, group_col]].dropna()
    groups = data[group_col].unique()
    if len(groups) != 2:
        raise ValueError(f"分组变量 {group_col} 必须恰好有 2 个水平，当前有 {len(groups)} 个。")

    g1, g2 = groups[0], groups[1]
    x1 = data.loc[data[group_col] == g1, value_col].astype(float)
    x2 = data.loc[data[group_col] == g2, value_col].astype(float)

    if len(x1) < 2 or len(x2) < 2:
        raise ValueError("每组至少需要 2 个观测值。")

    # Levene 方差齐性检验
    lev_stat, lev_p = stats.levene(x1, x2, center="median")
    equal_var = lev_p > alpha

    # t 检验
    t, p = stats.ttest_ind(x1, x2, equal_var=equal_var)

    # 自由度
    n1, n2 = len(x1), len(x2)
    v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
    if equal_var:
        df_t = n1 + n2 - 2
    else:
        df_t = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        )

    mean_diff = x1.mean() - x2.mean()

    # 均值差的标准误与置信区间
    if equal_var:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se_diff = np.sqrt(sp2 * (1 / n1 + 1 / n2))
    else:
        se_diff = np.sqrt(v1 / n1 + v2 / n2)

    t_crit = stats.t.ppf(1 - alpha / 2, df_t)
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    # Cohen's d
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    cohens_d = mean_diff / pooled_std if pooled_std != 0 else np.nan

    ad = abs(cohens_d)
    if ad < 0.2:
        effect_label = "极小"
    elif ad < 0.5:
        effect_label = "小"
    elif ad < 0.8:
        effect_label = "中"
    else:
        effect_label = "大"

    # 非参数备选
    u_stat, u_p = stats.mannwhitneyu(x1, x2, alternative="two-sided")

    if p < alpha:
        conclusion = (
            f"两组均值差异显著（t={t:.3f}, p={format_p(p)}），"
            f"{g1} 的均值 {'高于' if mean_diff > 0 else '低于'} {g2}。"
        )
    else:
        conclusion = (
            f"两组均值差异不显著（t={t:.3f}, p={format_p(p)}），"
            "没有足够证据说明两组均值不同。"
        )

    return {
        "分组变量": group_col,
        "因变量": value_col,
        "水平1": g1, "n1": n1, "mean1": x1.mean(), "std1": x1.std(ddof=1),
        "水平2": g2, "n2": n2, "mean2": x2.mean(), "std2": x2.std(ddof=1),
        "levene_stat": lev_stat, "levene_p": lev_p,
        "equal_var": equal_var,
        "检验方法": "Student t 检验" if equal_var else "Welch t 检验",
        "t": t, "df": df_t, "p": p,
        "mean_diff": mean_diff,
        "ci_low": ci_low, "ci_high": ci_high,
        "cohens_d": cohens_d, "effect_label": effect_label,
        "mannwhitney_u": u_stat, "mannwhitney_p": u_p,
        "结论": conclusion,
    }


def simple_linear_regression(df, x_col, y_col, alpha=0.05):
    """简单线性回归，返回字典。"""
    data = df[[x_col, y_col]].dropna()
    x = data[x_col].astype(float).values
    y = data[y_col].astype(float).values
    n = len(x)

    if n < 3:
        raise ValueError("至少需要 3 个观测值。")

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    y_pred = intercept + slope * x
    resid = y - y_pred

    r2 = r_value ** 2
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)
    f_stat = (r2 / 1) / ((1 - r2) / (n - 2)) if r2 < 1 else np.inf
    f_p = 1 - stats.f.cdf(f_stat, 1, n - 2) if np.isfinite(f_stat) else 0.0

    # 系数标准误
    x_mean = x.mean()
    s_err = np.sqrt(np.sum(resid ** 2) / (n - 2))
    se_slope = s_err / np.sqrt(np.sum((x - x_mean) ** 2))
    se_intercept = s_err * np.sqrt(1 / n + x_mean ** 2 / np.sum((x - x_mean) ** 2))

    t0 = intercept / se_intercept if se_intercept != 0 else np.nan
    t1 = slope / se_slope if se_slope != 0 else np.nan
    p0 = 2 * (1 - stats.t.cdf(abs(t0), n - 2)) if np.isfinite(t0) else np.nan
    p1 = 2 * (1 - stats.t.cdf(abs(t1), n - 2)) if np.isfinite(t1) else np.nan

    t_crit = stats.t.ppf(1 - alpha / 2, n - 2)
    ci0 = (intercept - t_crit * se_intercept, intercept + t_crit * se_intercept)
    ci1 = (slope - t_crit * se_slope, slope + t_crit * se_slope)

    pearson_r, pearson_p = stats.pearsonr(x, y)
    rmse = np.sqrt(np.mean(resid ** 2))
    equation = f"y = {intercept:.4f} + {slope:.4f} * x"

    if p1 < alpha:
        conclusion = (
            f"回归系数显著（β₁={slope:.4f}, p={format_p(p1)}），"
            f"{x_col} 对 {y_col} 有显著线性影响。"
        )
    else:
        conclusion = (
            f"回归系数不显著（β₁={slope:.4f}, p={format_p(p1)}），"
            f"未发现 {x_col} 与 {y_col} 的显著线性关系。"
        )

    return {
        "equation": equation, "n": n,
        "r2": r2, "adj_r2": adj_r2,
        "f_stat": f_stat, "f_p": f_p,
        "b0": intercept, "b1": slope,
        "se0": se_intercept, "se1": se_slope,
        "t0": t0, "t1": t1,
        "p0": p0, "p1": p1,
        "ci0": ci0, "ci1": ci1,
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "rmse": rmse,
        "残差": resid,
        "结论": conclusion,
    }


def chi_square_test(df, col1, col2, alpha=0.05):
    """卡方独立性检验：检验两个分类变量是否有关联。"""
    from scipy.stats import chi2_contingency
    tbl = pd.crosstab(df[col1], df[col2])
    if tbl.shape[0] < 2 or tbl.shape[1] < 2:
        raise ValueError("两个变量都至少需要 2 个类别。")
    chi2, p, dof, expected = chi2_contingency(tbl)
    n = int(tbl.to_numpy().sum())
    cramers = (np.sqrt(chi2 / (n * (min(tbl.shape) - 1)))
               if n > 0 and min(tbl.shape) > 1 else np.nan)
    conclusion = ("两变量有关联（不独立）" if p < alpha
                  else "没有足够证据说明两变量有关联")
    return {
        "列1": col1, "列2": col2, "样本量": n,
        "卡方统计量": chi2, "自由度": int(dof), "p 值": p,
        "Cramér's V": cramers,
        "结论": f"p={format_p(p)}，{conclusion}。",
        "列联表": tbl, "期望频数": expected,
    }


def anova_one_way(df, value_col, group_col, alpha=0.05):
    """单因素方差分析：检验多个组的均值是否相等。"""
    data = df[[value_col, group_col]].dropna()
    groups = [g for g in data[group_col].unique() if not pd.isna(g)]
    if len(groups) < 2:
        raise ValueError("分组变量至少需要 2 个水平。")
    samples = [data.loc[data[group_col] == g, value_col].astype(float).dropna()
               for g in groups]
    if any(len(s) < 2 for s in samples):
        raise ValueError("每个组至少需要 2 个观测值。")
    F, p = stats.f_oneway(*samples)
    all_vals = np.concatenate(samples)
    grand = all_vals.mean()
    ss_between = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
    ss_total = np.sum((all_vals - grand) ** 2)
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    conclusion = "组间均值差异显著" if p < alpha else "组间均值差异不显著"
    return {
        "因变量": value_col, "分组变量": group_col, "组数": len(groups),
        "F 统计量": F,
        "自由度": (len(groups) - 1, len(all_vals) - len(groups)),
        "p 值": p, "eta²": eta2,
        "结论": f"F={F:.4f}, p={format_p(p)}，{conclusion}。",
        "组均值": {str(g): float(s.mean()) for g, s in zip(groups, samples)},
    }


def correlation_test(df, x_col, y_col, method="Pearson", alpha=0.05):
    """相关性检验：Pearson（线性）或 Spearman（秩相关）。"""
    data = df[[x_col, y_col]].dropna()
    x = data[x_col].astype(float)
    y = data[y_col].astype(float)
    if method == "Spearman":
        r, p = stats.spearmanr(x, y)
    else:
        r, p = stats.pearsonr(x, y)
    ad = abs(r)
    strength = ("极强" if ad >= 0.8 else "强" if ad >= 0.6
                else "中等" if ad >= 0.4 else "弱" if ad >= 0.2 else "极弱")
    conclusion = "相关显著" if p < alpha else "相关不显著"
    return {
        "方法": method, "X": x_col, "Y": y_col, "样本量": len(x),
        "相关系数 r": r, "p 值": p, "强度": strength,
        "结论": f"r={r:.4f}, p={format_p(p)}，{strength}相关，{conclusion}。",
    }


def paired_t_test(df, before_col, after_col, alpha=0.05):
    """配对 t 检验：比较同一批对象的前测 / 后测差异。"""
    data = df[[before_col, after_col]].dropna()
    a = data[before_col].astype(float)
    b = data[after_col].astype(float)
    if len(a) < 3:
        raise ValueError("至少需要 3 对观测值。")
    t, p = stats.ttest_rel(a, b)
    d = a - b
    cohens_d = d.mean() / d.std(ddof=1) if d.std(ddof=1) != 0 else np.nan
    w_p = stats.wilcoxon(a, b).pvalue if len(a) >= 5 else np.nan
    mean_diff = b.mean() - a.mean()
    conclusion = "前后差异显著" if p < alpha else "前后差异不显著"
    direction = "后测更高" if mean_diff > 0 else ("后测更低" if mean_diff < 0 else "无变化")
    return {
        "前测列": before_col, "后测列": after_col, "样本对": len(a),
        "均值差(后-前)": mean_diff, "t 统计量": t,
        "自由度": len(a) - 1, "p 值": p, "Cohen's d": cohens_d,
        "Wilcoxon p": w_p,
        "结论": f"t={t:.4f}, p={format_p(p)}，{conclusion}（{direction}）。",
    }