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
    n_total = len(all_vals)
    k = len(groups)
    ss_between = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
    ss_total = np.sum((all_vals - grand) ** 2)
    ss_within = ss_total - ss_between
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    ms_within = ss_within / (n_total - k) if n_total > k else np.nan
    omega2 = ((ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
              if np.isfinite(ms_within) and ss_total + ms_within > 0 else np.nan)
    cohens_f = (np.sqrt(eta2 / (1 - eta2))
                if np.isfinite(eta2) and 0 < eta2 < 1 else np.nan)
    conclusion = "组间均值差异显著" if p < alpha else "组间均值差异不显著"
    return {
        "因变量": value_col, "分组变量": group_col, "组数": k,
        "F 统计量": F,
        "自由度": (k - 1, n_total - k),
        "p 值": p, "eta²": eta2, "omega²": omega2, "Cohen's f": cohens_f,
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


# ======================================================================
# v1.4.0 新增：效应量工具
# ======================================================================

def partial_eta_sq(ss_effect, ss_error):
    """偏 η² = SS效应 / (SS效应 + SS误差)。多因素 / 重复测量 ANOVA 常用。"""
    denom = ss_effect + ss_error
    return ss_effect / denom if denom > 0 else np.nan


# ======================================================================
# v1.4.0 新增：非参数检验独立入口
# ======================================================================

def _two_groups(df, value_col, group_col):
    """提取恰好两个水平的分组数据，返回 (g1, g2, x1, x2)。"""
    data = df[[value_col, group_col]].dropna()
    groups = [g for g in data[group_col].unique() if not pd.isna(g)]
    if len(groups) != 2:
        raise ValueError(f"分组变量 {group_col} 必须恰好有 2 个水平，当前有 {len(groups)} 个。")
    g1, g2 = groups[0], groups[1]
    x1 = data.loc[data[group_col] == g1, value_col].astype(float)
    x2 = data.loc[data[group_col] == g2, value_col].astype(float)
    if len(x1) < 1 or len(x2) < 1:
        raise ValueError("每组至少需要 1 个观测值。")
    return g1, g2, x1, x2


def mannwhitney_u_test(df, value_col, group_col, alpha=0.05):
    """Mann-Whitney U 检验：两独立样本分布差异（t 检验的非参替代）。

    适用：两组数据不满足正态性 / 方差齐性，或数据是有序分类（秩次）时。
    """
    g1, g2, x1, x2 = _two_groups(df, value_col, group_col)
    n1, n2 = len(x1), len(x2)
    u, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
    rb = 1 - 2 * u / (n1 * n2)          # 秩双列相关效应量（带方向）
    med1, med2 = x1.median(), x2.median()
    if p < alpha:
        cmp = "高于" if med1 > med2 else ("低于" if med1 < med2 else "与")
        conclusion = (
            f"两组分布差异显著（U={u:.1f}, p={format_p(p)}），"
            f"{g1} 的中位数 {cmp} {g2}。"
        )
    else:
        conclusion = (
            f"两组分布差异不显著（U={u:.1f}, p={format_p(p)}），"
            "没有足够证据说明两组的分布 / 中位数不同。"
        )
    return {
        "因变量": value_col, "分组变量": group_col,
        "水平1": g1, "n1": n1, "中位数1": med1,
        "水平2": g2, "n2": n2, "中位数2": med2,
        "U 统计量": u, "p 值": p,
        "效应量(秩双列相关)": rb,
        "结论": conclusion,
    }


def wilcoxon_signed_rank(df, before_col, after_col, alpha=0.05):
    """Wilcoxon 符号秩检验：两配对样本的中位数差异（配对 t 的非参替代）。

    适用：同一批对象的两次测量（前后测 / 配对对照），差值不满足正态性时。
    """
    data = df[[before_col, after_col]].dropna()
    a = data[before_col].astype(float)
    b = data[after_col].astype(float)
    d = b - a
    n = len(d)
    if n < 3:
        raise ValueError("至少需要 3 对观测值。")
    if len(d[d != 0]) < 3:
        raise ValueError("非零差异太少，无法进行 Wilcoxon 检验。")
    res = stats.wilcoxon(a, b, alternative="two-sided")
    w_stat = float(res.statistic)
    p = float(res.pvalue)
    # 效应量 r = Z / sqrt(n)，Z 由正态近似得到
    mu_w = n * (n + 1) / 4
    sigma_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_stat - mu_w) / sigma_w if sigma_w > 0 else np.nan
    r_eff = z / np.sqrt(n) if np.isfinite(z) else np.nan
    med_diff = float(np.median(d))
    if p < alpha:
        direction = "后测更高" if med_diff > 0 else ("后测更低" if med_diff < 0 else "前后无方向性差异")
        conclusion = f"配对中位数差异显著（W={w_stat:.1f}, p={format_p(p)}），{direction}。"
    else:
        conclusion = f"配对中位数差异不显著（W={w_stat:.1f}, p={format_p(p)}）。"
    return {
        "前测列": before_col, "后测列": after_col, "样本对": n,
        "中位数差(后-前)": med_diff,
        "W 统计量": w_stat, "p 值": p,
        "效应量 r": r_eff,
        "结论": conclusion,
    }


def kruskal_wallis_test(df, value_col, group_col, alpha=0.05):
    """Kruskal-Wallis H 检验：多组独立样本分布差异（单因素 ANOVA 的非参替代）。

    适用：3 组及以上，不满足正态性 / 方差齐性时。显著后建议再做 Tukey / Bonferroni 事后比较。
    """
    data = df[[value_col, group_col]].dropna()
    groups = [g for g in data[group_col].unique() if not pd.isna(g)]
    if len(groups) < 2:
        raise ValueError("分组变量至少需要 2 个水平。")
    samples = [data.loc[data[group_col] == g, value_col].astype(float).dropna()
               for g in groups]
    if any(len(s) < 1 for s in samples):
        raise ValueError("每个组至少需要 1 个观测值。")
    H, p = stats.kruskal(*samples)
    n_total = sum(len(s) for s in samples)
    k = len(groups)
    eps2 = (H - k + 1) / (n_total - k) if n_total > k else np.nan   # 效应量 ε²
    meds = {str(g): float(s.median()) for g, s in zip(groups, samples)}
    if p < alpha:
        conclusion = (f"至少有一个组的中位数显著不同（H={H:.4f}, p={format_p(p)}），"
                      "可进一步做事后多重比较确定具体差异组。")
    else:
        conclusion = f"各组中位数差异不显著（H={H:.4f}, p={format_p(p)}）。"
    return {
        "因变量": value_col, "分组变量": group_col, "组数": k, "样本量": n_total,
        "H 统计量": H, "自由度": k - 1, "p 值": p,
        "效应量 ε²": eps2,
        "组中位数": meds,
        "结论": conclusion,
    }


def friedman_test(df, cols, alpha=0.05):
    """Friedman 检验：多个配对 / 重复测量条件下的分布差异（重复测量 ANOVA 的非参替代）。

    输入：同一批对象在多个条件（列）下的测量值。
    """
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    k = len(cols)
    n = len(data)
    if k < 2:
        raise ValueError("至少需要 2 个测量条件（列）。")
    if n < 2:
        raise ValueError("至少需要 2 个观测对象（行）。")
    Q, p = stats.friedmanchisquare(*[data[c] for c in cols])
    w = Q / (n * (k - 1)) if n * (k - 1) > 0 else np.nan     # Kendall 协调系数
    meds = {str(c): float(data[c].median()) for c in cols}
    if p < alpha:
        conclusion = (f"不同条件下分布差异显著（Q={Q:.4f}, p={format_p(p)}），"
                      "可进一步做事后两两比较。")
    else:
        conclusion = f"不同条件下分布差异不显著（Q={Q:.4f}, p={format_p(p)}）。"
    return {
        "测量条件": [str(c) for c in cols], "样本量": n, "条件数": k,
        "Q 统计量": Q, "自由度": k - 1, "p 值": p,
        "Kendall's W": w,
        "各条件中位数": meds,
        "结论": conclusion,
    }


# ======================================================================
# v1.4.0 新增：事后多重比较
# ======================================================================

def posthoc_test(df, value_col, group_col, method="Tukey", alpha=0.05):
    """事后多重比较：ANOVA / Kruskal-Wallis 显著后，确定具体哪些组不同。

    method: "Tukey"（Tukey HSD，默认）/ "LSD" / "Bonferroni"。
    """
    data = df[[value_col, group_col]].dropna()
    vals = data[value_col].astype(float)
    groups = data[group_col]
    if len(groups.unique()) < 3:
        raise ValueError("事后多重比较至少需要 3 个组。")

    if method == "Tukey":
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        res = pairwise_tukeyhsd(vals, groups, alpha=alpha)
        rows = []
        raw = res._results_table.data[1:]
        for row in raw:
            rows.append({
                "组1": str(row[0]), "组2": str(row[1]),
                "均值差": float(row[2]), "p 值": float(row[3]),
                "95%CI低": float(row[4]), "95%CI高": float(row[5]),
                "显著": "是" if row[6] else "否",
            })
        method_name = "Tukey HSD"
    elif method in ("LSD", "Bonferroni"):
        from itertools import combinations
        method_name = "LSD（最小显著差）" if method == "LSD" else "Bonferroni 校正"
        u_groups = [g for g in data[group_col].unique() if not pd.isna(g)]
        pairs = list(combinations(u_groups, 2))
        rows = []
        for g1, g2 in pairs:
            x1 = data.loc[data[group_col] == g1, value_col].astype(float)
            x2 = data.loc[data[group_col] == g2, value_col].astype(float)
            n1, n2 = len(x1), len(x2)
            if n1 < 2 or n2 < 2:
                continue
            t_stat, p_raw = stats.ttest_ind(x1, x2)
            p_val = min(1.0, p_raw * len(pairs)) if method == "Bonferroni" else p_raw
            # 合并方差的两样本均值差置信区间
            v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
            sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
            se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
            t_crit = stats.t.ppf(1 - alpha / 2, n1 + n2 - 2)
            mdiff = float(x1.mean() - x2.mean())
            rows.append({
                "组1": str(g1), "组2": str(g2),
                "均值差": mdiff, "p 值": float(p_val),
                "95%CI低": mdiff - t_crit * se, "95%CI高": mdiff + t_crit * se,
                "显著": "是" if p_val < alpha else "否",
            })
    else:
        raise ValueError(f"未知事后比较方法：{method}，可选 Tukey / LSD / Bonferroni。")

    comp_df = pd.DataFrame(rows)
    sig_pairs = [f"{r['组1']} vs {r['组2']}" for _, r in comp_df.iterrows() if r["显著"] == "是"]
    if sig_pairs:
        conclusion = (f"{method_name} 共比较 {len(comp_df)} 对，"
                      f"显著不同的组对：{'、'.join(sig_pairs)}。")
    else:
        conclusion = f"{method_name} 未发现显著不同的组对（p 均 ≥ {alpha}）。"
    return {
        "因变量": value_col, "分组变量": group_col,
        "方法": method_name, "显著性水平": alpha,
        "比较对数": len(comp_df),
        "比较结果表": comp_df.set_index(["组1", "组2"]),
        "结论": conclusion,
    }


# ======================================================================
# v1.4.0 新增：单样本 t / 比例 / McNemar
# ======================================================================

def one_sample_t_test(df, col, mu=0.0, alpha=0.05):
    """单样本 t 检验：检验一个数值列的均值是否等于指定值 mu。"""
    s = df[col].astype(float).dropna()
    n = len(s)
    if n < 3:
        raise ValueError("至少需要 3 个观测值。")
    t, p = stats.ttest_1samp(s, mu)
    sd = s.std(ddof=1)
    se = sd / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
    ci_low, ci_high = s.mean() - t_crit * se, s.mean() + t_crit * se
    cohens_d = (s.mean() - mu) / sd if sd != 0 else np.nan
    ad = abs(cohens_d)
    label = ("大" if ad >= 0.8 else "中" if ad >= 0.5
             else "小" if ad >= 0.2 else "极小" if np.isfinite(ad) else "—")
    if p < alpha:
        direction = "高于" if s.mean() > mu else ("低于" if s.mean() < mu else "等于")
        conclusion = (f"均值与 {mu} 差异显著（t={t:.4f}, p={format_p(p)}），"
                      f"样本均值 {direction} {mu}。")
    else:
        conclusion = f"均值与 {mu} 差异不显著（t={t:.4f}, p={format_p(p)}）。"
    return {
        "变量": col, "样本量": n, "检验值 μ0": mu,
        "均值": s.mean(), "标准差": sd,
        "t 统计量": t, "自由度": n - 1, "p 值": p,
        "95%CI": (ci_low, ci_high),
        "Cohen's d": cohens_d, "效应量评价": label,
        "结论": conclusion,
    }


def proportion_test(df, col, success_value=None, p0=0.5, alpha=0.05):
    """二项比例检验：检验二分类列中某一水平（成功）所占比例是否等于 p0。

    success_value 为 None 时自动编码：数值列取 1，文本列取第二个水平为"成功"。
    """
    from scipy.stats import binomtest
    s = df[col].dropna()
    n = len(s)
    if n < 2:
        raise ValueError("至少需要 2 个观测值。")
    levels = [v for v in s.unique() if not pd.isna(v)]
    if len(levels) != 2:
        raise ValueError(f"{col} 必须是二分类列（当前有 {len(levels)} 个水平）。")
    if success_value is None:
        if pd.api.types.is_numeric_dtype(s):
            success = 1 if 1 in levels else levels[1]
        else:
            success = levels[1]
    else:
        success = success_value
    if success not in levels:
        raise ValueError(f"成功水平 {success} 不存在于列 {col} 中。")
    y = int((s == success).sum())
    phat = y / n
    res = binomtest(y, n, p0, alternative="two-sided")
    p_exact = float(res.pvalue)
    se = np.sqrt(p0 * (1 - p0) / n)
    z = (phat - p0) / se if se > 0 else np.nan
    p_z = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    # Wilson 置信区间
    zc = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + zc ** 2 / n
    center = (phat + zc ** 2 / (2 * n)) / denom
    half = zc * np.sqrt(phat * (1 - phat) / n + zc ** 2 / (4 * n ** 2)) / denom
    ci_low, ci_high = center - half, center + half
    if p_exact < alpha:
        direction = "高于" if phat > p0 else "低于"
        conclusion = (f"成功比例与 {p0} 差异显著（样本比例={phat:.4f}, "
                      f"精确 p={format_p(p_exact)}），明显 {direction} {p0}。")
    else:
        conclusion = (f"成功比例与 {p0} 差异不显著（样本比例={phat:.4f}, "
                      f"精确 p={format_p(p_exact)}）。")
    return {
        "变量": col, "成功水平": success, "样本量": n, "成功数": y,
        "样本比例": phat, "检验比例 p0": p0,
        "z 统计量(近似)": z, "近似 p 值": p_z,
        "精确 p 值(Binomial)": p_exact,
        "95%CI(Wilson)": (ci_low, ci_high),
        "结论": conclusion,
    }


def mcnemar_test(df, col1, col2, alpha=0.05):
    """McNemar 检验：两个二分类变量的配对卡方检验。

    适用：同一批对象在两种方法 / 两次判断下的"是否"结果，检验是否存在系统性差异。
    """
    from statsmodels.stats.contingency_tables import mcnemar
    data = df[[col1, col2]].dropna()
    c1, c2 = data[col1], data[col2]
    lv1 = [v for v in c1.unique() if not pd.isna(v)]
    lv2 = [v for v in c2.unique() if not pd.isna(v)]
    if len(lv1) != 2 or len(lv2) != 2:
        raise ValueError("两个变量都必须是二分类列（各有 2 个水平）。")

    def _enc(x, levels):
        return (x == levels[1]).astype(int)

    a, b = _enc(c1, lv1), _enc(c2, lv2)
    tbl = np.array([[int(((a == i) & (b == j)).sum()) for j in (0, 1)] for i in (0, 1)],
                   dtype=float)
    res = mcnemar(tbl, exact=False, correction=True)
    stat, p = float(res.statistic), float(res.pvalue)
    disc = int(tbl[0, 1] + tbl[1, 0])
    if disc > 0:
        p_exact = float(mcnemar(tbl, exact=True).pvalue)
    else:
        p_exact = 1.0
    n = int(tbl.sum())
    agreement = (tbl[0, 0] + tbl[1, 1]) / n if n > 0 else np.nan
    if p < alpha:
        conclusion = (f"两次判断 / 两种方法的结果存在显著系统性差异"
                      f"（McNemar χ²={stat:.4f}, p={format_p(p)}）。")
    else:
        conclusion = (f"两次判断 / 两种方法的结果无显著系统性差异"
                      f"（McNemar χ²={stat:.4f}, p={format_p(p)}）。")
    ct = pd.DataFrame(
        tbl.astype(int),
        index=[f"{col1} 否", f"{col1} 是"],
        columns=[f"{col2} 否", f"{col2} 是"],
    )
    return {
        "变量1": col1, "变量2": col2, "样本量": n,
        "McNemar χ²(校正)": stat, "p 值": p,
        "精确 p 值(Binomial)": p_exact,
        "不一致对数": disc, "一致率": agreement,
        "列联表": ct,
        "结论": conclusion,
    }


# ======================================================================
# v1.4.0 新增：逻辑回归 / 多元线性回归 / 信度分析
# ======================================================================

def logistic_regression(df, y_col, x_cols, alpha=0.05, success_value=None):
    """二分类逻辑回归（statsmodels Logit）。

    y_col 为二分类因变量；x_cols 为数值自变量列表。
    success_value 为 None 时自动编码：数值列取 1，文本列取第二个水平为"成功"。
    """
    import statsmodels.api as sm
    from sklearn.metrics import confusion_matrix, roc_auc_score
    if not x_cols:
        raise ValueError("至少需要 1 个自变量。")
    y = df[y_col]
    levels = [v for v in y.dropna().unique() if not pd.isna(v)]
    if len(levels) != 2:
        raise ValueError(f"因变量 {y_col} 必须是二分类列（当前有 {len(levels)} 个水平）。")
    if success_value is None:
        if pd.api.types.is_numeric_dtype(y):
            success = 1 if 1 in levels else levels[1]
        else:
            success = levels[1]
    else:
        success = success_value
    dat = pd.DataFrame({"__y": (y == success).astype(int)})
    for c in x_cols:
        dat[c] = pd.to_numeric(df[c], errors="coerce")
    dat = dat.dropna()
    if len(dat) < max(4, len(x_cols) + 2):
        raise ValueError("有效样本量不足，请检查缺失值。")
    X = sm.add_constant(dat[x_cols])
    try:
        model = sm.Logit(dat["__y"], X).fit(disp=0, maxiter=200)
    except Exception as e:
        raise ValueError(f"逻辑回归未收敛（数据可能存在完全分离）：{e}")
    params, pvals, bse = model.params, model.pvalues, model.bse
    ci = model.conf_int()
    ors = np.exp(params)
    or_ci = np.exp(ci.to_numpy())
    n = int(model.nobs)
    mcfadden = (1 - model.llf / model.llnull
                if model.llnull and np.isfinite(model.llnull) else np.nan)
    llr_p = float(model.llr_pvalue)
    prob = model.predict(X)
    pred = (prob > 0.5).astype(int)
    acc = float((pred == dat["__y"]).mean())
    cm = confusion_matrix(dat["__y"], pred)
    try:
        auc = float(roc_auc_score(dat["__y"], prob))
    except Exception:
        auc = np.nan
    coef_df = pd.DataFrame({
        "变量": ["常数项"] + list(x_cols),
        "系数 B": params.values,
        "标准误": bse.values,
        "z": (params / bse).values,
        "p 值": pvals.values,
        "OR": ors.values,
        "OR 95%CI低": or_ci[:, 0],
        "OR 95%CI高": or_ci[:, 1],
    })
    sig_preds = [c for c, pv in zip(x_cols, pvals.values[1:]) if pv < alpha]
    if llr_p < alpha:
        conclusion = (f"模型整体显著（似然比 p={format_p(llr_p)}，"
                      f"McFadden R²={mcfadden:.4f}）。")
        conclusion += (f"显著预测因子：{'、'.join(sig_preds)}。" if sig_preds
                       else "无单个显著预测因子。")
    else:
        conclusion = f"模型整体不显著（似然比 p={format_p(llr_p)}），自变量对 {y_col} 的预测力不足。"
    return {
        "因变量": y_col, "成功水平": success, "自变量": list(x_cols), "样本量": n,
        "系数表": coef_df,
        "McFadden R²": mcfadden, "整体似然比 p": llr_p,
        "准确率": acc, "AUC": auc,
        "混淆矩阵": cm,
        "预测概率": np.asarray(prob),
        "实际标签": dat["__y"].to_numpy(),
        "结论": conclusion,
    }


def multiple_linear_regression(df, y_col, x_cols, alpha=0.05):
    """多元线性回归（statsmodels OLS），含 VIF 共线性诊断与 Durbin-Watson 自相关检验。"""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson
    if len(x_cols) < 2:
        raise ValueError("多元线性回归至少需要 2 个自变量。")
    dat = df[[y_col] + x_cols].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(dat)
    k = len(x_cols)
    if n < k + 3:
        raise ValueError(f"有效样本量 {n} 不足（至少需要 {k + 3}）。")
    X = sm.add_constant(dat[x_cols])
    model = sm.OLS(dat[y_col], X).fit()
    ci = model.conf_int()
    coef_df = pd.DataFrame({
        "变量": ["常数项"] + list(x_cols),
        "系数 B": model.params.values,
        "标准误": model.bse.values,
        "t": model.tvalues.values,
        "p 值": model.pvalues.values,
        "95%CI低": ci[0].values,
        "95%CI高": ci[1].values,
    })
    vif_rows = []
    for i, c in enumerate(x_cols):
        vif_rows.append({"自变量": c, "VIF": float(variance_inflation_factor(X.values, i + 1))})
    vif_df = pd.DataFrame(vif_rows)
    dw = float(durbin_watson(model.resid))
    r2, adj_r2 = float(model.rsquared), float(model.rsquared_adj)
    f, f_p = float(model.fvalue), float(model.f_pvalue)
    rmse = float(np.sqrt(np.mean(model.resid ** 2)))
    aic, bic = float(model.aic), float(model.bic)
    sig_preds = [c for c, pv in zip(x_cols, model.pvalues.values[1:]) if pv < alpha]
    high_vif = [r["自变量"] for r in vif_rows if r["VIF"] >= 10]
    if f_p < alpha:
        conclusion = f"模型整体显著（F={f:.4f}, p={format_p(f_p)}），R²={r2:.4f}。"
        conclusion += (f"显著预测因子：{'、'.join(sig_preds)}。" if sig_preds
                       else "无单个显著预测因子。")
    else:
        conclusion = f"模型整体不显著（F={f:.4f}, p={format_p(f_p)}），自变量解释力不足。"
    if high_vif:
        conclusion += f"注意：{'、'.join(high_vif)} 存在严重共线性（VIF≥10）。"
    return {
        "因变量": y_col, "自变量": list(x_cols), "样本量": n,
        "R²": r2, "调整 R²": adj_r2,
        "F 统计量": f, "整体 p 值": f_p,
        "Durbin-Watson": dw,
        "RMSE": rmse, "AIC": aic, "BIC": bic,
        "系数表": coef_df,
        "VIF 表": vif_df,
        "残差": model.resid.to_numpy(),
        "结论": conclusion,
    }


def cronbach_alpha(df, cols):
    """Cronbach's α 信度分析：衡量一组题项（如李克特量表）的内部一致性。"""
    X = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    k = X.shape[1]
    n = len(X)
    if k < 2:
        raise ValueError("信度分析至少需要 2 个题项（列）。")
    if n < 2:
        raise ValueError("至少需要 2 个有效观测。")
    item_var = float(X.var(axis=0, ddof=1).sum())
    total_var = float(X.sum(axis=1).var(ddof=1))
    alpha = k / (k - 1) * (1 - item_var / total_var) if total_var > 0 else np.nan
    corr = X.corr().to_numpy()
    avg_r = (np.sum(corr) - k) / (k * (k - 1)) if k > 1 else np.nan
    alpha_std = (k * avg_r / (1 + (k - 1) * avg_r)
                 if np.isfinite(avg_r) else np.nan)
    del_rows = []
    for c in X.columns:
        Xc = X.drop(columns=[c])
        iv = float(Xc.var(axis=0, ddof=1).sum())
        tv = float(Xc.sum(axis=1).var(ddof=1))
        kc = Xc.shape[1]
        a_del = kc / (kc - 1) * (1 - iv / tv) if tv > 0 else np.nan
        del_rows.append({"删除的题项": str(c), "删除后 α": a_del})
    del_df = pd.DataFrame(del_rows).set_index("删除的题项")
    if np.isnan(alpha):
        conclusion = "数据方差为 0，无法计算 Cronbach's α（请检查题项是否有变异）。"
    elif alpha < 0.6:
        conclusion = f"Cronbach's α={alpha:.4f}，内部一致性较差（<0.6），建议考虑删除部分题项。"
    elif alpha < 0.7:
        conclusion = f"Cronbach's α={alpha:.4f}，内部一致性可接受（0.6~0.7）。"
    elif alpha < 0.9:
        conclusion = f"Cronbach's α={alpha:.4f}，内部一致性良好（0.7~0.9）。"
    else:
        conclusion = f"Cronbach's α={alpha:.4f}，内部一致性很高（≥0.9），需警惕题项冗余。"
    return {
        "样本量": n, "题项数": k,
        "Cronbach's α": alpha,
        "标准化 α": alpha_std,
        "平均项间相关": avg_r,
        "删除题项后 α": del_df,
        "结论": conclusion,
    }