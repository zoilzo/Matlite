# report.py
# -*- coding: utf-8 -*-
"""
用 python-docx 生成 Word 报告。
"""

import io

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


def _set_font(run, name="宋体", size=None):
    """设置中文字体。"""
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size:
        run.font.size = Pt(size)


def _add_dataframe_table(doc, df, title=None):
    """把 DataFrame 插入 Word 表格。"""
    if title:
        doc.add_heading(title, level=3)

    df_show = df.reset_index()
    table = doc.add_table(rows=1, cols=len(df_show.columns))
    table.style = "Table Grid"

    hdr = table.rows[0].cells
    for i, col in enumerate(df_show.columns):
        hdr[i].text = str(col)
        for p in hdr[i].paragraphs:
            for r in p.runs:
                _set_font(r, "宋体", 9)

    for _, row in df_show.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            if pd.isna(val):
                text = "—"
            elif isinstance(val, float):
                text = f"{val:.4f}"
            else:
                text = str(val)
            cells[i].text = text
            for p in cells[i].paragraphs:
                for r in p.runs:
                    _set_font(r, "宋体", 9)


def _add_figure(doc, fig, caption):
    """把 matplotlib 图插入 Word。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    doc.add_picture(buf, width=Inches(5.8))
    doc.add_paragraph(caption).alignment = WD_ALIGN_PARAGRAPH.CENTER
    buf.close()


def build_word_report(df, data_name, desc_table=None, norm_table=None,
                      ttest=None, reg=None, figures=None, alpha=0.05):
    """生成 Word 报告，返回 bytes。"""
    doc = Document()

    # 标题
    title = doc.add_heading("统计分析报告", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"数据文件：{data_name}")
    doc.add_paragraph(f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")
    doc.add_paragraph(f"显著性水平 α = {alpha}")

    # 一、数据概况
    doc.add_heading("一、数据概况", level=1)
    doc.add_paragraph(
        f"观测数：{df.shape[0]} 行；变量数：{df.shape[1]} 列；"
        f"缺失值总数：{int(df.isna().sum().sum())}。"
    )
    info = pd.DataFrame({
        "数据类型": df.dtypes.astype(str),
        "非缺失数": df.notna().sum(),
        "缺失数": df.isna().sum(),
        "缺失比例(%)": (df.isna().mean() * 100).round(2),
        "唯一值个数": df.nunique(dropna=True),
    })
    _add_dataframe_table(doc, info, "变量信息")

    # 二、描述统计
    if desc_table is not None and not desc_table.empty:
        doc.add_heading("二、描述统计", level=1)
        _add_dataframe_table(doc, desc_table, "描述统计表")

    # 三、正态性检验
    if norm_table is not None and not norm_table.empty:
        doc.add_heading("三、正态性检验", level=1)
        norm_show = norm_table.copy()
        if "p 值" in norm_show.columns:
            norm_show["p 值"] = norm_show["p 值"].apply(
                lambda p: "<0.001" if pd.notna(p) and p < 0.001
                else (f"{p:.4f}" if pd.notna(p) else "—")
            )
        _add_dataframe_table(doc, norm_show, "正态性检验结果")
        doc.add_paragraph("注：p > α 时不拒绝正态假设。")

    # 四、t 检验
    if ttest:
        doc.add_heading("四、独立样本 t 检验", level=1)
        r = ttest
        doc.add_paragraph(f"因变量：{r['因变量']}；分组变量：{r['分组变量']}。")
        doc.add_paragraph(
            f"{r['水平1']}：n={r['n1']}，均值={r['mean1']:.4f}，"
            f"标准差={r['std1']:.4f}。"
        )
        doc.add_paragraph(
            f"{r['水平2']}：n={r['n2']}，均值={r['mean2']:.4f}，"
            f"标准差={r['std2']:.4f}。"
        )
        doc.add_paragraph(
            f"Levene 方差齐性检验：统计量={r['levene_stat']:.4f}，"
            f"p={r['levene_p']:.4f}，方差{'齐性' if r['equal_var'] else '不齐性'}。"
        )
        doc.add_paragraph(
            f"检验方法：{r['检验方法']}；t={r['t']:.4f}，"
            f"df={r['df']:.2f}，p={r['p']:.4f}。"
        )
        doc.add_paragraph(
            f"均值差={r['mean_diff']:.4f}，"
            f"95% CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]。"
        )
        doc.add_paragraph(
            f"Cohen's d={r['cohens_d']:.4f}（{r['effect_label']}）。"
        )
        doc.add_paragraph(
            f"Mann-Whitney U：U={r['mannwhitney_u']:.1f}，"
            f"p={r['mannwhitney_p']:.4f}。"
        )
        doc.add_paragraph(f"结论：{r['结论']}")

    # 五、回归
    if reg:
        doc.add_heading("五、简单线性回归", level=1)
        r = reg
        doc.add_paragraph(f"回归方程：{r['equation']}，n={r['n']}。")
        doc.add_paragraph(
            f"R²={r['r2']:.4f}，调整 R²={r['adj_r2']:.4f}，"
            f"F={r['f_stat']:.4f}，p={r['f_p']:.4f}。"
        )
        doc.add_paragraph(
            f"常数项 β₀={r['b0']:.4f}（SE={r['se0']:.4f}, "
            f"t={r['t0']:.4f}, p={r['p0']:.4f}）。"
        )
        doc.add_paragraph(
            f"斜率 β₁={r['b1']:.4f}（SE={r['se1']:.4f}, "
            f"t={r['t1']:.4f}, p={r['p1']:.4f}）。"
        )
        doc.add_paragraph(
            f"Pearson r={r['pearson_r']:.4f}，p={r['pearson_p']:.4f}。"
        )
        doc.add_paragraph(f"结论：{r['结论']}")

    # 六、图形
    if figures:
        doc.add_heading("六、图形展示", level=1)
        for fig, caption in figures:
            _add_figure(doc, fig, caption)

    out = io.BytesIO()
    doc.save(out)
    out.seek(0)
    return out.getvalue()