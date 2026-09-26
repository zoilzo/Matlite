# -*- coding: utf-8 -*-
"""数据分析：导入 Excel/CSV，描述统计 / 正态性检验 / t 检验 / 回归 / 绘图 / 导出 Word 报告"""

import os
import sys
from tkinter import filedialog, messagebox

# 确保能导入父目录下的统计模块（stats_utils / stats_plots / stats_report）
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np
import pandas as pd
import tkinter.ttk as ttk
import customtkinter as ctk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from stats_utils import (
    descriptive_stats_table, normality_table,
    independent_t_test, simple_linear_regression, format_p,
    chi_square_test, anova_one_way, correlation_test, paired_t_test,
    mannwhitney_u_test, wilcoxon_signed_rank, kruskal_wallis_test,
    friedman_test, posthoc_test, one_sample_t_test, proportion_test,
    mcnemar_test, logistic_regression, multiple_linear_regression,
    cronbach_alpha,
)
import stats_plots as sp
from stats_report import build_word_report


ANALYSIS_TYPES = ["描述统计", "正态性检验", "t 检验", "回归分析",
                  "卡方检验", "方差分析 ANOVA", "相关性检验", "配对 t 检验",
                  "Mann-Whitney U", "Wilcoxon 符号秩", "Kruskal-Wallis",
                  "Friedman 检验", "事后多重比较", "单样本 t 检验",
                  "比例检验", "McNemar 检验", "逻辑回归", "多元线性回归",
                  "信度分析 (Cronbach's α)"]
PLOT_TYPES = ["散点图（含拟合线）", "直方图（含正态曲线）", "正态 Q-Q 图", "分组箱线图",
              "小提琴图（分组分布）", "误差棒图（均值±SE）", "相关矩阵热图", "回归森林图"]
ALPHA_CHOICES = ["0.05", "0.01", "0.10"]


class DataPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- 数据状态 ----
        self.df = None
        self.file_path = None
        self.file_name = None
        self.desc_table = None
        self.norm_table = None
        self.ttest = None
        self.reg = None
        self.reg_x = None
        self.chi2 = None
        self.anova = None
        self.corr = None
        self.paired = None
        self.figures = []           # [(fig, caption), ...]
        self._col_vars = {}
        self._canvas = None

        # ================= 左：控制面板 =================
        left = ctk.CTkScrollableFrame(self, width=360, corner_radius=12, label_text="数据分析")
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.grid_columnconfigure(0, weight=1)

        # 导入按钮 + 上次文件
        imp_row = ctk.CTkFrame(left, fg_color="transparent")
        imp_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 4))
        imp_row.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(imp_row, text="📂 导入数据", height=38, command=self._load).grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        self.last_btn = ctk.CTkButton(imp_row, text="↩ 上次文件", height=38, width=110,
                                      fg_color="gray40", command=self._load_last)
        self.last_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.last_btn.configure(state="disabled")
        self.file_label = ctk.CTkLabel(left, text="尚未导入数据", font=ctk.CTkFont(size=12),
                                       text_color="steelblue", wraplength=300, justify="left")
        self.file_label.grid(row=1, column=0, sticky="w", padx=14, pady=(2, 2))
        self.info_label = ctk.CTkLabel(left, text="", font=ctk.CTkFont(size=12), text_color="gray50")
        self.info_label.grid(row=2, column=0, sticky="w", padx=14, pady=(2, 6))

        # 分析类型（下拉选择，避免 8 个选项塞不下文字）
        ctk.CTkLabel(left, text="选择分析", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=3, column=0, sticky="w", padx=14, pady=(6, 4))
        self.type = ctk.CTkOptionMenu(left, values=ANALYSIS_TYPES,
                                      command=lambda _v: self._switch_type(),
                                      width=210)
        self.type.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 6))
        self.type.set(ANALYSIS_TYPES[0])

        # 动态参数区（随分析类型变化）
        self.dyn = ctk.CTkFrame(left, fg_color="transparent")
        self.dyn.grid(row=5, column=0, sticky="ew", padx=12, pady=4)
        self.dyn.grid_columnconfigure(0, weight=1)
        self.dyn_widgets = []

        # 显著性水平 + 运行
        alpha_row = ctk.CTkFrame(left, fg_color="transparent")
        alpha_row.grid(row=6, column=0, sticky="ew", padx=12, pady=(8, 4))
        alpha_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(alpha_row, text="显著性水平 α", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        self.alpha_dd = ctk.CTkOptionMenu(alpha_row, values=ALPHA_CHOICES, width=80)
        self.alpha_dd.grid(row=0, column=1, sticky="w")
        self.alpha_dd.set("0.05")

        ctk.CTkButton(left, text="▶ 运行分析", height=40, command=self._run).grid(
            row=7, column=0, sticky="ew", padx=12, pady=(4, 6))

        # 绘图区
        ctk.CTkLabel(left, text="绘制图形", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=8, column=0, sticky="w", padx=14, pady=(10, 4))
        self.plot_type_dd = ctk.CTkOptionMenu(left, values=PLOT_TYPES, command=lambda _v: self._build_plot_dyn())
        self.plot_type_dd.grid(row=9, column=0, sticky="ew", padx=12, pady=(0, 6))
        self.plot_type_dd.set(PLOT_TYPES[0])
        self.plot_dyn = ctk.CTkFrame(left, fg_color="transparent")
        self.plot_dyn.grid(row=10, column=0, sticky="ew", padx=12, pady=4)
        self.plot_dyn.grid_columnconfigure(1, weight=1)
        self.plot_widgets = []
        self._build_plot_dyn()

        ctk.CTkButton(left, text="📈 生成图形", height=36, fg_color="gray40", command=self._plot).grid(
            row=11, column=0, sticky="ew", padx=12, pady=(4, 4))

        # 导出报告
        ctk.CTkButton(left, text="📄 导出报告（Word / PDF）", height=40, fg_color="#2a7f5c",
                      command=self._export).grid(row=12, column=0, sticky="ew", padx=12, pady=(10, 4))

        # AI 解读
        ctk.CTkButton(left, text="🤖 AI 解读结果", height=40, fg_color="#5b4b8a",
                      command=self._ai_interpret).grid(row=13, column=0, sticky="ew", padx=12, pady=(4, 4))

        tip = ("使用步骤：\n"
               "① 点击「导入数据」选择 Excel 或 CSV；\n"
               "② 选择分析类型并勾选/选择变量；\n"
               "③ 点「运行分析」查看结果；\n"
               "④ 可在下方「绘制图形」出图；\n"
               "⑤ 「AI 解读结果」用本地 AI 讲人话；\n"
               "⑥ 最后「导出 Word 报告」一键成文。")
        ctk.CTkLabel(left, text=tip, justify="left", font=ctk.CTkFont(size=11),
                     text_color="gray45", anchor="w", wraplength=300).grid(
            row=14, column=0, sticky="w", padx=12, pady=(6, 14))

        # ================= 右：表格 + 结果 + 图形 =================
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=2)
        right.grid_rowconfigure(2, weight=2)
        right.grid_rowconfigure(4, weight=3)

        hd = ctk.CTkFrame(right, fg_color="transparent")
        hd.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        hd.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hd, text="分析结果（点击表头可排序）", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w")
        ctk.CTkButton(hd, text="💾 导出表格 CSV", width=140, height=30, fg_color="gray40",
                      command=self._export_table).grid(row=0, column=1, sticky="e")

        # 对齐的表格控件（数据预览 / 表格类结果都放这里）
        self.tv_frame = ctk.CTkFrame(right, fg_color="white", corner_radius=6)
        self.tv_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 6))
        self.tv_frame.grid_rowconfigure(0, weight=1)
        self.tv_frame.grid_columnconfigure(0, weight=1)
        self.tv = ttk.Treeview(self.tv_frame, show="headings")
        self.tv.grid(row=0, column=0, sticky="nsew")
        self.tv_scroll = ttk.Scrollbar(self.tv_frame, orient="vertical", command=self.tv.yview)
        self.tv_scroll.grid(row=0, column=1, sticky="ns")
        self.tv.configure(yscrollcommand=self.tv_scroll.set)
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.out = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=13))
        self.out.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 6))

        ctk.CTkLabel(right, text="图形预览", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=3, column=0, sticky="w", padx=16, pady=(6, 4))
        self.plot_frame = ctk.CTkFrame(right, fg_color="white", corner_radius=8)
        self.plot_frame.grid(row=4, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        self._write("欢迎使用数据分析。\n\n"
                    "① 点击左上角「导入 Excel / CSV 数据」加载你的数据文件；\n"
                    "② 选择分析类型并勾选变量，点「运行分析」；\n"
                    "③ 需要画图时，在左下角选择图形并「生成图形」；\n"
                    "④ 最后点「导出 Word 报告」一键生成统计报告。")

    # ================= 数据加载 =================
    def _load(self):
        f = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("数据文件", "*.xlsx *.xls *.csv *.txt"),
                       ("Excel 文件", "*.xlsx *.xls"),
                       ("CSV 文件", "*.csv"),
                       ("文本文件", "*.txt"),
                       ("所有文件", "*.*")],
        )
        if not f:
            return
        self._apply_file(f)

    def _apply_file(self, f):
        try:
            ext = f.lower().rsplit(".", 1)[-1] if "." in f else ""
            if ext in ("csv", "txt"):
                df = None
                for enc in ("utf-8-sig", "utf-8", "gbk"):
                    try:
                        if ext == "csv":
                            df = pd.read_csv(f, encoding=enc)
                        else:
                            # txt：自动嗅探分隔符（逗号 / 制表符 / 分号 / 空白）
                            df = pd.read_csv(f, encoding=enc, sep=None, engine="python")
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    raise ValueError("无法识别文件编码，请另存为 UTF-8 后重试。")
                if df.shape[1] <= 1 and ext == "txt":
                    # 单列结果通常是分隔符识别失败，退回按空白分隔
                    df = pd.read_csv(f, sep=r"\s+", engine="python")
            else:
                df = pd.read_excel(f)
        except Exception as e:
            self._write(f"导入失败：{e}")
            return
        self.df = df
        self.file_path = f
        self.file_name = os.path.basename(f)
        from modules import app_settings as S
        S.set_val("last_file", f)
        try:
            self.last_btn.configure(state="normal")
        except Exception:
            pass
        self._reset_results()
        self._update_info()
        self._switch_type()
        self._build_plot_dyn()
        self._preview()

    def _load_last(self):
        from modules import app_settings as S
        f = S.get("last_file")
        if not f or not os.path.exists(f):
            self._write("没有可用的上次文件（可能已移动或删除）。")
            return
        self._apply_file(f)

    def _reset_results(self):
        self.desc_table = None
        self.norm_table = None
        self.ttest = None
        self.reg = None
        self.reg_x = None
        self.chi2 = None
        self.anova = None
        self.corr = None
        self.paired = None
        self.mw = None
        self.wsr = None
        self.kw = None
        self.fried = None
        self.post = None
        self.osamp = None
        self.prop = None
        self.mcn = None
        self.logreg = None
        self.multireg = None
        self.cron = None
        self.figures = []
        self._clear_plot()

    def _update_info(self):
        if self.df is None:
            self.file_label.configure(text="尚未导入数据")
            self.info_label.configure(text="")
            return
        self.file_label.configure(text=self.file_name)
        n_num = len(self._numeric_columns())
        self.info_label.configure(
            text=f"{self.df.shape[0]} 行 × {self.df.shape[1]} 列\n数值列：{n_num} 个")

    def _preview(self):
        from modules import app_settings as _S
        n = min(int(_S.get("table_rows", 30)), len(self.df))
        self._show_table(self.df.head(n), f"数据预览（前 {n} 行，共 {self.df.shape[0]} 行）")
        self.out.delete("1.0", "end")
        self._write(f"数据文件：{self.file_name}\n")
        self._write(f"共 {self.df.shape[0]} 行 × {self.df.shape[1]} 列。\n")
        self._write("表格结果显示在上方（可拖动滚动条）。\n（导入成功，可选择左侧分析类型开始分析。）")

    def _clear_table(self):
        if hasattr(self, "tv"):
            self.tv.delete(*self.tv.get_children())

    def _show_table(self, df, title=""):
        """把 DataFrame 放进对齐的表格控件。"""
        self._clear_table()
        self._table_df = df
        self._table_title = title
        cols = list(df.columns)
        self.tv["columns"] = cols
        for c in cols:
            self.tv.heading(c, text=str(c), command=lambda col=c: self._sort_table(col))
            w = max(60, min(160, 14 * len(str(c))))
            self.tv.column(c, width=w, anchor="center", stretch=True)
        self._fill_rows(df)

    def _fill_rows(self, df):
        for row in df.itertuples(index=False):
            vals = []
            for v in row:
                if isinstance(v, float):
                    vals.append(f"{v:.4g}")
                elif pd.isna(v):
                    vals.append("—")
                else:
                    vals.append(str(v))
            self.tv.insert("", "end", values=vals)

    def _sort_table(self, col):
        if getattr(self, "_table_df", None) is None:
            self._write("当前没有表格数据。")
            return
        asc = getattr(self, "_sort_asc", True)
        self._sort_asc = not asc
        try:
            sdf = self._table_df.sort_values(by=col, ascending=asc).reset_index(drop=True)
        except Exception:
            sdf = self._table_df
        self._clear_table()
        self._fill_rows(sdf)

    def _export_table(self):
        df = getattr(self, "_table_df", None)
        if df is None:
            self._write("当前没有可导出的表格。")
            return
        f = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV 文件", "*.csv")],
                                         initialfile="表格数据.csv")
        if not f:
            return
        try:
            df.to_csv(f, index=False, encoding="utf-8-sig")
            self._write(f"✅ 已导出表格：{f}")
            messagebox.showinfo("导出完成", f"已保存到：\n{f}")
        except Exception as e:
            self._write(f"导出失败：{e}")

    # ================= 列工具 =================
    def _all_columns(self):
        return list(self.df.columns) if self.df is not None else []

    def _numeric_columns(self):
        if self.df is None:
            return []
        return [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]

    # ================= 动态参数区 =================
    def _clear(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    def _switch_type(self):
        self._clear(self.dyn)
        self.dyn_widgets = []
        self._col_vars = {}
        t = self.type.get()

        if self.df is None:
            ctk.CTkLabel(self.dyn, text="（导入数据后即可选择变量）",
                         font=ctk.CTkFont(size=12), text_color="gray45").grid(
                row=0, column=0, sticky="w", padx=4, pady=4)
            return

        if t in ("描述统计", "正态性检验"):
            cols = self._numeric_columns()
            if not cols:
                ctk.CTkLabel(self.dyn, text="（数据中没有数值列）",
                             font=ctk.CTkFont(size=12), text_color="gray45").grid(
                    row=0, column=0, sticky="w", padx=4, pady=4)
                return
            ctk.CTkLabel(self.dyn, text="勾选要分析的数值列：",
                         font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w", padx=4, pady=(2, 2))
            for i, c in enumerate(cols):
                var = ctk.BooleanVar(value=False)
                self._col_vars[c] = var
                ctk.CTkCheckBox(self.dyn, text=str(c), variable=var).grid(
                    row=i + 1, column=0, sticky="w", padx=4, pady=1)
        elif t == "t 检验":
            num = self._numeric_columns()
            allc = self._all_columns()
            grp = next((c for c in allc if c not in num), (allc[0] if allc else ""))
            self._dd(self.dyn, "因变量（数值列）", num, 0)
            self._dd(self.dyn, "分组变量", allc, 2, default=grp)
        elif t == "回归分析":
            num = self._numeric_columns()
            y_default = num[1] if len(num) > 1 else (num[0] if num else "")
            self._dd(self.dyn, "自变量 X（数值列）", num, 0)
            self._dd(self.dyn, "因变量 Y（数值列）", num, 2, default=y_default)
        elif t == "卡方检验":
            cat = self._cat_columns()
            if len(cat) < 2:
                ctk.CTkLabel(self.dyn, text="（需要至少 2 个分类列）",
                             font=ctk.CTkFont(size=12), text_color="gray45").grid(
                    row=0, column=0, sticky="w", padx=4, pady=4)
                return
            self._dd(self.dyn, "分类变量 1", cat, 0)
            self._dd(self.dyn, "分类变量 2", cat, 2, default=cat[1])
        elif t == "方差分析 ANOVA":
            num = self._numeric_columns()
            allc = self._all_columns()
            grp = next((c for c in allc if c not in num), (allc[0] if allc else ""))
            self._dd(self.dyn, "数值列（因变量）", num, 0)
            self._dd(self.dyn, "分组列（3 组以内）", allc, 2, default=grp)
        elif t == "相关性检验":
            num = self._numeric_columns()
            if len(num) < 2:
                ctk.CTkLabel(self.dyn, text="（需要至少 2 个数值列）",
                             font=ctk.CTkFont(size=12), text_color="gray45").grid(
                    row=0, column=0, sticky="w", padx=4, pady=4)
                return
            self._dd(self.dyn, "变量 1（X）", num, 0)
            self._dd(self.dyn, "变量 2（Y）", num, 2, default=num[1])
            ctk.CTkLabel(self.dyn, text="方法", font=ctk.CTkFont(size=12)).grid(
                row=4, column=0, sticky="w", padx=4, pady=(4, 1))
            dd = ctk.CTkOptionMenu(self.dyn, values=["Pearson", "Spearman"])
            dd.grid(row=5, column=0, sticky="ew", padx=4, pady=(0, 4))
            dd.set("Pearson")
            self.dyn_widgets.append(dd)
        elif t == "配对 t 检验":
            num = self._numeric_columns()
            if len(num) < 2:
                ctk.CTkLabel(self.dyn, text="（需要至少 2 个数值列）",
                             font=ctk.CTkFont(size=12), text_color="gray45").grid(
                    row=0, column=0, sticky="w", padx=4, pady=4)
                return
            self._dd(self.dyn, "前测列（如实验前）", num, 0)
            self._dd(self.dyn, "后测列（如实验后）", num, 2, default=num[1])
        elif t == "Mann-Whitney U":
            num = self._numeric_columns()
            allc = self._all_columns()
            grp = next((c for c in allc if c not in num), (allc[0] if allc else ""))
            self._dd(self.dyn, "因变量（数值列）", num, 0)
            self._dd(self.dyn, "分组变量（恰好 2 组）", allc, 2, default=grp)
        elif t == "Wilcoxon 符号秩":
            num = self._numeric_columns()
            if len(num) < 2:
                ctk.CTkLabel(self.dyn, text="（需要至少 2 个数值列）",
                             font=ctk.CTkFont(size=12), text_color="gray45").grid(
                    row=0, column=0, sticky="w", padx=4, pady=4)
                return
            self._dd(self.dyn, "前测列（如实验前）", num, 0)
            self._dd(self.dyn, "后测列（如实验后）", num, 2, default=num[1])
        elif t == "Kruskal-Wallis":
            num = self._numeric_columns()
            allc = self._all_columns()
            grp = next((c for c in allc if c not in num), (allc[0] if allc else ""))
            self._dd(self.dyn, "因变量（数值列）", num, 0)
            self._dd(self.dyn, "分组变量（≥2 组）", allc, 2, default=grp)
        elif t == "Friedman 检验":
            num = self._numeric_columns()
            if len(num) < 2:
                ctk.CTkLabel(self.dyn, text="（需要至少 2 个测量条件列）",
                             font=ctk.CTkFont(size=12), text_color="gray45").grid(
                    row=0, column=0, sticky="w", padx=4, pady=4)
                return
            ctk.CTkLabel(self.dyn, text="勾选测量条件列（同一批对象）：",
                         font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w", padx=4, pady=(2, 2))
            for i, c in enumerate(num):
                var = ctk.BooleanVar(value=False)
                self._col_vars[c] = var
                ctk.CTkCheckBox(self.dyn, text=str(c), variable=var).grid(
                    row=i + 1, column=0, sticky="w", padx=4, pady=1)
        elif t == "事后多重比较":
            num = self._numeric_columns()
            allc = self._all_columns()
            grp = next((c for c in allc if c not in num), (allc[0] if allc else ""))
            self._dd(self.dyn, "因变量（数值列）", num, 0)
            self._dd(self.dyn, "分组变量（≥3 组）", allc, 2, default=grp)
            self._dd(self.dyn, "比较方法", ["Tukey HSD", "LSD", "Bonferroni"], 4)
        elif t == "单样本 t 检验":
            num = self._numeric_columns()
            if not num:
                ctk.CTkLabel(self.dyn, text="（数据中没有数值列）",
                             font=ctk.CTkFont(size=12), text_color="gray45").grid(
                    row=0, column=0, sticky="w", padx=4, pady=4)
                return
            self._dd(self.dyn, "变量列", num, 0)
            ctk.CTkLabel(self.dyn, text="检验值 μ0（默认 0）", font=ctk.CTkFont(size=12)).grid(
                row=2, column=0, sticky="w", padx=4, pady=(4, 1))
            mu_entry = ctk.CTkEntry(self.dyn, placeholder_text="例如：5")
            mu_entry.grid(row=3, column=0, sticky="ew", padx=4, pady=(0, 4))
            mu_entry.insert(0, "0")
            self.dyn_widgets.append(mu_entry)
        elif t == "比例检验":
            cat = self._cat_columns()
            if not cat:
                ctk.CTkLabel(self.dyn, text="（数据中没有分类列）",
                             font=ctk.CTkFont(size=12), text_color="gray45").grid(
                    row=0, column=0, sticky="w", padx=4, pady=4)
                return
            self._dd(self.dyn, "分类变量（二分类）", cat, 0)
            ctk.CTkLabel(self.dyn, text="成功水平（留空自动）", font=ctk.CTkFont(size=12)).grid(
                row=2, column=0, sticky="w", padx=4, pady=(4, 1))
            succ_entry = ctk.CTkEntry(self.dyn, placeholder_text="自动：数值列取 1，文本取第 2 个水平")
            succ_entry.grid(row=3, column=0, sticky="ew", padx=4, pady=(0, 4))
            ctk.CTkLabel(self.dyn, text="检验比例 p0（默认 0.5）", font=ctk.CTkFont(size=12)).grid(
                row=4, column=0, sticky="w", padx=4, pady=(4, 1))
            p0_entry = ctk.CTkEntry(self.dyn, placeholder_text="例如：0.5")
            p0_entry.grid(row=5, column=0, sticky="ew", padx=4, pady=(0, 4))
            p0_entry.insert(0, "0.5")
            self.dyn_widgets += [succ_entry, p0_entry]
        elif t == "McNemar 检验":
            cat = self._cat_columns()
            if len(cat) < 2:
                ctk.CTkLabel(self.dyn, text="（需要至少 2 个二分类列）",
                             font=ctk.CTkFont(size=12), text_color="gray45").grid(
                    row=0, column=0, sticky="w", padx=4, pady=4)
                return
            self._dd(self.dyn, "变量 1（二分类）", cat, 0)
            self._dd(self.dyn, "变量 2（二分类）", cat, 2, default=cat[1])
        elif t == "逻辑回归":
            cat = self._cat_columns()
            num = self._numeric_columns()
            if not cat:
                ctk.CTkLabel(self.dyn, text="（数据中没有二分类因变量）",
                             font=ctk.CTkFont(size=12), text_color="gray45").grid(
                    row=0, column=0, sticky="w", padx=4, pady=4)
                return
            self._dd(self.dyn, "因变量 Y（二分类）", cat, 0)
            ctk.CTkLabel(self.dyn, text="勾选自变量（数值列）：",
                         font=ctk.CTkFont(size=12)).grid(row=2, column=0, sticky="w", padx=4, pady=(2, 2))
            for i, c in enumerate(num):
                var = ctk.BooleanVar(value=False)
                self._col_vars[c] = var
                ctk.CTkCheckBox(self.dyn, text=str(c), variable=var).grid(
                    row=i + 3, column=0, sticky="w", padx=4, pady=1)
        elif t == "多元线性回归":
            num = self._numeric_columns()
            if len(num) < 3:
                ctk.CTkLabel(self.dyn, text="（需要至少 3 个数值列：1 个因变量 + 2 个自变量）",
                             font=ctk.CTkFont(size=12), text_color="gray45").grid(
                    row=0, column=0, sticky="w", padx=4, pady=4)
                return
            self._dd(self.dyn, "因变量 Y（数值列）", num, 0)
            ctk.CTkLabel(self.dyn, text="勾选自变量（≥2 个数值列）：",
                         font=ctk.CTkFont(size=12)).grid(row=2, column=0, sticky="w", padx=4, pady=(2, 2))
            for i, c in enumerate(num):
                var = ctk.BooleanVar(value=False)
                self._col_vars[c] = var
                ctk.CTkCheckBox(self.dyn, text=str(c), variable=var).grid(
                    row=i + 3, column=0, sticky="w", padx=4, pady=1)
        elif t == "信度分析 (Cronbach's α)":
            num = self._numeric_columns()
            if len(num) < 2:
                ctk.CTkLabel(self.dyn, text="（需要至少 2 个题项列）",
                             font=ctk.CTkFont(size=12), text_color="gray45").grid(
                    row=0, column=0, sticky="w", padx=4, pady=4)
                return
            ctk.CTkLabel(self.dyn, text="勾选题项列（≥2）：",
                         font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w", padx=4, pady=(2, 2))
            for i, c in enumerate(num):
                var = ctk.BooleanVar(value=False)
                self._col_vars[c] = var
                ctk.CTkCheckBox(self.dyn, text=str(c), variable=var).grid(
                    row=i + 1, column=0, sticky="w", padx=4, pady=1)

    def _cat_columns(self):
        """分类列：非数值列，或取值很少的数值列。"""
        if self.df is None:
            return []
        out = []
        for c in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df[c]):
                out.append(c)
            elif self.df[c].nunique(dropna=True) <= 10:
                out.append(c)
        return out

    def _dd(self, frame, label, cols, row, default=None):
        ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=12)).grid(
            row=row, column=0, sticky="w", padx=4, pady=(4, 1))
        dd = ctk.CTkOptionMenu(frame, values=[str(c) for c in cols])
        dd.grid(row=row + 1, column=0, sticky="ew", padx=4, pady=(0, 4))
        if cols:
            dd.set(str(default) if default is not None and default in cols else str(cols[0]))
        self.dyn_widgets.append(dd)

    def _selected_cols(self):
        return [c for c, v in self._col_vars.items() if v.get()]

    def _distinct(self, a, b, what):
        if a == b:
            self._write(f"请选择两个不同的列（当前 {what} 都是「{a}」）。")
            return False
        return True

    # ================= 绘图参数区 =================
    def _build_plot_dyn(self):
        self._clear(self.plot_dyn)
        self.plot_widgets = []
        if self.df is None:
            ctk.CTkLabel(self.plot_dyn, text="（导入数据后即可绘图）",
                         font=ctk.CTkFont(size=12), text_color="gray45").grid(
                row=0, column=0, sticky="w", padx=4, pady=4)
            return
        pt = self.plot_type_dd.get()
        num = self._numeric_columns()
        if not num:
            ctk.CTkLabel(self.plot_dyn, text="（数据中没有数值列）",
                         font=ctk.CTkFont(size=12), text_color="gray45").grid(
                row=0, column=0, sticky="w", padx=4, pady=4)
            return
        if pt == "散点图（含拟合线）":
            self._dd2("X 列", num, 0)
            self._dd2("Y 列", num, 1, default=(num[1] if len(num) > 1 else num[0]))
        elif pt == "直方图（含正态曲线）":
            self._dd2("变量列", num, 0)
        elif pt == "正态 Q-Q 图":
            self._dd2("变量列", num, 0)
        elif pt == "分组箱线图":
            allc = self._all_columns()
            grp = next((c for c in allc if c not in num), (allc[0] if allc else ""))
            self._dd2("数值列", num, 0)
            self._dd2("分组列", allc, 1, default=grp)
        elif pt in ("小提琴图（分组分布）", "误差棒图（均值±SE）"):
            allc = self._all_columns()
            grp = next((c for c in allc if c not in num), (allc[0] if allc else ""))
            self._dd2("数值列", num, 0)
            self._dd2("分组列", allc, 1, default=grp)

    def _dd2(self, label, cols, row, default=None):
        ctk.CTkLabel(self.plot_dyn, text=label, font=ctk.CTkFont(size=12)).grid(
            row=row, column=0, sticky="w", padx=4, pady=(4, 1))
        dd = ctk.CTkOptionMenu(self.plot_dyn, values=[str(c) for c in cols])
        dd.grid(row=row, column=1, sticky="ew", padx=4, pady=(0, 4))
        if cols:
            dd.set(str(default) if default is not None and default in cols else str(cols[0]))
        self.plot_widgets.append(dd)

    # ================= 运行分析 =================
    def _run(self):
        if self.df is None:
            self._write("请先点击「导入 Excel / CSV 数据」加载文件。")
            return
        self.out.delete("1.0", "end")
        self._out_text = ""
        alpha = float(self.alpha_dd.get())
        t = self.type.get()
        try:
            if t == "描述统计":
                cols = self._selected_cols()
                if not cols:
                    self._write("请至少勾选一个数值列。")
                    return
                self.desc_table = descriptive_stats_table(self.df, cols)
                self._show_table(self.desc_table.reset_index(), "描述统计")
                self._write("【描述统计】结果见上方表格。")
                self._write("注：偏度>0 右偏，<0 左偏；峰度>0 尖峰（正态峰度为 0）。")
                return
            elif t == "正态性检验":
                cols = self._selected_cols()
                if not cols:
                    self._write("请至少勾选一个数值列。")
                    return
                self.norm_table = normality_table(self.df, cols, alpha=alpha)
                show = self.norm_table.reset_index()
                self._show_table(show, "正态性检验")
                self._write(f"【正态性检验】（α = {alpha}）结果见上方表格。")
                self._write("结论：p 值 > α 时不拒绝正态假设（可视为近似正态）。")
            elif t == "t 检验":
                val_dd, grp_dd = self.dyn_widgets
                val_col, grp_col = val_dd.get(), grp_dd.get()
                if not self._distinct(val_col, grp_col, "因变量与分组变量"):
                    return
                self.ttest = independent_t_test(self.df, val_col, grp_col, alpha=alpha)
                self._print_ttest(self.ttest, alpha)
            elif t == "回归分析":
                x_dd, y_dd = self.dyn_widgets
                x_col, y_col = x_dd.get(), y_dd.get()
                if not self._distinct(x_col, y_col, "X 与 Y"):
                    return
                self.reg_x, self.reg_y = x_col, y_col
                self.reg = simple_linear_regression(self.df, x_col, y_col, alpha=alpha)
                self._print_reg(self.reg, alpha)
            elif t == "卡方检验":
                c1_dd, c2_dd = self.dyn_widgets
                c1, c2 = c1_dd.get(), c2_dd.get()
                if not self._distinct(c1, c2, "两个分类变量"):
                    return
                self.chi2 = chi_square_test(self.df, c1, c2, alpha=alpha)
                self._print_chi2(self.chi2)
            elif t == "方差分析 ANOVA":
                v_dd, g_dd = self.dyn_widgets
                v_col, g_col = v_dd.get(), g_dd.get()
                if not self._distinct(v_col, g_col, "数值列与分组列"):
                    return
                self.anova = anova_one_way(self.df, v_col, g_col, alpha=alpha)
                self._print_anova(self.anova)
            elif t == "相关性检验":
                x_dd, y_dd, m_dd = self.dyn_widgets
                x_col, y_col = x_dd.get(), y_dd.get()
                if not self._distinct(x_col, y_col, "X 与 Y"):
                    return
                self.corr = correlation_test(self.df, x_col, y_col,
                                             method=m_dd.get(), alpha=alpha)
                self._print_corr(self.corr)
            elif t == "配对 t 检验":
                pre_dd, post_dd = self.dyn_widgets
                pre_c, post_c = pre_dd.get(), post_dd.get()
                if not self._distinct(pre_c, post_c, "前测列与后测列"):
                    return
                self.paired = paired_t_test(self.df, pre_c, post_c, alpha=alpha)
                self._print_paired(self.paired)
            elif t == "Mann-Whitney U":
                v_dd, g_dd = self.dyn_widgets
                if not self._distinct(v_dd.get(), g_dd.get(), "因变量与分组变量"):
                    return
                self.mw = mannwhitney_u_test(self.df, v_dd.get(), g_dd.get(), alpha=alpha)
                self._print_mannwhitney(self.mw)
            elif t == "Wilcoxon 符号秩":
                pre_dd, post_dd = self.dyn_widgets
                if not self._distinct(pre_dd.get(), post_dd.get(), "前测列与后测列"):
                    return
                self.wsr = wilcoxon_signed_rank(self.df, pre_dd.get(), post_dd.get(), alpha=alpha)
                self._print_wilcoxon(self.wsr)
            elif t == "Kruskal-Wallis":
                v_dd, g_dd = self.dyn_widgets
                if not self._distinct(v_dd.get(), g_dd.get(), "因变量与分组变量"):
                    return
                self.kw = kruskal_wallis_test(self.df, v_dd.get(), g_dd.get(), alpha=alpha)
                self._print_kruskal(self.kw)
            elif t == "Friedman 检验":
                cols = self._selected_cols()
                if len(cols) < 2:
                    self._write("请至少勾选 2 个测量条件列。")
                    return
                self.fried = friedman_test(self.df, cols, alpha=alpha)
                self._print_friedman(self.fried)
            elif t == "事后多重比较":
                v_dd, g_dd, m_dd = self.dyn_widgets
                if not self._distinct(v_dd.get(), g_dd.get(), "因变量与分组变量"):
                    return
                method_map = {"Tukey HSD": "Tukey", "LSD": "LSD", "Bonferroni": "Bonferroni"}
                self.post = posthoc_test(self.df, v_dd.get(), g_dd.get(),
                                         method=method_map[m_dd.get()], alpha=alpha)
                self._print_posthoc(self.post)
            elif t == "单样本 t 检验":
                col_dd, mu_entry = self.dyn_widgets
                mu = self._parse_float(mu_entry.get(), "检验值 μ0")
                self.osamp = one_sample_t_test(self.df, col_dd.get(), mu=mu, alpha=alpha)
                self._print_onesample(self.osamp)
            elif t == "比例检验":
                col_dd, succ_entry, p0_entry = self.dyn_widgets
                sv = succ_entry.get().strip() or None
                if sv is not None:
                    try:
                        sv = float(sv)
                    except ValueError:
                        pass
                p0 = self._parse_float(p0_entry.get(), "检验比例 p0")
                self.prop = proportion_test(self.df, col_dd.get(), success_value=sv, p0=p0, alpha=alpha)
                self._print_proportion(self.prop)
            elif t == "McNemar 检验":
                c1_dd, c2_dd = self.dyn_widgets
                if not self._distinct(c1_dd.get(), c2_dd.get(), "两个变量"):
                    return
                self.mcn = mcnemar_test(self.df, c1_dd.get(), c2_dd.get(), alpha=alpha)
                self._print_mcnemar(self.mcn)
            elif t == "逻辑回归":
                y_dd = self.dyn_widgets[0]
                y_col = y_dd.get()
                xs = [c for c in self._selected_cols() if c != y_col]
                if not xs:
                    self._write("请至少勾选 1 个自变量。")
                    return
                self.logreg = logistic_regression(self.df, y_col, xs, alpha=alpha)
                self._print_logistic(self.logreg)
            elif t == "多元线性回归":
                y_dd = self.dyn_widgets[0]
                y_col = y_dd.get()
                xs = [c for c in self._selected_cols() if c != y_col]
                if len(xs) < 2:
                    self._write("多元线性回归请至少勾选 2 个自变量。")
                    return
                self.multireg = multiple_linear_regression(self.df, y_col, xs, alpha=alpha)
                self._print_multireg(self.multireg)
            elif t == "信度分析 (Cronbach's α)":
                cols = self._selected_cols()
                if len(cols) < 2:
                    self._write("请至少勾选 2 个题项列。")
                    return
                self.cron = cronbach_alpha(self.df, cols)
                self._print_cronbach(self.cron)
        except Exception as e:
            self._write(f"出错：{e}")
        # 记录分析历史
        if getattr(self, "history", None):
            detail = ""
            if t in ("描述统计", "正态性检验"):
                detail = "变量: " + ", ".join(self._selected_cols())
            elif self.dyn_widgets:
                detail = "、".join(w.get() for w in self.dyn_widgets)
            self.history.log_analysis(t, detail, getattr(self, "_out_text", ""))

    def _print_ttest(self, r, alpha):
        self._write("【独立样本 t 检验】\n")
        self._write(f"因变量：{r['因变量']}；分组变量：{r['分组变量']}")
        self._write(f"{r['水平1']}：n={r['n1']}，均值={r['mean1']:.4f}，标准差={r['std1']:.4f}")
        self._write(f"{r['水平2']}：n={r['n2']}，均值={r['mean2']:.4f}，标准差={r['std2']:.4f}")
        self._write(f"\nLevene 方差齐性检验：F={r['levene_stat']:.4f}，p={format_p(r['levene_p'])}"
                    f"（{'方差齐' if r['equal_var'] else '方差不齐'}）")
        self._write(f"检验方法：{r['检验方法']}；t={r['t']:.4f}，df={r['df']:.2f}，p={format_p(r['p'])}")
        self._write(f"均值差={r['mean_diff']:.4f}，95% CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
        self._write(f"Cohen's d={r['cohens_d']:.4f}（{r['effect_label']}）")
        self._write(f"Mann-Whitney U：U={r['mannwhitney_u']:.1f}，p={format_p(r['mannwhitney_p'])}")
        self._write(f"\n结论：{r['结论']}")

    def _print_reg(self, r, alpha):
        self._write("【简单线性回归】\n")
        self._write(f"回归方程：{r['equation']}，n={r['n']}")
        self._write(f"R²={r['r2']:.4f}，调整 R²={r['adj_r2']:.4f}，F={r['f_stat']:.4f}，p={format_p(r['f_p'])}")
        self._write(f"\n常数项 β₀={r['b0']:.4f}（SE={r['se0']:.4f}，t={r['t0']:.4f}，p={format_p(r['p0'])}）")
        self._write(f"斜率　 β₁={r['b1']:.4f}（SE={r['se1']:.4f}，t={r['t1']:.4f}，p={format_p(r['p1'])}）")
        self._write(f"Pearson r={r['pearson_r']:.4f}，p={format_p(r['pearson_p'])}；RMSE={r['rmse']:.4f}")
        self._write(f"\n结论：{r['结论']}")

    def _print_chi2(self, r):
        self._show_table(r["列联表"].reset_index(), "卡方列联表（观测频数）")
        self._write("【卡方独立性检验】")
        self._write(f"卡方统计量 χ²={r['卡方统计量']:.4f}，自由度={r['自由度']}")
        cv = r["Cramér's V"]
        self._write(f"p 值={format_p(r['p 值'])}；Cramér's V={cv:.4f}")
        self._write(f"\n结论：{r['结论']}")
        self._write("\n（V 越大关联越强：0.1 弱 / 0.3 中等 / 0.5 强）")
        try:
            fig = sp.chi_square_bar_plot(r["列联表"], f"{r['列1']} 与 {r['列2']} 交叉频数")
            self.figures.append((fig, "卡方交叉频数图"))
            self._show_figure(fig)
        except Exception:
            pass

    def _print_anova(self, r):
        self._write("【单因素方差分析 ANOVA】")
        self._write(f"因变量：{r['因变量']}；分组变量：{r['分组变量']}（{r['组数']} 组）")
        gm = r["组均值"]
        self._write("各组均值：" + "；".join(f"{k}={v:.4f}" for k, v in gm.items()))
        self._write(f"F 统计量={r['F 统计量']:.4f}，自由度={r['自由度']}")
        self._write(f"p 值={format_p(r['p 值'])}；效应量 η²={r['eta²']:.4f}")
        self._write(f"\n结论：{r['结论']}")
        try:
            fig = sp.boxplot_by_group(self.df, r["因变量"], r["分组变量"])
            self.figures.append((fig, "各组分位数箱线图"))
            self._show_figure(fig)
        except Exception:
            pass

    def _print_corr(self, r):
        self._write(f"【相关性检验（{r['方法']}）】")
        self._write(f"变量：{r['X']} 与 {r['Y']}；样本量={r['样本量']}")
        self._write(f"相关系数 r={r['相关系数 r']:.4f}，p 值={format_p(r['p 值'])}，相关强度：{r['强度']}")
        self._write(f"\n结论：{r['结论']}")
        try:
            fig = sp.scatter_with_fit(self.df, r["X"], r["Y"])
            self.figures.append((fig, "相关散点图"))
            self._show_figure(fig)
        except Exception:
            pass

    def _print_paired(self, r):
        self._write("【配对 t 检验】")
        self._write(f"前测：{r['前测列']}；后测：{r['后测列']}；样本对={r['样本对']}")
        self._write(f"均值差(后-前)={r['均值差(后-前)']:.4f}")
        self._write(f"t 统计量={r['t 统计量']:.4f}，自由度={r['自由度']}，p 值={format_p(r['p 值'])}")
        cd = r["Cohen's d"]
        self._write(f"Cohen's d={cd:.4f}")
        if not pd.isna(r.get("Wilcoxon p", float("nan"))):
            self._write(f"Wilcoxon 非参数备选：p={format_p(r['Wilcoxon p'])}")
        self._write(f"\n结论：{r['结论']}")
        try:
            fig = sp.line_plot(self.df, [r["前测列"], r["后测列"]], title="前后测对比")
            self.figures.append((fig, "前后测对比折线图"))
            self._show_figure(fig)
        except Exception:
            pass

    def _parse_float(self, s, what):
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"请输入有效的{what}（当前：{s}）。")

    # ---------- v1.4.0 新增：非参 / 事后 / 单样本 / 比例 / McNemar / 回归 / 信度 ----------
    def _print_mannwhitney(self, r):
        self._write("【Mann-Whitney U 检验】")
        self._write(f"因变量：{r['因变量']}；分组变量：{r['分组变量']}")
        self._write(f"{r['水平1']}：n={r['n1']}，中位数={r['中位数1']:.4f}")
        self._write(f"{r['水平2']}：n={r['n2']}，中位数={r['中位数2']:.4f}")
        self._write(f"U 统计量={r['U 统计量']:.1f}，p 值={format_p(r['p 值'])}")
        self._write(f"效应量（秩双列相关）={r['效应量(秩双列相关)']:.4f}")
        self._write(f"\n结论：{r['结论']}")
        self._write("解读：比较两组中位数/分布是否不同。当数据不服从正态分布、样本量较小"
                    "或为有序等级数据时，用它替代独立样本 t 检验。")

    def _print_wilcoxon(self, r):
        self._write("【Wilcoxon 符号秩检验】")
        self._write(f"前测：{r['前测列']}；后测：{r['后测列']}；样本对={r['样本对']}")
        self._write(f"中位数差(后-前)={r['中位数差(后-前)']:.4f}")
        self._write(f"W 统计量={r['W 统计量']:.1f}，p 值={format_p(r['p 值'])}")
        reff = r["效应量 r"]
        self._write(f"效应量 r={reff:.4f}")
        self._write(f"\n结论：{r['结论']}")
        self._write("解读：同一批对象的两次测量（前后测/配对）比较中位数差。配对差值不服从"
                    "正态分布时，用它替代配对 t 检验。")

    def _print_kruskal(self, r):
        self._write("【Kruskal-Wallis H 检验】")
        self._write(f"因变量：{r['因变量']}；分组变量：{r['分组变量']}（{r['组数']} 组）")
        self._write(f"样本量={r['样本量']}，H 统计量={r['H 统计量']:.4f}，自由度={r['自由度']}")
        self._write(f"p 值={format_p(r['p 值'])}；效应量 ε²={r['效应量 ε²']:.4f}")
        self._write("各组中位数：" + "；".join(f"{k}={v:.4f}" for k, v in r["组中位数"].items()))
        self._write(f"\n结论：{r['结论']}")
        self._write("解读：多组独立样本（≥3 组）比较中位数/分布。数据不满足正态性、方差齐性时，"
                    "用它替代单因素 ANOVA；显著后应再做「事后多重比较」。")
        try:
            fig = sp.boxplot_by_group(self.df, r["因变量"], r["分组变量"])
            self.figures.append((fig, "Kruskal-Wallis 分组箱线图"))
            self._show_figure(fig)
        except Exception:
            pass

    def _print_friedman(self, r):
        self._write("【Friedman 检验】")
        self._write(f"测量条件：{'、'.join(r['测量条件'])}；样本量={r['样本量']}")
        self._write(f"Q 统计量={r['Q 统计量']:.4f}，自由度={r['自由度']}，p 值={format_p(r['p 值'])}")
        kendall_w = r["Kendall's W"]
        self._write(f"Kendall's W={kendall_w:.4f}")
        self._write("各条件中位数：" + "；".join(f"{k}={v:.4f}" for k, v in r["各条件中位数"].items()))
        self._write(f"\n结论：{r['结论']}")
        self._write("解读：同一批对象在 ≥3 个条件下重复测量（如 3 个时间点），比较不同条件的分布。"
                    "数据不满足正态/球对称假设时，用它替代重复测量 ANOVA。")
        try:
            fig = sp.line_plot(self.df, r["测量条件"], title="各条件测量值对比")
            self.figures.append((fig, "Friedman 各条件折线图"))
            self._show_figure(fig)
        except Exception:
            pass

    def _print_posthoc(self, r):
        self._write(f"【事后多重比较（{r['方法']}）】")
        self._write(f"因变量：{r['因变量']}；分组变量：{r['分组变量']}；共比较 {r['比较对数']} 对")
        self._show_table(r["比较结果表"].reset_index(), f"事后多重比较（{r['方法']}）")
        self._write(f"\n结论：{r['结论']}")
        self._write("解读：ANOVA 或 Kruskal-Wallis 显著后，用它确定具体哪两组不同。"
                    "Tukey 控制总体错误率（最常用）；Bonferroni 最保守、易漏检；LSD 最敏感但假阳性风险高。")

    def _print_onesample(self, r):
        self._write("【单样本 t 检验】")
        self._write(f"变量：{r['变量']}；样本量={r['样本量']}；检验值 μ0={r['检验值 μ0']}")
        self._write(f"均值={r['均值']:.4f}，标准差={r['标准差']:.4f}")
        self._write(f"t 统计量={r['t 统计量']:.4f}，自由度={r['自由度']}，p 值={format_p(r['p 值'])}")
        ci = r["95%CI"]
        self._write(f"均值 95%CI=[{ci[0]:.4f}, {ci[1]:.4f}]")
        cohens_d = r["Cohen's d"]
        self._write(f"Cohen's d={cohens_d:.4f}（{r['效应量评价']}）")
        self._write(f"\n结论：{r['结论']}")
        self._write("解读：检验单个变量的均值是否等于某个已知值（如标准值/理论值）。"
                    "数据近似正态且样本量≥30 时适用；否则可配合 Wilcoxon 符号秩检验。")

    def _print_proportion(self, r):
        self._write("【二项比例检验】")
        self._write(f"变量：{r['变量']}；成功水平={r['成功水平']}；样本量={r['样本量']}")
        self._write(f"成功数={r['成功数']}，样本比例={r['样本比例']:.4f}，检验比例 p0={r['检验比例 p0']}")
        self._write(f"近似 z={r['z 统计量(近似)']:.4f}（p={format_p(r['近似 p 值'])}）")
        self._write(f"精确 p 值（Binomial）={format_p(r['精确 p 值(Binomial)'])}")
        ci = r["95%CI(Wilson)"]
        self._write(f"95%CI(Wilson)=[{ci[0]:.4f}, {ci[1]:.4f}]")
        self._write(f"\n结论：{r['结论']}")
        self._write("解读：检验二分类数据中某一水平所占比例是否等于理论值 p0（如合格率是否≥90%）。"
                    "小样本看精确 p 值，大样本看近似 z 检验，二者通常结论一致。")
        try:
            fig = sp.proportion_bar_plot(
                [r["样本量"] - r["成功数"], r["成功数"]],
                [f"非{r['成功水平']}", str(r["成功水平"])],
                title=f"{r['变量']} 二分类比例分布")
            self.figures.append((fig, "比例分布柱状图"))
            self._show_figure(fig)
        except Exception:
            pass

    def _print_mcnemar(self, r):
        self._write("【McNemar 配对卡方检验】")
        self._write(f"变量1：{r['变量1']}；变量2：{r['变量2']}；样本量={r['样本量']}")
        self._show_table(r["列联表"].reset_index(), "McNemar 配对列联表")
        self._write(f"McNemar χ²(校正)={r['McNemar χ²(校正)']:.4f}，p 值={format_p(r['p 值'])}")
        self._write(f"精确 p 值（Binomial）={format_p(r['精确 p 值(Binomial)'])}")
        self._write(f"不一致对数={r['不一致对数']}，一致率={r['一致率']:.4f}")
        self._write(f"\n结论：{r['结论']}")
        self._write("解读：同一批对象在两种方法/两次判断下各自给出「是/否」结果，检验两者结果"
                    "是否存在系统性差异。只关注不一致的格子（对角一致格子不参与判断）。")

    def _print_logistic(self, r):
        self._write("【二分类逻辑回归】")
        self._write(f"因变量：{r['因变量']}（成功水平={r['成功水平']}）；自变量：{'、'.join(r['自变量'])}")
        self._write(f"样本量={r['样本量']}，McFadden R²={r['McFadden R²']:.4f}")
        self._write(f"整体似然比 p={format_p(r['整体似然比 p'])}，准确率={r['准确率']:.4f}，AUC={r['AUC']:.4f}")
        self._show_table(r["系数表"], "逻辑回归系数表")
        self._write(f"\n结论：{r['结论']}")
        self._write("解读：系数 B 的指数 exp(B)=OR 表示自变量每增加 1 单位，成功概率的胜算变化倍数"
                    "（OR>1 提高、<1 降低）。AUC 越接近 1 判别力越强（0.7 以上可接受）。"
                    "适用于因变量为二分类（如是否患病）的预测/影响因素研究。")
        try:
            fig = sp.roc_curve_plot(r["实际标签"], r["预测概率"],
                                    title=f"{r['因变量']} 逻辑回归 ROC 曲线")
            self.figures.append((fig, "逻辑回归 ROC 曲线"))
            self._show_figure(fig)
        except Exception:
            pass

    def _print_multireg(self, r):
        self._write("【多元线性回归】")
        self._write(f"因变量：{r['因变量']}；自变量：{'、'.join(r['自变量'])}；样本量={r['样本量']}")
        self._write(f"R²={r['R²']:.4f}，调整 R²={r['调整 R²']:.4f}，F={r['F 统计量']:.4f}，"
                    f"p={format_p(r['整体 p 值'])}")
        self._write(f"Durbin-Watson={r['Durbin-Watson']:.4f}（接近 2 说明残差无自相关）")
        self._write(f"RMSE={r['RMSE']:.4f}，AIC={r['AIC']:.1f}，BIC={r['BIC']:.1f}")
        self._show_table(r["系数表"], "多元回归系数表")
        self._show_table(r["VIF 表"], "共线性诊断（VIF）")
        self._write(f"\n结论：{r['结论']}")
        self._write("解读：多个自变量同时预测一个连续因变量。看单个自变量时用系数表和 p 值；"
                    "VIF≥10 提示该变量与其他自变量高度共线，建议剔除；DW 明显偏离 2 提示残差自相关。")
        try:
            fig = sp.regression_residual_plot(r)
            self.figures.append((fig, "多元回归残差图"))
            self._show_figure(fig)
        except Exception:
            pass

    def _print_cronbach(self, r):
        self._write("【信度分析（Cronbach's α）】")
        self._write(f"样本量={r['样本量']}，题项数={r['题项数']}")
        alpha_val = r["Cronbach's α"]
        self._write(f"Cronbach's α={alpha_val:.4f}，标准化 α={r['标准化 α']:.4f}")
        self._write(f"平均项间相关={r['平均项间相关']:.4f}")
        self._show_table(r["删除题项后 α"].reset_index(), "删除各题项后的 α")
        self._write(f"\n结论：{r['结论']}")
        self._write("解读：问卷/量表的内部一致性检验。若某题项「删除后 α」明显升高，说明该题"
                    "与其他题项不一致，可考虑删去。α≥0.7 一般可接受。")

    # ================= 绘图 =================
    def _plot(self):
        if self.df is None:
            self._write("请先导入数据。")
            return
        pt = self.plot_type_dd.get()
        try:
            w = self.plot_widgets
            if pt == "散点图（含拟合线）":
                if not self._distinct(w[0].get(), w[1].get(), "X 与 Y"):
                    return
                fig = sp.scatter_with_fit(self.df, w[0].get(), w[1].get())
            elif pt == "直方图（含正态曲线）":
                fig = sp.hist_with_normal(self.df[w[0].get()], w[0].get())
            elif pt == "正态 Q-Q 图":
                fig = sp.qq_plot(self.df[w[0].get()], w[0].get())
            elif pt == "分组箱线图":
                if not self._distinct(w[0].get(), w[1].get(), "数值列与分组列"):
                    return
                fig = sp.boxplot_by_group(self.df, w[0].get(), w[1].get())
            elif pt == "小提琴图（分组分布）":
                if not self._distinct(w[0].get(), w[1].get(), "数值列与分组列"):
                    return
                fig = sp.violin_by_group(self.df, w[0].get(), w[1].get())
            elif pt == "误差棒图（均值±SE）":
                if not self._distinct(w[0].get(), w[1].get(), "数值列与分组列"):
                    return
                fig = sp.errorbar_by_group(self.df, w[0].get(), w[1].get(), error="sd")
            elif pt == "相关矩阵热图":
                fig = sp.correlation_heatmap(self.df)
            elif pt == "回归森林图":
                coef = None
                if getattr(self, "multireg", None):
                    coef = self.multireg.get("系数表")
                elif getattr(self, "logreg", None):
                    coef = self.logreg.get("系数表")
                elif getattr(self, "reg", None):
                    # 简单线性回归：把 常数项/斜率 系数与 CI 拼成系数表
                    r = self.reg
                    coef = pd.DataFrame({
                        "变量": ["常数项", "斜率"],
                        "系数": [r["b0"], r["b1"]],
                        "95%下限": [r["b0"] - 1.96 * r["se0"], r["b1"] - 1.96 * r["se1"]],
                        "95%上限": [r["b0"] + 1.96 * r["se0"], r["b1"] + 1.96 * r["se1"]],
                    })
                if coef is None:
                    self._write("请先运行 回归分析 / 逻辑回归 / 多元回归，再绘制森林图。")
                    return
                fig = sp.regression_forest_plot(coef)
            else:
                return
            self.figures.append((fig, pt))
            self._show_figure(fig)
            if getattr(self, "history", None):
                self.history.save_plot(pt, fig)
        except Exception as e:
            self._write(f"绘图出错：{e}")

    def _show_figure(self, fig):
        self._clear_plot()
        self._canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._canvas.draw()

    def _clear_plot(self):
        if self._canvas is not None:
            try:
                self._canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self._canvas = None

    # ================= 导出报告 =================
    def _export(self):
        if self.df is None:
            self._write("请先导入数据。")
            return
        f = filedialog.asksaveasfilename(
            defaultextension=".docx",
            filetypes=[("Word 文档", "*.docx"), ("PDF 文档", "*.pdf")],
            initialfile="统计分析报告.docx",
        )
        if not f:
            return
        try:
            data = build_word_report(
                self.df, self.file_name,
                desc_table=self.desc_table,
                norm_table=self.norm_table,
                ttest=self.ttest,
                reg=self.reg,
                figures=self.figures or None,
                alpha=float(self.alpha_dd.get()),
                chi2=self.chi2, anova=self.anova, corr=self.corr,
                paired=self.paired, mw=self.mw, wsr=self.wsr,
                kw=self.kw, fried=self.fried, post=self.post,
                osamp=self.osamp, prop=self.prop, mcn=self.mcn,
                logreg=self.logreg, multireg=self.multireg, cron=self.cron,
            )
            if f.lower().endswith(".pdf"):
                self._save_pdf(data, f)
            else:
                with open(f, "wb") as fp:
                    fp.write(data)
                self._write(f"报告已导出：{f}")
                if getattr(self, "history", None):
                    self.history.log_op("数据分析", "导出Word报告", f)
                messagebox.showinfo("导出完成", f"Word 报告已保存到：\n{f}")
        except Exception as e:
            self._write(f"导出失败：{e}")
            messagebox.showerror("导出失败", str(e))

    def _save_pdf(self, docx_bytes, pdf_path):
        """把 Word 报告字节转成 PDF：借助本机 MS Word（docx2pdf）。

        无 Word 时优雅降级：另存一份 .docx 并提示用户可用 Word/WPS 打开后另存为 PDF。
        """
        tmp_docx = pdf_path.rsplit(".", 1)[0] + ".docx"
        with open(tmp_docx, "wb") as fp:
            fp.write(docx_bytes)
        try:
            from docx2pdf import convert
            convert(tmp_docx, pdf_path)
            self._write(f"PDF 报告已导出：{pdf_path}")
            if getattr(self, "history", None):
                self.history.log_op("数据分析", "导出PDF报告", pdf_path)
            messagebox.showinfo("导出完成", f"PDF 报告已保存到：\n{pdf_path}")
        except Exception as e:
            self._write(f"本机未安装 MS Word，无法直接转 PDF：{e}\n已同时保存 Word 版：{tmp_docx}")
            messagebox.showinfo(
                "PDF 未生成",
                f"需要本机安装 Microsoft Word 才能一键转 PDF。\n\n"
                f"已为您保存 Word 版报告：\n{tmp_docx}\n\n"
                f"打开后「另存为」选择 PDF 即可（或用 WPS 打开后导出 PDF）。")

    # ================= AI 解读 =================
    def _ai_interpret(self):
        try:
            if self.df is None:
                self._write("请先导入数据并运行分析，再点「AI 解读结果」。")
                return
            from modules.ai_page import AiPage
            ai = self.app.pages.get(AiPage) if hasattr(self, "app") else None
            if ai is None:
                self._write("AI 助手未就绪。")
                return
            ai._quick_interpret()
        except Exception as e:
            self._write(f"调用 AI 失败：{e}")

    # ================= 写结果 =================
    def _write(self, s):
        self.out.insert("end", s + "\n")
        if not hasattr(self, "_out_text"):
            self._out_text = ""
        self._out_text += s + "\n"
