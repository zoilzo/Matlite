# -*- coding: utf-8 -*-
"""矩阵与线性代数：行列式 / 逆矩阵 / 特征值 / 解方程组（基于 sympy，支持分数）"""

import customtkinter as ctk
from sympy import Matrix, MatrixBase, symbols, pretty, sympify, oo, latex
from modules.latex_view import sympy_to_image

DIM_CHOICES = ["1", "2", "3", "4", "5"]


class MatrixPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= 左侧：输入与控制 =================
        left = ctk.CTkScrollableFrame(self, width=440, corner_radius=12, label_text="矩阵输入")
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.grid_columnconfigure(1, weight=1)

        # ---- 矩阵 A ----
        ctk.CTkLabel(left, text="矩阵 A", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=(14, 0), pady=(8, 0))
        self.a_rows = ctk.CTkSegmentedButton(left, values=DIM_CHOICES, command=lambda _v: self._rebuild_a())
        self.a_rows.grid(row=0, column=1, sticky="w", padx=8)
        self.a_rows.set("2")
        self.a_cols = ctk.CTkSegmentedButton(left, values=DIM_CHOICES, command=lambda _v: self._rebuild_a())
        self.a_cols.grid(row=0, column=2, sticky="w", padx=(8, 14))
        self.a_cols.set("2")
        ctk.CTkLabel(left, text="行", font=ctk.CTkFont(size=11), text_color="gray").grid(row=1, column=1, sticky="w", padx=12)
        ctk.CTkLabel(left, text="列", font=ctk.CTkFont(size=11), text_color="gray").grid(row=1, column=2, sticky="w", padx=(8, 0))

        self.a_frame = ctk.CTkFrame(left, fg_color="transparent")
        self.a_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=14, pady=(6, 4))
        self.a_entries = []
        self._rebuild_a()

        ctk.CTkLabel(left, text="提示：单元格可填分数，如 1/2；留空按 0 处理。",
                     font=ctk.CTkFont(size=11), text_color="gray45").grid(
            row=3, column=0, columnspan=3, sticky="w", padx=14, pady=(0, 8))

        # ---- 矩阵 B（可选）----
        self.use_b = ctk.CTkCheckBox(left, text="使用矩阵 B（加减乘用）", command=self._toggle_b)
        self.use_b.grid(row=4, column=0, columnspan=3, sticky="w", padx=14, pady=(10, 2))
        self.b_frame = ctk.CTkFrame(left, fg_color="transparent")
        self.b_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=14, pady=(6, 4))
        self.b_rows = ctk.CTkSegmentedButton(left, values=DIM_CHOICES, command=lambda _v: self._rebuild_b())
        self.b_rows.grid(row=6, column=1, sticky="w", padx=8)
        self.b_rows.set("2")
        self.b_cols = ctk.CTkSegmentedButton(left, values=DIM_CHOICES, command=lambda _v: self._rebuild_b())
        self.b_cols.grid(row=6, column=2, sticky="w", padx=(8, 14))
        self.b_cols.set("2")
        self.b_entries = []
        self._rebuild_b()
        ctk.CTkLabel(left, text="矩阵 B 行", font=ctk.CTkFont(size=11), text_color="gray").grid(row=7, column=1, sticky="w", padx=12)
        ctk.CTkLabel(left, text="列", font=ctk.CTkFont(size=11), text_color="gray").grid(row=7, column=2, sticky="w", padx=(8, 0))

        # ---- 方程组 b 向量（可选）----
        self.use_bvec = ctk.CTkCheckBox(left, text="解方程组 A·x = b（填写 b 向量）", command=self._toggle_bvec)
        self.use_bvec.grid(row=8, column=0, columnspan=3, sticky="w", padx=14, pady=(10, 2))
        self.bv_frame = ctk.CTkFrame(left, fg_color="transparent")
        self.bv_frame.grid(row=9, column=0, columnspan=3, sticky="ew", padx=14, pady=(6, 4))
        self.bv_entries = []
        self._rebuild_bvec()

        # ---- 运算按钮 ----
        ctk.CTkLabel(left, text="选择运算", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=10, column=0, sticky="w", padx=14, pady=(14, 6))
        ops = [
            ("A + B", "add"), ("A − B", "sub"), ("A × B", "mul"),
            ("行列式 det(A)", "det"), ("逆矩阵 A⁻¹", "inv"), ("转置 Aᵀ", "trans"),
            ("秩 rank(A)", "rank"), ("特征值与向量", "eigen"), ("解方程组 Ax=b", "solve"),
        ]
        self.op_btns = {}
        for i, (txt, key) in enumerate(ops):
            btn = ctk.CTkButton(left, text=txt, height=38, command=lambda k=key: self._run_op(k))
            btn.grid(row=11 + i // 3, column=i % 3, sticky="ew", padx=6, pady=4)
            self.op_btns[key] = btn

        # ================= 右侧：结果 =================
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(right, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="计算结果（分数精确显示）", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w")
        self.ltx_btn = ctk.CTkButton(hdr, text="📋 复制 LaTeX", width=120, height=28,
                                     fg_color="gray40", command=self._copy_latex)
        self.ltx_btn.grid(row=0, column=1, sticky="e")

        # 专业公式预览（LaTeX 渲染）
        self.pv_frame = ctk.CTkFrame(right, fg_color="white", corner_radius=8, height=236)
        self.pv_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 6))
        self.pv_frame.grid_propagate(False)
        self.pv_frame.grid_rowconfigure(0, weight=1)
        self.pv_frame.grid_columnconfigure(0, weight=1)
        self.pv_label = ctk.CTkLabel(self.pv_frame, text="公式预览",
                                     font=ctk.CTkFont(size=13), text_color="gray50")
        self.pv_label.grid(row=0, column=0)

        self.out = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=14))
        self.out.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))

        self._pv = None
        self._last_latex = None
        self._pv_ctk = None

        self._write("欢迎使用矩阵计算器。\n\n"
                    "① 在左侧填写矩阵 A（可填分数）；\n"
                    "② 需要时勾选「使用矩阵 B」或「解方程组」；\n"
                    "③ 点下方任意运算按钮即可。")

    # ================= 重建控件 =================
    def _clear(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    def _rebuild_a(self):
        self._clear(self.a_frame)
        rows, cols = int(self.a_rows.get()), int(self.a_cols.get())
        self.a_entries = []
        for r in range(rows):
            row_entries = []
            for c in range(cols):
                e = ctk.CTkEntry(self.a_frame, width=56, height=34,
                                 font=ctk.CTkFont(size=14), justify="center")
                e.grid(row=r, column=c, padx=2, pady=2)
                row_entries.append(e)
            self.a_entries.append(row_entries)

    def _rebuild_b(self):
        self._clear(self.b_frame)
        rows, cols = int(self.b_rows.get()), int(self.b_cols.get())
        self.b_entries = []
        for r in range(rows):
            row_entries = []
            for c in range(cols):
                e = ctk.CTkEntry(self.b_frame, width=56, height=34,
                                 font=ctk.CTkFont(size=14), justify="center")
                e.grid(row=r, column=c, padx=2, pady=2)
                row_entries.append(e)
            self.b_entries.append(row_entries)

    def _rebuild_bvec(self):
        self._clear(self.bv_frame)
        self.bv_entries = []
        n = int(self.a_rows.get())
        for r in range(n):
            e = ctk.CTkEntry(self.bv_frame, width=56, height=34,
                             font=ctk.CTkFont(size=14), justify="center",
                             placeholder_text=f"b{r + 1}")
            e.grid(row=r, column=0, padx=2, pady=2)
            self.bv_entries.append(e)

    def _toggle_b(self):
        pass  # B 矩阵输入框保持可见即可，逻辑上仅运算时判断

    def _toggle_bvec(self):
        self._rebuild_bvec()

    # ================= 读取矩阵 =================
    def _read(self, entries):
        rows = len(entries)
        cols = len(entries[0]) if rows else 0
        data = []
        for r in range(rows):
            row = []
            for c in range(cols):
                t = entries[r][c].get().strip()
                if not t:
                    row.append(0)
                else:
                    try:
                        row.append(sympify(t))
                    except Exception:
                        raise ValueError(f"第 {r + 1} 行第 {c + 1} 列的 '{t}' 不是合法数字/分数")
            data.append(row)
        return Matrix(data)

    def _write(self, s):
        self.out.insert("end", s + "\n")
        if not hasattr(self, "_out_text"):
            self._out_text = ""
        self._out_text += s + "\n"

    def _print_matrix(self, m):
        return pretty(m)

    # ================= 运算 =================
    def _run_op(self, key):
        self.out.delete("1.0", "end")
        self._pv = None
        self._out_text = ""
        try:
            A = self._read(self.a_entries)
            use_b = bool(self.use_b.get())
            use_bv = bool(self.use_bvec.get())

            if key == "add":
                self._check_b(use_b); B = self._read(self.b_entries)
                if A.shape != B.shape:
                    self._write("维度不一致，无法相加。"); return
                self._pv = A + B
                self._write("A + B ="); self._write(self._print_matrix(A + B))
            elif key == "sub":
                self._check_b(use_b); B = self._read(self.b_entries)
                if A.shape != B.shape:
                    self._write("维度不一致，无法相减。"); return
                self._pv = A - B
                self._write("A − B ="); self._write(self._print_matrix(A - B))
            elif key == "mul":
                self._check_b(use_b); B = self._read(self.b_entries)
                if A.shape[1] != B.shape[0]:
                    self._write("A 的列数 ≠ B 的行数，无法相乘。"); return
                self._pv = A * B
                self._write("A × B ="); self._write(self._print_matrix(A * B))
            elif key == "det":
                if A.shape[0] != A.shape[1]:
                    self._write("行列式要求方阵。"); return
                self._pv = A.det()
                self._write("det(A) ="); self._write(pretty(A.det()))
            elif key == "inv":
                if A.shape[0] != A.shape[1]:
                    self._write("逆矩阵要求方阵。"); return
                self._pv = A.inv()
                self._write("A⁻¹ ="); self._write(self._print_matrix(A.inv()))
            elif key == "trans":
                self._pv = A.T
                self._write("Aᵀ ="); self._write(self._print_matrix(A.T))
            elif key == "rank":
                self._pv = A.rank()
                self._write("rank(A) ="); self._write(str(A.rank()))
            elif key == "eigen":
                if A.shape[0] != A.shape[1]:
                    self._write("特征值要求方阵。"); return
                ev = A.eigenvals()
                self._write("特征值（含重数）：")
                for val, mult in ev.items():
                    self._write(f"  λ = {pretty(val)}　（重数 {mult}）")
                self._write("\n特征向量：")
                for val, mult, vecs in A.eigenvects():
                    self._write(f"  λ = {pretty(val)}：")
                    for v in vecs:
                        self._write("    " + self._print_matrix(v).replace("\n", "\n    "))
            elif key == "solve":
                if not use_bv:
                    self._write("请先勾选「解方程组 A·x = b」并填写 b 向量。"); return
                if A.shape[0] != A.shape[1]:
                    self._write("方程组要求 A 为方阵。"); return
                b = Matrix([sympify(e.get().strip()) if e.get().strip() else 0
                            for e in self.bv_entries])
                if A.shape[0] != b.rows:
                    self._write("b 向量长度需等于 A 的行数。"); return
                sol = A.LUsolve(b)
                self._pv = sol
                self._write("方程组的解 x ="); self._write(self._print_matrix(sol))
        except Exception as e:
            self._write(f"出错：{e}")
        self._show_preview()
        # 记录历史
        if getattr(self, "history", None):
            self.history.log_matrix(key, getattr(self, "_out_text", ""),
                                    self._last_latex,
                                    getattr(self, "_last_matrix", None))

    def _check_b(self, use_b):
        if not use_b:
            raise ValueError("请先勾选「使用矩阵 B」并填写矩阵 B。")

    # ---- LaTeX 公式预览 ----
    def _show_preview(self):
        e = self._pv
        self._pv = None
        self._last_matrix = None
        if e is None:
            self.pv_label.configure(image=None, text="公式预览")
            return
        try:
            self._last_latex = latex(e)
            if isinstance(e, MatrixBase):
                self._last_matrix = e.tolist()
            img = sympy_to_image(e)
            scale = min(1.0, 245.0 / img.height)
            w = max(1, int(img.width * scale))
            h = max(1, int(img.height * scale))
            self._pv_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
            self.pv_label.configure(image=self._pv_ctk, text="")
        except Exception:
            self.pv_label.configure(image=None, text="（公式预览失败）")

    def _copy_latex(self):
        if self._last_latex:
            self.clipboard_clear()
            self.clipboard_append(self._last_latex)
            self.ltx_btn.configure(text="✅ 已复制")
            self.after(1500, lambda: self.ltx_btn.configure(text="📋 复制 LaTeX"))
        else:
            self.ltx_btn.configure(text="暂无内容")
            self.after(1500, lambda: self.ltx_btn.configure(text="📋 复制 LaTeX"))