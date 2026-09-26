# -*- coding: utf-8 -*-
"""矩阵与线性代数：行列式 / 逆矩阵 / 特征值 / 解方程组（基于 sympy，支持分数）"""

import customtkinter as ctk
from sympy import Matrix, MatrixBase, Symbol, symbols, Eq, pretty, sympify, oo, latex
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
            ("LU 分解", "lu"), ("QR 分解", "qr"), ("SVD 分解", "svd"),
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
        self.repro_btn = ctk.CTkButton(hdr, text="📦 导出复现", width=120, height=28,
                                       fg_color="steelblue", command=self._export_repro)
        self.repro_btn.grid(row=0, column=2, sticky="e", padx=(8, 0))
        hdr.grid_columnconfigure(1, weight=0)
        hdr.grid_columnconfigure(2, weight=0)

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
        self._last_op = None      # 记住最近一次运算的类型（导出复现用）
        self._last_A = None       # 记住最近一次运算的矩阵 A

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
        return self._fmt(m)

    def _fmt(self, m):
        """把 sympy 结果格式化为在等宽文本框里清晰对齐的文本。

        - 矩阵：按列对齐元素、外面套 [ ]（列/行向量也按此显示，保证竖排可读）；
        - 其它（标量、表达式）：交给 sympy pretty 输出。
        """
        if not isinstance(m, MatrixBase):
            return pretty(m)
        rows, cols = m.rows, m.cols
        cell = [[str(m[i, j]) for j in range(cols)] for i in range(rows)]
        widths = [max(len(cell[i][j]) for i in range(rows)) for j in range(cols)]
        lines = []
        for i in range(rows):
            parts = [cell[i][j].rjust(widths[j]) for j in range(cols)]
            lines.append("[ " + "  ".join(parts) + " ]")
        return "\n".join(lines)

    # ================= 运算 =================
    def _run_op(self, key):
        self.out.delete("1.0", "end")
        self._pv = None
        self._out_text = ""
        try:
            A = self._read(self.a_entries)
            self._last_op = key
            self._last_A = A
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
                self._write("特征值（含重数）：")
                for i, (val, mult, vecs) in enumerate(A.eigenvects(), 1):
                    self._write(f"  λ{i} = {val}　（重数 {mult}）")
                    if i == 1:
                        self._pv = Eq(Symbol("lambda"), val)
                self._write("\n特征向量：")
                for val, mult, vecs in A.eigenvects():
                    self._write(f"  λ = {val}：")
                    for v in vecs:
                        self._write("    " + self._fmt(v))
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
            elif key == "lu":
                if A.shape[0] != A.shape[1]:
                    self._write("LU 分解要求方阵。"); return
                L, U, perm = A.LUdecomposition()
                self._write("LU 分解：A = P·L·U（P 为置换矩阵；无行交换时 P = I）")
                if perm:
                    self._write(f"发生行交换：第 {perm} 行被换位（P ≠ I）")
                self._write("L（下三角，主对角线为 1） =")
                self._write(self._print_matrix(L))
                self._write("U（上三角） =")
                self._write(self._print_matrix(U))
                if not perm:
                    self._pv = L * U
                    self._write("验证 L·U =")
                    self._write(self._print_matrix(L * U))
                else:
                    self._pv = L
            elif key == "qr":
                if A.shape[0] != A.shape[1]:
                    self._write("QR 分解要求方阵。"); return
                Q, R = A.QRdecomposition()
                self._write("QR 分解：A = Q·R")
                self._write("Q（正交矩阵） =")
                self._write(self._print_matrix(Q))
                self._write("R（上三角） =")
                self._write(self._print_matrix(R))
                self._pv = R
                self._write("验证 Q·R =")
                self._write(self._print_matrix(Q * R))
            elif key == "svd":
                if A.shape[0] != A.shape[1]:
                    self._write("SVD 分解要求方阵。"); return
                U, S, V = A.singular_value_decomposition()
                self._write("SVD 分解：A = U·Σ·Vᵀ")
                self._write("U（正交矩阵） =")
                self._write(self._print_matrix(U))
                self._write("Σ（奇异值对角阵） =")
                self._write(self._print_matrix(S))
                self._write("V（正交矩阵） =")
                self._write(self._print_matrix(V.T))
                self._write("奇异值 = " + str(A.singular_values()))
                self._pv = S
                self._write("验证 U·Σ·Vᵀ =")
                self._write(self._print_matrix(U * S * V.T))
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
            self.pv_frame.configure(height=236)
            return
        try:
            self._last_latex = latex(e)
            if isinstance(e, MatrixBase):
                self._last_matrix = e.tolist()
            img = sympy_to_image(e)
            # 自适应缩放：填满预览框但不超过原始清晰度（1x），避免大矩阵/大式被裁或缩过头
            aw = int(self.pv_frame.winfo_width())
            aw = aw if aw > 100 else 560
            ah = int(self.pv_frame.winfo_height())
            ah = ah if ah > 100 else 248
            max_w = max(120, aw - 48)
            max_h = max(60, ah - 28)
            scale = min(1.0, max_w / img.width, max_h / img.height) if (img.width and img.height) else 1.0
            w = max(1, int(img.width * scale))
            h = max(1, int(img.height * scale))
            self._pv_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
            self.pv_label.configure(image=self._pv_ctk, text="")
            self.pv_frame.configure(height=min(max(h + 28, 140), 340))
        except Exception:
            self.pv_label.configure(image=None, text="（公式预览失败）")
            self.pv_frame.configure(height=236)

    def _copy_latex(self):
        if self._last_latex:
            self.clipboard_clear()
            self.clipboard_append(self._last_latex)
            self.ltx_btn.configure(text="✅ 已复制")
            self.after(1500, lambda: self.ltx_btn.configure(text="📋 复制 LaTeX"))
        else:
            self.ltx_btn.configure(text="暂无内容")
            self.after(1500, lambda: self.ltx_btn.configure(text="📋 复制 LaTeX"))

    # ---- 可复现工作流：导出 .py 脚本 + .md 配方 ----
    def _export_repro(self):
        from modules import repro
        try:
            A = self._read(self.a_entries)
        except Exception as e:
            self._write(f"无法读取矩阵：{e}")
            return
        op = self._last_op
        if op is None:
            self._write("请先点任意运算按钮计算结果，再导出复现脚本。")
            return
        label = {"add": "矩阵加法 A+B", "sub": "矩阵减法 A−B", "mul": "矩阵乘法 A×B",
                 "det": "行列式 det(A)", "inv": "逆矩阵 A⁻¹", "trans": "转置 Aᵀ",
                 "rank": "秩 rank(A)", "eigen": "特征值与向量", "solve": "解方程组 Ax=b",
                 "lu": "LU 分解", "qr": "QR 分解", "svd": "SVD 分解"}.get(op, "矩阵运算")

        def cell(x):
            if getattr(x, "is_Rational", False):
                return f"sp.Rational({x.p}, {x.q})"
            return repr(x)
        rows = ["[" + ", ".join(cell(A[i, j]) for j in range(A.cols)) + "]" for i in range(A.rows)]
        litA = "Matrix([" + ", ".join(rows) + "])"

        code = [repro.script_header(f"矩阵{label}", "矩阵与线性代数"),
                "import numpy as np", "from sympy import Matrix", "import sympy as sp", "",
                "A = " + litA]
        if op in ("add", "sub", "mul"):
            try:
                B = self._read(self.b_entries)
                rowsB = ["[" + ", ".join(cell(B[i, j]) for j in range(B.cols)) + "]" for i in range(B.rows)]
                code.append("B = Matrix([" + ", ".join(rowsB) + "])")
            except Exception:
                B = None
            expr = {"add": "A + B", "sub": "A - B", "mul": "A * B"}.get(op)
            if B is not None:
                code.append(f"print('结果 ='); print({expr})")
        elif op == "det":
            code.append("print('det(A) =', A.det())")
        elif op == "inv":
            code.append("print('A^(-1) ='); print(A.inv())")
        elif op == "trans":
            code.append("print('A^T ='); print(A.T)")
        elif op == "rank":
            code.append("print('rank(A) =', A.rank())")
        elif op == "eigen":
            code.append("print('特征值（含重数）：')")
            code.append("for val, mult, _ in A.eigenvects(): print(f'  lambda={val} (重数 {mult})')")
        elif op == "solve":
            bvals = [e.get().strip() or "0" for e in self.bv_entries]
            code.append("b = Matrix(" + repr(bvals) + ")")
            code.append("print('x ='); print(A.LUsolve(b))")
        elif op == "lu":
            code.append("L, U, perm = A.LUdecomposition()")
            code.append("print('L ='); print(L); print('U ='); print(U)")
        elif op == "qr":
            code.append("Q, R = A.QRdecomposition()")
            code.append("print('Q ='); print(Q); print('R ='); print(R)")
        elif op == "svd":
            code.append("U, S, V = A.singular_value_decomposition()")
            code.append("print('U ='); print(U); print('Sigma ='); print(S); print('V^T ='); print(V)")
        py_code = "\n".join(code) + "\n"

        result_lines = [ln for ln in getattr(self, "_out_text", "").splitlines() if ln.strip()][-10:]
        problem = [f"矩阵 A = {A.tolist()}"]
        try:
            B = self._read(self.b_entries) if op in ("add", "sub", "mul") else None
            if B is not None:
                problem.append(f"矩阵 B = {B.tolist()}")
        except Exception:
            pass
        repro.export_recipe(self, f"矩阵{label}", "矩阵与线性代数", problem, result_lines, py_code,
                            default_name=f"矩阵{label}", history=getattr(self, "history", None))