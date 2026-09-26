# -*- coding: utf-8 -*-
"""时间序列分析：数据折线 / 时序分解 / ACF·PACF / ARIMA 建模与预测。

统计引擎用 statsmodels，界面全中文，面向不懂代码的学生。
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import customtkinter as ctk

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OPS = ["原始数据折线", "时序分解（趋势/季节/残差）", "自相关 ACF 与偏自相关 PACF",
       "平稳性检验 ADF/KPSS", "白噪声检验 Ljung-Box",
       "自动定阶 (Auto ARIMA)", "ARIMA 建模与预测", "SARIMA 建模与预测（季节）"]


def nums(text):
    s = text.replace(",", " ").replace("，", " ").replace("；", " ").replace("\n", " ").strip()
    return [float(x) for x in s.split()]


def _sample_series():
    """生成 24 期带趋势和季节性的演示数据。"""
    t = np.arange(24.0)
    y = 20.0 + 0.5 * t + 2.0 * np.sin(t * 2.0 * np.pi / 4.0)
    return " ".join(f"{v:.1f}" for v in y)


class TimeSeriesPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= 左：参数 =================
        left = ctk.CTkScrollableFrame(self, width=390, corner_radius=12, label_text="时间序列")
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="时序数据（空格/逗号/换行分隔，一列数值）", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, sticky="w", padx=14, pady=(8, 2))
        self.data = ctk.CTkTextbox(left, height=110, font=ctk.CTkFont(family="Consolas", size=12))
        self.data.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))
        self.data.insert("1.0", _sample_series())

        ctk.CTkLabel(left, text="操作", font=ctk.CTkFont(size=12)).grid(
            row=2, column=0, sticky="w", padx=14, pady=(4, 2))
        self.op = ctk.CTkOptionMenu(left, values=OPS)
        self.op.grid(row=3, column=0, sticky="ew", padx=12)
        self.op.set(OPS[1])

        ctk.CTkLabel(left, text="周期（季节长度，分解用，如 4=季度 12=月度）", font=ctk.CTkFont(size=12)).grid(
            row=4, column=0, sticky="w", padx=14, pady=(8, 2))
        self.period = ctk.CTkEntry(left, height=32)
        self.period.grid(row=5, column=0, sticky="ew", padx=12)
        self.period.insert(0, "4")

        ctk.CTkLabel(left, text="ARIMA 阶数 p d q（空格分隔，如 1 1 1）", font=ctk.CTkFont(size=12)).grid(
            row=6, column=0, sticky="w", padx=14, pady=(8, 2))
        self.order = ctk.CTkEntry(left, height=32)
        self.order.grid(row=7, column=0, sticky="ew", padx=12)
        self.order.insert(0, "1 1 1")

        ctk.CTkLabel(left, text="季节阶数 P D Q（SARIMA 用，空=无季节）", font=ctk.CTkFont(size=12)).grid(
            row=8, column=0, sticky="w", padx=14, pady=(8, 2))
        self.seas = ctk.CTkEntry(left, height=32)
        self.seas.grid(row=9, column=0, sticky="ew", padx=12)
        self.seas.insert(0, "0 1 1")

        ctk.CTkLabel(left, text="预测期数", font=ctk.CTkFont(size=12)).grid(
            row=10, column=0, sticky="w", padx=14, pady=(8, 2))
        self.horizon = ctk.CTkEntry(left, height=32)
        self.horizon.grid(row=11, column=0, sticky="ew", padx=12)
        self.horizon.insert(0, "6")

        ctk.CTkButton(left, text="⚡ 分 析", height=42, command=self._run).grid(
            row=12, column=0, sticky="ew", padx=12, pady=(10, 2))

        tip = ("说明：\n"
               "· 分解：把序列拆成 趋势 + 季节 + 残差 三张子图；\n"
               "· ACF/PACF：看图判断适合的 ARIMA 阶数（截尾/拖尾）；\n"
               "· 平稳性：ADF/KPSS 判断序列是否平稳；\n"
               "· Ljung-Box：判断残差是否为白噪声；\n"
               "· ARIMA/SARIMA：输入或自动选择阶数拟合并预测；\n"
               "· 数据太少（< 2 个周期）分解会失败，请加长序列。")
        ctk.CTkLabel(left, text=tip, justify="left", font=ctk.CTkFont(size=11),
                     text_color="gray45", anchor="w", wraplength=360).grid(
            row=11, column=0, sticky="w", padx=14, pady=(6, 12))

        # ================= 右：结果 + 画布 =================
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(right, text="分析结果", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(12, 4))
        self.out = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=13))
        self.out.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 4))

        box = ctk.CTkFrame(right, fg_color="transparent")
        box.grid(row=2, column=0, sticky="nsew", padx=16, pady=(4, 4))
        box.grid_rowconfigure(0, weight=1)
        box.grid_columnconfigure(0, weight=1)
        self.figure = plt.Figure(figsize=(7, 4.6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=box)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
        toolbar = NavigationToolbar2Tk(self.canvas, box)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.ax.set_title("时序图形")
        self.canvas.draw()

    def _msg(self, s):
        self.out.configure(state="normal")
        self.out.insert("end", s + "\n")
        self.out.see("end")
        self.out.configure(state="disabled")

    def _clear_out(self):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.configure(state="disabled")

    def _get_y(self):
        v = nums(self.data.get("1.0", "end"))
        if len(v) < 8:
            raise ValueError("数据点太少（至少 8 个）。")
        return np.array(v, dtype=float)

    def _run(self):
        self._clear_out()
        op = self.op.get()
        try:
            y = self._get_y()
            if op == OPS[0]:
                self._plot_raw(y)
            elif op == OPS[1]:
                self._decompose(y)
            elif op == OPS[2]:
                self._acf_pacf(y)
            elif op == OPS[3]:
                self._stationarity(y)
            elif op == OPS[4]:
                self._ljungbox(y)
            elif op == OPS[5]:
                self._auto_arima(y)
            elif op == OPS[6]:
                self._arima(y)
            elif op == OPS[7]:
                self._sarima(y)
        except Exception as e:
            self._msg(f"出错：{e}")
        self.canvas.draw()
        if getattr(self, "history", None):
            self.history.log_op("时间序列", op, self.out.get("1.0", "end")[:200])

    def _plot_raw(self, y):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(y, lw=2, color="steelblue", marker="o", ms=3)
        self.ax.set_title("原始数据")
        self.ax.set_xlabel("期数")
        self.ax.set_ylabel("数值")
        self.ax.grid(True)
        self._msg(f"共 {len(y)} 个观测点，最小值 {y.min():.4g}，最大值 {y.max():.4g}。")

    def _decompose(self, y):
        from statsmodels.tsa.seasonal import seasonal_decompose
        per = int(float(self.period.get() or 4))
        if len(y) < 2 * per:
            raise ValueError(f"数据太短：需要至少 2 个周期（{2 * per} 个点），当前 {len(y)} 个。")
        res = seasonal_decompose(y, model="additive", period=per)
        self.figure.clear()
        self.ax = self.figure.add_subplot(411)
        self.ax.plot(res.observed, lw=1.6, color="k"); self.ax.set_title("观测值"); self.ax.grid(True)
        ax2 = self.figure.add_subplot(412, sharex=self.ax)
        ax2.plot(res.trend, lw=1.6, color="steelblue"); ax2.set_title("趋势"); ax2.grid(True)
        ax3 = self.figure.add_subplot(413, sharex=self.ax)
        ax3.plot(res.seasonal, lw=1.6, color="orange"); ax3.set_title("季节"); ax3.grid(True)
        ax4 = self.figure.add_subplot(414, sharex=self.ax)
        ax4.plot(res.resid, lw=1.6, color="crimson"); ax4.set_title("残差"); ax4.grid(True)
        self.figure.subplots_adjust(hspace=0.45)
        self._msg(f"分解完成（周期={per}）。趋势/季节/残差已绘制。")

    def _acf_pacf(self, y):
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        self.figure.clear()
        self.ax = self.figure.add_subplot(211)
        plot_acf(y, lags=min(20, len(y) // 2), ax=self.ax)
        self.ax.set_title("自相关 ACF")
        self.ax.grid(True)
        ax2 = self.figure.add_subplot(212)
        plot_pacf(y, lags=min(20, len(y) // 2), ax=ax2)
        ax2.set_title("偏自相关 PACF")
        ax2.grid(True)
        self.figure.subplots_adjust(hspace=0.45)
        self._msg("ACF 拖尾 + PACF 截尾→ 适合 AR；ACF 截尾 + PACF 拖尾→ 适合 MA。")

    def _parse_order(self, text):
        text = text.replace("，", ",").replace(" ", ",")
        parts = [int(x) for x in text.split(",") if x.strip()][:3]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)

    def _accuracy(self, resid, y):
        """残差精度指标：RMSE 与 MAPE。"""
        resid = np.asarray(resid, dtype=float)
        y = np.asarray(y, dtype=float)
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mask = y != 0
        mape = float(np.mean(np.abs(resid[mask] / y[mask])) * 100) if mask.any() else float("nan")
        return rmse, mape

    def _arima(self, y):
        from statsmodels.tsa.arima.model import ARIMA
        o = self._parse_order(self.order.get())
        steps = max(1, int(float(self.horizon.get() or 6)))
        model = ARIMA(y, order=o)
        fit = model.fit()
        fc = fit.get_forecast(steps)
        mean = fc.predicted_mean
        ci = fc.conf_int()
        rmse, mape = self._accuracy(fit.resid, y)
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        n = len(y)
        self.ax.plot(range(n), y, lw=2, color="steelblue", label="观测值")
        self.ax.plot(range(n, n + steps), mean, lw=2, color="crimson", marker="o", label="预测")
        self.ax.fill_between(range(n, n + steps), ci[:, 0], ci[:, 1], color="crimson", alpha=0.2, label="95% 置信区间")
        self.ax.axvline(n - 0.5, color="gray", ls="--", lw=1)
        self.ax.legend(fontsize=9)
        self.ax.set_title(f"ARIMA{o} 拟合与预测（下 {steps} 期）")
        self.ax.set_xlabel("期数")
        self.ax.set_ylabel("数值")
        self.ax.grid(True)
        self._msg(f"ARIMA 阶数：p={o[0]}，d={o[1]}，q={o[2]}")
        self._msg(f"AIC={fit.aic:.3g}，BIC={fit.bic:.3g}（越小越好）")
        self._msg(f"拟合误差：RMSE={rmse:.4g}，MAPE={mape:.3g}%")
        self._msg(f"预测 {steps} 期：{[f'{v:.3g}' for v in mean]}")

    def _stationarity(self, y):
        """ADF + KPSS 平稳性检验。"""
        from statsmodels.tsa.stattools import adfuller, kpss
        self._msg("平稳性检验：")
        adf, ap, *_ = adfuller(y, autolag="AIC")
        self._msg(f"ADF 检验：统计量={adf:.4g}，p={ap:.4g} → {'可认为平稳' if ap < 0.05 else '非平稳'}")
        try:
            kpv = kpss(y, regression="c", nlags="auto")
            self._msg(f"KPSS 检验：统计量={kpv[0]:.4g}，p={kpv[1]:.4g} → {'可认为平稳' if kpv[1] > 0.05 else '非平稳'}")
        except Exception as e:
            self._msg(f"KPSS 检验失败：{e}")
        self._msg("提示：ADF 的 H0 是非平稳，KPSS 的 H0 是平稳；两者一致时结论更可靠。")
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(y, lw=2, color="steelblue")
        self.ax.set_title("待检验序列")
        self.ax.grid(True)

    def _ljungbox(self, y):
        """Ljung-Box 白噪声检验：p>0.05 说明无自相关（近乎白噪声）。"""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        self._msg("Ljung-Box 白噪声检验：")
        lag = min(10, len(y) // 2)
        lb = acorr_ljungbox(y, lags=[lag], return_df=True)
        stat = float(lb["lb_stat"].iloc[0])
        pv = float(lb["lb_pvalue"].iloc[0])
        self._msg(f"滞后 {lag}：Q 统计量={stat:.4g}，p={pv:.4g} → {'白噪声（无自相关）' if pv > 0.05 else '存在自相关'}")
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.acorr(y, maxlags=min(20, len(y) // 2), lw=1.6)
        self.ax.set_title("序列自相关")
        self.ax.grid(True)

    def _auto_arima(self, y):
        """自动定阶：网格搜索 (p,d,q)，按 AIC 最小选最优，再预测。"""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        best = None
        for p in range(3):
            for q in range(3):
                for d in range(2):
                    try:
                        fit = SARIMAX(y, order=(p, d, q), enforce_stationarity=False,
                                      enforce_invertibility=False).fit(disp=False)
                        if best is None or fit.aic < best[0]:
                            best = (fit.aic, (p, d, q), fit)
                    except Exception:
                        continue
        if best is None:
            raise ValueError("自动定阶失败（样本过短或序列退化）。")
        aic, o, fit = best
        steps = max(1, int(float(self.horizon.get() or 6)))
        fc = fit.get_forecast(steps)
        mean = fc.predicted_mean
        ci = fc.conf_int()
        rmse, mape = self._accuracy(fit.resid, y)
        self._msg(f"自动定阶：ARIMA{o}（AIC={aic:.3g}）")
        self._msg(f"拟合误差：RMSE={rmse:.4g}，MAPE={mape:.3g}%")
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        n = len(y)
        self.ax.plot(range(n), y, lw=2, color="steelblue", label="观测值")
        self.ax.plot(range(n, n + steps), mean, lw=2, color="crimson", marker="o", label="预测")
        self.ax.fill_between(range(n, n + steps), ci[:, 0], ci[:, 1], color="crimson", alpha=0.2, label="95% 置信区间")
        self.ax.axvline(n - 0.5, color="gray", ls="--", lw=1)
        self.ax.legend(fontsize=9)
        self.ax.set_title(f"Auto ARIMA{o} 预测（下 {steps} 期）")
        self.ax.grid(True)
        self._msg(f"预测 {steps} 期：{[f'{v:.3g}' for v in mean]}")

    def _sarima(self, y):
        """SARIMA 建模：order=(p,d,q)，seasonal_order=(P,D,Q,s)。"""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        o = self._parse_order(self.order.get())
        s = max(2, int(float(self.period.get() or 4)))
        so = self._parse_order(self.seas.get()) if self.seas.get().strip() else (0, 0, 0)
        sorder = (so[0], so[1], so[2], s)
        steps = max(1, int(float(self.horizon.get() or 6)))
        model = SARIMAX(y, order=o, seasonal_order=sorder,
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        fc = fit.get_forecast(steps)
        mean = fc.predicted_mean
        ci = fc.conf_int()
        rmse, mape = self._accuracy(fit.resid, y)
        self._msg(f"SARIMA{o}×{sorder}：AIC={fit.aic:.3g}")
        self._msg(f"拟合误差：RMSE={rmse:.4g}，MAPE={mape:.3g}%")
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        n = len(y)
        self.ax.plot(range(n), y, lw=2, color="steelblue", label="观测值")
        self.ax.plot(range(n, n + steps), mean, lw=2, color="crimson", marker="o", label="预测")
        self.ax.fill_between(range(n, n + steps), ci[:, 0], ci[:, 1], color="crimson", alpha=0.2, label="95% 置信区间")
        self.ax.axvline(n - 0.5, color="gray", ls="--", lw=1)
        self.ax.legend(fontsize=9)
        self.ax.set_title(f"SARIMA{o}×{sorder} 预测（下 {steps} 期）")
        self.ax.grid(True)
        self._msg(f"预测 {steps} 期：{[f'{v:.3g}' for v in mean]}")

    def on_show(self):
        pass