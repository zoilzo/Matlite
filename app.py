# -*- coding: utf-8 -*-
"""
MatLite 数学工作台
==================
一个 MATLAB 风格的桌面计算工作台，面向不懂代码的人：
    * 公式计算器    —— 求值 / 求导 / 积分 / 解方程 / 极限
    * 矩阵与线性代数 —— 行列式 / 逆矩阵 / 特征值 / 解方程组
    * 绘图可视化    —— 2D 函数 / 3D 曲面 / 散点 / 折线 / 柱状 / 直方图
    * 数据分析统计  —— 导入 Excel/CSV，描述统计 / 正态性 / t 检验 / 回归 / 导出 Word 报告

运行方式：双击「启动工作台.bat」即可。
"""

import os
import sys
import threading

from modules.log_utils import logger

# 让程序无论从哪里启动都能找到同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# matplotlib 必须先指定 Tk 后端，再导入 pyplot
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# 全局中文字体（Windows 通用）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import customtkinter as ctk

from modules.calculator_page import CalculatorPage
from modules.matrix_page import MatrixPage
from modules.plot_page import PlotPage
from modules.data_page import DataPage
from modules.ai_page import AiPage
from modules.account import AccountManager
from modules.login_window import LoginWindow
from modules.history import HistoryStore
from modules.history_page import HistoryPage
from modules.numeric_page import NumericPage
from modules.calculus_page import CalculusPage
from modules.prob_page import ProbPage
from modules.settings_page import SettingsPage
from modules.help_page import HelpPage
from modules.complex_page import ComplexPage
from modules.time_series_page import TimeSeriesPage
from modules.ml_page import MlPage
from modules.ocr_page import OcrPage

from modules import app_settings as _S
ctk.set_appearance_mode("dark" if _S.get("theme", "浅色") == "深色" else "light")
ctk.set_default_color_theme("blue")       # 主题色

APP_TITLE = "MatLite 数学工作台"
APP_VERSION = _S.APP_VERSION  # 版本号唯一来源：modules/app_settings.py


class MainApp(ctk.CTk):
    def __init__(self, manager=None):
        super().__init__()
        self.mgr = manager or AccountManager()
        self.current_user = self.mgr.current_user
        self.title(APP_TITLE)
        self.geometry("1320x820")
        self.minsize(1100, 720)

        # ---- 主布局：左导航 + 右内容 ----
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 左侧导航栏
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        self.logo = ctk.CTkLabel(
            self.sidebar, text="🧮 MatLite\n数 学 工 作 台",
            font=ctk.CTkFont(size=21, weight="bold"),
        )
        self.logo.grid(row=0, column=0, padx=16, pady=(20, 12))

        # 当前账号栏
        user_box = ctk.CTkFrame(self.sidebar, corner_radius=8)
        user_box.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="ew")
        user_box.grid_columnconfigure(0, weight=1)
        who = "👤 游客" if self.mgr.is_guest() else f"👤 {self.current_user}"
        ctk.CTkLabel(user_box, text=who, font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(8, 2))
        ctk.CTkLabel(user_box, text="游客模式不记录历史" if self.mgr.is_guest()
                     else "已登录 · 历史将自动保存",
                     font=ctk.CTkFont(size=10), text_color="gray50").grid(
            row=1, column=0, sticky="w", padx=10, pady=(0, 4))
        ctk.CTkButton(user_box, text="切换账号", height=30, width=72,
                      command=self._switch_account).grid(row=0, column=1, rowspan=2, padx=(0, 8))

        # 意见反馈（放顶部，始终可见）
        ctk.CTkButton(self.sidebar, text="📮 意见反馈", height=36, fg_color="gray35",
                      command=self._open_feedback).grid(row=2, column=0, padx=12, pady=(0, 8), sticky="ew")

        nav = [
            ("🧮  公式计算", CalculatorPage),
            ("🔢  矩阵与线性代数", MatrixPage),
            ("🔬  方程与数值计算", NumericPage),
            ("∫  微积分深化", CalculusPage),
            ("🔮  复变函数", ComplexPage),
            ("🎲  概率与分布", ProbPage),
            ("📈  绘图可视化", PlotPage),
            ("📊  数据分析", DataPage),
            ("⏳  时间序列", TimeSeriesPage),
            ("🧠  机器学习", MlPage),
            ("🤖  AI 助手", AiPage),
            ("🎯  拍照识题", OcrPage),
            ("🗂  历史记录", HistoryPage),
            ("⚙️  设置", SettingsPage),
            ("📖  帮助中心", HelpPage),
        ]

        # 右侧内容容器
        self.container = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.container.grid(row=0, column=1, sticky="nsew")
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(0, weight=1)

        self.pages = {}
        self.nav_buttons = []
        # 历史记录存储（游客模式自动不记录）
        self.history = HistoryStore(self.mgr)
        # 导航区（可滚动：页面较多时嵌套滚动，避免与底部署名冲突）
        self.nav_frame = ctk.CTkScrollableFrame(self.sidebar, width=205, fg_color="transparent")
        self.nav_frame.grid(row=3, column=0, sticky="nsew", padx=2, pady=(6, 8))
        self.nav_frame.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(3, weight=1)
        for i, (text, cls) in enumerate(nav, start=0):
            btn = ctk.CTkButton(
                self.nav_frame, text=text, anchor="w", height=40,
                font=ctk.CTkFont(size=13),
                command=lambda c=cls: self.show_page(c),
            )
            btn.grid(row=i, column=0, padx=4, pady=3, sticky="ew")
            self.nav_buttons.append(btn)
            # 页面构建
            page = cls(self.container)
            page.app = self          # 让各页面能互相访问（AI 解读分析结果用）
            page.account = self.mgr  # 账号对象
            page.history = self.history
            self.pages[cls] = page

        foot = ctk.CTkLabel(
            self.sidebar, text="本地计算 · 数据不上传\nMade by zoilzo & Claude",
            font=ctk.CTkFont(size=10), text_color="gray",
        )
        foot.grid(row=4, column=0, padx=12, pady=12)

        self.show_page(CalculatorPage)

        # 启动后静默检查公告与更新（后台线程，失败不影响任何功能）
        try:
            threading.Thread(target=self._check_updates, daemon=True).start()
        except Exception as e:
            logger.warning("启动公告检查线程失败", exc_info=True)

    def _check_updates(self):
        from modules import announce
        try:
            data = announce.check_updates(APP_VERSION)
            if data:
                self.after(0, lambda: self.safe_show_updates(data))
        except Exception as e:
            logger.warning("公告检查异常", exc_info=True)

    def safe_show_updates(self, data):
        try:
            from modules import announce
            announce.show_updates(self, data)
        except Exception as e:
            logger.warning("展示公告对话框异常", exc_info=True)

    def _open_feedback(self):
        from modules.feedback import FeedbackDialog
        FeedbackDialog(self, history=self.history)

    def show_page(self, cls):
        for page in self.pages.values():
            page.grid_remove()
        page = self.pages[cls]
        page.grid(row=0, column=0, sticky="nsew")
        # 页面每次被切换时刷新一下数据（有的模块需要重新读取变量列表）
        if hasattr(page, "on_show"):
            page.on_show()
        # 记录页面切换日志
        if hasattr(self, "history"):
            self.history.log_op(cls.__name__, "切换到该页面")

    def _switch_account(self):
        """退出当前账号，回到登录窗口。"""
        self.destroy()
        win = LoginWindow(AccountManager())
        win.mainloop()
        if win.success:
            MainApp(win.manager).mainloop()


if __name__ == "__main__":
    try:
        mgr = AccountManager()
        win = LoginWindow(mgr)
        win.mainloop()
        if win.success:
            app = MainApp(win.manager)
            app.mainloop()
    except Exception:
        # 崩溃时把详细错误写入日志，并弹窗提示，避免闪退看不到原因
        import traceback
        import tkinter.messagebox as mb
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error_log.txt")
        info = traceback.format_exc()
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(info)
        except Exception:
            pass
        logger.critical("MatLite 崩溃:\n" + info)
        try:
            mb.showerror("MatLite 出错", "程序发生错误，详情已写入 error_log.txt：\n\n" + info[-1500:])
        except Exception:
            pass
        raise