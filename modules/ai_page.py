# -*- coding: utf-8 -*-
"""AI 助手：对接本地 Ollama / OpenAI 兼容服务，帮用户解读分析结果。

兼容两种接口：
  * Ollama 原生接口（/api/tags、/api/chat）——绝大多数本地部署
  * OpenAI 兼容接口（/v1/models、/v1/chat/completions）——LM Studio 等
刷新模型列表会自动回退尝试多个地址，连接失败时可直接手动输入模型名。
"""

import json
import os
import re
import threading
import time

import requests
import customtkinter as ctk

from modules import ai_tools as _at

MAX_TOOL_ROUNDS = 5  # 工具调用轮次上限，防止无限循环

DEFAULT_BASE = "http://localhost:11434"
MODE_NATIVE = "Ollama 原生接口（推荐）"
MODE_OPENAI = "OpenAI 兼容接口"
# 上下文窗口候选值（token）。Ollama 默认只有 2048，而「系统提示 + 52 个工具定义」就要
# 约 7600 token，默认值会把输出预算挤成 0，模型只能返回空 content（界面表现为回复空白）。
NUM_CTX_CHOICES = ["8192", "16384", "32768"]
DEFAULT_NUM_CTX = "16384"
# 空回复兜底文案：这些内容只是给用户看的提示，不该进入发给模型的历史
_PLACEHOLDER_PREFIXES = ("（本地模型未返回文字内容", "（工具调用次数过多", "✅ 已生成结果图")
# 精简 schema 时只保留这些格式关键参数的描述（其余参数名已自解释，删除省 token）
_KEEP_PARAM_DESC = ("data", "matrix", "params", "columns", "x_cols", "distribution")


def _is_filler(text):
    """判断是否为"只展示、不该进历史"的占位文案（空回复/兜底提示）。"""
    s = (text or "").strip()
    if not s:
        return True
    return any(s.startswith(p) for p in _PLACEHOLDER_PREFIXES)
# 本地 AI 服务一律直连，不走系统代理（很多电脑装了 SOCKS/HTTP 代理，
# 会让 requests 去连代理而报 "Missing dependencies for SOCKS support" 或连接超时）
_NO_PROXY = {"http": None, "https": None}
CONFIG_NAME = "config.json"
SYSTEM_PROMPT = (
    "你是一位耐心、亲切的中文数学与统计助教，服务对象是不懂编程的大学生。"
    "请用大白话解释数学和统计结果，多举例子，避免堆砌术语；"
    "涉及结论时给出明确判断和实际建议，条理清晰，篇幅适中。"
    "\n\n【内置能力】你可以调用 MatLite 内置的本地工具来精确计算："
    "求值/求导/积分/解方程/极限、矩阵运算、统计检验、概率分布、绘图、"
    "时序分析、机器学习、图片识别。"
    "遇到需要精确计算的任务时，优先调用对应工具（tool_calls），"
    "拿到工具结果后再用通俗语言向用户解释，不要把工具调用过程暴露给用户。"
)


def _result_summary(page):
    """把数据分析页的当前结果整理成文字摘要，喂给 AI。"""
    lines = []
    df = page.df
    if df is None:
        return "（尚未导入任何数据）"
    lines.append(f"数据文件：{page.file_name}；{df.shape[0]} 行 × {df.shape[1]} 列。")
    if page.desc_table is not None:
        lines.append("\n【描述统计】\n" + page.desc_table.round(4).to_string())
    if page.norm_table is not None:
        lines.append("\n【正态性检验】\n" + page.norm_table.to_string())
    if page.ttest is not None:
        r = page.ttest
        lines.append("\n【t 检验】")
        lines.append(f"  {r['水平1']}：均值={r['mean1']:.4f}，n={r['n1']}；"
                     f"{r['水平2']}：均值={r['mean2']:.4f}，n={r['n2']}")
        lines.append(f"  t={r['t']:.4f}，p={r['p']:.4f}，结论：{r['结论']}")
    if page.reg is not None:
        r = page.reg
        lines.append("\n【线性回归】")
        lines.append(f"  方程：{r['equation']}；R²={r['r2']:.4f}；斜率 p={r['p1']:.4f}")
        lines.append(f"  结论：{r['结论']}")
    return "\n".join(lines)


class AiPage(ctk.CTkFrame):
    def __init__(self, master, app=None):
        super().__init__(master, fg_color="transparent")
        self.app = app
        # 配置存用户目录（%APPDATA%\MatLite\config.json），打包安装后普通用户也能读写
        base_dir = os.environ.get("APPDATA") or os.path.expanduser("~")
        cfg_dir = os.path.join(base_dir, "MatLite")
        os.makedirs(cfg_dir, exist_ok=True)
        self.cfg_path = os.path.join(cfg_dir, CONFIG_NAME)
        self.cfg = self._load_config()
        self.messages = []          # [{"role": "user"/"assistant", "content": str}]
        self._busy = False
        self._stop = False
        self._ctx_pruned = False
        # tools 支持探测缓存：deepseek-r1 等模型不支持 tools，带 tools 会直接 400，
        # 首轮用最小请求探测一次（按模型名缓存），不支持则自动回退纯聊天
        self._tools_probe_model = None
        self._tools_probe_ok = None
        self._tools_fallback_hint = False

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= 左：设置 =================
        left = ctk.CTkScrollableFrame(self, width=350, corner_radius=12, label_text="本地 AI 设置")
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.grid_columnconfigure(0, weight=1)

        # 行计数器变量：所有控件排成一列，增加设置项时无需手动重排行号
        _r = [-1]
        def nxt(pad_top=4):
            _r[0] += 1
            return _r[0]

        ctk.CTkLabel(left, text="服务地址", font=ctk.CTkFont(size=12)).grid(
            row=nxt(), column=0, sticky="w", padx=12, pady=(4, 2))
        self.base = ctk.CTkEntry(left, height=34, placeholder_text="http://localhost:11434")
        self.base.grid(row=nxt(), column=0, sticky="ew", padx=12)
        self.base.insert(0, self.cfg.get("base_url", DEFAULT_BASE))

        ctk.CTkLabel(left, text="接口类型", font=ctk.CTkFont(size=12)).grid(
            row=nxt(), column=0, sticky="w", padx=12, pady=(8, 2))
        self.mode_dd = ctk.CTkOptionMenu(left, values=[MODE_NATIVE, MODE_OPENAI],
                                         command=lambda _v: self._mode_changed())
        self.mode_dd.grid(row=nxt(), column=0, sticky="ew", padx=12)
        self.mode_dd.set(self.cfg.get("mode", MODE_NATIVE))

        ctk.CTkLabel(left, text="模型名称（可直接手动输入）", font=ctk.CTkFont(size=12)).grid(
            row=nxt(), column=0, sticky="w", padx=12, pady=(8, 2))
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.grid(row=nxt(), column=0, sticky="ew", padx=12)
        row.grid_columnconfigure(0, weight=1)
        self.model_dd = ctk.CTkComboBox(row, values=[])
        self.model_dd.grid(row=0, column=0, sticky="ew")
        if self.cfg.get("model"):
            self.model_dd.set(self.cfg["model"])
        else:
            self.model_dd.set("qwen3-vl:8b")
        ctk.CTkButton(row, text="刷新", width=64, command=self._refresh_models).grid(
            row=0, column=1, padx=(6, 0))

        ctk.CTkLabel(left, text="API Key（本地服务一般留空）", font=ctk.CTkFont(size=12)).grid(
            row=nxt(), column=0, sticky="w", padx=12, pady=(8, 2))
        self.api_key = ctk.CTkEntry(left, height=34, show="●")
        self.api_key.grid(row=nxt(), column=0, sticky="ew", padx=12)
        self.api_key.insert(0, self.cfg.get("api_key", ""))

        ctk.CTkLabel(left, text=f"温度 temperature：{self.cfg.get('temperature', 0.7)}",
                     font=ctk.CTkFont(size=12)).grid(row=nxt(), column=0, sticky="w", padx=12, pady=(8, 2))
        self.temp_slider = ctk.CTkSlider(left, from_=0.0, to=1.5, number_of_steps=30)
        self.temp_slider.grid(row=nxt(), column=0, sticky="ew", padx=12)
        self.temp_slider.set(self.cfg.get("temperature", 0.7))
        self.temp_slider.configure(command=lambda v: self.temp_lab.configure(text=f"{v:.2f}"))
        self.temp_lab = ctk.CTkLabel(left, text=f"{self.temp_slider.get():.2f}", font=ctk.CTkFont(size=11))
        self.temp_lab.grid(row=nxt(), column=0, sticky="e", padx=16)

        ctk.CTkLabel(left, text="上下文窗口 num_ctx（token）", font=ctk.CTkFont(size=12)).grid(
            row=nxt(), column=0, sticky="w", padx=12, pady=(8, 2))
        self.ctx_dd = ctk.CTkOptionMenu(left, values=NUM_CTX_CHOICES)
        self.ctx_dd.grid(row=nxt(), column=0, sticky="ew", padx=12)
        self.ctx_dd.set(str(self.cfg.get("ai_num_ctx", DEFAULT_NUM_CTX)))
        ctk.CTkLabel(left, text="太小会导致 AI 回复空白；显存不足时选小一档",
                     font=ctk.CTkFont(size=11), text_color="gray45").grid(
            row=nxt(), column=0, sticky="w", padx=14, pady=(0, 2))

        # ---- 多模型协作（本地算 + 云端讲，可选）----
        ctk.CTkLabel(left, text="🔗 多模型协作（本地算 + 云端讲）",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=nxt(), column=0, sticky="w", padx=12, pady=(12, 4))
        self.collab_switch = ctk.CTkSwitch(left, text="开启：本地跑工具，云端出解释",
                                           command=self._collab_toggle)
        self.collab_switch.grid(row=nxt(), column=0, sticky="w", padx=12, pady=(0, 2))
        if self.cfg.get("collab_on", False):
            self.collab_switch.select()

        ctk.CTkLabel(left, text="云端地址（OpenAI 兼容，如 DeepSeek）", font=ctk.CTkFont(size=12)).grid(
            row=nxt(), column=0, sticky="w", padx=12, pady=(8, 2))
        self.cloud_base = ctk.CTkEntry(left, height=34, placeholder_text="https://api.deepseek.com/v1")
        self.cloud_base.grid(row=nxt(), column=0, sticky="ew", padx=12)
        self.cloud_base.insert(0, self.cfg.get("cloud_base", "https://api.deepseek.com/v1"))

        ctk.CTkLabel(left, text="云端 API Key（必填）", font=ctk.CTkFont(size=12)).grid(
            row=nxt(), column=0, sticky="w", padx=12, pady=(8, 2))
        self.cloud_key = ctk.CTkEntry(left, height=34, show="●")
        self.cloud_key.grid(row=nxt(), column=0, sticky="ew", padx=12)
        self.cloud_key.insert(0, self.cfg.get("cloud_key", ""))

        ctk.CTkLabel(left, text="云端模型名", font=ctk.CTkFont(size=12)).grid(
            row=nxt(), column=0, sticky="w", padx=12, pady=(8, 2))
        self.cloud_model = ctk.CTkEntry(left, height=34)
        self.cloud_model.grid(row=nxt(), column=0, sticky="ew", padx=8)
        self.cloud_model.insert(0, self.cfg.get("cloud_model", "deepseek-chat"))

        self.tools_switch = ctk.CTkSwitch(left, text="自动调用内置功能模块",
                                          command=self._tools_toggle)
        self.tools_switch.grid(row=nxt(), column=0, sticky="w", padx=12, pady=(10, 2))
        if self.cfg.get("ai_tools_on", True):
            self.tools_switch.select()

        ctk.CTkButton(left, text="🛠 测试连接", height=36, command=self._test_conn).grid(
            row=nxt(), column=0, sticky="ew", padx=8, pady=(8, 4))
        ctk.CTkButton(left, text="🔌 测试云端", height=36, fg_color="#5b4b8a",
                      command=self._test_cloud).grid(row=nxt(), column=0, sticky="ew", padx=8, pady=(8, 0))
        ctk.CTkButton(left, text="💾 保存设置", height=36, fg_color="gray40",
                      command=self._save_config).grid(row=nxt(), column=0, sticky="ew", padx=12, pady=4)
        ctk.CTkButton(left, text="💬 保存对话", height=34, fg_color="#2a7f5c",
                      command=self._save_chat_now).grid(row=nxt(), column=0, sticky="ew", padx=12, pady=4)
        ctk.CTkButton(left, text="📤 导出对话", height=34, fg_color="gray40",
                      command=self._export_chat).grid(row=nxt(), column=0, sticky="ew", padx=12, pady=4)

        tip = ("使用说明：\n"
               "① 确保本机已启动 Ollama（ollama serve）；\n"
               "② 点「刷新」选择已下载的模型；\n"
               "③ 若刷新失败，可在模型框直接输入模型名；\n"
               "④ 输入问题，或点右侧「解读当前分析」；\n"
               "⑤ 用 LM Studio 等其他服务时，改选\n"
               "   「OpenAI 兼容接口」并把地址末尾加 /v1。")
        ctk.CTkLabel(left, text=tip, justify="left", font=ctk.CTkFont(size=11),
                     text_color="gray45", anchor="w", wraplength=300).grid(
            row=nxt(), column=0, sticky="w", padx=12, pady=(6, 12))

        # ================= 右：对话 =================
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        head = ctk.CTkFrame(right, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(head, text="🤖 AI 助手", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, sticky="w")
        ctk.CTkButton(head, text="解读当前分析", height=32, command=self._quick_interpret).grid(
            row=0, column=1, padx=4)
        ctk.CTkButton(head, text="❓ 为什么", height=32, fg_color="#5b4b8a", command=self._quick_why).grid(
            row=0, column=2, padx=4)
        ctk.CTkButton(head, text="清空对话", height=33, fg_color="gray40", command=self._clear_chat).grid(
            row=0, column=3)

        self.chat = ctk.CTkTextbox(right, font=ctk.CTkFont(size=14), wrap="word")
        self.chat.grid(row=1, column=0, sticky="nsew", padx=12, pady=4)
        self.chat.configure(state="disabled")

        # 工具结果图片展示位（默认空、不占高度；有图时显示，引用存 self 防 GC）
        self.img_lab = ctk.CTkLabel(right, text="", font=ctk.CTkFont(size=12),
                                    text_color="gray50")
        self.img_lab.grid(row=2, column=0, sticky="ew", padx=12, pady=2)

        bottom = ctk.CTkFrame(right, fg_color="transparent")
        bottom.grid(row=3, column=0, sticky="ew", padx=12, pady=(4, 12))
        bottom.grid_columnconfigure(0, weight=1)
        self.input = ctk.CTkEntry(bottom, height=42, font=ctk.CTkFont(size=14),
                                  placeholder_text="输入你的问题，回车发送……")
        self.input.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.input.bind("<Return>", lambda _e: self._send())
        ctk.CTkButton(bottom, text="发送", width=90, height=42, command=self._send).grid(row=0, column=1)
        ctk.CTkButton(bottom, text="⏹ 停止", width=64, height=42, fg_color="gray40",
                      command=self._stop_generation).grid(row=0, column=2, padx=(6, 0))

        self.status = ctk.CTkLabel(right, text="", font=ctk.CTkFont(size=11), text_color="gray50")
        self.status.grid(row=4, column=0, sticky="w", padx=14, pady=(0, 6))

        # 启动时恢复上次对话（无则显示欢迎语）
        self._load_last_chat()
        if not self.messages:
            self._append_msg("system", "你好！我是本地 AI 助教。问我数学、统计问题，"
                             "或先到「数据分析」跑一次分析，再点「解读当前分析」。")

    # ================= 配置读写 =================
    def _load_config(self):
        from modules import app_settings as _AS
        return _AS.load()

    def _save_config(self):
        # 读-合并-写：只更新 AI 键，绝不覆盖设置页写进去的主题/公告等其他键
        from modules import app_settings as _AS
        self.cfg = _AS.load()
        self.cfg.update({
            "base_url": self.base.get().strip() or DEFAULT_BASE,
            "mode": self.mode_dd.get(),
            "model": self.model_dd.get(),
            "api_key": self.api_key.get().strip(),
            "temperature": float(self.temp_slider.get()),
            "ai_tools_on": bool(self.tools_switch.get()),
            "ai_num_ctx": self._num_ctx(),
            "collab_on": bool(self.collab_switch.get()),
            "cloud_base": self.cloud_base.get().strip(),
            "cloud_key": self.cloud_key.get().strip(),
            "cloud_model": self.cloud_model.get().strip(),
        })
        try:
            _AS.save(self.cfg)
            self.status.configure(text="设置已保存 ✓")
        except Exception as e:
            self.status.configure(text=f"保存失败：{e}")

    def _mode_changed(self):
        if self.mode_dd.get() == MODE_OPENAI:
            b = self.base.get().strip().rstrip("/")
            if not b.endswith("/v1"):
                self.base.delete(0, "end")
                self.base.insert(0, b + "/v1")
        else:
            b = self.base.get().strip().rstrip("/")
            if b.endswith("/v1"):
                self.base.delete(0, "end")
                self.base.insert(0, b[:-3])

    # ================= 连接服务 =================
    def _base(self):
        return self.base.get().strip() or DEFAULT_BASE

    def _headers(self):
        h = {"Content-Type": "application/json"}
        key = self.api_key.get().strip()
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def _chat_url(self):
        b = self._base().rstrip("/")
        return f"{b}/api/chat" if self.mode_dd.get() == MODE_NATIVE else f"{b}/chat/completions"

    def _num_ctx(self):
        """读取上下文窗口设置（非法值回退默认）。"""
        try:
            v = int(str(self.ctx_dd.get()).strip())
        except Exception:
            v = int(DEFAULT_NUM_CTX)
        return max(2048, v)

    def _gen_options(self):
        """统一的生成参数：必须带 num_ctx，否则 Ollama 用默认 2048 会把输出预算挤成 0。

        注意：不能叫 _options —— tkinter.Widget 内部用它做控件选项处理，
        重名会让控件创建直接崩（TypeError: _options() takes 1 positional argument）。
        """
        return {"temperature": float(self.temp_slider.get()), "num_ctx": self._num_ctx()}

    def _post_chat(self, payload, stream=False):
        """发一次对话请求，对连接中断做退避重试。

        切换 num_ctx 时 Ollama 需要重新加载模型，第一次请求常被直接 reset 掉
        （ConnectionResetError 10054），等一秒重试即可恢复，故重试 3 次。
        """
        last = None
        for attempt in range(3):
            try:
                r = requests.post(self._chat_url(), headers=self._headers(), json=payload,
                                  timeout=300, stream=stream, proxies=_NO_PROXY)
                r.raise_for_status()
                return r
            except (requests.ConnectionError, requests.Timeout) as e:
                last = e
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise last

    def _list_models(self):
        """拉取模型名列表，自动回退多个地址，兼容两种接口的返回格式。"""
        b = self._base().rstrip("/")
        if self.mode_dd.get() == MODE_NATIVE:
            urls = [f"{b}/api/tags", f"{b}/api/models", f"{b}/v1/models"]
        else:
            urls = [f"{b}/v1/models", f"{b}/models"]
        last = None
        for u in urls:
            try:
                r = requests.get(u, headers=self._headers(), timeout=4, proxies=_NO_PROXY)
                if r.status_code != 200:
                    last = Exception(f"HTTP {r.status_code}")
                    continue
                d = r.json()
                items = d.get("models") or d.get("data") or []
                names = [m.get("name") or m.get("id") or m.get("model")
                         for m in items if isinstance(m, dict)]
                names = [n for n in names if n]
                if names:
                    return names
            except Exception as e:
                last = e
        raise last or Exception("无法连接模型服务")

    def _refresh_models(self):
        if self._busy:
            return
        self.status.configure(text="正在连接模型服务……")
        threading.Thread(target=self._refresh_worker, daemon=True).start()

    def _refresh_worker(self):
        try:
            names = self._list_models()
            self.after(0, self._refresh_done, names)
        except Exception as e:
            self.after(0, self._refresh_fail, str(e))

    def _refresh_done(self, names):
        self._busy = False
        self.status.configure(text=f"发现 {len(names)} 个模型。")
        self.model_dd.configure(values=names)
        want = self.cfg.get("model")
        if want in names:
            self.model_dd.set(want)
        elif "qwen3-vl:8b" in names:
            self.model_dd.set("qwen3-vl:8b")
        else:
            self.model_dd.set(names[0])
        self._append_msg("assistant", f"✅ 连接成功！可用模型：{', '.join(names)}")

    def _refresh_fail(self, err):
        self._busy = False
        self.status.configure(text="连接失败，可直接手动输入模型名")
        self._append_msg("assistant",
                         f"❌ 刷新模型失败：{err}\n"
                         "请在模型框直接输入模型名称（如 qwen3-vl:8b）后发送，"
                         "或检查地址是否正确、Ollama 是否已启动。")

    def _test_conn(self):
        if self._busy:
            return
        self.status.configure(text="正在测试连接……")
        threading.Thread(target=self._test_worker, daemon=True).start()

    def _test_worker(self):
        try:
            names = self._list_models()
            self.after(0, self._append_msg, "assistant",
                       f"✅ 连接成功！可用模型：{', '.join(names) if names else '（暂无）'}")
            self.after(0, lambda: self.status.configure(text="连接正常 ✓"))
        except Exception as e:
            self.after(0, self._append_msg, "assistant",
                       f"❌ 连接失败：{e}\n请确认 Ollama 已启动，且地址正确。")

    def _test_cloud(self):
        """测试云端 OpenAI 兼容接口是否可用（需有效 API Key）。"""
        base = self.cloud_base.get().strip().rstrip("/")
        if base and not re.search(r"/v\d+$", base):
            base += "/v1"
        key = self.cloud_key.get().strip()
        model = self.cloud_model.get().strip()
        if not (key and model):
            self.status.configure(text="云端测试：请先填云端 Key 与模型名")
            return
        self.status.configure(text="正在测试云端连接……")
        threading.Thread(target=self._test_cloud_worker, args=(base, key, model), daemon=True).start()

    def _test_cloud_worker(self, base, key, model):
        try:
            r = requests.post(f"{base}/chat/completions",
                              headers={"Content-Type": "application/json",
                                       "Authorization": "Bearer " + key},
                              json={"model": model,
                                    "messages": [{"role": "user", "content": "你好，请回复 OK"}],
                                    "stream": False, "max_tokens": 8},
                              timeout=60, proxies=_NO_PROXY)
            r.raise_for_status()
            d = r.json()
            reply = (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            self.after(0, self._append_msg, "assistant",
                       f"✅ 云端模型连接成功！\n模型：{model}　回复：{(reply[:60] or '（模型已响应）')}")
            self.after(0, lambda: self.status.configure(
                text="✅ 云端连接成功：" + (reply[:40] or "（已响应）")))
        except Exception as e:
            self.after(0, self._append_msg, "assistant",
                       f"❌ 云端连接失败：{e}\n"
                       "请检查：地址是否为 OpenAI 兼容 /v1 接口；Key 是否有效；模型名是否正确。")
            self.after(0, lambda: self.status.configure(text="❌ 云端连接失败"))

    # ================= 对话 =================
    def _append_msg(self, role, text):
        self.messages.append({"role": role, "content": text})
        self.chat.configure(state="normal")
        name = {"user": "你", "assistant": "AI", "system": "系统"}.get(role, role)
        self.chat.insert("end", f"【{name}】 {text}\n\n")
        self.chat.see("end")
        self.chat.configure(state="disabled")
        if self.messages:
            self._save_chat()

    def _clear_chat(self):
        self.messages.clear()
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")

    def _collab_toggle(self):
        self._save_config()
        self.status.configure(text="多模型协作：" + ("开" if self.collab_switch.get() else "关"))

    def _quick_why(self):
        """AI 讲解「为什么」：让 AI 判断并解释当前该用/已用的统计方法。"""
        if self.app is None:
            self._append_msg("assistant", "当前没有可讲解的统计结果。")
            return
        from modules.data_page import DataPage
        dp = self.app.pages.get(DataPage)
        if dp is None or dp.df is None:
            self._append_msg("assistant", "请先到「数据分析」导入数据并运行分析，再点「为什么」。")
            return
        summary = _result_summary(dp)
        prompt = ("请先判断当前数据/分析最可能用到的统计方法，然后用大白话解释：\n"
                  "① 为什么推荐/采用了这个统计方法？它解决什么问题？\n"
                  "② 有哪些前提假设？当前数据满足吗？\n"
                  "③ 有没有更合适的方法？\n\n" + summary)
        self._send_text(prompt, as_user="（追问：为什么用这个统计方法？）")

    def _stop_generation(self):
        self._stop = True
        self.status.configure(text="已请求停止……")

    # ---- 对话持久化 / 导出 ----
    def _chats_dir(self):
        d = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "MatLite", "chats")
        os.makedirs(d, exist_ok=True)
        return d

    def _save_chat(self):
        try:
            p = os.path.join(self._chats_dir(), "last.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=1)
        except Exception:
            pass

    def _save_chat_now(self):
        self._save_chat()
        self.status.configure(text="对话已保存 ✓")

    def _load_last_chat(self):
        try:
            p = os.path.join(self._chats_dir(), "last.json")
            if os.path.exists(p):
                msgs = json.load(open(p, encoding="utf-8"))
                for m in msgs:
                    if isinstance(m, dict) and m.get("role") and m.get("content"):
                        self._append_msg(m["role"], m["content"])
        except Exception:
            pass

    def _export_chat(self):
        from tkinter import filedialog
        f = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown 对话", "*.md"), ("文本文件", "*.txt")],
            initialfile="MatLite-AI对话.md")
        if not f:
            return
        lines = []
        for m in self.messages:
            name = {"user": "你", "assistant": "AI", "system": "系统"}.get(m["role"], m["role"])
            lines.append(f"**【{name}】**：\n{m.get('content', '')}\n")
        with open(f, "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines))
        self.status.configure(text=f"对话已导出：{f}")

    # ---- 上下文长度管理 ----
    def _ctx_estimate(self, msgs=None):
        total = 0
        for m in (msgs if msgs is not None else self.messages):
            c = m.get("content") or ""
            total += len(c) // 2 + 16
            if m.get("tool_calls"):
                total += 400
        return total

    def _fit_context(self, msgs):
        """把发给模型的历史按 token 上限裁剪：保留系统提示 + 最近且装得下的消息。"""
        limit = int(self._num_ctx() * 0.7)
        system = msgs[0] if msgs and msgs[0].get("role") == "system" else None
        rest = msgs[1:] if system else list(msgs)
        keep, total = [], 0
        for m in reversed(rest):
            total += len(m.get("content") or "") // 2 + 24
            if total > limit:
                break
            keep.append(m)
        keep.reverse()
        out = ([system] if system else []) + keep
        if not hasattr(self, "_ctx_pruned"):
            self._ctx_pruned = False
        self._ctx_pruned = len(out) < len(msgs)
        return out

    # ---- 云端讲解（多模型协作）----
    def _cloud_prompt(self, user_text, tool_text, local_reply):
        parts = [f"用户问题：{user_text}"]
        if tool_text:
            parts.append(f"\n【已用内置工具算出的结果】\n{tool_text[:3200]}")
        if local_reply and local_reply.strip():
            parts.append(f"\n【本地模型的初步说明】\n{local_reply[:1500]}")
        parts.append("\n请用通俗中文向用户解释这个结果：给出明确结论、关键判断和可操作建议，"
                     "不要提及工具调用过程。")
        return "\n".join(parts)

    def _cloud_explain(self, user_text, tool_text, local_reply):
        """把（问题 + 工具结果）发给云端模型，换取自然语言讲解。"""
        base = self.cloud_base.get().strip().rstrip("/")
        if base and not re.search(r"/v\d+$", base):
            base += "/v1"
        key = self.cloud_key.get().strip()
        cmodel = self.cloud_model.get().strip()
        if not base or not cmodel:
            raise ValueError("云端模型未配置。")
        msgs = [{"role": "system", "content":
                 "你是中文理工科助教，负责把数学/统计计算的结果用大白话讲清楚，"
                 "面向不懂编程的大学生。只讲结论、判断与建议，不讲工具调用过程。"},
                {"role": "user", "content": self._cloud_prompt(user_text, tool_text, local_reply)}]
        payload = {"model": cmodel, "messages": msgs, "stream": False,
                   "temperature": float(self.temp_slider.get())}
        h = {"Content-Type": "application/json"}
        if key:
            h["Authorization"] = "Bearer " + key
        r = requests.post(f"{base}/chat/completions", headers=h, json=payload,
                          timeout=180, proxies=_NO_PROXY)
        r.raise_for_status()
        d = r.json()
        return (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""

    def _quick_interpret(self):
        if self.app is None:
            self._append_msg("assistant", "当前没有可解读的内容。请先在「数据分析」导入数据并运行分析。")
            return
        from modules.data_page import DataPage
        dp = self.app.pages.get(DataPage)
        if dp is None or dp.df is None:
            self._append_msg("assistant", "请先到「数据分析」导入数据并运行分析，再回来点「解读当前分析」。")
            return
        summary = _result_summary(dp)
        prompt = ("请用大白话解读以下分析结果，指出关键结论和需要注意的地方，"
                  "并给出一条可操作的建议：\n\n" + summary)
        self._send_text(prompt, as_user="（请求解读当前分析结果）")

    def _send(self):
        text = self.input.get().strip()
        if not text or self._busy:
            return
        self.input.delete(0, "end")
        self._send_text(text)

    def _send_text(self, text, as_user=None):
        self._append_msg("user", as_user or text)
        self._busy = True
        self.status.configure(text="AI 思考中……（本地大模型可能要几十秒，请稍候）")
        threading.Thread(target=self._worker, args=(text,), daemon=True).start()

    def _worker(self, user_text):
        """调度入口：工具开关开着走 function calling 路径，否则走原流式聊天路径。"""
        model = self.model_dd.get().strip()
        if not model:
            self.after(0, self._append_msg, "assistant",
                       "请先在左侧填写模型名称（点「刷新」选择，或直接输入）。")
            self._busy = False
            return
        self._stop = False
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs += [{"role": m["role"], "content": m["content"]}
                 for m in self.messages
                 if m["role"] in ("user", "assistant") and not _is_filler(m["content"])]
        # 关键：用真正的问题文本（可能带数据摘要）替换最后一条展示用的 user 消息，
        # 否则「AI 解读」时数据摘要发不到模型那里
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] = user_text
        else:
            msgs.append({"role": "user", "content": user_text})
        msgs = self._fit_context(msgs)
        collab = bool(self.collab_switch.get()) and self.cloud_model.get().strip() \
            and self.cloud_key.get().strip()
        try:
            want_tools = self.tools_switch.get() and self.mode_dd.get() == MODE_NATIVE
            if want_tools and self._tools_ok(model):
                reply, images, note = self._chat_with_tools(model, msgs, user_text)
                tool_text = "\n".join(m.get("content", "") for m in msgs
                                       if m.get("role") == "tool")
                # 多模型协作：本地已算完，交给云端做自然语言讲解
                if collab and (tool_text or not reply.strip()):
                    try:
                        explain = self._cloud_explain(user_text, tool_text, reply)
                        if explain and explain.strip():
                            reply = explain
                            note = ""  # 云端成功解释，去掉占位提示
                    except Exception:
                        pass  # 云端失败：保留本地结果
                self.after(0, self._stream_begin)
                self.after(0, self._stream_piece, reply + note)
                self.after(0, self._stream_end, reply)
                if images:
                    self.after(0, self._show_images, images)
            else:
                if want_tools:
                    self._tools_fallback_hint = True
                self._chat_stream(model, msgs)
        except Exception as e:
            self.after(0, self._append_msg, "assistant",
                       f"出错了：{e}\n请确认：① Ollama 已启动；② 模型名正确（可点「刷新」）。"
                       + ("③ 云端模型/Key 是否正确。" if collab else ""))
        if getattr(self, "history", None):
            try:
                self.history.log_op("AI助手", "任务", user_text[:120])
            except Exception:
                pass
        self._busy = False
        if getattr(self, "_ctx_pruned", False):
            self._ctx_pruned = False
            self.after(0, lambda: self.status.configure(text="历史过长，已自动省略较早对话"))
        elif self._tools_fallback_hint:
            self._tools_fallback_hint = False
            self.after(0, lambda: self.status.configure(
                text="当前模型不支持工具调用，已自动切换为纯聊天模式"))
        else:
            self.after(0, lambda: self.status.configure(text=""))

    def _chat_stream(self, model, msgs):
        """纯聊天路径：流式输出（保持原打字机体验）。

        num_ctx 必须显式带上：Ollama 默认只有 2048，系统提示一多就装不下，
        输出预算被挤成 0，模型只会返回空白（thinking 模型尤其明显）。
        """
        payload = {"model": model, "messages": msgs, "stream": True}
        if self.mode_dd.get() == MODE_NATIVE:
            payload["options"] = self._gen_options()
        else:
            payload["temperature"] = float(self.temp_slider.get())
        self.after(0, self._stream_begin)
        # 切换 num_ctx 时 Ollama 要重载模型，首个请求常被 reset 掉（10054），
        # 退避重试 3 次；已流式输出一部分时不再重试，避免对话区重复显示
        reply = ""
        last = None
        for attempt in range(3):
            try:
                r = requests.post(self._chat_url(), headers=self._headers(),
                                  json=payload, timeout=300, stream=True,
                                  proxies=_NO_PROXY)
                r.raise_for_status()
                for raw in r.iter_lines(decode_unicode=True):
                    if self._stop:
                        break
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    if not raw:
                        continue
                    line = raw.strip()
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    piece = ""
                    if isinstance(d, dict) and d.get("choices"):
                        piece = d["choices"][0].get("delta", {}).get("content") or ""
                    elif isinstance(d, dict) and d.get("message"):
                        piece = d["message"].get("content") or ""
                    elif isinstance(d, dict) and d.get("response"):
                        piece = d["response"] or ""
                    if piece:
                        reply += piece
                        self.after(0, self._stream_piece, piece)
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                if reply:
                    break  # 已输出一部分：重试会重复显示，保留已输出内容
                last = e
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        if not reply and last is not None:
            raise last
        # 兜底：模型返回空时给用户一个明确提示（只展示、不入历史）
        if not reply.strip():
            self.after(0, self._stream_piece, "（本地模型未返回文字内容，请换个说法再问一次。）")
        self.after(0, self._stream_end, reply)

    def _chat_with_tools(self, model, msgs, user_text):
        """工具路径：非流式轮询，function calling 优先，规则回退兜底。

        返回 (reply, images, note)：reply 是模型真正的回复（空则不入历史），
        note 是只展示给用户、不入历史的提示文案（空回复/工具次数超限兜底）。
        """
        images = []
        reply = ""
        note = ""
        rounds = 0
        while rounds < MAX_TOOL_ROUNDS:
            if self._stop:
                break
            rounds += 1
            payload = {"model": model, "messages": msgs, "stream": False,
                       "options": self._gen_options()}
            payload["tools"] = self._tool_schemas()
            r = self._post_chat(payload)
            d = r.json()
            msg = d.get("message") or (d.get("choices") or [{}])[0].get("message") or {}
            content = msg.get("content") or ""
            tcs = msg.get("tool_calls") or []
            if tcs:
                # 模型声明了工具调用：原样回传给服务端，再依次执行工具
                msgs.append({"role": "assistant", "content": content or "", "tool_calls": tcs})
                for tc in tcs:
                    fn = tc.get("function") or {}
                    name = fn.get("name") or ""
                    raw = fn.get("arguments") or {}
                    if isinstance(raw, str):
                        try:
                            args = json.loads(raw) or {}
                        except Exception:
                            args = {}
                    else:
                        args = raw if isinstance(raw, dict) else {}
                    out = _at.run_tool(name, args)
                    text = out.get("text", "")
                    if out.get("image") is not None:
                        images.append(out["image"])
                        text += "\n（已生成图片，见下方。）"
                    tmsg = {"role": "tool", "content": text}
                    if self.mode_dd.get() == MODE_NATIVE:
                        tmsg["name"] = name
                    else:
                        tmsg["tool_call_id"] = tc.get("id") or name
                    msgs.append(tmsg)
                continue
            # 模型未调用工具：首轮做规则回退（小模型不支持 function calling 时也能精确计算）
            if rounds == 1:
                tname = self._match_tool(user_text)
                args = {}
                if tname == "solve_eq":
                    ex = self._extract_expr(user_text)
                    if ex:
                        args = {"equation": ex}
                elif tname in ("diff_expr", "integrate_expr", "limit_expr"):
                    ex = self._extract_expr(user_text)
                    if ex:
                        args = {"expr": ex}
                elif tname == "matrix_det":
                    mx = self._extract_matrix(user_text)
                    if mx:
                        args = {"matrix": mx}
                if tname and args:
                    out = _at.run_tool(tname, args)
                    text = out.get("text", "")
                    if out.get("image") is not None:
                        images.append(out["image"])
                        text += "\n（已生成图片，见下方。）"
                    msgs.append({"role": "user",
                                 "content": f"[系统提示] 已自动调用内置模块「{tname}」计算结果：\n{text}\n\n"
                                            "请用通俗语言向用户解释这个结果，不要提及工具调用过程。"})
                    continue
            reply = content or ""
            break
        else:
            note = "（工具调用次数过多，已停止本轮。）"
        # 兜底：本地小模型常在调用工具后返回空 content（认为画完图任务就结束）。
        # 提示只展示、不入历史，避免空回复在对话历史里逐轮累积。
        if not reply.strip():
            if images:
                note = note or ("✅ 已生成结果图（见下方）。本地模型没有额外文字总结，"
                                "你可以继续追问「这张图说明了什么」。")
            else:
                note = note or "（本地模型未返回文字内容，请换个说法再问一次。）"
        return reply, images, note

    def _tools_ok(self, model):
        """当前模型是否支持 tools（结果按模型名缓存，探测失败不缓存、下次重试）。"""
        if model == self._tools_probe_model and self._tools_probe_ok is not None:
            return self._tools_probe_ok
        ok = self._probe_tools(model)
        if ok is not None:
            self._tools_probe_model = model
            self._tools_probe_ok = ok
        return bool(ok)

    def _probe_tools(self, model):
        """最小带 tools 请求探测模型是否支持函数调用。

        deepseek-r1 等模型不支持 tools，带上 tools 会直接返回 400；
        探测成功（200）才缓存为"支持"。返回 None 表示网络失败，下次再探。
        """
        payload = {"model": model,
                   "messages": [{"role": "user", "content": "你好"}],
                   "tools": [self._tool_schemas()[0]],
                   "stream": False,
                   "options": self._gen_options()}
        for attempt in range(3):
            try:
                r = requests.post(self._chat_url(), headers=self._headers(), json=payload,
                                  timeout=120, proxies=_NO_PROXY)
                return r.status_code == 200
            except (requests.ConnectionError, requests.Timeout):
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        return None

    def _tools_toggle(self):
        """工具开关：切换后立即保存。"""
        self._save_config()
        on = self.tools_switch.get()
        self.status.configure(text="内置功能模块调用：" + ("已开启" if on else "已关闭"))

    def _tool_schemas(self):
        """把工具注册表转成 Ollama/OpenAI 兼容的 tools 参数。

        52 个工具全部保留，但定义整体精简（实测 prompt 7637 → 5598 token）：
          * 工具描述只取首句并限长 40 字；
          * 参数描述只保留格式关键的 data/matrix/params/columns/x_cols/distribution，
            data 统一为短文案，其余参数名已自解释、删除以省 token。
        精简后 8192 上下文也能正常工具调用，并给对话历史留出更多空间。
        """
        def _short(doc, limit=40):
            if not doc:
                return ""
            line = re.split(r"[。；\n]", doc.strip())[0].strip()
            return line if len(line) <= limit else line[: limit - 1].rstrip("，。；,; ") + "…"

        out = []
        for t in _at.all_tools():
            props = {}
            for k, v in (t["parameters"].get("properties") or {}).items():
                v = dict(v)
                d = str(v.get("description") or "").strip().replace("\n", " ")
                if k == "data":
                    v["description"] = "数据：CSV 文本 / 字典 / 嵌套列表"
                elif k in _KEEP_PARAM_DESC and d:
                    v["description"] = d if len(d) <= 48 else d[:47].rstrip("，。；,; ") + "…"
                else:
                    v.pop("description", None)
                props[k] = v
            params = {"type": "object", "properties": props}
            if t["parameters"].get("required"):
                params["required"] = t["parameters"]["required"]
            out.append({"type": "function",
                        "function": {"name": t["name"],
                                     "description": _short(t["description"]),
                                     "parameters": params}})
        return out

    def _match_tool(self, text):
        """规则回退：按关键词匹配工具名（无参数时返回 None 或可自解析的工具）。"""
        rules = [
            (["求导", "导数"], "diff_expr"),
            (["积分"], "integrate_expr"),
            (["解方程", "求解", "方程"], "solve_eq"),
            (["极限"], "limit_expr"),
            (["行列式"], "matrix_det"),
            (["逆矩阵"], "matrix_inv"),
            (["特征值"], "matrix_eigen"),
        ]
        for kws, name in rules:
            if any(k in text for k in kws):
                return name
        return None

    def _extract_expr(self, text):
        """从自然语言里粗提取数学表达式（启发式，失败返回空串）。"""
        if "=" in text:
            head = text.split("=")[0]
            head = re.sub(r"[^\w\s+\-*/()^.0-9]", " ", head, flags=re.ASCII)
            head = " ".join(head.split())
            return head or ""
        cands = re.findall(r"[0-9a-zA-Z.^()+*/\-]{4,}", text)
        if cands:
            return max(cands, key=len)
        return ""

    def _extract_matrix(self, text):
        """从文本里提取 [[...]] 嵌套列表矩阵（失败返回 None）。"""
        m = re.search(r"\[\[.*?\]\]", text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _show_images(self, images):
        """把工具产出的图片显示到对话区下方展示位（保留引用防 GC）。"""
        if not images:
            return
        im = images[-1]
        w, h = im.size
        scale = min(1.0, 560.0 / w)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        im2 = im if scale >= 1.0 else im.resize((nw, nh))
        cimg = ctk.CTkImage(light_image=im2, dark_image=im2, size=(nw, nh))
        self._img_ref = cimg
        tag = (f"📊 本次共生成 {len(images)} 张图（显示最新一张）"
               if len(images) > 1 else "📊 结果图")
        self.img_lab.configure(image=cimg, text=tag)

    def _stream_begin(self):
        self.chat.configure(state="normal")
        self.chat.insert("end", "【AI】 ")
        self.chat.configure(state="disabled")

    def _stream_piece(self, piece):
        self.chat.configure(state="normal")
        self.chat.insert("end", piece)
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _stream_end(self, reply):
        self.chat.configure(state="normal")
        self.chat.insert("end", "\n\n")
        self.chat.configure(state="disabled")
        # 空回复/兜底提示只是给用户看，不该进对话历史（否则逐轮累积、越挤越满）
        if not _is_filler(reply):
            self.messages.append({"role": "assistant", "content": reply})
            self._save_chat()