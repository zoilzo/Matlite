# 🧮 MatLite 数学工作台

一个 MATLAB 风格的本地桌面数学与统计工具，面向不懂代码的大学生与初学者。**本地计算，数据不上传**。

## ✨ 功能一览

| 模块 | 说明 |
|------|------|
| 公式计算 | 求值 / 求导 / 积分 / 解方程 / 极限 |
| 矩阵与线性代数 | 行列式 / 逆矩阵 / 特征值 / 解方程组 / 矩阵乘法 |
| 方程与数值计算 | 方程求根 / 数值积分 / 线性规划 / 曲线拟合 |
| 微积分深化 | 泰勒展开 / 微分方程求解 / 参数方程 / 极坐标绘图 |
| 复变函数 | 复数运算 / 模曲面 / 复平面着色映射 / 留数 |
| 概率与分布 | 常见分布概率 / 分位数 / 随机数 / 分布拟合 |
| 绘图可视化 | 函数曲线 / 曲面 / 散点 / 折线 / 柱状 / 直方图 |
| 数据分析 | 导入 Excel/CSV，描述统计 / 正态性 / t 检验 / 回归 / Word 报告 |
| 时间序列 | 分解 / ACF/PACF / ARIMA 预测 |
| 机器学习 | K-Means 聚类 / 决策树 |
| AI 助手 | 对接本地 Ollama / 云端大模型，自动调用内置工具精确计算 |
| 拍照识题 | 拍照或粘贴图片，用多模态大模型识别数学题目并解答 |
| 历史记录 | 每账号独立保存计算结果、绘图、操作日志 |
| 设置 | 主题 / 精度 / DPI / 字体缩放 / 公告与更新提醒 |
| 帮助中心 | 使用说明与 FAQ |

## 🚀 快速开始

### 直接运行（需 Python 3.10+）

```bash
pip install -r requirements.txt
python app.py
```

或双击根目录的「启动工作台.bat」。

### 已发布版本

到 [Releases](https://github.com/zoilzo/MatLite/releases) 下载：
- `MatLiteSetup.exe` —— 正式安装包
- `MatLite-便携版.zip` —— 免安装绿色版

## 🛠 技术栈

Python · customtkinter · matplotlib · numpy · pandas · scipy · sympy · scikit-learn · statsmodels · python-docx · openpyxl · requests · Pillow

## 📦 打包发布

```bash
python bump_version.py [major|minor|1.3.0]  # 可选：先升级版本号（全链路同步）
python build_exe.py                          # PyInstaller 构建 EXE → dist\MatLite\
# 用 Inno Setup 编译 installer.iss           # 生成正式安装包
```

版本号**唯一来源**：`modules/app_settings.py` 的 `APP_VERSION`。
`bump_version.py` 会自动同步 `installer.iss`、`assets/version_info.txt`、`.version`。

## 🏗 目录结构

```
MatLite/
├── app.py                  # 入口：注册 15 个页面
├── modules/                # 各功能页面
│   ├── expr_utils.py       # 共享表达式工具（统一 SAFE 字典）
│   └── log_utils.py        # 统一日志系统
├── stats_utils.py          # 统计分析核心（根目录）
├── stats_plots.py          # 统计绘图
├── stats_report.py         # Word 报告生成
├── assets/                 # 图标、公告、安装条款、版本信息
├── build_exe.py            # PyInstaller 构建脚本
└── installer.iss           # Inno Setup 安装脚本
```

## 📄 许可

本工具免费供个人学习使用。详见安装条款与安装包内协议。

*Made by zoilzo & Claude*