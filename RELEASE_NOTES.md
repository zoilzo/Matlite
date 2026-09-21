# MatLite v1.3.0

**发布日期：2026-09-21**

工程全面优化 + AI 工具扩充版本。共 41 个内置工具。

## 🚀 新增功能

- **AI 工具扩充 5 个**（共 41 个）：数值求根 `numeric_solve`、曲线拟合 `curve_fit`、复变函数求值 `complex_eval`、复变函数渲染 `complex_f_value`、Word 报告导出 `export_report`
- **绘制大幅提速**：曲面/等高线绘图改为向量化求值，消除 25 万次逐点 Python 循环，复杂图形秒出

## ⚡ 性能与稳定性

- 配置文件读写加线程锁 + 原子写入，避免多线程并发丢失设置
- 历史记录写入加锁，防数据竞争
- 新增统一日志系统（`%APPDATA%\MatLite\app.log`），异常可追溯
- 统一数学函数字典（消除 4 份重复 SAFE 定义）

## 🐛 Bug 修复

- LaTeX 渲染后端冲突（模块级 matplotlib.use 副作用）
- 直方图正态曲线缩放错误（改用真实箱宽）
- 科学计数法 / `"2x3"` 等表达式解析错误

## 📦 下载

- [MatLiteSetup.exe（84.9 MB，安装包）](./MatLiteSetup.exe)
- [MatLite-便携版.zip（124 MB，免安装）](./MatLite-便携版.zip)

## 🏗 工程规范

- 新增 `.gitignore` / `README.md`
- 版本号全链路自动同步（bump_version.py 统一管理）
- 打包脚本单一入口（删除自动生成的 .spec）

---
*Made by zoilzo & Claude*
