@echo off
chcp 65001 >nul
cd /d "%~dp0"
title MatLite 打包程序

rem ============================================
rem  MatLite 数学工作台 —— 一键打包并自动归档版本
rem  用法：
rem    打包.bat            -> 主版本不变，补丁号+1（1.0.0 -> 1.0.1）
rem    打包.bat minor      -> 次版本+1（1.0.0 -> 1.1.0）
rem    打包.bat major      -> 主版本+1（1.0.0 -> 2.0.0）
rem    打包.bat 1.2.3      -> 直接指定为 1.2.3
rem ============================================
echo.

rem 1. 检查依赖
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 python，请先安装 Python 3.10+ 并勾选 Add to PATH
    pause & exit /b 1
)

rem 2. 递增版本号（可显式指定，否则默认补丁+1）
echo [1/5] 递增版本号 ...
python bump_version.py %1
if errorlevel 1 (
    echo [错误] 版本号处理失败
    pause & exit /b 1
)
set /p VER=<.version
echo     新版本：v%VER%

rem 3. 生成安装素材（图标/许可文件）
echo [2/5] 生成安装素材（图标/许可文件）...
python make_assets.py >nul 2>nul
if not exist "assets\languages\ChineseSimplified.isl" (
    echo     首次打包：下载中文语言文件（供安装向导使用）...
    mkdir assets\languages 2>nul
    curl -sSL -o assets\languages\ChineseSimplified.isl ^
        "https://raw.githubusercontent.com/jrsoftware/issrc/is-6_7_3/Files/Languages/Unofficial/ChineseSimplified.isl"
)

rem 4. 构建 EXE
echo [3/5] 构建 EXE（需要几分钟，请耐心等待）...
python build_exe.py
if errorlevel 1 (
    echo [错误] 打包失败，请查看上方错误信息
    pause & exit /b 1
)

rem 5. 用 Inno Setup 编译正式安装包
set "ISCC=C:\Users\19627\AppData\Local\Programs\Inno Setup 6\ISCC.exe"
echo [4/5] 编译安装包（Inno Setup 6）...
if exist "%ISCC%" (
    "%ISCC%" installer.iss
    if errorlevel 1 (
        echo [错误] 安装包编译失败，请查看上方信息
        pause & exit /b 1
    )
) else (
    echo [错误] 未找到 Inno Setup 6。请安装后运行：
    echo   "%ISCC%" installer.iss
    pause & exit /b 1
)

rem 6. 归档发布目录 releases\v%VER%\
set "REL=releases\v%VER%"
mkdir "%REL%" 2>nul
copy /y "dist\MatLiteSetup.exe" "%REL%\MatLiteSetup.exe" >nul
copy /y "assets\installer_license.txt" "%REL%\安装条款.txt" >nul
echo [5/5] 已归档：%REL%\

echo.
echo 打包完成！产物：
echo   %REL%\MatLiteSetup.exe   （正式安装包，含协议与自定义安装选项）
echo   %REL%\MatLite-便携版.zip （解压即用，可选）
echo.
echo 对方拿到程序后，首次使用请在「AI 助手」页填写自己的
echo Ollama 服务地址并点「刷新」选择模型，即可接入本地 AI。
echo.
pause
