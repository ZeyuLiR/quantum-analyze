@echo off
chcp 65001 >nul
title 量化工作台
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto run

echo ============================================
echo  首次运行：正在准备环境，大约需要几分钟...
echo ============================================
where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo [错误] 没有找到 Python。
  echo 请先从 https://www.python.org/downloads/ 安装 Python 3.10 或更新版本，
  echo 安装时务必勾选 "Add python.exe to PATH"，然后重新双击本文件。
  pause
  exit /b 1
)
python -m venv .venv
if errorlevel 1 (
  echo [错误] 创建虚拟环境失败。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
".venv\Scripts\python.exe" -m pip install -e ".[cn]" yfinance -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
  echo 国内镜像安装失败，改用官方源重试...
  ".venv\Scripts\python.exe" -m pip install -e ".[cn]" yfinance
)
if errorlevel 1 (
  echo [错误] 安装失败，请把上面的报错截图发给我。
  pause
  exit /b 1
)

:run
echo.
echo 正在启动量化工作台，浏览器会自动打开。
echo 使用期间请不要关闭这个黑色窗口；关闭它即可退出。
echo.
".venv\Scripts\python.exe" -m quantum_analyze ui
pause
