#!/bin/bash
# macOS / Linux: double-click (macOS) or run ./start_ui.command
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "首次运行：正在准备环境，大约需要几分钟..."
  python3 -m venv .venv || { echo "没有找到 python3，请先安装 Python 3.10+"; read -r; exit 1; }
  .venv/bin/python -m pip install -U pip setuptools wheel
  .venv/bin/python -m pip install -e ".[cn]" yfinance || { echo "安装失败"; read -r; exit 1; }
fi
echo "正在启动量化工作台，浏览器会自动打开。关闭本窗口即可退出。"
.venv/bin/python -m quantum_analyze ui
