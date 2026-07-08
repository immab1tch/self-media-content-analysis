@echo off
cd /d "%~dp0"

echo ================================
echo 自媒体账号数据分析系统
echo ================================
echo.

REM 检查 Python 是否可用
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未找到 Python！
    echo 请确保 Python 已安装并添加到系统 PATH
    pause
    exit /b 1
)

echo Python 版本：
python --version
echo.

REM 设置环境变量
set STREAMLIT_SERVER_HEADLESS=true
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

REM 启动 Streamlit
echo 正在启动应用...
echo.
python -m streamlit run src/app.py --server.port 8501

pause