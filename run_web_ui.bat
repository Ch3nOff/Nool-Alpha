@echo off
title Nool-Alpha-100M Local Web UI
color 0b
echo ====================================================================
echo        🚀 Menjalankan Nool-Alpha-100M Local Web Chat UI
echo ====================================================================
echo Model: C:\Users\Matthew Chen\Downloads\Nool_alpha model\sft\enx model
echo Web UI: http://127.0.0.1:7860
echo ====================================================================
echo.

start "" "http://127.0.0.1:7860"

".venv\Scripts\python.exe" web_playground.py --checkpoint "C:\Users\Matthew Chen\Downloads\Nool_alpha model\sft\enx model"

pause
