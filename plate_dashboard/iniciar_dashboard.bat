@echo off
title YOLO Plate Dashboard

REM Ir al proyecto
cd /d "C:\Users\jeron\OneDrive\Documentos\A_Semillero\Project_plates\plate_dashboard"

echo Activando entorno virtual...

REM Activar venv (PowerShell compatible llamado desde CMD)
call "venv\Scripts\activate.bat"

echo.
echo Ejecutando Streamlit...
echo.

streamlit run app.py

pause