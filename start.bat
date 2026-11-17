@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Turnstile detector

if exist ".venv\installed.ok" goto run

echo Первый запуск: ставлю Python-библиотеки, это займёт 5-10 минут...
where py >nul 2>nul
if %errorlevel%==0 (py -3 -m venv .venv) else (python -m venv .venv)
if not exist ".venv\Scripts\python.exe" goto nopython
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto noinstall
echo ok> ".venv\installed.ok"

:run
".venv\Scripts\python.exe" turnstile_detector.py %*
echo.
pause
exit /b

:nopython
echo Не найден Python. Установи Python 3.12 с python.org и поставь галочку "Add python.exe to PATH".
pause
exit /b 1

:noinstall
echo Не удалось установить библиотеки. Проверь интернет и запусти start.bat ещё раз.
pause
exit /b 1
