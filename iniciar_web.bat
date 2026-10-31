@echo off
setlocal
rem Sirve la carpeta web/ en http://localhost:8000 y abre el navegador.
rem Si Anaconda no esta en el PATH, intenta activarlo desde su ruta por defecto.
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
cd /d "%~dp0web"
echo Sirviendo %CD% en http://localhost:8000  (Ctrl+C para detener)
start "" cmd /c "timeout /t 2 >nul & start http://localhost:8000"
python -m http.server 8000
pause
