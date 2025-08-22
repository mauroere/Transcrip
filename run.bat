@echo off
echo Iniciando Transcriptor de Audios Movistar...
echo.

python app.py

if errorlevel 1 (
    echo.
    echo Error al ejecutar la aplicacion.
    echo Asegurate de haber ejecutado install.bat primero.
    echo.
    pause
)
