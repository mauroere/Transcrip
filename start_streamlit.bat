@echo off
REM Transcriptor de Audios Movistar - Streamlit Launcher
REM Ejecuta la aplicación Streamlit

echo.
echo ====================================================
echo   🎯 Transcriptor de Audios - Streamlit
echo ====================================================
echo.

REM Verificar si Python está disponible
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python no encontrado en PATH
    echo    Instala Python 3.13+ desde https://python.org
    pause
    exit /b 1
)

echo ✅ Python encontrado
echo.

REM Verificar si Streamlit está instalado
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Streamlit no encontrado. Instalando...
    pip install streamlit
    if errorlevel 1 (
        echo ❌ Error instalando Streamlit
        pause
        exit /b 1
    )
)

echo ✅ Streamlit disponible
echo.

REM Verificar modelo Whisper
python -c "import whisper" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  OpenAI Whisper no encontrado. Instalando...
    pip install openai-whisper
    if errorlevel 1 (
        echo ❌ Error instalando Whisper
        pause
        exit /b 1
    )
)

echo ✅ Whisper disponible

REM Verificar procesador de audio (Python 3.13 compatible)
python -c "import librosa" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Librosa no encontrado. Instalando para mejor compatibilidad...
    pip install librosa soundfile
    if errorlevel 1 (
        echo ⚠️ Warning: No se pudo instalar librosa, funcionalidad básica disponible
    )
)

echo ✅ Procesador de audio configurado
echo.

REM Ejecutar aplicación
echo 🚀 Iniciando aplicación Streamlit...
echo.
echo 📱 La aplicación se abrirá en: http://localhost:8504
echo 🛑 Para cerrar: Presiona Ctrl+C en esta ventana
echo.

python -m streamlit run streamlit_app.py --server.port 8504

echo.
echo 👋 ¡Aplicación cerrada!
pause
