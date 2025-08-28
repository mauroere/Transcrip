@echo off
echo Instalando dependencias para el Transcriptor de Audios...
echo.

python --version
if errorlevel 1 (
    echo Error: Python no esta instalado o no esta en el PATH
    echo Por favor instala Python 3.8 o superior desde https://python.org
    pause
    exit /b 1
)

echo Actualizando pip...
python -m pip install --upgrade pip

echo.
echo Verificando FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ⚠️ FFmpeg no encontrado - Es necesario para procesar archivos de audio
    echo Ejecutando instalador de FFmpeg...
    python install_ffmpeg.py
    echo.
    echo Si FFmpeg no se instaló correctamente, consulta TROUBLESHOOTING.md
    echo.
)

echo.
echo Instalando PyTorch primero (esto puede tomar unos minutos)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Instalando otras dependencias...
pip install Flask>=2.3.0
pip install Werkzeug>=2.3.0
pip install openai-whisper>=20240930
pip install pydub>=0.25.1
pip install moviepy>=1.0.3
pip install Jinja2>=3.1.0
pip install MarkupSafe>=2.1.0
pip install itsdangerous>=2.1.0
pip install click>=8.1.0
pip install blinker>=1.6.0
pip install ffmpeg-python>=0.2.0

echo.
echo Verificando instalacion...
python -c "import whisper; print('Whisper instalado correctamente')"
python -c "import flask; print('Flask instalado correctamente')"

echo.
echo ¡Instalacion completada!
echo.
echo Para ejecutar la aplicacion, usa: python app.py
echo Luego abre tu navegador en: http://localhost:5000
echo.
pause
