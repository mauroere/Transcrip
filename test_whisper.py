#!/usr/bin/env python3
"""
Test de Whisper - Diagnóstico completo
"""

print("🔍 DIAGNÓSTICO DE WHISPER")
print("=" * 50)

# 1. Verificar importación
try:
    import whisper
    print("✅ Whisper importado correctamente")
    print(f"   Versión: {whisper.__version__}")
except ImportError as e:
    print(f"❌ Error importando Whisper: {e}")
    exit(1)

# 2. Verificar torch (dependencia de Whisper)
try:
    import torch
    print("✅ PyTorch disponible")
    print(f"   Versión: {torch.__version__}")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Error con PyTorch: {e}")

# 3. Listar modelos disponibles
print("\n📋 MODELOS DISPONIBLES:")
available_models = whisper.available_models()
for model in available_models:
    print(f"   • {model}")

# 4. Intentar cargar modelo base
print("\n🔄 PROBANDO CARGA DE MODELO...")
try:
    model = whisper.load_model("base")
    print("✅ Modelo 'base' cargado exitosamente")
    print(f"   Tipo: {type(model)}")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    print("   Posibles soluciones:")
    print("   • Verificar conexión a internet")
    print("   • Reinstalar whisper: pip uninstall openai-whisper && pip install openai-whisper")

# 5. Verificar pydub
try:
    from pydub import AudioSegment
    print("✅ Pydub disponible")
except ImportError as e:
    print(f"⚠️  Pydub no disponible: {e}")
    print("   Instalar con: pip install pydub")

# 6. Verificar ffmpeg
import subprocess
try:
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ FFmpeg disponible")
    else:
        print("⚠️  FFmpeg no encontrado")
except FileNotFoundError:
    print("⚠️  FFmpeg no instalado")
    print("   Instalar desde: https://ffmpeg.org/download.html")

print("\n" + "=" * 50)
print("🎯 DIAGNÓSTICO COMPLETADO")
