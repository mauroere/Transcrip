import os
import whisper

# Configurar FFmpeg si está en directorio local
ffmpeg_local = os.path.join(os.getcwd(), 'ffmpeg', 'ffmpeg.exe')
if os.path.exists(ffmpeg_local):
    ffmpeg_dir = os.path.dirname(ffmpeg_local)
    if ffmpeg_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
    print(f"✓ FFmpeg configurado desde: {ffmpeg_local}")
else:
    print("⚠️ FFmpeg local no encontrado, usando PATH del sistema")

# Cargar modelo
print("Cargando modelo Whisper...")
model = whisper.load_model("base")
print("Modelo cargado")

# Probar el primer archivo
uploads_dir = "uploads"
files = os.listdir(uploads_dir)
if files:
    test_file = os.path.join(uploads_dir, files[0])
    print(f"\nProbando archivo: {files[0]}")
    print(f"Ruta completa: {test_file}")
    print(f"Tamaño: {os.path.getsize(test_file)} bytes")
    
    try:
        result = model.transcribe(test_file, language='es')
        print(f"\n✅ ÉXITO!")
        print(f"Idioma detectado: {result.get('language', 'N/A')}")
        print(f"Texto: {result['text']}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
else:
    print("No hay archivos para probar")
