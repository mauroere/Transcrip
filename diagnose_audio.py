"""
Script de diagnóstico para archivos de audio problemáticos
"""
import os
import whisper
import traceback

def diagnose_audio_file(filepath):
    """Diagnostica problemas con un archivo de audio"""
    print(f"\n=== DIAGNÓSTICO: {os.path.basename(filepath)} ===")
    
    # Verificar existencia
    if not os.path.exists(filepath):
        print("❌ El archivo no existe")
        return False
    
    print("✓ El archivo existe")
    
    # Verificar tamaño
    file_size = os.path.getsize(filepath)
    print(f"📏 Tamaño: {file_size} bytes ({file_size/(1024*1024):.2f} MB)")
    
    if file_size == 0:
        print("❌ El archivo está vacío")
        return False
    
    if file_size > 100 * 1024 * 1024:
        print("⚠️ Archivo muy grande (>100MB)")
    
    # Verificar header del archivo
    try:
        with open(filepath, 'rb') as f:
            header = f.read(12)
            print(f"📋 Header: {header[:8].hex()}")
            
            # Detectar formato por header
            if header.startswith(b'RIFF'):
                print("✓ Formato detectado: WAV")
            elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):
                print("✓ Formato detectado: MP3")
            elif header.startswith(b'fLaC'):
                print("✓ Formato detectado: FLAC")
            else:
                print(f"⚠️ Formato no reconocido o archivo corrupto")
    
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
        return False
    
    # Probar con Whisper
    print("\n🎯 Probando con Whisper...")
    try:
        model = whisper.load_model("base")
        
        # Intento básico
        print("Intento 1: Configuración estándar")
        result = model.transcribe(filepath, language='es', verbose=False)
        
        if result and result.get('text', '').strip():
            text = result['text']
            print(f"✅ ÉXITO - Texto transcrito ({len(text)} caracteres):")
            print(f"'{text[:100]}{'...' if len(text) > 100 else ''}'")
            return True
        else:
            print("⚠️ Transcripción vacía")
    
    except Exception as e:
        print(f"❌ Error en Whisper: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    
    # Intento con auto-detección de idioma
    try:
        print("\nIntento 2: Auto-detección de idioma")
        result = model.transcribe(filepath, verbose=False)
        
        if result and result.get('text', '').strip():
            text = result['text']
            language = result.get('language', 'unknown')
            print(f"✅ ÉXITO con idioma {language} - Texto ({len(text)} caracteres):")
            print(f"'{text[:100]}{'...' if len(text) > 100 else ''}'")
            return True
    
    except Exception as e:
        print(f"❌ Error en segundo intento: {e}")
    
    print("❌ FALLO TOTAL - No se pudo transcribir")
    return False

def main():
    """Función principal para diagnosticar archivos"""
    print("🔍 DIAGNÓSTICO DE ARCHIVOS DE AUDIO")
    print("=" * 50)
    
    # Buscar archivos en uploads
    uploads_dir = "uploads"
    if os.path.exists(uploads_dir):
        print(f"\n📁 Buscando archivos en {uploads_dir}...")
        files = os.listdir(uploads_dir)
        audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg'))]
        
        if audio_files:
            print(f"Encontrados {len(audio_files)} archivos de audio:")
            for file in audio_files:
                filepath = os.path.join(uploads_dir, file)
                diagnose_audio_file(filepath)
        else:
            print("No se encontraron archivos de audio en uploads/")
    else:
        print(f"❌ Directorio {uploads_dir} no existe")
    
    # También permitir diagnosticar archivos específicos
    print(f"\n💡 Para diagnosticar un archivo específico:")
    print(f"python diagnose_audio.py 'ruta/al/archivo.mp3'")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Diagnosticar archivo específico
        filepath = sys.argv[1]
        diagnose_audio_file(filepath)
    else:
        # Buscar y diagnosticar todos los archivos
        main()
