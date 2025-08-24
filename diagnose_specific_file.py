#!/usr/bin/env python3
"""
Script diagnóstico específico para el archivo "nasif daniel 20-8 13.29hs.wav"
que está causando errores de tensor
"""

import os
import sys
import tempfile
import warnings

# Suprimir warnings de Streamlit
warnings.filterwarnings("ignore")

def mock_streamlit_functions():
    """Crear funciones mock para reemplazar las de Streamlit durante las pruebas"""
    
    class MockStreamlit:
        @staticmethod
        def warning(text):
            print(f"⚠️ {text}")
        
        @staticmethod
        def info(text):
            print(f"ℹ️ {text}")
        
        @staticmethod
        def success(text):
            print(f"✅ {text}")
        
        @staticmethod
        def error(text):
            print(f"❌ {text}")
    
    # Reemplazar funciones de Streamlit en el módulo
    import streamlit_app
    streamlit_app.st = MockStreamlit()

def diagnose_audio_file(file_path):
    """Diagnosticar un archivo de audio específico"""
    
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        return False
    
    print(f"🔍 Analizando archivo: {os.path.basename(file_path)}")
    
    try:
        # Información básica del archivo
        file_size = os.path.getsize(file_path)
        print(f"📊 Tamaño del archivo: {file_size / 1024 / 1024:.2f} MB")
        
        # Analizar con librosa si está disponible
        try:
            import librosa
            import soundfile as sf
            
            # Cargar información del audio sin cargar todo el archivo
            info = sf.info(file_path)
            print(f"📊 Duración: {info.duration:.2f} segundos")
            print(f"📊 Sample rate: {info.samplerate} Hz")
            print(f"📊 Canales: {info.channels}")
            print(f"📊 Formato: {info.format}")
            print(f"📊 Subtype: {info.subtype}")
            
            # Cargar una pequeña muestra para análisis
            print("🔍 Cargando muestra del audio...")
            y, sr = librosa.load(file_path, sr=None, duration=5.0)  # Solo primeros 5 segundos
            
            print(f"📊 Rango de valores: {y.min():.4f} a {y.max():.4f}")
            print(f"📊 Valor RMS: {librosa.feature.rms(y=y).mean():.4f}")
            
            # Detectar problemas potenciales
            problems = []
            suggestions = []
            
            if info.samplerate > 48000:
                problems.append(f"Sample rate muy alto: {info.samplerate} Hz")
                suggestions.append("Reducir sample rate a 16000 Hz")
            
            if info.channels > 1:
                problems.append(f"Audio estéreo/multicanal: {info.channels} canales")
                suggestions.append("Convertir a mono")
            
            if info.duration > 600:  # Más de 10 minutos
                problems.append(f"Archivo muy largo: {info.duration:.1f} segundos")
                suggestions.append("Dividir en segmentos más pequeños")
            
            if abs(y.max()) > 0.9 or abs(y.min()) > 0.9:
                problems.append("Posible clipping en el audio")
                suggestions.append("Normalizar el audio")
            
            if problems:
                print(f"\n⚠️ PROBLEMAS DETECTADOS:")
                for problem in problems:
                    print(f"  • {problem}")
                
                print(f"\n💡 SUGERENCIAS:")
                for suggestion in suggestions:
                    print(f"  • {suggestion}")
            else:
                print(f"\n✅ No se detectaron problemas obvios en el archivo")
            
            return True
            
        except ImportError:
            print("❌ Librosa no disponible - análisis limitado")
            return False
            
    except Exception as e:
        print(f"❌ Error analizando archivo: {e}")
        return False

def test_specific_file_with_improvements(file_path):
    """Probar el archivo específico con las mejoras implementadas"""
    
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        return False
    
    try:
        # Mock de funciones de Streamlit
        mock_streamlit_functions()
        
        # Importar las funciones necesarias
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from streamlit_app import transcribe_with_enhanced_quality
        
        # Cargar modelo Whisper
        print("📥 Cargando modelo Whisper...")
        import whisper
        
        try:
            model = whisper.load_model("tiny")  # Usar modelo más pequeño para pruebas
            if model is None:
                print("❌ No se pudo cargar el modelo Whisper")
                return False
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
        
        print("✅ Modelo cargado correctamente")
        
        print(f"\n🔬 Probando transcripción con mejoras...")
        
        try:
            transcription, error = transcribe_with_enhanced_quality(model, file_path)
            
            if transcription:
                print(f"🎉 ¡ÉXITO! Transcripción completada")
                print(f"📏 Longitud: {len(transcription)} caracteres")
                print(f"📄 Primeras 200 caracteres: {transcription[:200]}...")
                return True
            else:
                print(f"❌ Falló la transcripción: {error}")
                
                # Análisis del error
                if error and "tensor" in error.lower():
                    print(f"\n🔧 ERROR DE TENSOR DETECTADO")
                    print(f"💡 Soluciones recomendadas:")
                    print(f"   1. Convertir con FFmpeg:")
                    print(f"      ffmpeg -i \"{os.path.basename(file_path)}\" -ar 16000 -ac 1 -c:a pcm_s16le \"fixed_{os.path.basename(file_path)}\"")
                    print(f"   2. Dividir en segmentos:")
                    print(f"      ffmpeg -i \"{os.path.basename(file_path)}\" -f segment -segment_time 300 -ar 16000 -ac 1 \"segmento_%03d.wav\"")
                
                return False
                
        except Exception as e:
            print(f"❌ Error inesperado durante transcripción: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False

def generate_ffmpeg_commands(file_path):
    """Generar comandos FFmpeg específicos para el archivo"""
    
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    print(f"\n🔧 COMANDOS FFMPEG RECOMENDADOS PARA: {filename}")
    print("=" * 60)
    
    print(f"\n1️⃣ CONVERSIÓN BÁSICA (más compatible):")
    print(f"ffmpeg -i \"{filename}\" -ar 16000 -ac 1 -c:a pcm_s16le \"{name_without_ext}_fixed.wav\"")
    
    print(f"\n2️⃣ DIVISIÓN EN SEGMENTOS (5 minutos cada uno):")
    print(f"ffmpeg -i \"{filename}\" -f segment -segment_time 300 -ar 16000 -ac 1 \"{name_without_ext}_parte_%03d.wav\"")
    
    print(f"\n3️⃣ CONVERSIÓN A MP3 (menor tamaño):")
    print(f"ffmpeg -i \"{filename}\" -ar 22050 -ac 1 -b:a 64k \"{name_without_ext}_fixed.mp3\"")
    
    print(f"\n4️⃣ NORMALIZACIÓN DE AUDIO:")
    print(f"ffmpeg -i \"{filename}\" -ar 16000 -ac 1 -filter:a \"volume=0.5\" \"{name_without_ext}_normalized.wav\"")

def main():
    """Función principal"""
    print("🚀 DIAGNÓSTICO ESPECÍFICO PARA ERRORES DE TENSOR")
    print("=" * 60)
    
    # Buscar archivos de audio en el directorio actual
    audio_files = []
    for file in os.listdir('.'):
        if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
            audio_files.append(file)
    
    if not audio_files:
        print("❌ No se encontraron archivos de audio en el directorio actual")
        print("💡 Copia tu archivo de audio a este directorio y ejecuta el script nuevamente")
        return
    
    print(f"📁 Archivos de audio encontrados:")
    for i, file in enumerate(audio_files, 1):
        print(f"   {i}. {file}")
    
    if len(audio_files) == 1:
        selected_file = audio_files[0]
        print(f"\n🎯 Seleccionado automáticamente: {selected_file}")
    else:
        try:
            choice = int(input(f"\n🎯 Selecciona un archivo (1-{len(audio_files)}): ")) - 1
            if 0 <= choice < len(audio_files):
                selected_file = audio_files[choice]
            else:
                print("❌ Selección inválida")
                return
        except ValueError:
            print("❌ Entrada inválida")
            return
    
    print(f"\n🔍 ANALIZANDO: {selected_file}")
    print("-" * 40)
    
    # 1. Diagnóstico del archivo
    print("📊 PASO 1: Análisis del archivo")
    diagnose_audio_file(selected_file)
    
    # 2. Generar comandos FFmpeg
    print(f"\n🔧 PASO 2: Comandos de corrección")
    generate_ffmpeg_commands(selected_file)
    
    # 3. Probar transcripción con mejoras
    print(f"\n🧪 PASO 3: Prueba de transcripción")
    print("¿Quieres probar la transcripción con las mejoras implementadas? (s/n): ", end="")
    
    try:
        response = input().lower().strip()
        if response in ['s', 'si', 'sí', 'y', 'yes']:
            test_specific_file_with_improvements(selected_file)
        else:
            print("✅ Análisis completado. Usa los comandos FFmpeg de arriba para corregir el archivo.")
    except KeyboardInterrupt:
        print("\n👋 Análisis cancelado.")

if __name__ == "__main__":
    main()
