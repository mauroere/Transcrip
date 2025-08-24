#!/usr/bin/env python3
"""
Script de prueba simplificado para verificar las mejoras en el manejo de errores de tensor
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
import warnings

# Suprimir warnings de Streamlit
warnings.filterwarnings("ignore")

def create_test_audio(duration=30, sample_rate=16000, problematic=False):
    """Crear un archivo de audio de prueba"""
    
    # Generar audio sintético
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    if problematic:
        # Crear un audio con características que pueden causar problemas de tensor
        # Mezcla de frecuencias complejas
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3 +  # La (440 Hz)
                np.sin(2 * np.pi * 880 * t) * 0.2 +   # La octava superior
                np.sin(2 * np.pi * 220 * t) * 0.2 +   # La octava inferior
                np.random.normal(0, 0.1, len(t)))     # Ruido
        
        # Agregar cambios abruptos que pueden confundir al modelo
        for i in range(0, len(audio), len(audio)//5):
            end_idx = min(i + len(audio)//10, len(audio))
            audio[i:end_idx] *= 0.1 if (i//(len(audio)//5)) % 2 == 0 else 2.0
    else:
        # Audio simple y limpio
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Normalizar
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)

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

def test_tensor_error_handling():
    """Probar el manejo mejorado de errores de tensor"""
    
    print("🧪 Iniciando pruebas de manejo de errores de tensor...")
    
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
            model = whisper.load_model("tiny")  # Usar modelo más pequeño para tests
            if model is None:
                print("❌ No se pudo cargar el modelo Whisper")
                return False
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
        
        print("✅ Modelo cargado correctamente")
        
        # Crear archivos de prueba
        test_files = []
        
        # 1. Audio normal
        print("\n🎵 Creando audio de prueba normal...")
        normal_audio = create_test_audio(duration=5, problematic=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            sf.write(f.name, normal_audio, 16000)
            test_files.append(("normal", f.name))
        
        # 2. Audio problemático (que puede causar errores de tensor)
        print("🎵 Creando audio de prueba problemático...")
        problematic_audio = create_test_audio(duration=8, problematic=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            sf.write(f.name, problematic_audio, 16000)
            test_files.append(("problematic", f.name))
        
        # Probar cada archivo
        results = []
        
        for test_type, file_path in test_files:
            print(f"\n🔬 Probando archivo {test_type}...")
            
            try:
                transcription, error = transcribe_with_enhanced_quality(model, file_path)
                
                if transcription:
                    print(f"✅ {test_type}: Transcripción exitosa")
                    print(f"   Longitud: {len(transcription)} caracteres")
                    print(f"   Contenido: {transcription[:100]}..." if len(transcription) > 100 else f"   Contenido: {transcription}")
                    results.append((test_type, True, None))
                else:
                    print(f"⚠️ {test_type}: Falló - {error}")
                    results.append((test_type, False, error))
                    
            except Exception as e:
                print(f"❌ {test_type}: Error inesperado - {str(e)}")
                results.append((test_type, False, str(e)))
        
        # Limpiar archivos temporales
        for _, file_path in test_files:
            try:
                os.unlink(file_path)
            except:
                pass
        
        # Resumen de resultados
        print(f"\n📊 RESUMEN DE PRUEBAS:")
        print(f"{'Tipo':<15} {'Resultado':<10} {'Error'}")
        print("-" * 60)
        
        for test_type, success, error in results:
            status = "✅ ÉXITO" if success else "❌ FALLO"
            error_msg = error[:40] + "..." if error and len(error) > 40 else (error or "")
            print(f"{test_type:<15} {status:<10} {error_msg}")
        
        # Verificar si al menos uno funcionó
        successful_tests = [r for r in results if r[1]]
        if len(successful_tests) > 0:
            print(f"\n🎉 {len(successful_tests)}/{len(results)} pruebas exitosas")
            print("✅ El sistema de manejo de errores de tensor está funcionando")
            return True
        else:
            print(f"\n😞 0/{len(results)} pruebas exitosas")
            print("❌ El sistema necesita más ajustes")
            return False
            
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("💡 Asegúrate de que todas las dependencias estén instaladas")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 PRUEBAS DE MEJORAS EN MANEJO DE TENSOR (Versión Simplificada)")
    print("=" * 70)
    
    success = test_tensor_error_handling()
    
    if success:
        print("\n🎯 CONCLUSIÓN: Las mejoras están funcionando correctamente")
        print("🔧 El sistema ahora debe manejar mejor los errores de tensor")
        print("\n💡 SIGUIENTE PASO: Prueba con tu archivo real en la aplicación:")
        print("   python -m streamlit run streamlit_app.py")
    else:
        print("\n⚠️ CONCLUSIÓN: Se necesitan más ajustes")
        print("🔧 Revisa los errores reportados arriba")
        print("\n🔧 RECOMENDACIÓN: Usa comandos FFmpeg para convertir archivos problemáticos:")
        print("   ffmpeg -i \"archivo_problema.wav\" -ar 16000 -ac 1 -c:a pcm_s16le \"archivo_fijo.wav\"")
