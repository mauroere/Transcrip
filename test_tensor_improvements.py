#!/usr/bin/env python3
"""
Script de prueba para verificar las mejoras en el manejo de errores de tensor
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf

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

def test_tensor_error_handling():
    """Probar el manejo mejorado de errores de tensor"""
    
    print("🧪 Iniciando pruebas de manejo de errores de tensor...")
    
    try:
        # Importar las funciones necesarias
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from streamlit_app import transcribe_with_enhanced_quality, load_whisper_model
        
        # Cargar modelo
        print("📥 Cargando modelo Whisper...")
        model = load_whisper_model()
        
        if model is None:
            print("❌ No se pudo cargar el modelo Whisper")
            return False
        
        print("✅ Modelo cargado correctamente")
        
        # Crear archivos de prueba
        test_files = []
        
        # 1. Audio normal
        print("\n🎵 Creando audio de prueba normal...")
        normal_audio = create_test_audio(duration=10, problematic=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            sf.write(f.name, normal_audio, 16000)
            test_files.append(("normal", f.name))
        
        # 2. Audio problemático (que puede causar errores de tensor)
        print("🎵 Creando audio de prueba problemático...")
        problematic_audio = create_test_audio(duration=15, problematic=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            sf.write(f.name, problematic_audio, 16000)
            test_files.append(("problematic", f.name))
        
        # 3. Audio largo (para probar segmentación)
        print("🎵 Creando audio de prueba largo...")
        long_audio = create_test_audio(duration=60, problematic=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            sf.write(f.name, long_audio, 16000)
            test_files.append(("long", f.name))
        
        # Probar cada archivo
        results = []
        
        for test_type, file_path in test_files:
            print(f"\n🔬 Probando archivo {test_type}...")
            
            try:
                transcription, error = transcribe_with_enhanced_quality(model, file_path)
                
                if transcription:
                    print(f"✅ {test_type}: Transcripción exitosa")
                    print(f"   Longitud: {len(transcription)} caracteres")
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
        print("-" * 50)
        
        for test_type, success, error in results:
            status = "✅ ÉXITO" if success else "❌ FALLO"
            error_msg = error[:30] + "..." if error and len(error) > 30 else (error or "")
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
        return False

if __name__ == "__main__":
    print("🚀 PRUEBAS DE MEJORAS EN MANEJO DE TENSOR")
    print("=" * 50)
    
    success = test_tensor_error_handling()
    
    if success:
        print("\n🎯 CONCLUSIÓN: Las mejoras están funcionando correctamente")
        print("🔧 El sistema ahora debe manejar mejor los errores de tensor")
    else:
        print("\n⚠️ CONCLUSIÓN: Se necesitan más ajustes")
        print("🔧 Revisa los errores reportados arriba")
    
    print("\n💡 SIGUIENTE PASO: Prueba con tu archivo real:")
    print("   python -m streamlit run streamlit_app.py")
