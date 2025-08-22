#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script - Verificación de Compatibilidad Tensor
=================================================
Este script verifica que las mejoras implementadas para resolver
los errores de tensor en Whisper funcionen correctamente.

Desarrollado por: Mauro Rementeria - mauroere@gmail.com
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

def test_basic_imports():
    """Probar importaciones básicas"""
    print("🔧 Probando importaciones básicas...")
    
    try:
        import whisper
        print("✅ Whisper importado correctamente")
    except Exception as e:
        print(f"❌ Error importando Whisper: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} importado correctamente")
    except Exception as e:
        print(f"❌ Error importando NumPy: {e}")
        return False
        
    try:
        import librosa
        print("✅ Librosa importado correctamente")
    except Exception as e:
        print(f"❌ Error importando Librosa: {e}")
        return False
    
    return True

def test_model_loading():
    """Probar carga de modelos con fallback"""
    print("\n🤖 Probando carga de modelos...")
    
    import whisper
    
    # Intentar cargar modelo base
    try:
        print("📥 Intentando cargar modelo 'base'...")
        model = whisper.load_model("base")
        print("✅ Modelo 'base' cargado exitosamente")
        return model
    except Exception as e:
        print(f"⚠️ Error con modelo 'base': {e}")
        
        # Fallback a tiny
        try:
            print("📥 Intentando cargar modelo 'tiny' (fallback)...")
            model = whisper.load_model("tiny")
            print("✅ Modelo 'tiny' cargado exitosamente")
            return model
        except Exception as e:
            print(f"❌ Error con modelo 'tiny': {e}")
            return None

def test_tensor_compatibility():
    """Probar compatibilidad de tensores"""
    print("\n🔬 Probando compatibilidad de tensores...")
    
    model = test_model_loading()
    if model is None:
        print("❌ No se pudo cargar ningún modelo")
        return False
    
    # Crear un audio de prueba simple
    try:
        import numpy as np
        
        # Audio silencioso de prueba (1 segundo)
        sample_rate = 16000
        duration = 1.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        print("🎵 Audio de prueba creado")
        
        # Probar transcripción básica
        try:
            result = model.transcribe(audio, language="es")
            print("✅ Transcripción básica exitosa")
            return True
        except Exception as e:
            print(f"❌ Error en transcripción básica: {e}")
            
            # Si hay error de tensor, intentar con opciones mínimas
            try:
                result = model.transcribe(audio)
                print("✅ Transcripción con configuración mínima exitosa")
                return True
            except Exception as e2:
                print(f"❌ Error persistente: {e2}")
                return False
                
    except Exception as e:
        print(f"❌ Error creando audio de prueba: {e}")
        return False

def test_enhanced_transcription():
    """Probar función mejorada de transcripción"""
    print("\n⚡ Probando función mejorada de transcripción...")
    
    # Importar función desde streamlit_app
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from streamlit_app import transcribe_with_enhanced_quality, load_whisper_model
        print("✅ Funciones importadas correctamente")
    except Exception as e:
        print(f"❌ Error importando funciones: {e}")
        return False
    
    # Cargar modelo
    model = load_whisper_model()
    if model is None:
        print("❌ No se pudo cargar el modelo")
        return False
    
    print("✅ Modelo cargado correctamente")
    
    # Crear archivo de audio temporal
    try:
        import tempfile
        import numpy as np
        import soundfile as sf
        
        # Crear audio de prueba más realista
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Tono simple + ruido blanco suave
        audio = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.normal(0, 1, len(t))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, audio, sample_rate)
            tmp_path = tmp_file.name
        
        print("🎵 Archivo de audio temporal creado")
        
        # Probar función mejorada
        transcription, error = transcribe_with_enhanced_quality(model, tmp_path)
        
        # Limpiar archivo temporal
        os.unlink(tmp_path)
        
        if error:
            print(f"❌ Error en transcripción mejorada: {error}")
            return False
        else:
            print(f"✅ Transcripción mejorada exitosa")
            print(f"📝 Resultado: '{transcription[:100]}...' (truncado)")
            return True
            
    except Exception as e:
        print(f"❌ Error en prueba de transcripción mejorada: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("="*60)
    print("🧪 SCRIPT DE VERIFICACIÓN - TENSOR COMPATIBILITY FIX")
    print("="*60)
    print("Desarrollado por: Mauro Rementeria - mauroere@gmail.com")
    print()
    
    tests = [
        ("Importaciones Básicas", test_basic_imports),
        ("Compatibilidad de Tensores", test_tensor_compatibility),
        ("Transcripción Mejorada", test_enhanced_transcription)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error crítico en {test_name}: {e}")
            results.append((test_name, False))
        
        print()
    
    # Resumen final
    print("="*60)
    print("📊 RESUMEN DE PRUEBAS")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print()
    print(f"Resultado: {passed}/{len(results)} pruebas exitosas")
    
    if passed == len(results):
        print("🎉 ¡Todas las pruebas pasaron! El sistema está listo.")
        return True
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa los errores anteriores.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
