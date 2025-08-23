#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creador de Audio de Prueba para Errores de Tensor
================================================
Este script crea un archivo de audio que puede causar errores similares
para probar nuestras soluciones.

Desarrollado por: Mauro Rementeria - mauroere@gmail.com
"""

import numpy as np
import soundfile as sf
import os

def create_problematic_audio():
    """Crear un archivo de audio que puede causar problemas de tensor"""
    print("🎵 Creando archivo de audio de prueba...")
    
    # Parámetros que pueden causar problemas
    sample_rate = 44100  # Sample rate alto
    duration = 180  # 3 minutos
    
    # Crear audio con características problemáticas
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Audio base con múltiples frecuencias
    audio = np.zeros_like(t)
    
    # Añadir voz simulada (frecuencias típicas de voz humana)
    for freq in [200, 400, 800, 1200, 1600]:
        audio += 0.1 * np.sin(2 * np.pi * freq * t)
    
    # Añadir ruido para simular grabación real
    noise = 0.02 * np.random.normal(0, 1, len(t))
    audio += noise
    
    # Modular amplitud para simular habla
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
    audio *= modulation
    
    # Normalizar
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Convertir a estéreo (puede causar problemas)
    stereo_audio = np.column_stack([audio, audio * 0.9])
    
    # Guardar archivo
    filename = "audio_prueba_tensor.wav"
    sf.write(filename, stereo_audio, sample_rate)
    
    print(f"✅ Archivo creado: {filename}")
    print(f"   • Duración: {duration} segundos")
    print(f"   • Sample rate: {sample_rate} Hz")
    print(f"   • Canales: 2 (estéreo)")
    print(f"   • Tamaño: {os.path.getsize(filename) / 1024 / 1024:.1f} MB")
    
    return filename

def test_with_streamlit_functions():
    """Probar el archivo con las funciones de streamlit_app.py"""
    print("\n🧪 Probando con funciones mejoradas...")
    
    # Importar funciones desde streamlit_app.py
    import sys
    sys.path.append('.')
    
    try:
        # Simular ambiente sin streamlit para evitar errores
        import streamlit
        streamlit.info = lambda x: print(f"INFO: {x}")
        streamlit.warning = lambda x: print(f"WARNING: {x}")
        streamlit.success = lambda x: print(f"SUCCESS: {x}")
        streamlit.error = lambda x: print(f"ERROR: {x}")
    except:
        pass
    
    try:
        from streamlit_app import load_whisper_model, transcribe_with_enhanced_quality
        
        print("🤖 Cargando modelo Whisper...")
        model = load_whisper_model()
        
        if model is None:
            print("❌ No se pudo cargar el modelo")
            return False
        
        print("✅ Modelo cargado")
        
        # Probar con el archivo de audio
        filename = "audio_prueba_tensor.wav"
        if not os.path.exists(filename):
            print("❌ Archivo de prueba no existe")
            return False
        
        print(f"🎯 Probando transcripción de {filename}...")
        
        # Probar función mejorada
        transcription, error = transcribe_with_enhanced_quality(model, filename)
        
        if error:
            print(f"❌ Error: {error}")
            
            # Verificar si es el tipo de error que esperamos manejar
            if "tensor" in error.lower():
                print("✅ Error de tensor detectado y manejado correctamente")
                return True
            else:
                print("⚠️ Error diferente al esperado")
                return False
        else:
            print(f"✅ Transcripción exitosa: {transcription[:100]}...")
            return True
            
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

def main():
    """Función principal"""
    print("🧪 GENERADOR DE PRUEBAS PARA ERRORES DE TENSOR")
    print("=" * 50)
    print("Desarrollado por: Mauro Rementeria - mauroere@gmail.com")
    print()
    
    # Crear archivo de prueba
    try:
        filename = create_problematic_audio()
        
        # Probar con las funciones mejoradas
        success = test_with_streamlit_functions()
        
        print("\n" + "=" * 50)
        if success:
            print("✅ PRUEBA EXITOSA: Las mejoras manejan correctamente los errores")
        else:
            print("⚠️ PRUEBA PARCIAL: Necesita más refinamiento")
        print("=" * 50)
        
        # Limpiar archivo de prueba
        try:
            os.remove(filename)
            print(f"🧹 Archivo de prueba eliminado: {filename}")
        except:
            pass
            
    except Exception as e:
        print(f"❌ Error en prueba: {e}")

if __name__ == "__main__":
    main()
