#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Diagnóstico Específico para Errores de Tensor
=======================================================
Este script diagnóstica el archivo específico que está causando errores.

Para usar: python diagnose_tensor_error.py "Ramallo leonardo 20-8 16.05hs.wav"

Desarrollado por: Mauro Rementeria - mauroere@gmail.com
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

def analyze_problematic_file(file_path):
    """Analizar específicamente el archivo que causa errores de tensor"""
    print(f"🔍 ANÁLISIS ESPECÍFICO: {os.path.basename(file_path)}")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        return False
    
    # Información básica
    file_size = os.path.getsize(file_path)
    print(f"📏 Tamaño: {file_size / 1024 / 1024:.2f} MB")
    
    # Análisis detallado con librosa
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        
        print("\n📊 ANÁLISIS CON LIBROSA:")
        print("-" * 30)
        
        # Metadatos del archivo
        info = sf.info(file_path)
        print(f"📋 Metadatos:")
        print(f"   • Formato: {info.format} ({info.subtype})")
        print(f"   • Canales: {info.channels}")
        print(f"   • Sample Rate: {info.samplerate} Hz")
        print(f"   • Duración: {info.duration:.2f} segundos ({info.duration/60:.1f} minutos)")
        print(f"   • Frames totales: {info.frames:,}")
        
        # Problemas potenciales identificados
        problems = []
        if info.channels > 2:
            problems.append(f"Demasiados canales ({info.channels})")
        if info.samplerate > 48000:
            problems.append(f"Sample rate muy alto ({info.samplerate} Hz)")
        if info.duration > 1200:  # 20 minutos
            problems.append(f"Archivo muy largo ({info.duration/60:.1f} minutos)")
        if info.frames > 50000000:  # 50M frames
            problems.append(f"Demasiados frames ({info.frames:,})")
            
        if problems:
            print("\n⚠️ PROBLEMAS POTENCIALES DETECTADOS:")
            for problem in problems:
                print(f"   • {problem}")
        
        # Cargar muestra para análisis
        print(f"\n🧪 CARGANDO MUESTRA DE AUDIO (10 segundos):")
        try:
            y, sr = librosa.load(file_path, sr=None, duration=10.0, mono=False)
            
            if y.ndim == 1:
                print(f"   • Audio mono cargado")
                y_analysis = y
            else:
                print(f"   • Audio estéreo cargado, shape: {y.shape}")
                y_analysis = librosa.to_mono(y)
            
            print(f"   • Shape final: {y_analysis.shape}")
            print(f"   • Dtype: {y_analysis.dtype}")
            print(f"   • Rango: [{y_analysis.min():.6f}, {y_analysis.max():.6f}]")
            print(f"   • RMS promedio: {np.sqrt(np.mean(y_analysis**2)):.6f}")
            
            # Detectar problemas en el audio
            if np.abs(y_analysis).max() < 0.001:
                problems.append("Audio muy silencioso")
            if np.isnan(y_analysis).any():
                problems.append("Contiene valores NaN")
            if np.isinf(y_analysis).any():
                problems.append("Contiene valores infinitos")
                
        except Exception as e:
            print(f"   ❌ Error al cargar muestra: {e}")
            problems.append(f"Error de carga: {e}")
            
    except ImportError:
        print("⚠️ Librosa no disponible para análisis detallado")
    
    # Pruebas específicas con Whisper
    print(f"\n🤖 PRUEBAS CON WHISPER:")
    print("-" * 30)
    
    try:
        import whisper
        
        # Cargar modelo tiny para pruebas rápidas
        print("🔄 Cargando modelo Whisper (tiny)...")
        model = whisper.load_model("tiny")
        print("✅ Modelo cargado")
        
        # Configuraciones específicas para errores de tensor
        test_configs = [
            {"name": "Configuración Vacía", "config": {}},
            {"name": "Solo Idioma", "config": {"language": "es"}},
            {"name": "Sin FP16", "config": {"language": "es", "fp16": False}},
            {"name": "Temperatura 0", "config": {"language": "es", "temperature": 0.0, "fp16": False}},
            {"name": "Sin Beam Search", "config": {"language": "es", "beam_size": 1, "fp16": False}},
            {"name": "Task Explícita", "config": {"language": "es", "task": "transcribe", "fp16": False}},
        ]
        
        print(f"\n🧪 PROBANDO {len(test_configs)} CONFIGURACIONES:")
        
        results = []
        tensor_error_count = 0
        
        for i, test in enumerate(test_configs, 1):
            print(f"\n   {i}. {test['name']}:")
            try:
                # Intentar transcripción
                result = model.transcribe(file_path, **test['config'])
                text = result.get("text", "").strip()
                
                if text:
                    print(f"      ✅ ÉXITO: '{text[:40]}...'")
                    results.append({"config": test['name'], "status": "success", "text": text[:50]})
                else:
                    print(f"      ⚠️ Sin texto generado")
                    results.append({"config": test['name'], "status": "empty", "text": ""})
                    
            except Exception as e:
                error_msg = str(e)
                print(f"      ❌ ERROR: {error_msg[:60]}...")
                
                if "tensor" in error_msg.lower() and "size" in error_msg.lower():
                    print(f"      🔥 ERROR DE TENSOR CONFIRMADO")
                    tensor_error_count += 1
                    results.append({"config": test['name'], "status": "tensor_error", "error": error_msg})
                else:
                    results.append({"config": test['name'], "status": "other_error", "error": error_msg})
        
        # Resumen de resultados
        print(f"\n📊 RESUMEN DE PRUEBAS:")
        print(f"   • Configuraciones exitosas: {len([r for r in results if r['status'] == 'success'])}")
        print(f"   • Errores de tensor: {tensor_error_count}")
        print(f"   • Otros errores: {len([r for r in results if r['status'] == 'other_error'])}")
        print(f"   • Sin texto: {len([r for r in results if r['status'] == 'empty'])}")
        
        if tensor_error_count > 0:
            print(f"\n🚨 CONFIRMADO: ARCHIVO CON PROBLEMAS DE TENSOR")
            return False
        elif len([r for r in results if r['status'] == 'success']) > 0:
            print(f"\n✅ El archivo funciona con algunas configuraciones")
            return True
        else:
            print(f"\n❌ El archivo no funciona con ninguna configuración")
            return False
            
    except Exception as e:
        print(f"❌ Error crítico con Whisper: {e}")
        return False

def generate_fix_commands(file_path):
    """Generar comandos específicos para arreglar el archivo"""
    print(f"\n🛠️ COMANDOS PARA ARREGLAR EL ARCHIVO:")
    print("=" * 40)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_name = f"{base_name}_fixed.wav"
    
    print(f"💡 Ejecuta CUALQUIERA de estos comandos para arreglar el archivo:")
    print()
    
    print(f"1️⃣ CONVERSIÓN BÁSICA (Recomendado):")
    print(f"   ffmpeg -i \"{file_path}\" -ar 16000 -ac 1 -c:a pcm_s16le \"{output_name}\"")
    print()
    
    print(f"2️⃣ CONVERSIÓN + LIMPIEZA:")
    print(f"   ffmpeg -i \"{file_path}\" -ar 16000 -ac 1 -af \"highpass=f=80\" \"{output_name}\"")
    print()
    
    print(f"3️⃣ DIVISIÓN EN SEGMENTOS (si es muy largo):")
    print(f"   ffmpeg -i \"{file_path}\" -f segment -segment_time 300 -ar 16000 -ac 1 \"{base_name}_part%03d.wav\"")
    print()
    
    print(f"4️⃣ CONVERSIÓN A MP3 (alternativa):")
    print(f"   ffmpeg -i \"{file_path}\" -ar 22050 -ac 1 -b:a 64k \"{base_name}_fixed.mp3\"")
    print()
    
    print("📋 DESPUÉS DE CONVERTIR:")
    print("   • Usa el archivo convertido en la aplicación")
    print("   • Debería procesar sin errores de tensor")
    print("   • Si sigue fallando, intenta con los segmentos divididos")

def main():
    """Función principal"""
    print("🔍 DIAGNÓSTICO ESPECÍFICO DE ERRORES DE TENSOR")
    print("=" * 50)
    print("Desarrollado por: Mauro Rementeria - mauroere@gmail.com")
    print()
    
    # Verificar argumentos
    if len(sys.argv) != 2:
        print("❌ Uso incorrecto")
        print("💡 Arrastra el archivo al script o usa:")
        print("   python diagnose_tensor_error.py \"nombre_archivo.wav\"")
        print()
        print("📁 Para el archivo específico que tienes:")
        print("   python diagnose_tensor_error.py \"Ramallo leonardo 20-8 16.05hs.wav\"")
        return
    
    file_path = sys.argv[1]
    
    # Verificar que existe
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        print()
        
        # Buscar archivos similares
        directory = os.path.dirname(file_path) if os.path.dirname(file_path) else "."
        print(f"🔍 Archivos de audio en {directory}:")
        try:
            files = [f for f in os.listdir(directory) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]
            for f in files:
                print(f"   📄 {f}")
        except:
            print("   No se pudieron listar archivos")
        return
    
    # Ejecutar análisis
    print(f"🎯 Analizando archivo específico que causa errores...")
    print()
    
    success = analyze_problematic_file(file_path)
    
    # Siempre mostrar comandos de arreglo
    generate_fix_commands(file_path)
    
    # Conclusión
    print("\n" + "=" * 60)
    if success:
        print("✅ CONCLUSIÓN: El archivo puede funcionar con configuraciones específicas")
        print("💡 Intenta procesar de nuevo en la aplicación")
    else:
        print("🚨 CONCLUSIÓN: El archivo necesita ser convertido antes de usar")
        print("💡 Ejecuta uno de los comandos de arriba para arreglarlo")
    print("=" * 60)

if __name__ == "__main__":
    main()
