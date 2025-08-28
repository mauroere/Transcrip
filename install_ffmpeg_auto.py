#!/usr/bin/env python3
"""
Instalador automático de FFmpeg para Windows
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

def install_ffmpeg():
    print("🔧 INSTALADOR DE FFMPEG PARA WINDOWS")
    print("=" * 50)
    
    # Detectar arquitectura
    import platform
    arch = platform.machine().lower()
    if arch in ['amd64', 'x86_64']:
        ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        print("✅ Detectado: Windows 64-bit")
    else:
        print("❌ Arquitectura no soportada")
        return False
    
    # Directorio de instalación
    install_dir = Path.home() / "ffmpeg"
    
    print(f"📁 Directorio de instalación: {install_dir}")
    
    # Crear directorio si no existe
    install_dir.mkdir(exist_ok=True)
    
    # Descargar FFmpeg
    zip_path = install_dir / "ffmpeg.zip"
    
    print("📥 Descargando FFmpeg...")
    try:
        urllib.request.urlretrieve(ffmpeg_url, zip_path)
        print("✅ Descarga completada")
    except Exception as e:
        print(f"❌ Error descargando: {e}")
        return False
    
    # Extraer archivo
    print("📦 Extrayendo archivos...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(install_dir)
        
        # Encontrar la carpeta extraída
        extracted_folders = [f for f in install_dir.iterdir() if f.is_dir() and f.name.startswith('ffmpeg')]
        if extracted_folders:
            ffmpeg_folder = extracted_folders[0]
            bin_folder = ffmpeg_folder / "bin"
            
            # Mover ejecutables al directorio principal
            if bin_folder.exists():
                for exe in bin_folder.glob("*.exe"):
                    shutil.move(str(exe), str(install_dir / exe.name))
                
                print("✅ Archivos extraídos correctamente")
            else:
                print("❌ No se encontró la carpeta bin")
                return False
        else:
            print("❌ No se encontró la carpeta de FFmpeg")
            return False
            
    except Exception as e:
        print(f"❌ Error extrayendo: {e}")
        return False
    
    # Limpiar archivos temporales
    try:
        zip_path.unlink()
        shutil.rmtree(ffmpeg_folder)
        print("🧹 Archivos temporales limpiados")
    except:
        pass
    
    # Verificar instalación
    ffmpeg_exe = install_dir / "ffmpeg.exe"
    if ffmpeg_exe.exists():
        print(f"✅ FFmpeg instalado en: {ffmpeg_exe}")
        
        # Agregar al PATH del sistema
        current_path = os.environ.get('PATH', '')
        if str(install_dir) not in current_path:
            print("\n🔧 CONFIGURACIÓN DEL PATH:")
            print("Para que FFmpeg funcione globalmente, agrega esta ruta al PATH del sistema:")
            print(f"   {install_dir}")
            print("\nO ejecuta este comando en PowerShell como Administrador:")
            print(f'   setx PATH "$env:PATH;{install_dir}" /M')
        
        return True
    else:
        print("❌ Error: FFmpeg no se instaló correctamente")
        return False

def test_ffmpeg():
    print("\n🧪 PROBANDO FFMPEG...")
    
    # Probar desde la ubicación local
    ffmpeg_local = Path.home() / "ffmpeg" / "ffmpeg.exe"
    
    if ffmpeg_local.exists():
        print(f"✅ FFmpeg encontrado en: {ffmpeg_local}")
        
        # Agregar al PATH temporal
        ffmpeg_dir = str(ffmpeg_local.parent)
        current_path = os.environ.get('PATH', '')
        os.environ['PATH'] = f"{ffmpeg_dir};{current_path}"
        
        # Probar pydub
        try:
            from pydub import AudioSegment
            print("✅ Pydub puede usar FFmpeg ahora")
            return True
        except Exception as e:
            print(f"⚠️  Pydub aún tiene problemas: {e}")
            return False
    else:
        print("❌ FFmpeg no encontrado")
        return False

if __name__ == "__main__":
    if install_ffmpeg():
        test_ffmpeg()
        print("\n🎉 ¡INSTALACIÓN COMPLETADA!")
        print("Reinicia la aplicación Streamlit para usar FFmpeg")
    else:
        print("\n❌ Instalación fallida")
        print("Soluciones alternativas:")
        print("1. Descargar manualmente desde: https://ffmpeg.org/download.html")
        print("2. Usar chocolatey: choco install ffmpeg")
        print("3. Usar winget: winget install ffmpeg")
