"""
Instalador automático de FFmpeg para Windows
"""
import os
import subprocess
import sys
import urllib.request
import zipfile
import shutil

def check_ffmpeg():
    """Verifica si FFmpeg está disponible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg ya está instalado")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ FFmpeg no encontrado")
    return False

def download_ffmpeg():
    """Descarga e instala FFmpeg portable"""
    print("📥 Descargando FFmpeg...")
    
    # URL de FFmpeg builds para Windows
    ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    
    try:
        # Crear directorio para FFmpeg
        ffmpeg_dir = "ffmpeg"
        if os.path.exists(ffmpeg_dir):
            shutil.rmtree(ffmpeg_dir)
        os.makedirs(ffmpeg_dir)
        
        # Descargar archivo
        zip_path = os.path.join(ffmpeg_dir, "ffmpeg.zip")
        print("Descargando... (esto puede tomar unos minutos)")
        urllib.request.urlretrieve(ffmpeg_url, zip_path)
        
        # Extraer
        print("Extrayendo archivos...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ffmpeg_dir)
        
        # Buscar el ejecutable
        for root, dirs, files in os.walk(ffmpeg_dir):
            if 'ffmpeg.exe' in files:
                ffmpeg_exe = os.path.join(root, 'ffmpeg.exe')
                # Mover a directorio principal
                final_path = os.path.join(ffmpeg_dir, 'ffmpeg.exe')
                shutil.copy2(ffmpeg_exe, final_path)
                print(f"✅ FFmpeg instalado en: {os.path.abspath(final_path)}")
                
                # Limpiar archivos temporales
                os.remove(zip_path)
                
                return os.path.abspath(final_path)
        
        print("❌ No se pudo encontrar ffmpeg.exe en el archivo descargado")
        return None
        
    except Exception as e:
        print(f"❌ Error descargando FFmpeg: {e}")
        return None

def add_ffmpeg_to_path(ffmpeg_path):
    """Agrega FFmpeg al PATH de la sesión actual"""
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    current_path = os.environ.get('PATH', '')
    
    if ffmpeg_dir not in current_path:
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + current_path
        print(f"✅ FFmpeg agregado al PATH de la sesión")

def manual_instructions():
    """Proporciona instrucciones manuales para instalar FFmpeg"""
    print("\n" + "="*60)
    print("📋 INSTRUCCIONES MANUALES PARA INSTALAR FFMPEG")
    print("="*60)
    print()
    print("OPCIÓN 1 - Usando Chocolatey (Recomendado):")
    print("1. Abre PowerShell como Administrador")
    print("2. Ejecuta: Set-ExecutionPolicy Bypass -Scope Process -Force")
    print("3. Ejecuta: iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))")
    print("4. Ejecuta: choco install ffmpeg")
    print()
    print("OPCIÓN 2 - Descarga manual:")
    print("1. Ve a: https://ffmpeg.org/download.html")
    print("2. Descarga 'Windows builds from gyan.dev'")
    print("3. Extrae el archivo ZIP")
    print("4. Agrega la carpeta 'bin' al PATH del sistema")
    print()
    print("OPCIÓN 3 - Usando winget:")
    print("1. Abre PowerShell")
    print("2. Ejecuta: winget install FFmpeg")
    print()
    print("Después de instalar, reinicia VS Code y ejecuta la aplicación nuevamente.")
    print("="*60)

def main():
    """Función principal"""
    print("🔧 CONFIGURADOR DE FFMPEG PARA TRANSCRIPTOR MOVISTAR")
    print("="*55)
    
    if check_ffmpeg():
        print("✅ Todo está listo para usar!")
        return True
    
    print("\n🛠️ FFmpeg es necesario para procesar archivos de audio.")
    print("¿Deseas intentar la instalación automática? (s/n): ", end="")
    
    try:
        response = input().lower().strip()
        if response in ['s', 'si', 'y', 'yes']:
            ffmpeg_path = download_ffmpeg()
            if ffmpeg_path:
                add_ffmpeg_to_path(ffmpeg_path)
                
                # Verificar nuevamente
                if check_ffmpeg():
                    print("\n🎉 ¡FFmpeg instalado correctamente!")
                    print("✅ Ahora puedes usar la aplicación de transcripción")
                    return True
                else:
                    print("\n⚠️ La instalación automática no funcionó completamente")
            
            manual_instructions()
        else:
            manual_instructions()
    
    except KeyboardInterrupt:
        print("\n\n❌ Instalación cancelada")
        manual_instructions()
    
    return False

if __name__ == "__main__":
    main()
