"""
Script para generar un archivo de audio de prueba
"""
import numpy as np
import wave
import struct

def create_test_audio():
    """Crea un archivo WAV de prueba con una señal sinusoidal"""
    
    # Parámetros del audio
    sample_rate = 16000  # 16 kHz
    duration = 3  # 3 segundos
    frequency = 440  # La nota A4
    
    # Generar señal sinusoidal
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave_data = np.sin(2 * np.pi * frequency * t)
    
    # Normalizar y convertir a int16
    wave_data = (wave_data * 32767).astype(np.int16)
    
    # Guardar archivo WAV
    filename = "test_audio.wav"
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes por muestra (16 bits)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(wave_data.tobytes())
    
    print(f"Archivo de prueba creado: {filename}")
    print(f"Duración: {duration} segundos")
    print(f"Frecuencia de muestreo: {sample_rate} Hz")
    
    return filename

if __name__ == "__main__":
    try:
        import numpy as np
        create_test_audio()
    except ImportError:
        print("NumPy no está instalado. Instalando...")
        import subprocess
        subprocess.check_call(["pip", "install", "numpy"])
        import numpy as np
        create_test_audio()
