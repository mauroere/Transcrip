# Transcriptor de Audios

Una aplicación web para transcribir y analizar audios de llamadas de asesores comerciales y técnicos

## Características

- **Transcripción automática**: Utiliza Whisper AI para transcribir audios con alta precisión
- **Análisis de sentimientos**: Analiza el tono de las conversaciones
- **Métricas de calidad**: Identifica palabras clave relacionadas con el servicio al cliente
- **Dashboard analítico**: Visualiza estadísticas y tendencias
- **Múltiples formatos**: Soporta WAV, MP3, MP4, AVI, MOV, FLAC, M4A, OGG, WEBM
- **Interfaz intuitiva**: Drag & drop para subir archivos fácilmente

## Instalación

1. **Clona o descarga el proyecto**
   ```bash
   cd c:\Users\rementeriama\Downloads\Code\Transcrip
   ```

2. **Instala las dependencias**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecuta la aplicación**
   ```bash
   python app.py
   ```

4. **Abre tu navegador**
   - Ve a: `http://localhost:5000`

## Uso

### Transcribir Audios
1. Arrastra y suelta archivos de audio en la zona de carga
2. O haz clic para seleccionar archivos manualmente
3. Espera a que se complete la transcripción
4. Revisa los resultados y análisis

### Dashboard
- Ve al dashboard para ver estadísticas generales
- Analiza tendencias en palabras clave
- Revisa el sentimiento general de las llamadas
- Descarga transcripciones en formato JSON

## Análisis Incluidos

### Palabras Clave
- **Saludos**: "buenos días", "buenas tardes", "hola"
- **Agradecimientos**: "gracias", "agradezco"
- **Disculpas**: "disculpe", "perdón", "lo siento"
- **Servicios**: "servicio", "plan", "línea", "internet", "televisión"
- **Problemas Técnicos**: "problema", "falla", "error", "no funciona", "lento"

### Sentimientos
- **Positivo**: "excelente", "perfecto", "bien", "satisfecho"
- **Negativo**: "mal", "terrible", "molesto", "insatisfecho"

## Estructura del Proyecto

```
Transcrip/
├── app.py                 # Aplicación principal Flask
├── requirements.txt       # Dependencias Python
├── templates/
│   ├── index.html        # Página principal
│   └── dashboard.html    # Dashboard de análisis
├── uploads/              # Archivos temporales (se crea automáticamente)
├── transcriptions/       # Transcripciones guardadas (se crea automáticamente)
└── README.md            # Este archivo
```

## Requisitos del Sistema

- **Python 3.8+**
- **PyTorch** (para Whisper)
- **FFmpeg** (para procesamiento de audio)
- **Memoria RAM**: Mínimo 4GB recomendado
- **Espacio en disco**: Al menos 2GB libres

## Notas Técnicas

- Los archivos se convierten automáticamente a WAV para mejor compatibilidad
- El modelo Whisper "base" se descarga automáticamente en la primera ejecución
- Las transcripciones se guardan en formato JSON con metadatos completos
- La aplicación optimiza el audio (mono, 16kHz) para mejor rendimiento

## Solución de Problemas

### Error de instalación de PyTorch
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Error con FFmpeg
- Windows: Descarga FFmpeg desde https://ffmpeg.org/download.html
- Agrega FFmpeg al PATH del sistema

### Rendimiento lento
- Usa archivos de audio más cortos (< 10 minutos)
- Considera usar el modelo Whisper "tiny" para mayor velocidad

## Licencia
MIT
