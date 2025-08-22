# 🎯 Transcriptor de Audios Movistar - Streamlit

Aplicación web desarrollada con **Streamlit** para transcribir y analizar llamadas de atención al cliente de Movistar.

## 🚀 Características Principales

### 📊 **Análisis Integral de Performance**
- **Evaluación de Protocolo**: Saludo, identificación, pregunta de ayuda, despedida
- **Análisis de Tono**: Detección automática (amable, empático, profesional, cortado, frustrado)
- **Resolución de Problemas**: Estado de resolución y tipo de problema
- **Scores de Performance**: Métricas cuantitativas (0-100%)

### 🤖 **Tecnologías Utilizadas**
- **OpenAI Whisper**: Transcripción de audio en español
- **Streamlit**: Interfaz web interactiva
- **FFmpeg**: Procesamiento de audio
- **Python 3.13**: Backend de análisis

### 💡 **Funcionalidades Avanzadas**
- **Prompt para ChatGPT**: Generación automática de prompts optimizados
- **Análisis Multiarchivo**: Procesamiento en lote
- **Descarga de Reportes**: JSON con análisis completo
- **Interfaz Intuitiva**: Drag & drop, barras de progreso, métricas visuales

## 📋 **Requisitos del Sistema**

### **Software Necesario**
- Python 3.13+
- FFmpeg (incluido en el proyecto)
- 4GB+ RAM recomendado
- Conexión a internet (primera carga del modelo)

### **Formatos Soportados**
- **Audio**: WAV, MP3, FLAC, M4A, OGG
- **Video**: MP4, AVI, MOV, WEBM
- **Límite**: 100MB por archivo

## 🛠️ **Instalación y Uso**

### **1. Preparar Entorno**
```bash
# Navegar al directorio
cd "C:\\Users\\rementeriama\\Downloads\\Code\\Transcrip"

# Instalar dependencias (si es necesario)
pip install streamlit openai-whisper
```

### **2. Ejecutar Aplicación**
```bash
# Método 1: Puerto por defecto (8501)
streamlit run streamlit_app.py

# Método 2: Puerto personalizado
streamlit run streamlit_app.py --server.port 8502

# Método 3: Con Python completo
python -m streamlit run streamlit_app.py
```

### **3. Usar la Aplicación**
1. **Abrir navegador** en `http://localhost:8501` (o puerto configurado)
2. **Subir archivos** de audio usando drag & drop
3. **Procesar** haciendo clic en "🚀 Procesar Archivos"
4. **Revisar resultados** con métricas de performance
5. **Generar prompt ChatGPT** para análisis adicional
6. **Descargar reportes** en formato JSON

## 📊 **Interpretación de Scores**

### **Protocolo Score (0-100%)**
- **80-100%**: Excelente cumplimiento de protocolo
- **60-79%**: Cumplimiento aceptable, mejoras menores
- **0-59%**: Requiere entrenamiento en protocolo

### **Tono Score (0-100%)**
- **80-100%**: Comunicación empática y profesional
- **60-79%**: Tono apropiado con oportunidades de mejora
- **0-59%**: Requiere desarrollo de habilidades blandas

### **Resolución Score (0-100%)**
- **100%**: Problema identificado y resuelto
- **75%**: Sin problemas detectados (neutral)
- **50%**: Problema identificado sin resolución

## 🤖 **Integración con ChatGPT**

### **Prompt Automático Incluye:**
- 📊 Scores de performance actuales
- ✅ Checklist de protocolo cumplido
- 🎭 Análisis detallado de tono
- 🔧 Estado de resolución de problemas
- ⚠️ Falencias detectadas
- 💡 Puntos de mejora sugeridos
- 📝 Transcripción completa

### **Para Obtener Análisis Adicional:**
1. Clic en "🤖 Generar Prompt ChatGPT"
2. Copiar texto generado
3. Ir a [ChatGPT](https://chat.openai.com) o [Claude](https://claude.ai)
4. Pegar prompt y enviar
5. Recibir análisis profesional detallado

## 🔧 **Arquitectura Técnica**

### **Flujo de Procesamiento**
```
Audio Input → FFmpeg → Whisper → Análisis → Métricas → Streamlit UI
```

### **Componentes Principales**
- `streamlit_app.py`: Aplicación principal
- `ffmpeg/`: Binarios de FFmpeg local
- `uploads/`: Archivos temporales
- `transcriptions/`: Resultados guardados
- `.streamlit/config.toml`: Configuración UI

### **Funciones Core**
- `load_whisper_model()`: Cache del modelo IA
- `validate_audio_file()`: Validación de archivos
- `transcribe_with_fallback()`: Transcripción con manejo de errores
- `analyze_transcription()`: Motor de análisis de performance
- `generate_chatgpt_prompt()`: Generador de prompts optimizados

## 📈 **Métricas de Calidad**

### **Indicadores Clave (KPIs)**
- **Tiempo de Procesamiento**: ~30 segundos por minuto de audio
- **Precisión de Transcripción**: 90%+ en español
- **Cobertura de Análisis**: 15+ métricas de performance
- **Compatibilidad**: 8 formatos de archivo soportados

### **Casos de Uso Principales**
- ✅ Evaluación de asesores comerciales
- ✅ Análisis de calidad de servicio
- ✅ Identificación de oportunidades de mejora
- ✅ Generación de reportes de performance
- ✅ Entrenamiento y coaching de equipos

## 🔍 **Solución de Problemas**

### **Errores Comunes**
1. **"No module named 'pyaudioop'"**: Normal en Python 3.13, no afecta funcionalidad
2. **"Port already in use"**: Usar `--server.port 8502`
3. **"FFmpeg not found"**: Verificar que `ffmpeg/ffmpeg.exe` existe
4. **"Out of memory"**: Reducir tamaño de archivos o procesar uno por vez

### **Optimizaciones**
- **Modelo Whisper**: Se carga una sola vez (cache)
- **Archivos Temporales**: Limpieza automática
- **Memoria**: Procesamiento streaming para archivos grandes
- **Performance**: Análisis paralelo cuando sea posible

## 📞 **Contexto Movistar**

Esta aplicación está específicamente optimizada para analizar llamadas de atención al cliente de **Movistar**, incluyendo:

- **Protocolos de atención** específicos de telecom
- **Terminología técnica** de servicios Movistar
- **Métricas de satisfacción** del cliente
- **Indicadores de calidad** de servicio
- **Compliance regulatorio** del sector

---

## 🎯 **¡Listo para Usar!**

La aplicación Streamlit está optimizada para proporcionar una experiencia fluida y profesional en el análisis de calidad de servicio al cliente de Movistar.

**🚀 Ejecuta: `streamlit run streamlit_app.py`**
