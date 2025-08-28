import streamlit as st
import os
import sys
import traceback
import tempfile
import json
from datetime import datetime
import uuid
import warnings
import io
import time
from pathlib import Path
import pandas as pd
from io import BytesIO
import re

# Importaciones para audio
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError as e:
    WHISPER_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError as e:
    PYDUB_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Análisis de Performance",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Verificar compatibilidad de Python
python_version = sys.version_info

def check_ffmpeg_available():
    """Verificar si FFmpeg está disponible silenciosamente"""
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except:
        return False

def check_audio_format(filename):
    """Verificar formato de audio y dar recomendaciones"""
    ext = filename.lower().split('.')[-1]
    
    format_info = {
        'wav': {'compatible': True, 'note': 'Formato óptimo - Compatible al 100%'},
        'mp3': {'compatible': True, 'note': 'Buena compatibilidad - Requiere FFmpeg para conversión'},
        'mp4': {'compatible': True, 'note': 'Buena compatibilidad - Requiere FFmpeg'},
        'm4a': {'compatible': True, 'note': 'Buena compatibilidad - Requiere FFmpeg'},
        'flac': {'compatible': True, 'note': 'Alta calidad - Requiere FFmpeg'},
        'ogg': {'compatible': True, 'note': 'Buena compatibilidad - Requiere FFmpeg'},
        'webm': {'compatible': False, 'note': 'Formato complejo - Convertir a WAV recomendado'},
        'avi': {'compatible': False, 'note': 'Formato de video - Extraer audio primero'}
    }
    
    return format_info.get(ext, {'compatible': False, 'note': 'Formato no reconocido - Usar WAV'})

# Configuraciones y funciones auxiliares
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'avi', 'mov', 'flac', 'm4a', 'ogg', 'webm'}

# Crear directorios si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Verificar si el archivo tiene una extensión permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_whisper_model_no_cache():
    """Cargar modelo Whisper sin cache para debugging"""
    if not WHISPER_AVAILABLE:
        return None
    
    try:
        import whisper
        # Intentar modelos de menor a mayor
        models_to_try = ["tiny", "base", "small"]
        
        for model_name in models_to_try:
            try:
                st.info(f"Intentando cargar modelo '{model_name}'...")
                model = whisper.load_model(model_name)
                st.success(f"✅ Modelo '{model_name}' cargado exitosamente")
                return model
            except Exception as e:
                st.warning(f"⚠️ Modelo '{model_name}' falló: {str(e)}")
                continue
        
        return None
    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        return None

@st.cache_resource
def load_whisper_model():
    """Cargar modelo Whisper con cache y manejo robusto de errores"""
    if not WHISPER_AVAILABLE:
        return None
    
    try:
        with st.spinner("🔄 Cargando modelo Whisper (puede tomar unos momentos la primera vez)..."):
            # Usar modelo tiny para pruebas rápidas, base para producción
            model = whisper.load_model("tiny")  # Cambiar a "base" para mejor calidad
            return model
    except Exception as e:
        st.error(f"Error específico cargando Whisper: {str(e)}")
        
        # Intentar con modelo más pequeño
        try:
            st.info("Intentando con modelo más ligero...")
            model = whisper.load_model("tiny")
            st.success("✅ Modelo ligero cargado exitosamente")
            return model
        except Exception as e2:
            st.error(f"Error crítico: {str(e2)}")
            
            # Diagnóstico adicional
            st.warning("🔍 **Diagnóstico del problema:**")
            st.markdown("• Problema de conectividad para descargar modelo")
            st.markdown("• Espacio insuficiente en disco")
            st.markdown("• Permisos de escritura en el directorio")
            
            return None

def convert_audio_to_wav(file_path):
    """Convertir audio a WAV si es necesario"""
    # Verificar si ya es WAV
    if file_path.lower().endswith('.wav'):
        return file_path
    
    if not PYDUB_AVAILABLE:
        st.warning("⚠️ Conversión de audio no disponible. Sube un archivo WAV directamente.")
        return file_path
    
    try:
        # Intentar conversión con pydub
        audio = AudioSegment.from_file(file_path)
        wav_path = file_path.replace(os.path.splitext(file_path)[1], '.wav')
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        st.info(f"💡 No se pudo convertir el audio automáticamente. Usa archivos WAV para mejores resultados.")
        # Intentar usar el archivo original directamente con Whisper
        return file_path

def transcribe_audio(model, file_path):
    """Transcribir audio con Whisper - versión robusta"""
    if not model:
        return None, "Modelo Whisper no disponible"
    
    try:
        # Whisper puede manejar muchos formatos sin conversión
        # Usar fp16=False para compatibilidad con CPU
        result = model.transcribe(
            file_path, 
            language="es", 
            fp16=False,
            verbose=False
        )
        return result, None
    except FileNotFoundError as e:
        if "ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower():
            return None, "FFmpeg no está instalado. Usar archivos WAV o instalar FFmpeg."
        return None, f"Archivo no encontrado: {str(e)}"
    except Exception as e:
        error_msg = str(e)
        
        # Manejo específico de errores comunes
        if "ffmpeg" in error_msg.lower():
            return None, "Error: FFmpeg requerido para este formato. Usar archivos WAV."
        elif "decode" in error_msg.lower():
            return None, "Error: Formato de audio no soportado. Usar WAV, MP3 o M4A."
        elif "permission" in error_msg.lower():
            return None, "Error: Sin permisos para acceder al archivo."
        else:
            return None, f"Error de transcripción: {error_msg}"

def format_dialogue(segments):
    """Formatear segmentos como diálogo"""
    dialogue = []
    current_speaker = 1
    last_end = 0
    
    for i, segment in enumerate(segments):
        start = segment['start']
        text = segment['text'].strip()
        
        # Simple detección de cambio de speaker basada en pausas
        if start - last_end > 2.0:  # Pausa de más de 2 segundos
            current_speaker = 2 if current_speaker == 1 else 1
        
        speaker_name = "ASESOR" if current_speaker == 1 else "CLIENTE"
        timestamp = f"{int(start//60):02d}:{int(start%60):02d}"
        
        dialogue.append({
            'timestamp': timestamp,
            'speaker': speaker_name,
            'text': text
        })
        
        last_end = segment['end']
    
    return dialogue

def analyze_performance(text):
    """Análisis completo de performance del asesor"""
    text_lower = text.lower()
    
    # Análisis de protocolo
    protocol_analysis = {
        "saludo_inicial": any(saludo in text_lower for saludo in [
            "hola", "buenos días", "buenas tardes", "buenas noches", 
            "bienvenido", "gracias por contactar", "habla con"
        ]),
        "identificacion": any(id_phrase in text_lower for id_phrase in [
            "su nombre", "me puede dar", "necesito sus datos", "podría confirmar",
            "su número de documento", "su dni", "su identificación"
        ]),
        "pregunta_ayuda": any(ayuda in text_lower for ayuda in [
            "en qué puedo ayudar", "cómo puedo ayudar", "cuál es su consulta",
            "qué necesita", "en qué le puedo asistir", "motivo de su llamada"
        ]),
        "despedida": any(despedida in text_lower for despedida in [
            "que tenga buen día", "gracias por contactar", "hasta luego",
            "que esté bien", "nos vemos", "chau", "adiós"
        ])
    }
    
    protocol_score = (sum(protocol_analysis.values()) / len(protocol_analysis)) * 100
    
    # Análisis de tono
    tono_positivo = sum([text_lower.count(x) for x in ['por favor', 'con gusto', 'perfecto', 'excelente']])
    tono_negativo = sum([text_lower.count(x) for x in ['no puede', 'imposible', 'no funciona']])
    
    if tono_positivo + tono_negativo > 0:
        tono_score = (tono_positivo / (tono_positivo + tono_negativo)) * 100
    else:
        tono_score = 50
    
    return {
        'protocol_analysis': protocol_analysis,
        'protocol_score': protocol_score,
        'tone_score': tono_score,
        'overall_score': (protocol_score + tono_score) / 2
    }

def generate_ai_prompt(dialogue, analysis, transcription):
    """Generar prompt optimizado para IA"""
    dialogue_text = "\n".join([
        f"[{item['timestamp']}] {item['speaker']}: {item['text']}"
        for item in dialogue
    ])
    
    prompt = f"""🎯 ANÁLISIS DE LLAMADA DE ATENCIÓN AL CLIENTE

📊 SCORES AUTOMÁTICOS:
• Protocolo: {analysis['protocol_score']:.1f}%
• Tono: {analysis['tone_score']:.1f}%
• General: {analysis['overall_score']:.1f}%

📋 PROTOCOLO EVALUADO:
• Saludo inicial: {'✓ SÍ' if analysis['protocol_analysis']['saludo_inicial'] else '✗ NO'}
• Identificación: {'✓ SÍ' if analysis['protocol_analysis']['identificacion'] else '✗ NO'}
• Pregunta de ayuda: {'✓ SÍ' if analysis['protocol_analysis']['pregunta_ayuda'] else '✗ NO'}
• Despedida: {'✓ SÍ' if analysis['protocol_analysis']['despedida'] else '✗ NO'}

💬 DIÁLOGO POR INTERLOCUTORES:
{dialogue_text}

📝 TRANSCRIPCIÓN COMPLETA:
"{transcription}"

🤖 SOLICITUD PARA IA:
Por favor analiza esta llamada de atención al cliente y proporciona:

1. Análisis detallado del desempeño del asesor
2. Recomendaciones específicas para mejorar la atención
3. Evaluación de la satisfacción del cliente
4. Sugerencias de entrenamiento o coaching
5. Puntos positivos que el asesor debería mantener
6. Calificación general del 1-10 con justificación
7. Análisis de las dinámicas de conversación
8. Evaluación del equilibrio en la participación

Contexto: Análisis de calidad de servicio al cliente para mejorar la atención.
"""
    return prompt

# APLICACIÓN PRINCIPAL
st.title("🎙️ Sistema de Análisis de Performance")
st.markdown("### 📊 Análisis Profesional de Atención al Cliente")

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📁 Subir Audio", "📝 Análisis Manual", "📊 Resultados"])

with tab1:
    st.header("📁 Dashboard de Subida de Audio")
    
    # Verificar estado de las dependencias
    ffmpeg_status = check_ffmpeg_available()
    
    if not ffmpeg_status:
        st.info("💡 **Estado del sistema:** FFmpeg no detectado. Archivos WAV funcionarán perfectamente.")
    
    # Botón de diagnóstico
    if st.button("🔍 Ejecutar Diagnóstico del Sistema"):
        with st.expander("📋 Resultados del Diagnóstico", expanded=True):
            # Verificar Whisper
            if WHISPER_AVAILABLE:
                st.success("✅ Whisper disponible")
                try:
                    import whisper
                    st.write(f"   Versión: {whisper.__version__}")
                    
                    # Intentar cargar modelo
                    try:
                        with st.spinner("Probando carga de modelo..."):
                            test_model = whisper.load_model("tiny")
                        st.success("✅ Modelo Whisper carga correctamente")
                    except Exception as e:
                        st.error(f"❌ Error cargando modelo: {e}")
                except Exception as e:
                    st.error(f"❌ Error con Whisper: {e}")
            else:
                st.error("❌ Whisper no disponible")
            
            # Verificar FFmpeg
            if ffmpeg_status:
                st.success("✅ FFmpeg disponible")
            else:
                st.warning("⚠️ FFmpeg no disponible")
            
            # Verificar Pydub
            if PYDUB_AVAILABLE:
                st.success("✅ Pydub disponible")
            else:
                st.warning("⚠️ Pydub no disponible")
            
            # Información del sistema
            import sys
            st.info(f"🐍 Python: {sys.version}")
            
            # Verificar espacio en disco
            import shutil
            free_space = shutil.disk_usage('.').free / (1024**3)
            st.info(f"💾 Espacio libre: {free_space:.1f} GB")
            
            if free_space < 1:
                st.warning("⚠️ Poco espacio en disco - puede afectar descarga de modelos")
            
            # Botón para limpiar cache
            if st.button("🧹 Limpiar Cache de Streamlit"):
                st.cache_resource.clear()
                st.success("✅ Cache limpiada - recarga la página")
    
    uploaded_file = st.file_uploader(
        "Selecciona un archivo de audio:",
        type=['wav', 'mp3', 'mp4', 'avi', 'mov', 'flac', 'm4a', 'ogg', 'webm'],
        help="Formatos soportados: WAV, MP3, MP4, AVI, MOV, FLAC, M4A, OGG, WEBM"
    )
    
    if uploaded_file is not None:
        # Verificar formato de archivo
        format_info = check_audio_format(uploaded_file.name)
        
        # Mostrar información del archivo
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Nombre", uploaded_file.name)
        with col2:
            st.metric("📏 Tamaño", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
        with col3:
            st.metric("📋 Tipo", uploaded_file.type)
        
        # Mostrar compatibilidad del formato
        if format_info['compatible']:
            st.success(f"✅ **Formato compatible:** {format_info['note']}")
        else:
            st.warning(f"⚠️ **Formato problemático:** {format_info['note']}")
            st.info("💡 **Recomendación:** Convierte el archivo a WAV para mejor compatibilidad")
        
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        st.success("✅ Archivo subido correctamente")
        
        # Procesar transcripción
        if WHISPER_AVAILABLE:
            # Botón inteligente según el formato
            if uploaded_file.name.lower().endswith('.wav'):
                button_text = "🎙️ Transcribir Audio WAV (Recomendado)"
                button_type = "primary"
            else:
                button_text = "🎙️ Intentar Transcripción (Puede requerir FFmpeg)"
                button_type = "secondary"
                
            if st.button(button_text, type=button_type):
                with st.spinner("🔄 Procesando audio... Esto puede tomar unos minutos"):
                    # Cargar modelo
                    model = load_whisper_model()
                    
                    # Si falla con cache, intentar sin cache
                    if not model:
                        st.info("🔄 Intentando carga alternativa sin cache...")
                        model = load_whisper_model_no_cache()
                    
                    if model:
                        # Intentar transcripción directa
                        result, error = transcribe_audio(model, temp_path)
                        
                        if result and not error:
                            st.session_state['transcription_result'] = result
                            st.session_state['audio_file'] = uploaded_file.name
                            st.success("✅ Transcripción completada exitosamente")
                            st.rerun()
                        else:
                            st.error(f"❌ {error}")
                            
                            # Mostrar soluciones específicas según el error
                            if "FFmpeg" in error:
                                st.info("🔧 **Soluciones para FFmpeg:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Opción 1: Instalar FFmpeg**")
                                    st.code("winget install ffmpeg")
                                    st.markdown("Luego reinicia VS Code")
                                with col2:
                                    st.markdown("**Opción 2: Usar WAV**")
                                    st.markdown("Convierte tu archivo a WAV:")
                                    st.markdown("• [Online-Convert.com](https://audio.online-convert.com/es/convertir-a-wav)")
                                    st.markdown("• [CloudConvert.com](https://cloudconvert.com/mp3-to-wav)")
                            
                            elif "formato" in error.lower():
                                st.info("📄 **Solución de formato:**")
                                st.markdown("• Convierte el archivo a **WAV** o **MP3**")
                                st.markdown("• Usa herramientas como Audacity o convertidores online")
                            
                            st.markdown("**💡 Mientras tanto, puedes usar el análisis manual en la siguiente pestaña**")
                    else:
                        st.error("❌ No se pudo cargar el modelo Whisper")
                        st.info("🔄 Reinicia la aplicación si el problema persiste")
        else:
            st.info("💡 **Transcripción automática disponible**")
            st.markdown("**Notas importantes:**")
            st.markdown("• Whisper puede procesar la mayoría de formatos de audio directamente")
            st.markdown("• Para mejor compatibilidad, usa archivos WAV")
            st.markdown("• Si hay errores, instala FFmpeg: `winget install ffmpeg`")
            st.markdown("• **Formatos soportados:** MP3, WAV, MP4, M4A, FLAC, OGG")
            
            if st.button("🎙️ Transcribir Audio Directamente", type="primary"):
                with st.spinner("🔄 Procesando audio... Esto puede tomar unos minutos"):
                    # Cargar modelo
                    model = load_whisper_model()
                    
                    if model:
                        # Intentar transcripción directa (Whisper maneja muchos formatos)
                        result, error = transcribe_audio(model, temp_path)
                        
                        if result and not error:
                            st.session_state['transcription_result'] = result
                            st.session_state['audio_file'] = uploaded_file.name
                            st.success("✅ Transcripción completada")
                            st.rerun()
                        else:
                            st.error(f"❌ {error}")
                            st.info("💡 **Soluciones:**")
                            st.markdown("1. Instala FFmpeg: `winget install ffmpeg`")
                            st.markdown("2. Convierte tu audio a WAV usando un convertidor online")
                            st.markdown("3. Usa el análisis manual con texto transcrito")
                    else:
                        st.error("❌ No se pudo cargar el modelo Whisper")

with tab2:
    st.header("📝 Análisis Manual de Texto")
    
    manual_text = st.text_area(
        "Ingresa el texto transcrito manualmente:",
        height=300,
        placeholder="Ejemplo: Hola, buenos días, habla con María del departamento de atención al cliente, en qué puedo ayudarle..."
    )
    
    if manual_text and st.button("🔍 Analizar Texto Manual"):
        # Crear resultado manual
        fake_segments = [{"start": 0, "end": len(manual_text)/10, "text": manual_text}]
        result = {
            "text": manual_text,
            "segments": fake_segments
        }
        st.session_state['transcription_result'] = result
        st.session_state['audio_file'] = "Texto Manual"
        st.success("✅ Texto procesado para análisis")
        st.rerun()

with tab3:
    st.header("📊 Resultados del Análisis")
    
    if 'transcription_result' in st.session_state:
        result = st.session_state['transcription_result']
        filename = st.session_state.get('audio_file', 'Audio')
        
        st.info(f"📁 **Archivo procesado:** {filename}")
        
        # Extraer texto y segmentos
        transcription = result['text']
        segments = result.get('segments', [])
        
        # Formatear como diálogo
        dialogue = format_dialogue(segments)
        
        # Análisis de performance
        analysis = analyze_performance(transcription)
        
        # Mostrar métricas
        st.subheader("📊 Métricas de Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Protocolo", f"{analysis['protocol_score']:.1f}%")
        with col2:
            st.metric("🎭 Tono", f"{analysis['tone_score']:.1f}%")
        with col3:
            st.metric("🎯 General", f"{analysis['overall_score']:.1f}%")
        with col4:
            st.metric("📝 Palabras", len(transcription.split()))
        
        # Mostrar diálogo formateado
        st.subheader("💬 Diálogo por Interlocutores")
        
        # Crear visualización tipo chat
        for item in dialogue:
            if "ASESOR" in item['speaker']:
                st.markdown(f"""
                <div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0; border-left: 4px solid #2196f3;'>
                    <strong>🎧 {item['speaker']} [{item['timestamp']}]:</strong><br>
                    {item['text']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #f3e5f5; padding: 10px; border-radius: 10px; margin: 5px 0; border-left: 4px solid #9c27b0;'>
                    <strong>👤 {item['speaker']} [{item['timestamp']}]:</strong><br>
                    {item['text']}
                </div>
                """, unsafe_allow_html=True)
        
        # Protocolo de atención
        st.subheader("📋 Evaluación de Protocolo")
        protocol_labels = {
            'saludo_inicial': 'Saludo inicial',
            'identificacion': 'Solicitud de identificación',
            'pregunta_ayuda': 'Pregunta de ayuda',
            'despedida': 'Despedida profesional'
        }
        
        for key, value in analysis['protocol_analysis'].items():
            label = protocol_labels.get(key, key)
            status = '✅ CUMPLIDO' if value else '❌ NO CUMPLIDO'
            st.write(f"**{label}:** {status}")
        
        # Prompt para IA
        st.subheader("🤖 Prompt para Inteligencia Artificial")
        ai_prompt = generate_ai_prompt(dialogue, analysis, transcription)
        
        # Mostrar prompt en área de texto copiable
        st.text_area(
            "Copia este texto completo y pégalo en ChatGPT, Claude, o cualquier IA:",
            value=ai_prompt,
            height=400,
            help="Selecciona todo el texto (Ctrl+A) y cópialo (Ctrl+C) para usar en tu IA favorita"
        )
        
        # Botón para limpiar
        if st.button("🗑️ Limpiar Resultados"):
            del st.session_state['transcription_result']
            if 'audio_file' in st.session_state:
                del st.session_state['audio_file']
            st.rerun()
    
    else:
        st.info("👆 Sube un archivo de audio en la primera pestaña o ingresa texto manualmente en la segunda pestaña para ver los resultados aquí.")

# Footer
st.markdown("---")
st.markdown("*Sistema de Análisis de Performance para Call Center | Optimizado para Python 3.13*")
