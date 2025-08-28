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
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Análisis de Performance - Movistar",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Verificar compatibilidad de Python
python_version = sys.version_info

# Configuraciones y funciones auxiliares
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'avi', 'mov', 'flac', 'm4a', 'ogg', 'webm'}

# Crear directorios si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Verificar si el archivo tiene una extensión permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@st.cache_resource
def load_whisper_model():
    """Cargar modelo Whisper con cache"""
    if not WHISPER_AVAILABLE:
        return None
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"Error cargando Whisper: {e}")
        return None

def convert_audio_to_wav(file_path):
    """Convertir audio a WAV si es necesario"""
    if not PYDUB_AVAILABLE:
        return file_path
    
    try:
        audio = AudioSegment.from_file(file_path)
        wav_path = file_path.replace(os.path.splitext(file_path)[1], '.wav')
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        st.warning(f"No se pudo convertir el audio: {e}")
        return file_path

def transcribe_audio(model, file_path):
    """Transcribir audio con Whisper"""
    if not model:
        return None, "Modelo Whisper no disponible"
    
    try:
        result = model.transcribe(file_path, language="es")
        return result, None
    except Exception as e:
        return None, str(e)

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
        
        speaker_name = "ASESOR MOVISTAR" if current_speaker == 1 else "CLIENTE"
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
    
    prompt = f"""🎯 ANÁLISIS DE LLAMADA DE ATENCIÓN AL CLIENTE - MOVISTAR

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
Por favor analiza esta llamada de atención al cliente de Movistar y proporciona:

1. Análisis detallado del desempeño del asesor
2. Recomendaciones específicas para mejorar la atención
3. Evaluación de la satisfacción del cliente
4. Sugerencias de entrenamiento o coaching
5. Puntos positivos que el asesor debería mantener
6. Calificación general del 1-10 con justificación
7. Análisis de las dinámicas de conversación
8. Evaluación del equilibrio en la participación

Contexto: Somos Movistar y queremos mejorar la calidad de nuestro servicio al cliente.
"""
    return prompt

# APLICACIÓN PRINCIPAL
st.title("🎙️ Sistema de Análisis de Performance - Movistar")
st.markdown("### 📊 Análisis Profesional de Atención al Cliente")

# Sidebar con información del sistema
with st.sidebar:
    st.header("ℹ️ Estado del Sistema")
    
    if python_version >= (3, 10):
        st.success(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    st.info(f"✅ Streamlit funcionando")
    
    if WHISPER_AVAILABLE:
        st.success("✅ Whisper disponible")
    else:
        st.warning("⚠️ Whisper no disponible")
    
    if PYDUB_AVAILABLE:
        st.success("✅ Pydub disponible")
    else:
        st.warning("⚠️ Pydub no disponible")
    
    st.markdown("---")
    st.markdown("**Funcionalidades:**")
    st.markdown("• 📁 Subir archivos de audio")
    st.markdown("• 🎙️ Transcripción automática")
    st.markdown("• 💬 Diálogos por interlocutores")
    st.markdown("• 📊 Análisis de performance")
    st.markdown("• 🤖 Prompt para IA")

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📁 Subir Audio", "📝 Análisis Manual", "📊 Resultados"])

with tab1:
    st.header("📁 Dashboard de Subida de Audio")
    
    uploaded_file = st.file_uploader(
        "Selecciona un archivo de audio:",
        type=['wav', 'mp3', 'mp4', 'avi', 'mov', 'flac', 'm4a', 'ogg', 'webm'],
        help="Formatos soportados: WAV, MP3, MP4, AVI, MOV, FLAC, M4A, OGG, WEBM"
    )
    
    if uploaded_file is not None:
        # Mostrar información del archivo
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Nombre", uploaded_file.name)
        with col2:
            st.metric("📏 Tamaño", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
        with col3:
            st.metric("📋 Tipo", uploaded_file.type)
        
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        st.success("✅ Archivo subido correctamente")
        
        # Procesar transcripción
        if st.button("🎙️ Iniciar Transcripción Profesional", type="primary"):
            if WHISPER_AVAILABLE:
                with st.spinner("🔄 Procesando audio... Esto puede tomar unos minutos"):
                    # Cargar modelo
                    model = load_whisper_model()
                    
                    if model:
                        # Convertir audio si es necesario
                        wav_path = convert_audio_to_wav(temp_path)
                        
                        # Transcribir
                        result, error = transcribe_audio(model, wav_path)
                        
                        if result and not error:
                            st.session_state['transcription_result'] = result
                            st.session_state['audio_file'] = uploaded_file.name
                            st.success("✅ Transcripción completada")
                            st.rerun()
                        else:
                            st.error(f"❌ Error en transcripción: {error}")
                    else:
                        st.error("❌ No se pudo cargar el modelo Whisper")
            else:
                st.error("❌ Whisper no está disponible. Usa el análisis manual en la pestaña correspondiente.")

with tab2:
    st.header("📝 Análisis Manual de Texto")
    
    manual_text = st.text_area(
        "Ingresa el texto transcrito manualmente:",
        height=300,
        placeholder="Ejemplo: Hola, buenos días, habla con María de Movistar, en qué puedo ayudarle..."
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
st.markdown("*Sistema de Análisis de Performance para Call Center Movistar | Optimizado para Python 3.13*")
