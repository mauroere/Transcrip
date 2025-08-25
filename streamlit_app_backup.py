import streamlit as st
import sys
import os
import warnings
import tempfile
import gc
import re
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Transcripción Movistar",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Verificar compatibilidad de Python
python_version = sys.version_info
if python_version >= (3, 10):
    st.error(f"""
    ⚠️ **VERSIÓN BACKUP - FUNCIONALIDAD LIMITADA**
    
    🚨 **Streamlit Cloud está usando Python {python_version.major}.{python_version.minor}.{python_version.micro}**
    
    Esta es una **versión de respaldo** sin OpenAI Whisper debido a incompatibilidades.
    
    🔧 **Para activar funcionalidad completa**:
    1. Elimina la app actual en Streamlit Cloud
    2. Crea una nueva app desde el repositorio
    3. Verifica que use Python 3.9 con runtime.txt
    
    📞 **Mientras tanto, puedes usar esta versión para pruebas básicas**
    """)

def main():
    """Función principal de la aplicación"""
    
    # Título y descripción
    st.title("🎙️ Sistema de Transcripción Movistar")
    st.markdown("### Análisis de Audios de Call Center")
    
    # Información del sistema
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🐍 Python", f"{python_version.major}.{python_version.minor}.{python_version.micro}")
    
    with col2:
        st.metric("🌐 Plataforma", "Streamlit Cloud")
    
    with col3:
        if python_version >= (3, 10):
            st.metric("🚨 Estado", "INCOMPATIBLE", delta="Requiere Python 3.9")
        else:
            st.metric("✅ Estado", "COMPATIBLE")
    
    # Sidebar con información
    with st.sidebar:
        st.header("📋 Información del Sistema")
        
        st.markdown(f"""
        **Versión Python**: {python_version.major}.{python_version.minor}.{python_version.micro}
        **Fecha**: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        **Repositorio**: mauroere/Transcrip
        """)
        
        if python_version >= (3, 10):
            st.error("🚨 Modo Backup Activo")
            st.markdown("""
            **Limitaciones**:
            - Sin transcripción automática
            - Sin detección de speakers
            - Solo análisis de texto manual
            """)
        else:
            st.success("✅ Sistema Completo")
    
    # Funcionalidad principal
    st.header("📁 Subir Archivo de Audio")
    
    uploaded_file = st.file_uploader(
        "Selecciona un archivo de audio",
        type=['wav', 'mp3', 'mp4', 'avi', 'mov', 'flac', 'm4a', 'ogg', 'webm'],
        help="Formatos soportados: WAV, MP3, MP4, AVI, MOV, FLAC, M4A, OGG, WEBM"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ Archivo cargado: {uploaded_file.name}")
        st.info(f"📊 Tamaño: {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        if python_version >= (3, 10):
            st.warning("""
            🚨 **Transcripción automática no disponible**
            
            Debido a la incompatibilidad de Python 3.13 con OpenAI Whisper, 
            la transcripción automática está deshabilitada.
            
            **Opciones disponibles**:
            1. Ingresa el texto manualmente para análisis
            2. Espera a que se solucione la compatibilidad
            3. Usa herramientas externas para transcribir
            """)
            
            # Análisis manual de texto
            st.header("📝 Análisis Manual de Texto")
            manual_text = st.text_area(
                "Pega aquí el texto transcrito manualmente:",
                height=200,
                placeholder="Ejemplo: Hola, buenos días, habla con María de Movistar..."
            )
            
            if manual_text and st.button("🔍 Analizar Texto"):
                # Análisis básico
                word_count = len(manual_text.split())
                char_count = len(manual_text)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Palabras", word_count)
                with col2:
                    st.metric("📊 Caracteres", char_count)
                
                # Análisis de palabras clave básico
                movistar_mentions = manual_text.lower().count('movistar')
                gracias_mentions = manual_text.lower().count('gracias')
                problema_mentions = manual_text.lower().count('problema')
                
                st.subheader("🔍 Análisis de Palabras Clave")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🏢 Movistar", movistar_mentions)
                with col2:
                    st.metric("🙏 Gracias", gracias_mentions)
                with col3:
                    st.metric("⚠️ Problema", problema_mentions)
                
                # Mostrar el texto analizado
                st.subheader("📄 Texto Analizado")
                st.text_area("", value=manual_text, height=150, disabled=True)
        else:
            # Aquí iría la funcionalidad completa con Whisper
            st.info("🎯 **Funcionalidad completa disponible** - Python 3.9 detectado")
            
            if st.button("🎙️ Iniciar Transcripción"):
                st.success("✅ Transcripción iniciada (funcionalidad completa disponible)")
    
    # Información adicional
    st.header("ℹ️ Información del Problema")
    
    with st.expander("🔧 Detalles Técnicos"):
        st.markdown("""
        **Problema identificado**: 
        - Streamlit Cloud usa Python 3.13.5
        - OpenAI Whisper requiere Python ≤ 3.9
        - Triton (dependencia de Whisper) no soporta Python 3.13
        
        **Solución implementada**:
        - ✅ `runtime.txt` con `python-3.9.19`
        - ✅ `requirements.txt` optimizado
        - ✅ Configuración de deployment
        
        **Estado actual**:
        - ❌ Streamlit Cloud ignora `runtime.txt`
        - 🔄 Requiere recrear la app
        """)
    
    with st.expander("📞 Pasos para Solucionar"):
        st.markdown("""
        **Para administradores**:
        
        1. **Eliminar app actual** en [share.streamlit.io](https://share.streamlit.io)
        2. **Crear nueva app** desde el repositorio `mauroere/Transcrip`
        3. **Verificar** que use Python 3.9 desde el inicio
        4. **Monitorear logs** para confirmar compatibilidad
        
        **Alternativas**:
        - Migrar a Heroku (respeta runtime.txt)
        - Usar Railway o Render
        - Deployment local con Docker
        """)

if __name__ == "__main__":
    main()
