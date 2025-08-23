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

# Importación condicional de psutil (para monitoreo del sistema)
try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

warnings.filterwarnings("ignore")

# Configurar FFmpeg si está en directorio local
ffmpeg_local = os.path.join(os.getcwd(), 'ffmpeg', 'ffmpeg.exe')
if os.path.exists(ffmpeg_local):
    ffmpeg_dir = os.path.dirname(ffmpeg_local)
    if ffmpeg_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
    # st.sidebar.success(f"✓ FFmpeg configurado desde: {ffmpeg_local}")
else:
    # st.sidebar.warning("⚠️ FFmpeg local no encontrado, usando PATH del sistema")
    pass

try:
    import whisper
    # st.sidebar.success("✓ Whisper importado correctamente")
except ImportError as e:
    st.sidebar.error(f"✗ Error importando Whisper: {e}")
    st.stop()

try:
    from pydub import AudioSegment
    # st.sidebar.success("✓ Pydub importado correctamente")
    AUDIO_PROCESSOR = "pydub"
except ImportError:
    try:
        import librosa
        import soundfile as sf
        # st.sidebar.success("✓ Librosa importado correctamente (alternativa a Pydub)")
        AudioSegment = None
        AUDIO_PROCESSOR = "librosa"
    except ImportError as e:
        # st.sidebar.info("ℹ️ Sin procesador de audio adicional - Whisper maneja los formatos directamente")
        AudioSegment = None
        AUDIO_PROCESSOR = "none"

# Configuración
UPLOAD_FOLDER = 'uploads'
TRANSCRIPTIONS_FOLDER = 'transcriptions'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'avi', 'mov', 'flac', 'm4a', 'ogg', 'webm'}

# Crear directorios si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTIONS_FOLDER, exist_ok=True)

# Cache del modelo Whisper
@st.cache_resource
def load_whisper_model():
    """Cargar modelo Whisper con cache y manejo robusto de errores"""
    try:
        with st.spinner("🤖 Inicializando sistema de transcripción..."):
            # Intentar cargar el modelo base primero
            try:
                model = whisper.load_model("base")
                # Verificar que el modelo se cargó correctamente
                if model is not None:
                    return model
            except Exception as e:
                st.warning(f"⚠️ Error con modelo 'base': {str(e)}")
            
            # Fallback: intentar con modelo tiny
            try:
                st.info("🔄 Intentando con modelo alternativo...")
                model = whisper.load_model("tiny")
                if model is not None:
                    st.success("✅ Modelo alternativo cargado correctamente")
                    return model
            except Exception as e:
                st.warning(f"⚠️ Error con modelo 'tiny': {str(e)}")
            
            # Si ambos fallan, devolver None
            return None
            
    except Exception as e:
        st.error(f"❌ Error crítico al cargar modelo Whisper: {str(e)}")
        return None

def allowed_file(filename):
    """Verificar si el archivo tiene una extensión permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_audio_file(file_path):
    """Validar que el archivo de audio sea válido"""
    try:
        # Verificar que el archivo existe
        if not os.path.exists(file_path):
            return False, f"Archivo no encontrado: {file_path}"
        
        # Verificar que no está vacío
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "El archivo está vacío"
        except OSError as e:
            return False, f"Error al acceder al archivo: {str(e)}"
        
        # Verificación básica de extensión
        valid_extensions = ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg']
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in valid_extensions:
            return True, f"Archivo válido: {file_size/1024/1024:.1f} MB"
        else:
            # Intentar validación con librosa si está disponible
            try:
                if AUDIO_PROCESSOR == "librosa":
                    import librosa
                    y, sr = librosa.load(file_path, sr=None, duration=1.0)  # Solo cargar 1 segundo para validar
                    if len(y) > 0:
                        return True, f"Archivo de audio válido: {file_size/1024/1024:.1f} MB"
                    else:
                        return False, "El archivo no contiene datos de audio válidos"
                else:
                    # Si no tenemos procesador, aceptar el archivo
                    return True, f"Archivo: {file_size/1024/1024:.1f} MB"
            except Exception as audio_error:
                return False, f"Error al validar audio: {str(audio_error)}"
            
    except FileNotFoundError as e:
        return False, f"Archivo no encontrado: {str(e)}"
    except PermissionError as e:
        return False, f"Sin permisos para acceder al archivo: {str(e)}"
    except Exception as e:
        return False, f"Error al validar archivo: {str(e)}"

def convert_audio_format(input_path, output_path):
    """Convertir formato de audio usando el procesador disponible"""
    try:
        if AUDIO_PROCESSOR == "pydub" and AudioSegment:
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_path, format="wav")
            return True, None
        elif AUDIO_PROCESSOR == "librosa":
            import librosa
            import soundfile as sf
            # Cargar audio
            y, sr = librosa.load(input_path, sr=16000, mono=True)
            # Guardar como WAV
            sf.write(output_path, y, sr)
            return True, None
        else:
            return False, "No hay procesador de audio disponible para conversión"
    except Exception as e:
        return False, f"Error en conversión: {str(e)}"

def enhance_audio_quality(file_path):
    """Mejorar calidad de audio para mejor transcripción"""
    try:
        if AUDIO_PROCESSOR == "librosa":
            import librosa
            import soundfile as sf
            import numpy as np
            
            # Cargar audio
            y, sr = librosa.load(file_path, sr=None, mono=False)
            
            # Convertir a mono si es estéreo
            if y.ndim > 1:
                y = librosa.to_mono(y)
            
            # Normalizar audio
            y = librosa.util.normalize(y)
            
            # Reducir ruido (filtro pasa-altos simple)
            y = librosa.effects.preemphasis(y)
            
            # Resamplear a 16kHz (óptimo para Whisper)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Guardar archivo mejorado
            enhanced_path = file_path.replace('.wav', '_enhanced.wav')
            sf.write(enhanced_path, y, sr)
            
            return enhanced_path, None
        else:
            return file_path, None  # Sin mejora si no hay librosa
    except Exception as e:
        return file_path, f"Error mejorando audio: {str(e)}"

def clean_transcription_text(text):
    """Limpiar y profesionalizar el texto transcrito"""
    if not text:
        return text
    
    # Diccionario de correcciones específicas para call center
    corrections = {
        # Saludos comunes
        r'\bbuen[ao]s? d[ií]as?\b': 'buenos días',
        r'\bbuen[ao]s? tard[eé]s?\b': 'buenas tardes',
        r'\bbuen[ao]s? noch[eé]s?\b': 'buenas noches',
        
        # Empresa y servicios
        r'\bmov[ií]star\b': 'Movistar',
        r'\bfactur[ao]?\b': 'factura',
        r'\bl[ií]ne[ao]?\b': 'línea',
        r'\bservici[eo]s?\b': 'servicio',
        r'\binter?net\b': 'internet',
        r'\btelev[ií]si[óo]n?\b': 'televisión',
        r'\bplan[eé]s?\b': 'plan',
        
        # Palabras comunes mal transcritas
        r'\bgracias?\b': 'gracias',
        r'\bdiscul[pq][ae]?\b': 'disculpe',
        r'\bperd[óo]n?\b': 'perdón',
        r'\bproblemas?\b': 'problema',
        r'\bsoluci[óo]n?\b': 'solución',
        r'\brespuest[ao]?\b': 'respuesta',
        r'\bmom[eé]nto?\b': 'momento',
        
        # Números mal transcritos
        r'\bun[oa]?\b': 'uno',
        r'\bdos?\b': 'dos',
        r'\btr[eé]s?\b': 'tres',
        r'\bcuatro?\b': 'cuatro',
        r'\bcinco?\b': 'cinco',
        
        # Expresiones comunes
        r'\b[¿c][óo]m[eo]?\s+est[áa]?\b': 'cómo está',
        r'\ben\s+qu[eé]\s+pued[eo]\s+ayud[ao]r?\b': 'en qué puedo ayudar',
        r'\bpor\s+favor\b': 'por favor',
        r'\bmuch[ao]s?\s+gracias?\b': 'muchas gracias',
    }
    
    # Aplicar correcciones con regex
    cleaned_text = text.lower()
    for pattern, replacement in corrections.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
    
    # Limpiar símbolos y caracteres problemáticos
    cleaned_text = re.sub(r'[♪♫♪]', '', cleaned_text)  # Remover símbolos musicales
    cleaned_text = re.sub(r'\s*\[.*?\]\s*', ' ', cleaned_text)  # Remover [texto entre corchetes]
    cleaned_text = re.sub(r'\s*\(.*?\)\s*', ' ', cleaned_text)  # Remover (texto entre paréntesis) largo
    cleaned_text = re.sub(r'[^\w\sáéíóúüñ¿¡.,!?:-]', '', cleaned_text)  # Solo caracteres válidos
    
    # Normalizar espacios y puntuación
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Espacios múltiples
    cleaned_text = re.sub(r'\s*([.!?])\s*', r'\1 ', cleaned_text)  # Espacios alrededor de puntuación
    cleaned_text = re.sub(r'([.!?])\s*([.!?])', r'\1', cleaned_text)  # Puntuación duplicada
    
    # Capitalizar apropiadamente
    sentences = re.split(r'([.!?]+)', cleaned_text)
    capitalized_sentences = []
    
    for i, sentence in enumerate(sentences):
        if i % 2 == 0:  # Texto (no puntuación)
            sentence = sentence.strip()
            if sentence:
                # Capitalizar primera letra de cada oración
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                
                # Capitalizar después de signos específicos
                sentence = re.sub(r'([.!?]\s+)([a-z])', 
                                lambda m: m.group(1) + m.group(2).upper(), sentence)
                
                # Capitalizar nombres propios conocidos
                sentence = re.sub(r'\bmovistar\b', 'Movistar', sentence, flags=re.IGNORECASE)
                
        capitalized_sentences.append(sentence)
    
    # Unir y limpiar espacios finales
    final_text = ''.join(capitalized_sentences).strip()
    
    return final_text

def transcribe_with_enhanced_quality(model, file_path):
    """Transcripción mejorada con manejo específico de errores de tensor y múltiples estrategias de recuperación"""
    try:
        # Verificar archivo
        if not os.path.exists(file_path):
            return None, f"Archivo no existe para transcripción: {file_path}"
        
        # Verificar tamaño del archivo
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return None, "El archivo está vacío"
        
        # Estrategias de procesamiento ordenadas de más a menos agresivas
        processing_strategies = [
            {
                "name": "Archivo Original + Configuración Completa",
                "use_enhanced": False,
                "configs": [
                    {"language": "es", "task": "transcribe", "temperature": 0.0, "beam_size": 5, "best_of": 5, "fp16": False},
                    {"language": "es", "task": "transcribe", "temperature": 0.2, "fp16": False},
                    {"language": "es", "fp16": False},
                    {"language": "es"}
                ]
            },
            {
                "name": "Audio Mejorado + Configuración Robusta",
                "use_enhanced": True,
                "configs": [
                    {"language": "es", "temperature": 0.1, "fp16": False},
                    {"language": "es", "fp16": False},
                    {"language": "es"}
                ]
            },
            {
                "name": "Procesamiento Mínimo + Configuraciones Básicas",
                "use_enhanced": False,
                "configs": [
                    {"language": "es"},
                    {"task": "transcribe"},
                    {}
                ]
            },
            {
                "name": "Estrategia de Emergencia - Segmentación",
                "use_enhanced": False,
                "configs": [{"language": "es"}],
                "segment_audio": True
            }
        ]
        
        last_error = None
        
        for strategy_idx, strategy in enumerate(processing_strategies):
            try:
                # Determinar qué archivo usar
                if strategy["use_enhanced"]:
                    enhanced_path, enhance_error = enhance_audio_quality(file_path)
                    if enhance_error:
                        continue  # Saltar esta estrategia si falla la mejora
                    target_path = enhanced_path
                else:
                    target_path = file_path
                
                # Mostrar progreso de estrategia
                if strategy_idx > 0:
                    st.warning(f"🔄 Estrategia {strategy_idx + 1}: {strategy['name']}")
                
                # Procesamiento especial para segmentación
                if strategy.get("segment_audio", False):
                    result_text = process_audio_segments(model, target_path, strategy["configs"][0])
                    if result_text:
                        return clean_transcription_text(result_text), None
                    else:
                        last_error = f"Estrategia {strategy_idx + 1}: Segmentación falló"
                        continue
                
                # Probar configuraciones normales
                for config_idx, config in enumerate(strategy["configs"]):
                    try:
                        # Limpiar memoria antes de cada intento
                        import gc
                        gc.collect()
                        
                        # Mensaje específico para errores de tensor
                        if "tensor" in str(last_error).lower() and config_idx == 0:
                            st.info("🧠 Aplicando solución para compatibilidad de tensores...")
                        
                        result = model.transcribe(target_path, **config)
                        raw_text = result.get("text", "")
                        
                        if raw_text and raw_text.strip():
                            # Limpiar archivo temporal si se creó
                            if strategy["use_enhanced"] and target_path != file_path and os.path.exists(target_path):
                                try:
                                    os.unlink(target_path)
                                except:
                                    pass
                            
                            cleaned_text = clean_transcription_text(raw_text)
                            st.success(f"✅ Transcripción exitosa con {strategy['name']}")
                            return cleaned_text, None
                        else:
                            last_error = f"Estrategia {strategy_idx + 1}, Config {config_idx + 1}: Texto vacío"
                            continue
                            
                    except Exception as e:
                        error_msg = str(e)
                        last_error = f"Estrategia {strategy_idx + 1}, Config {config_idx + 1}: {error_msg}"
                        
                        # Manejo específico de errores de tensor
                        if "tensor" in error_msg.lower() or "size" in error_msg.lower():
                            st.warning(f"⚠️ Error de tensor detectado en configuración {config_idx + 1}")
                            # Forzar limpieza agresiva de memoria
                            try:
                                import gc
                                import torch
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except:
                                pass
                            continue
                        
                        # Para otros errores, continuar con siguiente configuración
                        continue
                
                # Limpiar archivo temporal de esta estrategia
                if strategy["use_enhanced"] and target_path != file_path and os.path.exists(target_path):
                    try:
                        os.unlink(target_path)
                    except:
                        pass
                        
            except Exception as e:
                last_error = f"Estrategia {strategy_idx + 1}: {str(e)}"
                continue
        
        # Si llegamos aquí, todas las estrategias fallaron
        return None, f"Error en todas las estrategias de transcripción. Último error: {last_error}"
        
    except Exception as e:
        return None, f"Error crítico en transcripción: {str(e)}"

def process_audio_segments(model, file_path, config):
    """Procesar audio en segmentos para evitar errores de tensor en archivos largos"""
    try:
        import librosa
        import soundfile as sf
        import tempfile
        
        # Cargar audio completo
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        
        # Calcular duración
        duration = len(y) / sr
        
        # Si es muy corto, procesar normalmente
        if duration < 30:
            return None
        
        # Dividir en segmentos de 30 segundos con overlap de 2 segundos
        segment_length = 30 * sr  # 30 segundos
        overlap = 2 * sr  # 2 segundos de overlap
        step = segment_length - overlap
        
        segments_text = []
        
        for start in range(0, len(y), step):
            end = min(start + segment_length, len(y))
            segment = y[start:end]
            
            # Saltar segmentos muy cortos
            if len(segment) < sr:  # Menos de 1 segundo
                continue
            
            # Guardar segmento temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                sf.write(tmp_file.name, segment, sr)
                tmp_path = tmp_file.name
            
            try:
                # Transcribir segmento
                result = model.transcribe(tmp_path, **config)
                segment_text = result.get("text", "").strip()
                
                if segment_text:
                    segments_text.append(segment_text)
                    
            except Exception as e:
                # Si falla un segmento, continuar con el siguiente
                pass
            finally:
                # Limpiar archivo temporal
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        # Unir todos los segmentos
        if segments_text:
            return " ".join(segments_text)
        else:
            return None
            
    except Exception as e:
        return None

def transcribe_with_fallback(model, file_path):
    """Mantener compatibilidad - usar nueva función mejorada"""
    return transcribe_with_enhanced_quality(model, file_path)

def analyze_transcription(text):
    """Análisis completo de la transcripción con evaluación de performance"""
    # Métricas básicas
    word_count = len(text.split())
    character_count = len(text)
    estimated_duration = word_count / 150  # ~150 palabras por minuto

    # Análisis de palabras clave específicas
    keywords = {
        "saludos": count_keywords(text, ["hola", "buenos días", "buenas tardes", "buenas noches", "bienvenido"]),
        "agradecimientos": count_keywords(text, ["gracias", "agradezco", "muchas gracias", "te agradezco"]),
        "disculpas": count_keywords(text, ["disculpa", "perdón", "lo siento", "disculpe", "perdone"]),
        "servicios_movistar": count_keywords(text, ["movistar", "plan", "servicio", "factura", "línea", "internet", "móvil"]),
        "problemas_tecnicos": count_keywords(text, ["problema", "error", "falla", "no funciona", "caído", "lento"])
    }

    # Análisis de sentimiento básico
    sentiment_indicators = {
        "positive": count_keywords(text, ["excelente", "perfecto", "genial", "bien", "bueno", "contento", "satisfecho"]),
        "negative": count_keywords(text, ["mal", "terrible", "horrible", "molesto", "enojado", "frustrado", "problema"])
    }

    # ANÁLISIS AVANZADO DE PERFORMANCE
    performance = analyze_performance(text)

    return {
        "word_count": word_count,
        "character_count": character_count,
        "estimated_duration": estimated_duration,
        "keywords": keywords,
        "sentiment_indicators": sentiment_indicators,
        "performance": performance
    }

def analyze_performance(text):
    """Análisis completo de performance del asesor"""
    text_lower = text.lower()
    
    # 1. ANÁLISIS DE PROTOCOLO
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
    
    # Calcular score de protocolo
    protocol_score = (sum(protocol_analysis.values()) / len(protocol_analysis)) * 100

    # 2. ANÁLISIS DE TONO
    tone_indicators = {
        "amable": count_keywords(text, ["por favor", "con gusto", "encantado", "perfecto", "claro que sí"]),
        "empatico": count_keywords(text, ["entiendo", "comprendo", "me imagino", "tiene razón", "lo siento"]),
        "profesional": count_keywords(text, ["señor", "señora", "usted", "permítame", "procedemos"]),
        "cortado": count_keywords(text, ["sí", "no", "ok", "bien"]) - count_keywords(text, ["sí señor", "claro que sí"]),
        "frustrado": count_keywords(text, ["pero", "sin embargo", "es que", "ya le dije", "otra vez"]),
        "agresivo": count_keywords(text, ["no puede", "imposible", "no se puede", "tiene que"])
    }
    
    # Determinar tono principal
    primary_tone = max(tone_indicators.keys(), key=lambda k: tone_indicators[k])
    if tone_indicators[primary_tone] == 0:
        primary_tone = "neutral"
    
    # Score de tono (positivo vs negativo)
    positive_indicators = tone_indicators["amable"] + tone_indicators["empatico"] + tone_indicators["profesional"]
    negative_indicators = tone_indicators["cortado"] + tone_indicators["frustrado"] + tone_indicators["agresivo"]
    
    if positive_indicators + negative_indicators > 0:
        tone_score = (positive_indicators / (positive_indicators + negative_indicators)) * 100
    else:
        tone_score = 50  # Neutral si no hay indicadores

    # 3. ANÁLISIS DE RESOLUCIÓN DE PROBLEMAS
    resolution_indicators = {
        "solucion_ofrecida": any(sol in text_lower for sol in [
            "vamos a solucionar", "le voy a ayudar", "podemos hacer", "voy a verificar",
            "le ofrezco", "una opción es", "podría", "vamos a revisar"
        ]),
        "seguimiento": any(seg in text_lower for seg in [
            "le voy a llamar", "estaremos en contacto", "le envío", "quedamos en",
            "voy a hacer el seguimiento", "en las próximas horas"
        ]),
        "confirmacion": any(conf in text_lower for conf in [
            "quedó claro", "está de acuerdo", "confirma", "está bien",
            "entendió", "alguna duda", "algo más"
        ])
    }
    
    problem_resolved = any(res in text_lower for res in [
        "solucionado", "resuelto", "listo", "perfecto", "ya está",
        "problema resuelto", "ya funciona"
    ])
    
    follow_up_needed = any(follow in text_lower for follow in [
        "voy a verificar", "le confirmo", "estaremos pendientes",
        "en 24 horas", "mañana", "próximamente"
    ])
    
    # Determinar tipo de problema
    problem_type = "otro"
    if any(tech in text_lower for tech in ["internet", "wifi", "conexión", "señal"]):
        problem_type = "técnico"
    elif any(bill in text_lower for bill in ["factura", "cobro", "pago", "dinero"]):
        problem_type = "facturación"
    elif any(plan in text_lower for plan in ["plan", "cambio", "upgrade", "servicio"]):
        problem_type = "comercial"
    
    resolution_score = (sum(resolution_indicators.values()) / len(resolution_indicators)) * 100
    if problem_resolved:
        resolution_score = min(resolution_score + 30, 100)

    # 4. DETECCIÓN DE FALENCIAS
    falencias = []
    
    if not protocol_analysis["saludo_inicial"]:
        falencias.append("No realizó saludo inicial apropiado")
    if not protocol_analysis["identificacion"]:
        falencias.append("No solicitó identificación del cliente")
    if not protocol_analysis["pregunta_ayuda"]:
        falencias.append("No preguntó específicamente cómo ayudar")
    if tone_indicators["cortado"] > 3:
        falencias.append("Tono muy cortante o seco en las respuestas")
    if tone_indicators["frustrado"] > 2:
        falencias.append("Mostró signos de frustración")
    if not resolution_indicators["confirmacion"]:
        falencias.append("No confirmó la comprensión del cliente")
    if not problem_resolved and not follow_up_needed:
        falencias.append("No ofreció solución ni seguimiento al problema")

    # 5. PUNTOS DE MEJORA
    puntos_mejora = []
    
    if protocol_score < 75:
        puntos_mejora.append("Mejorar adherencia al protocolo de atención estándar")
    if tone_score < 60:
        puntos_mejora.append("Trabajar en tono más amable y empático")
    if resolution_score < 70:
        puntos_mejora.append("Fortalecer habilidades de resolución de problemas")
    if tone_indicators["profesional"] < 2:
        puntos_mejora.append("Usar lenguaje más formal y profesional")
    if not resolution_indicators["seguimiento"]:
        puntos_mejora.append("Implementar seguimiento proactivo a las consultas")

    # 6. SCORE GENERAL
    overall_score = (protocol_score * 0.3 + tone_score * 0.4 + resolution_score * 0.3)

    return {
        "protocol_analysis": protocol_analysis,
        "protocol_score": round(protocol_score, 1),
        "tone_analysis": {
            "primary_tone": primary_tone,
            "indicators": tone_indicators
        },
        "tone_score": round(tone_score, 1),
        "problem_resolution": {
            "resolved": problem_resolved,
            "type": problem_type,
            "follow_up_needed": follow_up_needed,
            "indicators": resolution_indicators
        },
        "resolution_score": round(resolution_score, 1),
        "overall_score": round(overall_score, 1),
        "falencias": falencias,
        "puntos_mejora": puntos_mejora
    }

def count_keywords(text, keywords):
    """Contar ocurrencias de palabras clave en el texto"""
    text_lower = text.lower()
    return sum(text_lower.count(keyword.lower()) for keyword in keywords)

def format_bytes(bytes_size):
    """Formatear tamaño en bytes a formato legible"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def generate_chatgpt_prompt(transcription, analysis, filename):
    """Generar prompt optimizado para ChatGPT"""
    perf = analysis.get('performance', {})
    
    prompt = f"""🎯 ANÁLISIS DE LLAMADA DE ATENCIÓN AL CLIENTE

📁 ARCHIVO: {filename}
📅 FECHA: {datetime.now().strftime('%d/%m/%Y')}

📊 SCORES ACTUALES:
• Protocolo: {round(perf.get('protocol_score', 0))}%
• Tono: {round(perf.get('tone_score', 0))}%
• Resolución: {round(perf.get('resolution_score', 0))}%
• General: {round(perf.get('overall_score', 0))}%

✅ PROTOCOLO CUMPLIDO:"""
    
    protocol_labels = {
        'saludo_inicial': 'Saludo inicial',
        'identificacion': 'Identificación',
        'pregunta_ayuda': 'Pregunta de ayuda',
        'despedida': 'Despedida'
    }
    
    for key, value in perf.get('protocol_analysis', {}).items():
        label = protocol_labels.get(key, key)
        status = '✓ SÍ' if value else '✗ NO'
        prompt += f"\n• {label}: {status}"
    
    tone_analysis = perf.get('tone_analysis', {})
    prompt += f"""

🎭 ANÁLISIS DE TONO:
• Tono principal detectado: {tone_analysis.get('primary_tone', 'No detectado')}"""
    
    for tone, count in tone_analysis.get('indicators', {}).items():
        if count > 0:
            prompt += f"\n• {tone}: {count} menciones"
    
    problem_resolution = perf.get('problem_resolution', {})
    prompt += f"""

🔧 RESOLUCIÓN DE PROBLEMAS:
• ¿Problema resuelto?: {'✓ SÍ' if problem_resolution.get('resolved', False) else '✗ NO'}"""
    
    if problem_resolution.get('type'):
        prompt += f"\n• Tipo de problema: {problem_resolution['type']}"
    if problem_resolution.get('follow_up_needed'):
        prompt += f"\n• ⚠️ Requiere seguimiento"
    
    falencias = perf.get('falencias', [])
    prompt += f"""

⚠️ FALENCIAS DETECTADAS:"""
    if falencias:
        for falencia in falencias:
            prompt += f"\n• {falencia}"
    else:
        prompt += f"\n• No se detectaron falencias críticas"
    
    mejoras = perf.get('puntos_mejora', [])
    prompt += f"""

💡 PUNTOS DE MEJORA SUGERIDOS:"""
    if mejoras:
        for mejora in mejoras:
            prompt += f"\n• {mejora}"
    else:
        prompt += f"\n• Se requiere análisis más detallado"
    
    prompt += f"""

📝 TRANSCRIPCIÓN COMPLETA:
"{transcription}"

🤖 SOLICITUD PARA CHATGPT:
Por favor analiza esta llamada de atención al cliente y proporciona:

1. Un análisis más profundo del desempeño del asesor
2. Recomendaciones específicas para mejorar la atención
3. Evaluación de la satisfacción del cliente
4. Sugerencias de entrenamiento o coaching
5. Puntos positivos que el asesor debería mantener
6. Una calificación general del 1-10 con justificación

Contexto: Somos y queremos mejorar la calidad de nuestro servicio al cliente."""
    
    return prompt

def generate_excel_report(result, analysis):
    """Generar reporte en formato Excel"""
    # Crear DataFrame con los datos principales
    main_data = {
        'Archivo': [result.get('filename', '')],
        'Fecha de Análisis': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Palabras': [analysis.get('word_count', 0)],
        'Caracteres': [analysis.get('character_count', 0)],
        'Duración Estimada (min)': [analysis.get('estimated_duration', 0)],
    }
    
    # Agregar métricas de performance si existen
    if 'performance' in analysis:
        perf = analysis['performance']
        main_data.update({
            'Score Protocolo (%)': [perf.get('protocol_score', 0)],
            'Score Tono (%)': [perf.get('tone_score', 0)],
            'Score Resolución (%)': [perf.get('resolution_score', 0)],
            'Score General (%)': [perf.get('overall_score', 0)]
        })
    
    df_main = pd.DataFrame(main_data)
    
    # Crear DataFrame para palabras clave
    keywords_data = []
    if 'keywords' in analysis:
        for category, count in analysis['keywords'].items():
            keywords_data.append({
                'Categoría': category.replace('_', ' ').title(),
                'Cantidad': count
            })
    
    df_keywords = pd.DataFrame(keywords_data)
    
    # Crear archivo Excel en memoria
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Hoja principal
        df_main.to_excel(writer, sheet_name='Análisis Principal', index=False)
        
        # Hoja de palabras clave
        if not df_keywords.empty:
            df_keywords.to_excel(writer, sheet_name='Palabras Clave', index=False)
        
        # Hoja de transcripción
        df_transcript = pd.DataFrame({
            'Transcripción Completa': [result.get('transcription', '')]
        })
        df_transcript.to_excel(writer, sheet_name='Transcripción', index=False)
    
    output.seek(0)
    return output

def generate_csv_report(result, analysis):
    """Generar reporte en formato CSV"""
    data = {
        'archivo': result.get('filename', ''),
        'fecha_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'palabras': analysis.get('word_count', 0),
        'caracteres': analysis.get('character_count', 0),
        'duracion_estimada_min': analysis.get('estimated_duration', 0),
        'transcripcion': result.get('transcription', '')
    }
    
    # Agregar métricas de performance
    if 'performance' in analysis:
        perf = analysis['performance']
        data.update({
            'score_protocolo': perf.get('protocol_score', 0),
            'score_tono': perf.get('tone_score', 0),
            'score_resolucion': perf.get('resolution_score', 0),
            'score_general': perf.get('overall_score', 0)
        })
    
    # Agregar palabras clave
    if 'keywords' in analysis:
        for category, count in analysis['keywords'].items():
            data[f'palabras_clave_{category}'] = count
    
    df = pd.DataFrame([data])
    return df.to_csv(index=False)

def create_copy_button(text, button_text, button_id, success_message="✅ Copiado al portapapeles"):
    """Crear un botón de copiado que no cause rerun de Streamlit"""
    # Escapar el texto para JavaScript
    text_safe = text.replace('\\', '\\\\').replace('`', '\\`').replace('\n', '\\n').replace('\r', '\\r').replace('"', '\\"').replace("'", "\\'")
    
    button_html = f"""
    <div style="margin-bottom: 10px;">
        <button onclick="copyText_{button_id}()" style="
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            width: 100%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        " 
        onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.3)'"
        onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.2)'"
        >{button_text}</button>
    </div>
    <script>
    function copyText_{button_id}() {{
        const text = `{text_safe}`;
        if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text).then(function() {{
                showSuccessMessage_{button_id}();
            }}, function(err) {{
                fallbackCopy_{button_id}(text);
            }});
        }} else {{
            fallbackCopy_{button_id}(text);
        }}
    }}
    
    function fallbackCopy_{button_id}(text) {{
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {{
            document.execCommand('copy');
            showSuccessMessage_{button_id}();
        }} catch (err) {{
            alert('No se pudo copiar automáticamente. Por favor, copia manualmente del área de texto.');
        }}
        
        document.body.removeChild(textArea);
    }}
    
    function showSuccessMessage_{button_id}() {{
        // Crear notificación temporal
        const notification = document.createElement('div');
        notification.innerHTML = '{success_message}';
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            z-index: 9999;
            font-family: Arial, sans-serif;
            font-size: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        `;
        document.body.appendChild(notification);
        
        // Remover notificación después de 3 segundos
        setTimeout(function() {{
            if (notification.parentNode) {{
                notification.parentNode.removeChild(notification);
            }}
        }}, 3000);
    }}
    </script>
    """
    
    return button_html

def generate_word_report(result, analysis):
    """Generar reporte en formato HTML para Word"""
    filename = result.get('filename', 'Audio')
    transcription = result.get('transcription', '')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Análisis de Audio - {filename}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #d32f2f; border-bottom: 2px solid #d32f2f; }}
            h2 {{ color: #1976d2; margin-top: 30px; }}
            .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            .transcription {{ background: #fafafa; padding: 20px; border-radius: 5px; font-style: italic; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>📊 Análisis de Performance</h1>
        
        <h2>📁 Información del Archivo</h2>
        <div class="metric"><strong>Archivo:</strong> {filename}</div>
        <div class="metric"><strong>Fecha de Análisis:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <h2>📈 Métricas Básicas</h2>
        <div class="metric"><strong>Palabras:</strong> {analysis.get('word_count', 0)}</div>
        <div class="metric"><strong>Caracteres:</strong> {analysis.get('character_count', 0)}</div>
        <div class="metric"><strong>Duración Estimada:</strong> {analysis.get('estimated_duration', 0):.1f} minutos</div>
    """
    
    # Agregar métricas de performance si existen
    if 'performance' in analysis:
        perf = analysis['performance']
        html_content += f"""
        <h2>🎯 Scores de Performance</h2>
        <table>
            <tr><th>Métrica</th><th>Puntuación</th></tr>
            <tr><td>Protocolo</td><td>{perf.get('protocol_score', 0):.1f}%</td></tr>
            <tr><td>Tono</td><td>{perf.get('tone_score', 0):.1f}%</td></tr>
            <tr><td>Resolución</td><td>{perf.get('resolution_score', 0):.1f}%</td></tr>
            <tr><td><strong>Score General</strong></td><td><strong>{perf.get('overall_score', 0):.1f}%</strong></td></tr>
        </table>
        """
    
    # Agregar palabras clave si existen
    if 'keywords' in analysis:
        html_content += """
        <h2>🔑 Palabras Clave Identificadas</h2>
        <table>
            <tr><th>Categoría</th><th>Cantidad</th></tr>
        """
        for category, count in analysis['keywords'].items():
            category_name = category.replace('_', ' ').title()
            html_content += f"<tr><td>{category_name}</td><td>{count}</td></tr>"
        html_content += "</table>"
    
    # Agregar transcripción
    html_content += f"""
        <h2>📄 Transcripción Completa</h2>
        <div class="transcription">
            {transcription.replace('\n', '<br>')}
        </div>
        
        <hr>
        <p><em>Reporte generado automáticamente por el Sistema de Análisis</em></p>
    </body>
    </html>
    """
    
    return html_content

def display_performance_metrics(analysis):
    """Mostrar métricas de performance en formato compacto"""
    if 'performance' not in analysis:
        return
    
    perf = analysis['performance']
    
    # Scores principales en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = perf.get('protocol_score', 0)
        st.metric("🔖 Protocolo", f"{score:.1f}%")
    with col2:
        score = perf.get('tone_score', 0)
        st.metric("🎭 Tono", f"{score:.1f}%")
    with col3:
        score = perf.get('resolution_score', 0)
        st.metric("🔧 Resolución", f"{score:.1f}%")
    with col4:
        score = perf.get('overall_score', 0)
        st.metric("⭐ General", f"{score:.1f}%")

def main():
    # Configuración de página
    st.set_page_config(
        page_title="Transcriptor - Análisis de Performance",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS personalizado
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    # Encabezado principal
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Transcriptor de Audios</h1>
        <p>Análisis Avanzado de Performance para Asesores de Atención al Cliente</p>
        <div style="text-align: right; margin-top: 10px; font-size: 0.8em; opacity: 0.8;">
            <i>Desarrollado por <strong>Mauro Rementeria</strong> | mauroere@gmail.com</i>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar con información
    with st.sidebar:
        st.header("📊 Panel de Control")
        
        # Mostrar información del sistema
        st.subheader("🔧 Estado del Sistema")
        
        # Cargar modelo con manejo robusto de errores
        try:
            model = load_whisper_model()
            if model is None:
                st.error("❌ Error crítico: No se pudo cargar el sistema de transcripción")
                st.warning("🔧 Posibles soluciones:")
                st.markdown("- Verifica tu conexión a internet")
                st.markdown("- Reinicia la aplicación")
                st.markdown("- Contacta al desarrollador si el problema persiste")
                st.stop()
            
            # Solo mostrar que el sistema está listo
            st.success("✅ Sistema de transcripción inicializado correctamente")
            
            # Información del sistema para Cloud
            if st.sidebar.checkbox("ℹ️ Info del Sistema", value=False):
                if PSUTIL_AVAILABLE:
                    try:
                        memory_gb = psutil.virtual_memory().available / (1024**3)
                        st.sidebar.info(f"📊 Memoria disponible: {memory_gb:.1f} GB")
                        st.sidebar.info("🔧 Modelo optimizado para Cloud")
                    except Exception:
                        st.sidebar.info("📊 Monitor de sistema temporalmente no disponible")
                        st.sidebar.info("🔧 Modelo optimizado para Cloud")
                else:
                    st.sidebar.info("📊 Información del sistema no disponible en esta versión")
                    st.sidebar.info("🔧 Modelo optimizado para Cloud")
                
        except Exception as e:
            st.error(f"❌ Error inicializando sistema: {e}")
            st.stop()

        st.subheader("📝 Formatos Soportados")
        st.info("WAV, MP3, MP4, AVI, MOV, FLAC, M4A, OGG, WEBM")
        
        st.subheader("📏 Límites")
        st.info("Tamaño máximo: 100MB por archivo")
        
        st.subheader("✨ Mejoras de Transcripción")
        st.success("""
        **🚀 Transcripción Profesional:**
        • Optimización automática de audio
        • Corrección de términos específicos
        • Limpieza de símbolos y ruido
        • Capitalización inteligente
        • Mayor precisión en español
        """)
        
        st.subheader("🎯 Análisis Incluido")
        st.info("""
        • **Performance del Asesor**
        • **Protocolo de Atención**
        • **Calidad del Tono**
        • **Resolución de Problemas**
        • **Palabras Clave**
        """)
        
        # Créditos del desarrollador
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #E10A68, #0066CC); 
                    color: white; border-radius: 8px; margin-top: 20px;">
            <h4 style="margin: 0;">👨‍💻 Desarrollador</h4>
            <p style="margin: 5px 0; font-size: 0.9em;">
                <strong>Mauro Rementeria</strong><br>
                <a href="mailto:mauroere@gmail.com" style="color: #FFD700; text-decoration: none;">
                    📧 mauroere@gmail.com
                </a>
            </p>
            <p style="font-size: 0.8em; margin: 5px 0; opacity: 0.9;">
                🚀 Soluciones IA
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Área principal
    uploaded_files = st.file_uploader(
        "🎵 Sube uno o varios archivos de audio",
        type=['wav', 'mp3', 'mp4', 'avi', 'mov', 'flac', 'm4a', 'ogg', 'webm'],
        accept_multiple_files=True,
        help="Arrastra y suelta archivos aquí o haz clic para seleccionar"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} archivo(s) cargado(s)")
        
        # Botón de procesamiento con estado mejorado
        processing_key = "processing_audio"
        
        # Inicializar estado si no existe
        if processing_key not in st.session_state:
            st.session_state[processing_key] = False
        
        # Determinar si mostrar botón o estado de procesamiento
        is_processing = st.session_state[processing_key]
        
        if not is_processing:
            # Mostrar botón solo si no está procesando
            process_button = st.button(
                "🚀 Procesar Archivos",
                type="primary",
                key="process_button",
                help="Iniciar transcripción y análisis de todos los archivos"
            )
            
            if process_button:
                # Marcar como procesando inmediatamente
                st.session_state[processing_key] = True
                st.rerun()  # Necesario para actualizar el estado del botón
        else:
            # Mostrar estado de procesamiento en lugar del botón
            st.info("⏳ **Procesando archivos...** Por favor espera a que termine el proceso actual.")
            st.warning("🚫 **No refresques la página** - El procesamiento está en curso")
        
        # Solo ejecutar procesamiento si el estado lo indica
        if st.session_state[processing_key]:
            # Progress bars
            main_progress = st.progress(0)
            status_text = st.empty()
            
            results = []
            total_files = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Actualizar progreso general
                overall_progress = i / total_files
                main_progress.progress(overall_progress)
                
                # Crear contenedor para progreso detallado
                progress_container = st.container()
                with progress_container:
                    st.markdown(f"### � Procesando: {uploaded_file.name}")
                    file_progress = st.progress(0)
                    file_status = st.empty()
                
                # Paso 1: Preparando archivo
                file_progress.progress(0.1)
                file_status.text("📋 Preparando archivo para procesamiento...")
                
                # Crear archivo temporal de forma más robusta
                file_extension = os.path.splitext(uploaded_file.name)[1]
                if not file_extension:
                    file_extension = '.wav'  # Default extension
                
                try:
                    # Paso 2: Creando archivo temporal
                    file_progress.progress(0.2)
                    file_status.text("💾 Guardando archivo temporal...")
                    
                    # Crear archivo temporal
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        # Escribir contenido del archivo
                        file_content = uploaded_file.getbuffer()
                        tmp_file.write(file_content)
                        tmp_file.flush()  # Asegurar que se escriba al disco
                        os.fsync(tmp_file.fileno())  # Forzar escritura al disco
                        tmp_path = tmp_file.name
                    
                    # Pequeña pausa para asegurar que el archivo esté disponible
                    import time
                    time.sleep(0.1)
                    
                    # Paso 3: Verificando archivo
                    file_progress.progress(0.3)
                    file_status.text("🔍 Verificando integridad del archivo...")
                    
                    # Verificar que el archivo se creó correctamente
                    if not os.path.exists(tmp_path):
                        results.append({
                            "filename": uploaded_file.name,
                            "success": False,
                            "error": "No se pudo crear el archivo temporal"
                        })
                        continue
                    
                    # Verificar tamaño del archivo
                    file_size = os.path.getsize(tmp_path)
                    if file_size == 0:
                        results.append({
                            "filename": uploaded_file.name,
                            "success": False,
                            "error": "El archivo temporal está vacío"
                        })
                        continue
                        
                except Exception as e:
                    results.append({
                        "filename": uploaded_file.name,
                        "success": False,
                        "error": f"Error al crear archivo temporal: {str(e)}"
                    })
                    continue
                
                try:
                    # Paso 4: Validando formato de audio
                    file_progress.progress(0.4)
                    file_status.text("✅ Validando formato de audio...")
                    
                    is_valid, validation_msg = validate_audio_file(tmp_path)
                    
                    if not is_valid:
                        results.append({
                            "filename": uploaded_file.name,
                            "success": False,
                            "error": validation_msg
                        })
                        continue
                    
                    # Paso 5: Iniciando transcripción mejorada
                    file_progress.progress(0.5)
                    file_status.text("🎙️ Iniciando transcripción avanzada...")
                    
                    # Progreso detallado para transcripción mejorada
                    progress_steps = [
                        (0.55, "🔧 Optimizando calidad de audio..."),
                        (0.65, "🧠 Procesando con IA avanzada..."),
                        (0.75, "📝 Convirtiendo audio a texto..."),
                        (0.85, "✨ Limpiando y profesionalizando texto...")
                    ]
                    
                    for progress_value, message in progress_steps:
                        file_progress.progress(progress_value)
                        file_status.text(message)
                        time.sleep(0.3)  # Pausa para mostrar progreso
                    
                    # Uso de función mejorada con fallback robusto
                    transcription, error = transcribe_with_enhanced_quality(model, tmp_path)
                    
                    if error:
                        # Error específico con sugerencias mejoradas
                        st.error(f"❌ Error en transcripción de {uploaded_file.name}")
                        st.markdown(f"**Detalle del error**: {error}")
                        
                        # Análisis específico del error para dar mejores sugerencias
                        error_lower = error.lower()
                        
                        if "tensor" in error_lower and "size" in error_lower:
                            st.warning("� **Problema de Compatibilidad de Tensor Detectado**")
                            st.info("""
                            💡 **Sugerencias para resolver este error**:
                            • Este archivo tiene una estructura que causa conflictos de tensor
                            • Intenta convertir el audio a formato WAV con menor calidad
                            • Reduce la duración del archivo (divide en partes más pequeñas)
                            • Usa un software como Audacity para re-exportar el audio
                            """)
                            
                        elif "memory" in error_lower:
                            st.info("""
                            💡 **Problema de Memoria**:
                            • El archivo es muy grande para procesar
                            • Intenta con archivos más pequeños (menos de 10 minutos)
                            • Divide el audio en segmentos más cortos
                            """)
                            
                        elif "format" in error_lower:
                            st.info("""
                            💡 **Problema de Formato**:
                            • El formato de audio no es compatible
                            • Convierte a MP3, WAV o M4A
                            • Verifica que el archivo no esté corrupto
                            """)
                            
                        elif "estrategia" in error_lower:
                            st.warning("⚠️ **Error en Todas las Estrategias**")
                            st.info("""
                            💡 **Opciones disponibles**:
                            • El archivo puede estar corrupto o tener un formato problemático
                            • Intenta re-grabar o re-exportar el audio
                            • Usa un formato más estándar como MP3 o WAV
                            • Verifica que el audio contenga voz humana claramente audible
                            """)
                        else:
                            st.info("""
                            💡 **Sugerencias generales**:
                            • Verifica que el archivo contenga audio válido
                            • Intenta con un formato diferente (MP3, WAV, M4A)
                            • Asegúrate de que el audio no esté corrupto
                            • Reduce el ruido de fondo si es posible
                            """)
                        
                        results.append({
                            "filename": uploaded_file.name,
                            "success": False,
                            "error": error
                        })
                        continue
                    
                    # Paso 6: Analizando performance
                    file_progress.progress(0.9)
                    file_status.text("📊 Analizando performance del asesor...")
                    
                    analysis = analyze_transcription(transcription)
                    
                    # Paso 7: Finalizando
                    file_progress.progress(1.0)
                    file_status.text("✅ ¡Análisis completado exitosamente!")
                    
                    # Guardar resultado
                    file_id = str(uuid.uuid4())
                    result_data = {
                        "file_id": file_id,
                        "filename": uploaded_file.name,
                        "transcription": transcription,
                        "analysis": analysis,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Guardar en archivo JSON
                    result_path = os.path.join(TRANSCRIPTIONS_FOLDER, f"{file_id}.json")
                    with open(result_path, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=2)
                    
                    results.append({
                        "filename": uploaded_file.name,
                        "success": True,
                        "transcription": transcription,
                        "analysis": analysis,
                        "file_id": file_id
                    })
                    
                except Exception as e:
                    st.error(f"Error procesando {uploaded_file.name}: {str(e)}")
                    results.append({
                        "filename": uploaded_file.name,
                        "success": False,
                        "error": str(e)
                    })
                
                finally:
                    # Limpiar archivo temporal
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            
            # Completar progreso
            main_progress.progress(1.0)
            status_text.text("🎉 ¡Todos los archivos han sido procesados!")
            
            # Resetear estado de procesamiento
            st.session_state[processing_key] = False
            
            # Limpiar contenedores de progreso individual
            time.sleep(1)  # Breve pausa para que el usuario vea el éxito
            
            # Mostrar resultados
            st.markdown("---")
            st.header("📊 Resultados del Análisis")
            
            # Mensaje de resumen amigable
            total_processed = len(results)
            successful = [r for r in results if r.get('success', False)]
            failed = [r for r in results if not r.get('success', False)]
            
            if len(successful) == total_processed:
                st.success(f"🎉 ¡Excelente! Todos los {total_processed} archivos fueron procesados exitosamente")
            elif len(successful) > 0:
                st.warning(f"⚠️ Se procesaron {len(successful)} de {total_processed} archivos. {len(failed)} tuvieron errores.")
            else:
                st.error(f"❌ No se pudo procesar ningún archivo. Revisa los errores a continuación.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ Exitosos", len(successful))
            with col2:
                st.metric("❌ Errores", len(failed))
            
            # Mostrar errores
            if failed:
                st.subheader("❌ Archivos con Errores")
                for result in failed:
                    st.error(f"**{result['filename']}**: {result.get('error', 'Error desconocido')}")
            
            # Mostrar resultados exitosos
            for result in successful:
                display_result(result)

def display_result(result):
    """Mostrar resultado individual"""
    st.markdown("---")
    st.subheader(f"🎵 {result['filename']}")
    
    analysis = result.get('analysis', {})
    
    # Mostrar métricas de performance si existen
    if 'performance' in analysis:
        display_performance_metrics(analysis)
    
    # Estadísticas básicas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Palabras", analysis.get('word_count', 0))
    with col2:
        st.metric("⏱️ Duración est.", f"{analysis.get('estimated_duration', 0):.1f} min")
    with col3:
        st.metric("🔤 Caracteres", analysis.get('character_count', 0))
    
    # Transcripción
    st.subheader("📄 Transcripción")
    st.text_area(
        "Contenido:",
        value=result.get('transcription', ''),
        height=150,
        key=f"transcript_{result.get('file_id', 'unknown')}"
    )
    
    # Botones de acción
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Botón de copiado sin rerun usando la nueva función
        transcript_text = result.get('transcription', '')
        file_id = result.get('file_id', 'unknown').replace('-', '_')
        copy_button_html = create_copy_button(
            text=transcript_text,
            button_text="📋 Copiar Transcripción",
            button_id=f"transcript_{file_id}",
            success_message="✅ Transcripción copiada al portapapeles"
        )
        st.markdown(copy_button_html, unsafe_allow_html=True)
    
    with col2:
        # Usar expander en lugar de session state para evitar reruns
        with st.expander("🤖 ANALIZAR CON IA"):
            prompt = generate_chatgpt_prompt(
                result.get('transcription', ''),
                result.get('analysis', {}),
                result.get('filename', '')
            )
            
            # Botón de copiar para el prompt usando la nueva función
            file_id = result.get('file_id', 'unknown').replace('-', '_')
            copy_prompt_html = create_copy_button(
                text=prompt,
                button_text="📋 Copiar Prompt para IA",
                button_id=f"prompt_{file_id}",
                success_message="✅ Prompt copiado. ¡Pégalo en ChatGPT!"
            )
            st.markdown(copy_prompt_html, unsafe_allow_html=True)
            
            st.info("📋 Usa el botón de arriba para copiar automáticamente, o selecciona el texto manualmente:")
            st.text_area(
                "Prompt:", 
                value=prompt, 
                height=300, 
                key=f"prompt_{result.get('file_id', 'unknown')}",
                help="Selecciona todo (Ctrl+A) y copia (Ctrl+C)"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.link_button("🔗 Abrir ChatGPT", "https://chat.openai.com")
            with col_b:
                st.link_button("🔗 Abrir Claude", "https://claude.ai")
    
    with col3:
        # Descargar análisis en múltiples formatos
        st.subheader("📊 Descargar Análisis")
        
        file_base_name = result.get('filename', 'archivo').split('.')[0]
        
        # Excel
        excel_data = generate_excel_report(result, analysis)
        st.download_button(
            label="📈 Excel (.xlsx)",
            data=excel_data,
            file_name=f"analisis_{file_base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_excel_{result.get('file_id', 'unknown')}"
        )
        
        # CSV
        csv_data = generate_csv_report(result, analysis)
        st.download_button(
            label="📋 CSV (.csv)",
            data=csv_data,
            file_name=f"analisis_{file_base_name}.csv",
            mime="text/csv",
            key=f"download_csv_{result.get('file_id', 'unknown')}"
        )
    
    with col4:
        st.subheader("📄 Descargar Completo")
        
        # Word/HTML
        word_data = generate_word_report(result, analysis)
        st.download_button(
            label="� Word (.html)",
            data=word_data,
            file_name=f"reporte_{file_base_name}.html",
            mime="text/html",
            key=f"download_word_{result.get('file_id', 'unknown')}"
        )
        
        # JSON (mantener opción original)
        report_data = {
            'archivo': result.get('filename', ''),
            'fecha_analisis': datetime.now().isoformat(),
            'transcripcion': result.get('transcription', ''),
            'analisis_performance': analysis.get('performance', {}),
            'metricas_basicas': {
                'palabras': analysis.get('word_count', 0),
                'caracteres': analysis.get('character_count', 0),
                'duracion_estimada': analysis.get('estimated_duration', 0)
            }
        }
        
        st.download_button(
            label="� JSON (.json)",
            data=json.dumps(report_data, ensure_ascii=False, indent=2),
            file_name=f"datos_{file_base_name}.json",
            mime="application/json",
            key=f"download_json_{result.get('file_id', 'unknown')}"
        )

def process_files(uploaded_files, model):
    """Procesar archivos de audio"""
    
    # Progress bars
    main_progress = st.progress(0)
    status_text = st.empty()
    
    results = []
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Actualizar progreso
        progress = i / total_files
        main_progress.progress(progress)
        status_text.text(f"📁 Procesando {uploaded_file.name} ({i+1}/{total_files})")
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            # Validar archivo
            is_valid, validation_msg = validate_audio_file(tmp_path)
            
            if not is_valid:
                results.append({
                    "filename": uploaded_file.name,
                    "success": False,
                    "error": validation_msg
                })
                continue
            
            # Transcribir
            status_text.text(f"🎙️ Transcribiendo {uploaded_file.name}...")
            transcription, error = transcribe_with_fallback(model, tmp_path)
            
            if error:
                results.append({
                    "filename": uploaded_file.name,
                    "success": False,
                    "error": error
                })
                continue
            
            # Analizar
            status_text.text(f"📊 Analizando {uploaded_file.name}...")
            analysis = analyze_transcription(transcription)
            
            # Guardar resultado
            file_id = str(uuid.uuid4())
            result_data = {
                "file_id": file_id,
                "filename": uploaded_file.name,
                "transcription": transcription,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            # Guardar en archivo JSON
            result_path = os.path.join(TRANSCRIPTIONS_FOLDER, f"{file_id}.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            results.append({
                "filename": uploaded_file.name,
                "success": True,
                "transcription": transcription,
                "analysis": analysis,
                "file_id": file_id
            })
            
        except Exception as e:
            results.append({
                "filename": uploaded_file.name,
                "success": False,
                "error": f"Error inesperado: {str(e)}"
            })
        
        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    # Completar progreso
    main_progress.progress(1.0)
    status_text.text("✅ ¡Procesamiento completado!")
    
    # Mostrar resultados
    display_results(results)

def display_results(results):
    """Mostrar resultados del procesamiento"""
    
    st.markdown("---")
    st.header("📋 Resultados del Análisis")
    
    # Estadísticas generales
    successful = len([r for r in results if r["success"]])
    failed = len(results) - successful
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Procesados", len(results))
    with col2:
        st.metric("✅ Exitosos", successful)
    with col3:
        st.metric("❌ Fallidos", failed)
    
    # Mostrar cada resultado
    for result in results:
        if result["success"]:
            display_successful_result(result)
        else:
            display_error_result(result)

def display_successful_result(result):
    """Mostrar resultado exitoso"""
    analysis = result["analysis"]
    perf = analysis.get("performance", {})
    
    with st.expander(f"📁 {result['filename']} - Score: {round(perf.get('overall_score', 0))}%", expanded=True):
        
        # Métricas de performance
        st.subheader("📊 Evaluación de Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score = perf.get('protocol_score', 0)
            color = "success" if score >= 80 else "warning" if score >= 60 else "danger"
            st.metric("🔖 Protocolo", f"{round(score)}%")
            
        with col2:
            score = perf.get('tone_score', 0)
            color = "success" if score >= 80 else "warning" if score >= 60 else "danger"
            st.metric("🎭 Tono", f"{round(score)}%")
            
        with col3:
            score = perf.get('resolution_score', 0)
            color = "success" if score >= 80 else "warning" if score >= 60 else "danger"
            st.metric("🔧 Resolución", f"{round(score)}%")
            
        with col4:
            score = perf.get('overall_score', 0)
            color = "success" if score >= 80 else "warning" if score >= 60 else "danger"
            st.metric("⭐ General", f"{round(score)}%")
        
        # Análisis detallado en pestañas
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Protocolo", "🎭 Tono", "🔧 Resolución", "⚠️ Falencias", "📝 Transcripción"
        ])
        
        with tab1:
            st.subheader("✅ Cumplimiento de Protocolo")
            protocol = perf.get('protocol_analysis', {})
            protocol_labels = {
                'saludo_inicial': 'Saludo inicial',
                'identificacion': 'Identificación del cliente',
                'pregunta_ayuda': 'Pregunta sobre necesidades',
                'despedida': 'Despedida apropiada'
            }
            
            for key, value in protocol.items():
                label = protocol_labels.get(key, key)
                if value:
                    st.success(f"✅ {label}")
                else:
                    st.error(f"❌ {label}")
        
        with tab2:
            st.subheader("🎭 Análisis de Tono")
            tone_analysis = perf.get('tone_analysis', {})
            
            primary_tone = tone_analysis.get('primary_tone', 'No detectado')
            tone_icons = {
                'amable': '😊', 'empatico': '❤️', 'profesional': '💼',
                'cortado': '😐', 'frustrado': '😤', 'agresivo': '😠', 'neutral': '😶'
            }
            icon = tone_icons.get(primary_tone, '🤔')
            
            st.info(f"{icon} **Tono principal:** {primary_tone}")
            
            indicators = tone_analysis.get('indicators', {})
            if indicators:
                st.subheader("📈 Indicadores por tipo:")
                for tone, count in indicators.items():
                    if count > 0:
                        st.write(f"• **{tone.title()}:** {count} menciones")
        
        with tab3:
            st.subheader("🔧 Resolución de Problemas")
            resolution = perf.get('problem_resolution', {})
            
            if resolution.get('resolved', False):
                st.success("✅ Problema resuelto")
            else:
                st.error("❌ Problema no resuelto")
            
            if resolution.get('type'):
                st.info(f"📋 **Tipo:** {resolution['type']}")
            
            if resolution.get('follow_up_needed'):
                st.warning("⚠️ Requiere seguimiento")
        
        with tab4:
            st.subheader("⚠️ Falencias Detectadas")
            falencias = perf.get('falencias', [])
            mejoras = perf.get('puntos_mejora', [])
            
            if falencias:
                st.error("**Falencias encontradas:**")
                for falencia in falencias:
                    st.write(f"• {falencia}")
            else:
                st.success("✅ No se detectaron falencias críticas")
            
            if mejoras:
                st.info("**Puntos de mejora sugeridos:**")
                for mejora in mejoras:
                    st.write(f"• {mejora}")
        
        with tab5:
            st.subheader("📝 Transcripción Completa")
            st.text_area("Texto transcrito:", result["transcription"], height=200, disabled=True)
            
            # Estadísticas básicas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Palabras", analysis["word_count"])
            with col2:
                st.metric("⏱️ Duración estimada", f"{analysis['estimated_duration']:.1f} min")
            with col3:
                st.metric("🔤 Caracteres", analysis["character_count"])
        
        # Botones de acción
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(f"📋 Copiar Transcripción", key=f"copy_{result['file_id']}"):
                st.code(result["transcription"])
                st.success("✅ Transcripción mostrada arriba para copiar")
        
        with col2:
            # Usar expander también aquí para evitar reruns
            with st.expander("🤖 Generar Prompt ChatGPT"):
                chatgpt_prompt = generate_chatgpt_prompt(
                    result["transcription"], 
                    analysis, 
                    result["filename"]
                )
                
                # Botón de copiar específico para el prompt
                if st.button("📋 Copiar Prompt", key=f"copy_prompt2_{result['file_id']}"):
                    st.code(chatgpt_prompt, language=None)
                    st.success("✅ Prompt mostrado arriba. Selecciona todo (Ctrl+A) y copia (Ctrl+C)")
                
                st.info("📋 Copia este texto y pégalo en ChatGPT:")
                st.text_area(
                    "Prompt optimizado:",
                    chatgpt_prompt,
                    height=300,
                    key=f"prompt2_{result['file_id']}",
                    help="Selecciona todo (Ctrl+A) y copia (Ctrl+C)"
                )
                
                # Enlaces directos
                col_a, col_b = st.columns(2)
                with col_a:
                    st.link_button("🔗 Abrir ChatGPT", "https://chat.openai.com")
                with col_b:
                    st.link_button("🔗 Abrir Claude", "https://claude.ai")
        
        with col3:
            # Múltiples opciones de descarga
            st.subheader("📊 Descargar Reporte")
            
            file_base_name = result["filename"].split('.')[0]
            
            # Excel
            excel_data = generate_excel_report(result, analysis)
            st.download_button(
                "📈 Descargar Excel",
                excel_data,
                file_name=f"analisis_{file_base_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_excel2_{result['file_id']}"
            )
            
            # CSV
            csv_data = generate_csv_report(result, analysis)
            st.download_button(
                "� Descargar CSV",
                csv_data,
                file_name=f"analisis_{file_base_name}.csv",
                mime="text/csv",
                key=f"download_csv2_{result['file_id']}"
            )
            
            # Word/HTML
            word_data = generate_word_report(result, analysis)
            st.download_button(
                "📝 Descargar Word",
                word_data,
                file_name=f"reporte_{file_base_name}.html",
                mime="text/html",
                key=f"download_word2_{result['file_id']}"
            )
            
            # JSON (para desarrolladores)
            report = {
                "archivo": result["filename"],
                "fecha_analisis": datetime.now().isoformat(),
                "transcripcion": result["transcription"],
                "analisis_performance": perf,
                "metricas_basicas": {
                    "palabras": analysis["word_count"],
                    "caracteres": analysis["character_count"],
                    "duracion_estimada": analysis["estimated_duration"]
                },
                "resumen_ejecutivo": {
                    "score_general": perf.get('overall_score', 0),
                    "protocolo_cumplido": perf.get('protocol_score', 0),
                    "calidad_tono": perf.get('tone_score', 0),
                    "resolucion_efectiva": perf.get('resolution_score', 0),
                    "problema_resuelto": perf.get('problem_resolution', {}).get('resolved', False)
                }
            }
            
            json_str = json.dumps(report, ensure_ascii=False, indent=2)
            st.download_button(
                "� Descargar JSON",
                json_str,
                file_name=f"datos_{file_base_name}.json",
                mime="application/json",
                key=f"download_json2_{result['file_id']}"
            )

def display_error_result(result):
    """Mostrar resultado con error"""
    with st.expander(f"❌ {result['filename']} - Error", expanded=False):
        st.error(f"**Error:** {result['error']}")
        st.info("💡 **Sugerencias:**\n- Verifica que el archivo no esté corrupto\n- Asegúrate de que el formato sea compatible\n- Intenta con un archivo más pequeño")

if __name__ == "__main__":
    main()
    
    # Footer profesional con créditos del desarrollador
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #E10A68, #0066CC); 
                color: white; border-radius: 10px; margin-top: 30px;">
        <h3>🚀 Transcriptor de Audios</h3>
        <p style="margin: 10px 0;">
            <strong>Desarrollado por:</strong> Mauro Rementeria<br>
            <strong>Email:</strong> <a href="mailto:mauroere@gmail.com" style="color: #FFD700;">mauroere@gmail.com</a><br>
            <strong>Tecnologías:</strong> Python | Streamlit | OpenAI Whisper | IA
        </p>
        <p style="font-size: 0.9em; opacity: 0.9; margin-top: 15px;">
            ⚡ Transcripción profesional con análisis de performance | 
            🎯 Optimizado para atención al cliente | 
            📊 Insights automatizados
        </p>
        <p style="font-size: 0.8em; margin-top: 10px;">
            © 2025 - Todos los derechos reservados
        </p>
    </div>
    """, unsafe_allow_html=True)
