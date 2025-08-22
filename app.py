import os
import sys
import traceback

# Configurar FFmpeg si está en directorio local
ffmpeg_local = os.path.join(os.getcwd(), 'ffmpeg', 'ffmpeg.exe')
if os.path.exists(ffmpeg_local):
    ffmpeg_dir = os.path.dirname(ffmpeg_local)
    if ffmpeg_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
    print(f"✓ FFmpeg configurado desde: {ffmpeg_local}")
else:
    print("⚠️ FFmpeg local no encontrado, usando PATH del sistema")

try:
    import whisper
    print("✓ Whisper importado correctamente")
except ImportError as e:
    print(f"✗ Error importando Whisper: {e}")
    sys.exit(1)

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

try:
    from pydub import AudioSegment
    print("✓ Pydub importado correctamente")
except ImportError as e:
    print(f"✗ Error importando Pydub: {e}")
    print("Instala ffmpeg: https://ffmpeg.org/download.html")
    AudioSegment = None

import tempfile
import json
from datetime import datetime
import uuid
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'uploads'
TRANSCRIPTIONS_FOLDER = 'transcriptions'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'avi', 'mov', 'flac', 'm4a', 'ogg', 'webm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TRANSCRIPTIONS_FOLDER'] = TRANSCRIPTIONS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Crear directorios si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTIONS_FOLDER, exist_ok=True)

# Cargar modelo Whisper
print("Cargando modelo Whisper...")
whisper_model = whisper.load_model("base")
print("Modelo Whisper cargado exitosamente")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_audio_file(filepath):
    """Valida que el archivo de audio sea procesable"""
    try:
        if not os.path.exists(filepath):
            return False, "Archivo no encontrado"
        
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            return False, "Archivo vacío"
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            return False, "Archivo demasiado grande (>100MB)"
        
        # Intentar leer los primeros bytes para verificar que no esté corrupto
        try:
            with open(filepath, 'rb') as f:
                header = f.read(12)
                if len(header) < 4:
                    return False, "Archivo muy pequeño o corrupto"
        except Exception as e:
            return False, f"No se puede leer el archivo: {str(e)}"
        
        return True, "OK"
    except Exception as e:
        return False, f"Error validando archivo: {str(e)}"

def convert_to_wav(input_path):
    """Convierte el archivo de audio a formato WAV"""
    if AudioSegment is None:
        print("Warning: pydub no disponible, intentando usar el archivo directamente")
        return input_path
    
    try:
        audio = AudioSegment.from_file(input_path)
        # Convertir a mono si es estéreo para reducir tamaño
        if audio.channels > 1:
            audio = audio.set_channels(1)
        # Reducir sample rate si es muy alto
        if audio.frame_rate > 16000:
            audio = audio.set_frame_rate(16000)
        
        output_path = input_path.rsplit('.', 1)[0] + '.wav'
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Error convirtiendo archivo: {e}")
        print("Intentando usar archivo original...")
        return input_path
    """Convierte el archivo de audio a formato WAV"""
    if AudioSegment is None:
        print("Warning: pydub no disponible, intentando usar el archivo directamente")
        return input_path
    
    try:
        audio = AudioSegment.from_file(input_path)
        # Convertir a mono si es estéreo para reducir tamaño
        if audio.channels > 1:
            audio = audio.set_channels(1)
        # Reducir sample rate si es muy alto
        if audio.frame_rate > 16000:
            audio = audio.set_frame_rate(16000)
        
        output_path = input_path.rsplit('.', 1)[0] + '.wav'
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Error convirtiendo archivo: {e}")
        print("Intentando usar archivo original...")
        return input_path

def transcribe_with_fallback(audio_path):
    """Intenta transcribir con diferentes configuraciones si falla"""
    
    # Configuraciones para intentar
    configs = [
        {'language': 'es', 'task': 'transcribe', 'verbose': False},
        {'language': None, 'task': 'transcribe', 'verbose': False},  # Auto-detect language
        {'language': 'es', 'task': 'transcribe', 'verbose': False, 'fp16': False},  # Sin FP16
    ]
    
    for i, config in enumerate(configs):
        try:
            print(f"Intento {i+1}/3 con configuración: {config}")
            result = whisper_model.transcribe(audio_path, **config)
            
            if result and result.get('text', '').strip():
                print(f"✓ Transcripción exitosa en intento {i+1}")
                return {
                    'text': result['text'],
                    'segments': result.get('segments', []),
                    'language': result.get('language', 'es')
                }
            else:
                print(f"Intento {i+1} devolvió texto vacío")
                
        except Exception as e:
            print(f"Intento {i+1} falló: {str(e)}")
            if i == len(configs) - 1:  # Último intento
                print(f"Todos los intentos fallaron")
                print(f"Último error: {traceback.format_exc()}")
    
    return None

def transcribe_with_whisper(audio_path):
    """Transcribe audio usando Whisper con logging detallado"""
    try:
        print(f"=== INICIANDO TRANSCRIPCIÓN ===")
        print(f"Archivo: {audio_path}")
        print(f"Existe: {os.path.exists(audio_path)}")
        
        if not os.path.exists(audio_path):
            print(f"ERROR: El archivo {audio_path} no existe")
            return None
        
        # Verificar tamaño del archivo
        file_size = os.path.getsize(audio_path)
        print(f"Tamaño del archivo: {file_size} bytes ({file_size/(1024*1024):.2f} MB)")
        
        if file_size == 0:
            print(f"ERROR: El archivo está vacío")
            return None
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            print(f"WARNING: Archivo muy grande, esto puede tomar mucho tiempo")
        
        print(f"Iniciando transcripción con Whisper...")
        result = whisper_model.transcribe(
            audio_path, 
            language='es',
            task='transcribe',
            verbose=False
        )
        
        print(f"✓ Transcripción exitosa")
        print(f"Texto transcrito: {result['text'][:100]}...")
        
        return {
            'text': result['text'],
            'segments': result.get('segments', []),
            'language': result.get('language', 'es')
        }
    except Exception as e:
        print(f"❌ ERROR en transcripción Whisper:")
        print(f"Tipo de error: {type(e).__name__}")
        print(f"Mensaje: {str(e)}")
        print(f"Traceback completo:")
        print(traceback.format_exc())
        return None
    """Transcribe audio usando Whisper"""
    try:
        print(f"=== INICIANDO TRANSCRIPCIÓN ===")
        print(f"Archivo: {audio_path}")
        print(f"Existe: {os.path.exists(audio_path)}")
        
        if not os.path.exists(audio_path):
            print(f"ERROR: El archivo {audio_path} no existe")
            return None
        
        # Verificar tamaño del archivo
        file_size = os.path.getsize(audio_path)
        print(f"Tamaño del archivo: {file_size} bytes ({file_size/(1024*1024):.2f} MB)")
        
        if file_size == 0:
            print(f"ERROR: El archivo está vacío")
            return None
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            print(f"WARNING: Archivo muy grande, esto puede tomar mucho tiempo")
        
        print(f"Iniciando transcripción con Whisper...")
        result = whisper_model.transcribe(
            audio_path, 
            language='es',
            task='transcribe',
            verbose=False
        )
        
        print(f"✓ Transcripción exitosa")
        print(f"Texto transcrito: {result['text'][:100]}...")
        
        return {
            'text': result['text'],
            'segments': result.get('segments', []),
            'language': result.get('language', 'es')
        }
    except Exception as e:
        print(f"❌ ERROR en transcripción Whisper:")
        print(f"Tipo de error: {type(e).__name__}")
        print(f"Mensaje: {str(e)}")
        print(f"Traceback completo:")
        print(traceback.format_exc())
        return None

def analyze_transcription(text):
    """Analiza la transcripción para extraer insights avanzados de performance del asesor"""
    
    # Convertir a minúsculas para análisis
    text_lower = text.lower()
    
    # Métricas básicas
    words = text.split()
    word_count = len(words)
    character_count = len(text)
    estimated_duration = word_count / 150  # ~150 palabras por minuto
    
    # === ANÁLISIS DE PROTOCOLO DE ATENCIÓN ===
    protocol_analysis = {
        'saludo_inicial': bool(
            any(phrase in text_lower for phrase in [
                'buenos días', 'buenas tardes', 'buenas noches', 'hola',
                'bienvenido', 'gracias por llamar', 'movistar le saluda'
            ])
        ),
        'identificacion': bool(
            any(phrase in text_lower for phrase in [
                'mi nombre es', 'soy', 'le habla', 'le atiende',
                'identificación', 'número de documento', 'dni'
            ])
        ),
        'pregunta_ayuda': bool(
            any(phrase in text_lower for phrase in [
                '¿en qué puedo ayudar?', '¿cómo puedo ayudar?',
                '¿cuál es su consulta?', '¿qué necesita?'
            ])
        ),
        'despedida': bool(
            any(phrase in text_lower for phrase in [
                'que tenga un buen día', 'hasta luego', 'gracias por llamar',
                'fue un placer', 'cualquier otra consulta'
            ])
        )
    }
    
    # === ANÁLISIS DE TONO Y ACTITUD ===
    tone_indicators = {
        'amable': sum([
            text_lower.count('por favor'),
            text_lower.count('con mucho gusto'),
            text_lower.count('será un placer'),
            text_lower.count('por supuesto'),
            text_lower.count('encantado'),
            text_lower.count('perfecto'),
            text_lower.count('excelente')
        ]),
        'empatico': sum([
            text_lower.count('entiendo'),
            text_lower.count('comprendo'),
            text_lower.count('me imagino'),
            text_lower.count('debe ser frustrante'),
            text_lower.count('lamento'),
            text_lower.count('disculpe'),
            text_lower.count('tiene razón')
        ]),
        'profesional': sum([
            text_lower.count('permítame'),
            text_lower.count('voy a revisar'),
            text_lower.count('voy a verificar'),
            text_lower.count('déjeme consultar'),
            text_lower.count('le confirmo'),
            text_lower.count('según veo'),
            text_lower.count('procedimiento')
        ]),
        'cortado': sum([
            text_lower.count('no puedo'),
            text_lower.count('no es posible'),
            text_lower.count('eso no'),
            text_lower.count('imposible'),
            text_lower.count('no se puede')
        ]),
        'frustrado': sum([
            text_lower.count('ya le dije'),
            text_lower.count('como le expliqué'),
            text_lower.count('tiene que entender'),
            text_lower.count('es obvio'),
            text_lower.count('claramente')
        ])
    }
    
    # === ANÁLISIS DE RESOLUCIÓN DE PROBLEMAS ===
    problem_resolution = {
        'identifico_problema': bool(
            any(phrase in text_lower for phrase in [
                '¿cuál es el problema?', '¿qué está pasando?',
                'cuénteme qué sucede', 'explíqueme', 'describe',
                'veo que tiene', 'entiendo que'
            ])
        ),
        'propuso_solucion': bool(
            any(phrase in text_lower for phrase in [
                'vamos a', 'podemos', 'le sugiero', 'recomiendo',
                'la solución es', 'voy a', 'haremos', 'procederemos'
            ])
        ),
        'siguio_protocolo': bool(
            any(phrase in text_lower for phrase in [
                'voy a revisar', 'consulto el sistema', 'verifico',
                'según el procedimiento', 'déjeme ver'
            ])
        ),
        'escalo_problema': bool(
            any(phrase in text_lower for phrase in [
                'supervisor', 'área técnica', 'especialista',
                'nivel superior', 'escalamiento', 'derivar'
            ])
        ),
        'confirmo_resolucion': bool(
            any(phrase in text_lower for phrase in [
                '¿se solucionó?', '¿funcionó?', '¿está bien ahora?',
                '¿necesita algo más?', '¿quedó resuelto?'
            ])
        )
    }
    
    # === ANÁLISIS DE CONOCIMIENTO TÉCNICO ===
    technical_knowledge = {
        'servicios_movistar': sum([
            text_lower.count('plan'),
            text_lower.count('línea'),
            text_lower.count('internet'),
            text_lower.count('televisión'),
            text_lower.count('móvil'),
            text_lower.count('fibra'),
            text_lower.count('roaming')
        ]),
        'terminos_tecnicos': sum([
            text_lower.count('configuración'),
            text_lower.count('router'),
            text_lower.count('modem'),
            text_lower.count('wifi'),
            text_lower.count('señal'),
            text_lower.count('cobertura'),
            text_lower.count('velocidad')
        ]),
        'procesos_internos': sum([
            text_lower.count('sistema'),
            text_lower.count('base de datos'),
            text_lower.count('historial'),
            text_lower.count('cuenta'),
            text_lower.count('facturación')
        ])
    }
    
    # === ANÁLISIS DE PROBLEMAS DETECTADOS ===
    problems_detected = {
        'problemas_tecnicos': sum([
            text_lower.count('no funciona'),
            text_lower.count('falla'),
            text_lower.count('error'),
            text_lower.count('problema'),
            text_lower.count('lento'),
            text_lower.count('intermitente'),
            text_lower.count('corte')
        ]),
        'problemas_facturacion': sum([
            text_lower.count('cobro'),
            text_lower.count('factura'),
            text_lower.count('descuento'),
            text_lower.count('cargo'),
            text_lower.count('deuda')
        ]),
        'problemas_servicio': sum([
            text_lower.count('atención'),
            text_lower.count('demora'),
            text_lower.count('espera'),
            text_lower.count('reclamo'),
            text_lower.count('queja')
        ])
    }
    
    # === CÁLCULO DE SCORES ===
    
    # Score de protocolo (0-100)
    protocol_score = (sum(protocol_analysis.values()) / len(protocol_analysis)) * 100
    
    # Score de tono (0-100)
    positive_tone = tone_indicators['amable'] + tone_indicators['empatico'] + tone_indicators['profesional']
    negative_tone = tone_indicators['cortado'] + tone_indicators['frustrado']
    total_tone_indicators = positive_tone + negative_tone
    
    if total_tone_indicators > 0:
        tone_score = (positive_tone / total_tone_indicators) * 100
    else:
        tone_score = 50  # Neutro si no hay indicadores
    
    # Score de resolución (0-100)
    resolution_score = (sum(problem_resolution.values()) / len(problem_resolution)) * 100
    
    # Score general (promedio ponderado)
    overall_score = (protocol_score * 0.3 + tone_score * 0.4 + resolution_score * 0.3)
    
    # === DETERMINACIÓN DE TONO DOMINANTE ===
    max_tone = max(tone_indicators.items(), key=lambda x: x[1])
    dominant_tone = max_tone[0] if max_tone[1] > 0 else 'neutral'
    
    # === IDENTIFICACIÓN DE FALENCIAS ===
    falencias = []
    
    if not protocol_analysis['saludo_inicial']:
        falencias.append("No realizó saludo inicial apropiado")
    if not protocol_analysis['identificacion']:
        falencias.append("No se identificó correctamente")
    if not protocol_analysis['pregunta_ayuda']:
        falencias.append("No preguntó cómo podía ayudar")
    if not protocol_analysis['despedida']:
        falencias.append("No realizó despedida apropiada")
    
    if tone_indicators['cortado'] > tone_indicators['amable']:
        falencias.append("Tono cortante o poco amable")
    if tone_indicators['frustrado'] > 2:
        falencias.append("Mostró signos de frustración")
    if tone_indicators['empatico'] == 0:
        falencias.append("Falta de empatía hacia el cliente")
    
    if not problem_resolution['identifico_problema']:
        falencias.append("No identificó claramente el problema")
    if not problem_resolution['propuso_solucion']:
        falencias.append("No propuso soluciones claras")
    if not problem_resolution['confirmo_resolucion']:
        falencias.append("No confirmó la resolución del problema")
    
    # === PUNTOS DE MEJORA ===
    mejoras = []
    
    if protocol_score < 75:
        mejoras.append("Mejorar seguimiento del protocolo de atención")
    if tone_score < 60:
        mejoras.append("Trabajar en un tono más amable y empático")
    if resolution_score < 70:
        mejoras.append("Fortalecer habilidades de resolución de problemas")
    if technical_knowledge['servicios_movistar'] < 3:
        mejoras.append("Ampliar conocimiento sobre servicios Movistar")
    if tone_indicators['empatico'] < 2:
        mejoras.append("Desarrollar mayor empatía hacia los clientes")
    
    # === ANÁLISIS AVANZADO ===
    analysis = {
        'word_count': word_count,
        'character_count': character_count,
        'estimated_duration': estimated_duration,
        
        'scores': {
            'protocol_score': round(protocol_score, 1),
            'tone_score': round(tone_score, 1),
            'resolution_score': round(resolution_score, 1),
            'overall_score': round(overall_score, 1)
        },
        
        'protocol_analysis': protocol_analysis,
        'tone_indicators': tone_indicators,
        'dominant_tone': dominant_tone,
        'problem_resolution': problem_resolution,
        'technical_knowledge': technical_knowledge,
        'problems_detected': problems_detected,
        
        'falencias': falencias,
        'puntos_mejora': mejoras,
        
        'resolvio_problema': problem_resolution['confirmo_resolucion'] and problem_resolution['propuso_solucion'],
        
        'keywords': {
            'saludos': sum([
                text_lower.count('buenos días'),
                text_lower.count('buenas tardes'),
                text_lower.count('hola')
            ]),
            'agradecimientos': sum([
                text_lower.count('gracias'),
                text_lower.count('agradezco')
            ]),
            'disculpas': sum([
                text_lower.count('disculpe'),
                text_lower.count('perdón'),
                text_lower.count('lo siento')
            ]),
            'servicios_movistar': technical_knowledge['servicios_movistar'],
            'problemas_tecnicos': problems_detected['problemas_tecnicos']
        },
        
        'sentiment_indicators': {
            'positive': positive_tone,
            'negative': negative_tone
        },
        
        'evaluation_summary': {
            'nivel_performance': (
                'Excelente' if overall_score >= 85 else
                'Bueno' if overall_score >= 70 else
                'Regular' if overall_score >= 55 else
                'Necesita Mejora'
            ),
            'areas_fortaleza': [
                area for area, value in {
                    'Protocolo de Atención': protocol_score,
                    'Tono y Actitud': tone_score,
                    'Resolución de Problemas': resolution_score
                }.items() if value >= 75
            ],
            'areas_criticas': [
                area for area, value in {
                    'Protocolo de Atención': protocol_score,
                    'Tono y Actitud': tone_score,
                    'Resolución de Problemas': resolution_score
                }.items() if value < 60
            ]
        }
    }
    
    return analysis

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No se encontraron archivos'}), 400
    
    files = request.files.getlist('files[]')
    results = []
    
    for file in files:
        if file.filename == '':
            continue
            
        if file and allowed_file(file.filename):
            # Generar nombre único para el archivo
            file_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            original_extension = filename.rsplit('.', 1)[1].lower()
            safe_filename = f"{file_id}.{original_extension}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            
            try:
                file.save(filepath)
                print(f"Archivo guardado: {filepath}")
                print(f"Tamaño del archivo: {os.path.getsize(filepath)} bytes")
                
                # Validar archivo antes de procesar
                is_valid, validation_message = validate_audio_file(filepath)
                if not is_valid:
                    print(f"Archivo inválido: {validation_message}")
                    results.append({
                        'filename': filename,
                        'error': f'Archivo inválido: {validation_message}'
                    })
                    try:
                        os.remove(filepath)
                    except:
                        pass
                    continue
                    
            except Exception as e:
                print(f"Error guardando archivo: {e}")
                results.append({
                    'filename': filename,
                    'error': f'Error al guardar archivo: {str(e)}'
                })
                continue
            
            # Convertir a WAV si es necesario y es posible
            converted_path = input_path = filepath
            if original_extension != 'wav' and AudioSegment is not None:
                converted_path = convert_to_wav(filepath)
                if converted_path != filepath:
                    # Eliminar archivo original después de conversión exitosa
                    try:
                        os.remove(filepath)
                        filepath = converted_path
                    except:
                        pass
            
            # Transcribir audio
            try:
                transcription_result = transcribe_with_fallback(filepath)
            except Exception as e:
                print(f"Error en transcripción: {e}")
                transcription_result = None
            
            if transcription_result:
                # Analizar transcripción
                analysis = analyze_transcription(transcription_result['text'])
                
                # Guardar transcripción y análisis
                transcription_data = {
                    'file_id': file_id,
                    'original_filename': filename,
                    'transcription': transcription_result['text'],
                    'segments': transcription_result.get('segments', []),
                    'language': transcription_result.get('language', 'es'),
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                }
                
                transcription_file = os.path.join(
                    app.config['TRANSCRIPTIONS_FOLDER'], 
                    f"{file_id}.json"
                )
                
                with open(transcription_file, 'w', encoding='utf-8') as f:
                    json.dump(transcription_data, f, ensure_ascii=False, indent=2)
                
                results.append({
                    'file_id': file_id,
                    'filename': filename,
                    'transcription': transcription_result['text'],
                    'analysis': analysis,
                    'success': True
                })
                
                # Limpiar archivo temporal
                try:
                    os.remove(filepath)
                except:
                    pass  # No es crítico si no se puede eliminar
            else:
                results.append({
                    'filename': filename,
                    'error': 'Error al transcribir el archivo'
                })
        else:
            results.append({
                'filename': file.filename,
                'error': 'Formato de archivo no permitido'
            })
    
    return jsonify({'results': results})

@app.route('/transcription/<file_id>')
def get_transcription(file_id):
    transcription_file = os.path.join(app.config['TRANSCRIPTIONS_FOLDER'], f"{file_id}.json")
    
    if os.path.exists(transcription_file):
        with open(transcription_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({'error': 'Transcripción no encontrada'}), 404

@app.route('/transcriptions')
def list_transcriptions():
    transcriptions = []
    
    for filename in os.listdir(app.config['TRANSCRIPTIONS_FOLDER']):
        if filename.endswith('.json'):
            file_path = os.path.join(app.config['TRANSCRIPTIONS_FOLDER'], filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                transcriptions.append({
                    'file_id': data['file_id'],
                    'original_filename': data['original_filename'],
                    'timestamp': data['timestamp'],
                    'word_count': data['analysis']['word_count'],
                    'estimated_duration': data['analysis']['estimated_duration']
                })
    
    # Ordenar por timestamp descendente
    transcriptions.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify({'transcriptions': transcriptions})

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
