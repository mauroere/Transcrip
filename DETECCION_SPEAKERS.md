# 🎭 Detección de Interlocutores - Nueva Funcionalidad

## 📋 Resumen de la Implementación

He implementado un sistema completo de **detección de interlocutores** que convierte las transcripciones tradicionales en **diálogos estructurados** con identificación automática de quién dice qué.

## 🚀 Características Implementadas

### 1. **Extracción de Características de Voz**
- **Análisis de Pitch (Frecuencia Fundamental)**: Detecta diferencias en el tono de voz
- **Coeficientes MFCC**: Analiza características espectrales únicas de cada voz
- **Características Espectrales**: Centroide, rolloff y zero-crossing rate
- **Análisis de Energía**: Detecta patrones de volumen y intensidad
- **Segmentación Temporal**: Divide el audio en segmentos de 2 segundos para análisis

### 2. **Clustering Inteligente de Speakers**
- **Machine Learning**: Usa K-Means con optimización automática
- **Detección Automática**: Determina el número óptimo de interlocutores (2-4)
- **Validación de Calidad**: Usa métricas de silhouette para validar la detección
- **Normalización**: Estandariza características para mejor precisión

### 3. **Formato de Diálogo Avanzado**
- **Timestamps Precisos**: Cada intervención tiene su momento temporal
- **Identificación Automática**: Clasifica automáticamente como ASESOR/CLIENTE
- **Agrupación Inteligente**: Combina segmentos consecutivos del mismo speaker
- **Formato Legible**: Presenta conversaciones en formato natural

### 4. **Análisis de Dinámicas Conversacionales**
- **Distribución de Participación**: Porcentaje de palabras por interlocutor
- **Patrones de Dominancia**: Detecta quién domina la conversación
- **Tipo de Conversación**: Clasifica como explicativo, consultivo o colaborativo
- **Calidad de Interacción**: Evalúa duración y fluidez
- **Recomendaciones Específicas**: Sugerencias para mejorar dinámicas

## 📊 Ejemplo de Output

### Diálogo Formateado:
```
[00:15] 🎧 ASESOR: Hola buenos días, gracias por contactar a Movistar. Mi nombre es Carlos, ¿en qué puedo ayudarle hoy?

[00:23] 👤 CLIENTE: Hola, tengo un problema con mi internet que no está funcionando desde ayer.

[00:30] 🎧 ASESOR: Entiendo su situación. Permítame verificar el estado de su línea. ¿Podría proporcionarme su número de documento?

[00:45] 👤 CLIENTE: Sí, es 12345678.
```

### Análisis de Participación:
- **🎧 ASESOR**: 65% (234 palabras, 8 intervenciones)
- **👤 CLIENTE**: 35% (126 palabras, 6 intervenciones)

### Recomendaciones:
- ⚠️ El asesor habla demasiado. Debería dar más espacio al cliente para expresarse.

## 🔧 Implementación Técnica

### Funciones Principales Agregadas:

1. **`extract_speaker_features()`**: Extrae características de voz del audio
2. **`detect_speakers()`**: Aplica clustering para identificar interlocutores
3. **`format_dialogue_transcription()`**: Convierte texto en formato diálogo
4. **`analyze_conversation_dynamics()`**: Analiza patrones conversacionales
5. **`transcribe_with_speaker_detection()`**: Función principal integrada

### Librerías Utilizadas:
- **librosa**: Análisis de audio y extracción de características
- **scikit-learn**: Machine learning para clustering
- **numpy**: Operaciones matemáticas y estadísticas

## 🎯 Casos de Uso en Call Center

### Para Supervisores:
- **Monitoreo de Calidad**: Identificar quién habla demasiado/poco
- **Evaluación de Dinámicas**: Analizar balance conversacional
- **Training Personalizado**: Feedback específico por asesor

### Para Asesores:
- **Auto-evaluación**: Ver distribución propia de participación
- **Mejora de Técnicas**: Entender patrones conversacionales
- **Seguimiento de Progreso**: Comparar diferentes llamadas

### Para Analistas:
- **Reportes Detallados**: Estadísticas por interlocutor
- **Patrones de Comportamiento**: Análisis de tendencias
- **Optimización de Procesos**: Identificar best practices

## 📈 Métricas Disponibles

### Métricas Básicas:
- Número de interlocutores detectados
- Turnos de conversación
- Distribución de palabras por speaker

### Métricas Avanzadas:
- Tipo de conversación (explicativo, consultivo, colaborativo)
- Patrón de dominancia (balanceado, asesor dominante, cliente dominante)
- Calidad de interacción (limitada, adecuada, extensa)

### Recomendaciones Automáticas:
- Sugerencias para mejorar balance conversacional
- Alertas sobre conversaciones muy largas/cortas
- Recomendaciones de técnicas de comunicación

## 🔄 Integración con Sistema Existente

La nueva funcionalidad está **totalmente integrada** con el sistema actual:

- ✅ **Compatible con transcripción normal**: Si falla la detección, usa transcripción tradicional
- ✅ **Preserva análisis existente**: Mantiene todos los análisis de performance actuales
- ✅ **Mejora prompts de ChatGPT**: Incluye información de diálogo en análisis de IA
- ✅ **Export completo**: Incluye datos de diálogo en reportes Excel/CSV
- ✅ **Copy buttons mejorados**: Botones separados para diálogo y transcripción normal

## 🎨 Interfaz de Usuario

### Nuevas Secciones:
1. **🎭 Análisis de Interlocutores**: Métricas de conversación
2. **📊 Distribución de Participación**: Visualización de balance
3. **💬 Diálogo por Interlocutores**: Pestañas para formato diálogo vs normal
4. **💡 Recomendaciones de Dinámicas**: Sugerencias específicas

### Experiencia de Usuario:
- **Pestañas intuitivas**: Fácil navegación entre formatos
- **Visualización clara**: Barras de progreso para distribución
- **Copy buttons específicos**: Para diálogo y transcripción por separado
- **Información contextual**: Tooltips y explicaciones

## 🔍 Precisión y Limitaciones

### Funciona Mejor Con:
- ✅ Audio de buena calidad
- ✅ Voces claramente diferenciadas
- ✅ Conversaciones de 2-4 interlocutores
- ✅ Audio con poco ruido de fondo

### Limitaciones:
- ⚠️ Requiere librosa y scikit-learn instalados
- ⚠️ Puede tener dificultades con voces muy similares
- ⚠️ Funciona mejor con audio mono o convertido a mono
- ⚠️ Segmentos muy cortos pueden no detectarse correctamente

## 🚀 Próximas Mejoras Posibles

1. **Entrenamiento con datos específicos** de call center Movistar
2. **Identificación de emociones** por interlocutor
3. **Detección de interrupciones** y solapamientos
4. **Análisis de velocidad de habla** por speaker
5. **Integración con modelos de NLP** específicos para diálogos

## 🎉 Resultado Final

La implementación convierte audios simples en **análisis conversacionales completos**, proporcionando insights valiosos para mejorar la calidad del servicio al cliente en Movistar. Los supervisores ahora pueden evaluar no solo QUÉ se dice, sino también CÓMO se desarrolla la conversación entre asesor y cliente.

---

**Desarrollado por**: Mauro Rementeria  
**Email**: mauroere@gmail.com  
**Fecha**: Agosto 2025
