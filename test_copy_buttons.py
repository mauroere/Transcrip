#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script - Verificación de Botones de Copiado
============================================
Este script verifica que los botones de copiado no causen rerun en Streamlit.

Desarrollado por: Mauro Rementeria - mauroere@gmail.com
"""

import streamlit as st

# Configurar página
st.set_page_config(
    page_title="Test Botones de Copiado",
    page_icon="📋",
    layout="wide"
)

st.title("🧪 Test - Botones de Copiado Sin Rerun")

# Crear función de copiado (copiada de streamlit_app.py)
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

# Inicializar contador en session state
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# Mostrar contador para verificar que no hay rerun
st.metric("Contador de Reruns", st.session_state.counter)

# Botón normal que causa rerun
if st.button("🔄 Botón Normal (causa rerun)"):
    st.session_state.counter += 1
    st.success("Botón normal presionado - contador incrementado")

st.markdown("---")

# Texto de prueba
sample_text = """Hola, buenos días. Mi nombre es Juan Pérez y estoy llamando porque tengo un problema con mi servicio de internet de Movistar. La conexión se ha estado cortando frecuentemente desde hace tres días.

¿Podrían ayudarme a solucionarlo?

Gracias por su tiempo."""

# Área de texto para mostrar el contenido
st.subheader("📝 Texto de Prueba")
st.text_area("Contenido:", value=sample_text, height=150, key="sample_text")

# Botón de copiado que NO causa rerun
st.subheader("📋 Botón de Copiado (SIN rerun)")
copy_button_html = create_copy_button(
    text=sample_text,
    button_text="📋 Copiar Texto de Prueba",
    button_id="test_copy",
    success_message="✅ ¡Texto copiado sin rerun!"
)
st.markdown(copy_button_html, unsafe_allow_html=True)

st.info("""
🧪 **Instrucciones de Prueba:**
1. Observa el contador de reruns al inicio
2. Haz clic en el "Botón Normal" - verás que el contador aumenta
3. Haz clic en el "Botón de Copiado" - el contador NO debe aumentar
4. Deberías ver una notificación verde temporal
5. El texto debe copiarse al portapapeles
""")

st.success("""
✅ **Si el botón de copiado funciona sin incrementar el contador, 
   entonces la solución es exitosa y puede aplicarse a la aplicación principal.**
""")

# Mostrar instrucciones adicionales
st.markdown("---")
st.subheader("🔧 Cómo Verificar")
st.markdown("""
1. **Verifica el contador**: Debe mantenerse igual al usar el botón de copiado
2. **Busca la notificación**: Aparece una notificación verde temporal
3. **Pega el texto**: Ctrl+V en cualquier editor para verificar que se copió
4. **Prueba en diferentes navegadores**: Chrome, Firefox, Edge, etc.
""")

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 15px; background: #f0f2f6; border-radius: 8px;">
    <strong>👨‍💻 Desarrollado por Mauro Rementeria</strong><br>
    <em>mauroere@gmail.com</em>
</div>
""", unsafe_allow_html=True)
