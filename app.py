
import os
import json
import hashlib
from pathlib import Path

import streamlit as st
import torch
import timm
from google import genai
from PIL import Image, ImageOps
from torchvision import transforms


# =========================================================
# 1. CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="DeepDetect 2026",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CARPETA_MODELO = Path(__file__).resolve().parent / "modelo"


# =========================================================
# 2. DISEÑO VISUAL
# =========================================================

st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(
                ellipse at 85% 0%,
                rgba(47, 73, 149, 0.22),
                transparent 42%
            ),
            #0B1020;
        color: #F4F7FF;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    h1, h2, h3, p, label {
        color: #F4F7FF;
    }

    /* Identidad */
    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 25px;
    }

    .brand-icon {
        background: #24315C;
        border: 1px solid #4059A2;
        padding: 11px 14px;
        border-radius: 13px;
        font-size: 26px;
    }

    .brand-name {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #FFFFFF;
    }

    .brand-name span {
        color: #89AAFC;
    }

    .brand-subtitle {
        font-size: 11px;
        color: #A0B2D5;
        letter-spacing: 2px;
        margin-top: 4px;
    }

    /* Encabezado principal: propósito claro y referencia visual */
    .hero {
        background: linear-gradient(125deg, #162747 0%, #101B34 58%, #182B51 100%);
        border: 1px solid #344B77;
        border-radius: 22px;
        padding: 34px 36px;
        margin-bottom: 25px;
        box-shadow: 0 14px 38px rgba(0, 0, 0, 0.22);
        overflow: hidden;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.25fr) minmax(260px, 0.85fr);
        gap: 30px;
        align-items: center;
    }

    .eyebrow {
        color: #A5BDFF;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 1.8px;
        margin-bottom: 14px;
    }

    .hero-title {
        color: #FFFFFF;
        font-size: clamp(29px, 3.7vw, 44px);
        font-weight: 800;
        letter-spacing: -1.3px;
        line-height: 1.16;
        margin-bottom: 16px;
    }

    .hero-description {
        color: #C1CDE2;
        font-size: 15px;
        line-height: 1.7;
        margin-bottom: 20px;
        max-width: 640px;
    }

    .tag {
        display: inline-block;
        background: #233A60;
        border: 1px solid #3D5687;
        border-radius: 8px;
        padding: 7px 11px;
        color: #D7E4FF;
        font-size: 12px;
        margin-right: 7px;
        margin-bottom: 7px;
    }

    /* Gráfico ilustrativo: NO representa una detección realizada */
    .hero-visual {
        background: linear-gradient(165deg, #22385D 0%, #121E37 100%);
        border: 1px solid #46649C;
        border-radius: 20px;
        padding: 18px;
        position: relative;
        min-width: 0;
    }

    .visual-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        color: #BCD0F5;
        font-size: 10px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .visual-top strong {
        color: #9EBBFA;
        font-weight: 700;
    }

    .face-frame {
        height: 190px;
        border: 1px solid rgba(138, 171, 238, 0.35);
        border-radius: 15px;
        background:
            linear-gradient(transparent 98%, rgba(137, 170, 252, .10) 99%),
            linear-gradient(90deg, transparent 98%, rgba(137, 170, 252, .10) 99%),
            radial-gradient(ellipse at center, #2B426C 0%, #172643 67%);
        background-size: 100% 25px, 25px 100%, auto;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        overflow: hidden;
    }

    .face-illustration {
        position: relative;
        width: 128px;
        height: 154px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-end;
    }

    .face-head {
        width: 67px;
        height: 80px;
        border-radius: 44% 44% 46% 46%;
        border: 2px solid #A9C6FF;
        background: linear-gradient(160deg, #789AD0, #415F9B);
        position: relative;
        z-index: 2;
        box-shadow: inset -9px -7px 13px rgba(9, 18, 38, .28);
    }

    .face-head::before,
    .face-head::after {
        content: "";
        position: absolute;
        top: 35px;
        height: 3px;
        width: 9px;
        border-radius: 3px;
        background: #182C50;
    }
    .face-head::before {left: 15px;}
    .face-head::after {right: 15px;}

    .face-shoulders {
        height: 57px;
        width: 121px;
        margin-top: -4px;
        border: 2px solid #A9C6FF;
        border-bottom: 0;
        border-radius: 62px 62px 0 0;
        background: linear-gradient(160deg, #587EC0, #293E68);
    }

    .frame-corner {
        position: absolute;
        width: 24px;
        height: 24px;
        border-color: #A9C6FF;
        border-style: solid;
    }
    .corner-tl {top: 13px; left: 13px; border-width: 2px 0 0 2px;}
    .corner-tr {top: 13px; right: 13px; border-width: 2px 2px 0 0;}
    .corner-bl {bottom: 13px; left: 13px; border-width: 0 0 2px 2px;}
    .corner-br {bottom: 13px; right: 13px; border-width: 0 2px 2px 0;}

    .scan-line {
        position: absolute;
        top: 49%;
        left: 10%;
        right: 10%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #8CE8FF, transparent);
        box-shadow: 0 0 14px rgba(140, 232, 255, .75);
        z-index: 3;
    }

    .visual-bottom {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding-top: 15px;
        color: #B8C8E4;
        font-size: 12px;
    }

    .visual-classes {
        display: flex;
        gap: 7px;
    }

    .visual-class {
        background: #213A5F;
        color: #E1EAFF;
        border: 1px solid #476499;
        border-radius: 7px;
        padding: 5px 9px;
        font-size: 11px;
    }

    .visual-note {
        margin-top: 10px;
        font-size: 10px;
        color: #90A7CF;
        line-height: 1.5;
    }

    /* Títulos y paneles */
    .section-heading {
        color: #F4F7FF;
        font-weight: 750;
        font-size: 19px;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .section-heading span {
        color: #89AAFC;
        font-size: 13px;
        margin-right: 8px;
        letter-spacing: 1px;
    }

    .panel {
        background: #15203A;
        border: 1px solid #293A5D;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .panel-small {
        background: #172440;
        border: 1px solid #304775;
        border-radius: 15px;
        padding: 20px;
    }

    /* Diagnóstico */
    .diagnostic {
        background: #3B2E23;
        border: 1px solid #6D5437;
        border-radius: 13px;
        padding: 20px;
        margin-bottom: 18px;
    }

    .diagnostic.real {
        background: #17352F;
        border-color: #276B59;
    }

    .diagnostic-label {
        color: #F8C78F;
        font-size: 12px;
        letter-spacing: 1px;
        font-weight: 750;
    }

    .diagnostic.real .diagnostic-label {
        color: #85E2BE;
    }

    .diagnostic-score {
        color: #FFFFFF;
        font-size: 40px;
        font-weight: 800;
        line-height: 1.3;
        margin-top: 6px;
    }

    .diagnostic-caption {
        color: #C9B9A7;
        font-size: 12px;
    }

    .diagnostic.real .diagnostic-caption {
        color: #AAD1C5;
    }

    /* Barras de clasificación */
    .score-row {
        display: flex;
        justify-content: space-between;
        color: #DCE5F5;
        font-size: 14px;
        margin: 16px 0 7px;
    }

    .score-track {
        height: 10px;
        background: #303B55;
        border-radius: 20px;
        overflow: hidden;
    }

    .score-fill {
        height: 100%;
        border-radius: 20px;
    }

    /* Métricas */
    .metric-card {
        background: #15203A;
        border: 1px solid #293A5D;
        border-radius: 15px;
        padding: 22px;
        margin-top: 12px;
    }

    .metric-value {
        color: #FFFFFF;
        font-size: 30px;
        font-weight: 800;
    }

    .metric-label {
        color: #A7B7D3;
        font-size: 12px;
        margin-top: 5px;
    }

    .fine-print {
        color: #98AAC9;
        font-size: 12px;
        line-height: 1.6;
    }

    /* Botones */
    div.stButton > button {
        background: #5267C7;
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 11px 20px;
    }

    div.stButton > button:hover {
        background: #657BE3;
        color: #FFFFFF;
        border: none;
    }

    /* Cargador de imágenes */
    [data-testid="stFileUploader"] {
        background: #15203A;
        border: 1px dashed #5271AF;
        border-radius: 13px;
        padding: 14px;
    }

    [data-testid="stImage"] img {
        border-radius: 13px;
    }

    /* Respuesta generada por Gemini */
    .st-key-gemini_result {
        background: #151E35;
        border: 1px solid #6275C8 !important;
        border-radius: 18px;
        padding: 22px;
        margin-top: 14px;
        margin-bottom: 20px;
    }

    .st-key-gemini_result [data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }

    .st-key-gemini_result h3 {
        color: #DCE4FF;
        font-size: 23px;
        margin: 0;
        padding: 0;
    }

    .st-key-gemini_result p,
    .st-key-gemini_result li {
        line-height: 1.55;
    }

    /* Desplegable de información */
    [data-testid="stExpander"] {
        background: #111B33;
        border: 1px solid #293A5D;
        border-radius: 12px;
    }

    [data-testid="stExpander"] summary {
        background: #172440;
        color: #E5ECFC;
        border-radius: 10px;
    }

    /* Encabezado adaptable a dispositivos pequeños */
    @media (max-width: 900px) {
        .hero-grid { grid-template-columns: 1fr; }
        .hero-visual { max-width: 480px; width: 100%; }
    }

    /* Adaptación a pantallas pequeñas */
    @media (max-width: 700px) {
        .block-container {
            padding: 1.2rem 1rem;
        }

        .hero {
            padding: 25px 20px;
        }

        .hero-description {
            font-size: 14px;
        }

        .metric-value {
            font-size: 24px;
        }

        .st-key-gemini_result {
            padding: 15px;
        }
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. CARGA DEL MODELO Y MÉTRICAS
# =========================================================

@st.cache_resource
def cargar_modelo():
    with open(
        CARPETA_MODELO / "config.json",
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    modelo = timm.create_model(
        config["architecture"],
        pretrained=False,
        num_classes=1,
    )

    checkpoint = torch.load(
        CARPETA_MODELO / "modelo_final.pth",
        map_location="cpu",
        weights_only=True,
    )

    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()

    return modelo, config


@st.cache_data
def cargar_metricas():
    with open(
        CARPETA_MODELO / "metricas.json",
        encoding="utf-8"
    ) as f:
        return json.load(f)


modelo, config = cargar_modelo()
metricas = cargar_metricas()


def analizar_imagen(imagen):
    transformacion = transforms.Compose([
        transforms.Resize(
            (config["image_size"], config["image_size"])
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["mean"],
            std=config["std"],
        ),
    ])

    tensor = transformacion(imagen).unsqueeze(0)

    with torch.inference_mode():
        logit = modelo(tensor).flatten()[0]
        p_real = torch.sigmoid(logit).item()

    p_fake = 1 - p_real

    clase = (
        "real"
        if p_real >= config["threshold_real"]
        else "fake"
    )

    return clase, p_real, p_fake


# =========================================================
# 4. ENCABEZADO Y PRESENTACIÓN
# =========================================================

st.markdown("""
<div class="brand">
    <div class="brand-icon">◎</div>
    <div>
        <div class="brand-name">DeepDetect<span> 2026</span></div>
        <div class="brand-subtitle">ANÁLISIS DE IMÁGENES FACIALES</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-grid">
        <div>
            <div class="eyebrow">DETECCIÓN DE IMÁGENES GENERADAS POR IA</div>
            <div class="hero-title">
                ¿Es una fotografía real o generada por IA?
            </div>
            <div class="hero-description">
                Sube una imagen facial y revisa cómo la clasifica nuestro
                modelo ResNet-50. Podrás ver la puntuación de cada clase y
                obtener recomendaciones para comprobar su procedencia.
                El resultado es orientativo, no una prueba de autenticidad.
            </div>
            <span class="tag">Imágenes faciales</span>
            <span class="tag">ResNet-50</span>
            <span class="tag">Real / Fake</span>
            <span class="tag">Gemini</span>
        </div>
        <div class="hero-visual" aria-label="Ilustración de un rostro y las categorías real y fake">
            <div class="visual-top">
                <span>Vista de análisis</span>
                <strong>DEEPDETECT / 2026</strong>
            </div>
            <div class="face-frame">
                <span class="frame-corner corner-tl"></span>
                <span class="frame-corner corner-tr"></span>
                <span class="frame-corner corner-bl"></span>
                <span class="frame-corner corner-br"></span>
                <div class="face-illustration">
                    <div class="face-head"></div>
                    <div class="face-shoulders"></div>
                </div>
                <div class="scan-line"></div>
            </div>
            <div class="visual-bottom">
                <span>Clasificación binaria</span>
                <div class="visual-classes">
                    <span class="visual-class">REAL</span>
                    <span class="visual-class">FAKE</span>
                </div>
            </div>
            <div class="visual-note">
                Ilustración de la interfaz; no corresponde a una predicción.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 5. CARGADOR DE IMÁGENES Y CLASIFICACIÓN
# =========================================================

st.markdown(
    '<div class="section-heading">'
    '<span>01 /</span> Cargar imagen'
    '</div>',
    unsafe_allow_html=True,
)

imagen_subida = st.file_uploader(
    "Selecciona una fotografía facial para analizar",
    type=["jpg", "jpeg", "png", "webp"],
)

if imagen_subida is not None:
    imagen = ImageOps.exif_transpose(
        Image.open(imagen_subida)
    ).convert("RGB")

    clase, p_real, p_fake = analizar_imagen(imagen)

    col_imagen, col_resultado = st.columns(
        [1.1, 0.9],
        gap="large",
    )

    with col_imagen:
        st.markdown(
            '<div class="section-heading">'
            '<span>02 /</span> Vista previa'
            '</div>',
            unsafe_allow_html=True,
        )

        st.image(
            imagen,
            caption="Imagen proporcionada para el análisis",
            use_container_width=True,
        )

    with col_resultado:
        st.markdown(
            '<div class="section-heading">'
            '<span>03 /</span> Resultado del análisis'
            '</div>',
            unsafe_allow_html=True,
        )

        if clase == "real":
            estilo = "real"
            etiqueta = "POSIBLEMENTE REAL"
            puntuacion = p_real
            descripcion = "Puntuación de la clase real"
        else:
            estilo = ""
            etiqueta = "POSIBLEMENTE GENERADA POR IA"
            puntuacion = p_fake
            descripcion = "Puntuación de la clase fake"

        # Resultado principal
        st.markdown(
            f'<div class="diagnostic {estilo}">'
            f'<div class="diagnostic-label">'
            f'{etiqueta}</div>'
            f'<div class="diagnostic-score">'
            f'{puntuacion:.2%}</div>'
            f'<div class="diagnostic-caption">'
            f'{descripcion}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Barras de puntuación
        st.markdown(
            f'<div class="panel">'
            f'<div class="score-row">'
            f'<span>IA / Fake</span>'
            f'<strong>{p_fake:.2%}</strong>'
            f'</div>'
            f'<div class="score-track">'
            f'<div class="score-fill" '
            f'style="width:{p_fake * 100:.2f}%; '
            f'background:#DBA873;">'
            f'</div></div>'
            f'<div class="score-row">'
            f'<span>Real</span>'
            f'<strong>{p_real:.2%}</strong>'
            f'</div>'
            f'<div class="score-track">'
            f'<div class="score-fill" '
            f'style="width:{p_real * 100:.2f}%; '
            f'background:#729FFF;">'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

        # Advertir si la decisión está cerca del umbral o la clase
        # asignada tiene una puntuación baja (sin calibración).
        cerca_umbral = abs(p_real - config["threshold_real"]) < 0.10
        puntuacion_baja = puntuacion < 0.60
        resultado_incierto = cerca_umbral or puntuacion_baja

        if resultado_incierto:
            st.warning(
                "La clasificación requiere cautela: la puntuación "
                "es baja o está cerca del umbral de decisión. "
                "Contrasta la fuente y considera una revisión humana."
            )

        st.markdown(
            '<p class="fine-print">'
            'Las puntuaciones son salidas del modelo, '
            'no probabilidades calibradas ni pruebas '
            'definitivas de autenticidad.'
            '</p>',
            unsafe_allow_html=True,
        )

else:
    st.markdown(
        '<div class="panel">'
        '<p style="font-size:20px; margin-bottom:7px;">'
        '◎ Elige una imagen para empezar'
        '</p>'
        '<p class="fine-print">'
        'Sube una fotografía de una persona para ver la '
        'clasificación, ambas puntuaciones y las '
        'recomendaciones de verificación.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# 6. IA GENERATIVA: RECOMENDACIONES CON GEMINI
# =========================================================

st.markdown(
    '<div class="section-heading">'
    '<span>04 /</span> Recomendaciones con IA'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="panel-small">'
    '<div style="font-size:20px; font-weight:750; '
    'color:#F4F7FF; margin-bottom:8px;">'
    'Recomendaciones para verificar la imagen'
    '</div>'
    '<p style="color:#B6C5DF; font-size:14px;">'
    'Consulta sugerencias para interpretar el resultado y '
    'revisar la procedencia de la imagen.'
    '</p>'
    '</div>',
    unsafe_allow_html=True,
)


# Obtener la clave sin escribirla en el código
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except (FileNotFoundError, KeyError):
        api_key = None


# Conservar la respuesta mientras se visualiza la misma imagen
if "gemini_respuesta" not in st.session_state:
    st.session_state.gemini_respuesta = None

if "gemini_imagen_id" not in st.session_state:
    st.session_state.gemini_imagen_id = None


if imagen_subida is None:
    st.session_state.gemini_respuesta = None
    st.session_state.gemini_imagen_id = None

    st.info(
        "Primero sube una imagen para habilitar "
        "las recomendaciones con Gemini."
    )

else:
    imagen_id = hashlib.sha256(
        imagen_subida.getvalue()
    ).hexdigest()

    # Evitar mostrar recomendaciones de una imagen anterior
    if st.session_state.gemini_imagen_id != imagen_id:
        st.session_state.gemini_respuesta = None
        st.session_state.gemini_imagen_id = imagen_id

    if st.button(
        "✨ Generar recomendaciones con IA",
        use_container_width=True,
    ):

        if not api_key:
            st.error(
                "Gemini todavía no está configurado "
                "en esta aplicación."
            )

        else:
            etiqueta_resultado = (
                "posiblemente real"
                if clase == "real"
                else "posiblemente generada por IA"
            )

            cerca_umbral = abs(p_real - config["threshold_real"]) < 0.10
            puntuacion_baja = (
                p_real if clase == "real" else p_fake
            ) < 0.60

            prompt = f"""
            Actúa como asistente educativo especializado
            en verificación de imágenes digitales.

            Una aplicación llamada DeepDetect 2026
            analizó una imagen mediante ResNet-50.

            RESULTADOS:
            - Clasificación: {etiqueta_resultado}
            - Puntuación real: {p_real:.2%}
            - Puntuación fake: {p_fake:.2%}
            - Umbral para clasificar como real:
              {config["threshold_real"]:.0%}
            - ¿Cerca del umbral?: {cerca_umbral}
            - ¿Puntuación baja para la clase asignada?: {puntuacion_baja}

            IMPORTANTE:
            No estás viendo la imagen original.
            Solamente conoces la salida del modelo.
            No inventes detalles visuales.
            No asegures que la imagen es auténtica
            ni que fue generada por IA.
            Las puntuaciones no están calibradas.

            Responde en español con un máximo
            de 120 palabras en total.

            Usa exactamente esta estructura:

            **Interpretación:** explica en 1 o 2
            frases qué indica la clasificación.

            **Limitaciones:** explica en 1 frase
            por qué el resultado no es definitivo.

            **Recomendaciones:**
            - Primera acción concreta.
            - Segunda acción concreta.
            - Tercera acción concreta.

            Si está cerca del umbral o la puntuación asignada
            es baja, recomienda revisión humana.

            No escribas introducciones, conclusiones
            ni títulos adicionales.
            No repitas información.
            """

            with st.spinner(
                "Gemini está generando recomendaciones..."
            ):
                try:
                    cliente = genai.Client(api_key=api_key)

                    respuesta = cliente.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt,
                    )

                    if respuesta.text:
                        st.session_state.gemini_respuesta = (
                            respuesta.text.strip()
                        )
                    else:
                        st.session_state.gemini_respuesta = None
                        st.warning(
                            "Gemini no devolvió una respuesta."
                        )

                except Exception as e:
                    st.session_state.gemini_respuesta = None
                    st.error(
                        "No fue posible generar las recomendaciones. "
                        "Inténtalo nuevamente."
                    )
                    st.caption(
                        f"Detalle técnico: {type(e).__name__}"
                    )

    # Mostrar la respuesta dentro de un recuadro independiente
    if st.session_state.gemini_respuesta:
        with st.container(
            border=True,
            key="gemini_result",
        ):
            st.markdown(
                "### ✨ Análisis generado por Gemini"
            )

            st.caption(
                "Sugerencias basadas en la clasificación de DeepDetect 2026."
            )

            st.markdown(
                st.session_state.gemini_respuesta
            )

            st.caption(
                "🤖 Contenido generado mediante Gemini API. "
                "No sustituye la verificación humana."
            )


# =========================================================
# 7. DESEMPEÑO GLOBAL DEL MODELO
# =========================================================

st.markdown(
    '<div class="section-heading">'
    '<span>05 /</span> Rendimiento del modelo'
    '</div>',
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3, gap="medium")

with m1:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">'
        f'{metricas["accuracy"]:.2%}</div>'
        f'<div class="metric-label">'
        f'Exactitud en test</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">'
        f'{metricas["recall_fake"]:.2%}</div>'
        f'<div class="metric-label">'
        f'Recall de imágenes fake</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-value">3.238</div>'
        '<div class="metric-label">'
        'Imágenes en test</div>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<p class="fine-print" style="margin-top:15px;">'
    'Estas métricas fueron obtenidas al evaluar ResNet-50 '
    'sobre el conjunto de prueba del proyecto. '
    'Describen el desempeño global del modelo y no deben '
    'confundirse con las puntuaciones de una imagen individual.'
    '</p>',
    unsafe_allow_html=True,
)


# =========================================================
# 8. INFORMACIÓN Y LIMITACIONES
# =========================================================

with st.expander("ℹ️ Acerca del modelo y sus limitaciones"):
    st.markdown("""
**Arquitectura:** ResNet-50 preentrenada en ImageNet
y ajustada para clasificación binaria.

**Dataset:** Deep Detect Dataset 2026, disponible en Kaggle.

**Clases:** fake (generada por IA) y real.

**Partición:** 70 % entrenamiento, 15 % validación
y 15 % prueba, agrupando imágenes con hashes
perceptuales idénticos.

**Limitaciones:** el sistema puede cometer errores,
especialmente con imágenes modificadas, comprimidas,
capturas de pantalla o imágenes distintas de las
utilizadas durante el entrenamiento.

**Uso previsto:** demostración académica y apoyo
exploratorio. No constituye una herramienta
de autenticación forense.
    """)

st.markdown(
    '<div style="border-top:1px solid #263858; '
    'padding-top:22px; margin-top:35px; '
    'text-align:center; color:#8092B1; font-size:12px;">'
    'DeepDetect 2026 · Proyecto académico de clasificación '
    'de imágenes mediante inteligencia artificial'
    '</div>',
    unsafe_allow_html=True,
)
