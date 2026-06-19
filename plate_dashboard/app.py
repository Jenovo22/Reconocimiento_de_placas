import streamlit as st
from pathlib import Path
import re
import unicodedata
from uuid import uuid4

from config import (
    BASE_DIR,
    INPUT_DIR,
    USE_SAHI_IMAGE,
    USE_SAHI_VIDEO,
    SAHI_GRID_ROWS,
    SAHI_GRID_COLS,
)

from src.model_loader import load_model
from src.video_processor import process_video, get_video_metadata
from src.image_processor import process_image
from src.camera_processor import run_camera


# ============================================================
# COMPATIBILIDAD CON CONFIG.PY
# ============================================================

try:
    from config import INPUT_VIDEO_DIR, INPUT_IMAGE_DIR
except ImportError:
    INPUT_VIDEO_DIR = INPUT_DIR
    INPUT_IMAGE_DIR = BASE_DIR / "inputs" / "images"


# ============================================================
# HELPERS
# ============================================================

def slugify_filename(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name or uuid4().hex


def save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(uploaded_file.name)
    safe_stem = slugify_filename(original_name.stem)
    suffix = original_name.suffix.lower()
    unique_id = uuid4().hex[:8]

    save_path = target_dir / f"{safe_stem}_{unique_id}{suffix}"

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return save_path


def go_to_page(page_name: str):
    st.session_state.current_page = page_name
    st.rerun()


def reset_video_state():
    st.session_state.input_video_path = None
    st.session_state.processed_video_path = None
    st.session_state.uploaded_video_signature = None
    st.session_state.processed_video_signature = None
    st.session_state.processed_video_mode = None


def reset_image_state():
    st.session_state.input_image_path = None
    st.session_state.processed_image_path = None
    st.session_state.uploaded_image_signature = None
    st.session_state.processed_image_signature = None
    st.session_state.processed_image_mode = None


def seconds_to_mmss(seconds: float) -> str:
    seconds = int(seconds)
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:02d}"


def get_video_mode_label(use_sahi_video: bool) -> str:
    if use_sahi_video:
        return f"SAHI grid {SAHI_GRID_ROWS}x{SAHI_GRID_COLS}"

    return "YOLO + ByteTrack"


def get_image_mode_label(use_sahi_image: bool) -> str:
    if use_sahi_image:
        return f"SAHI grid {SAHI_GRID_ROWS}x{SAHI_GRID_COLS}"

    return "YOLO normal"


# ============================================================
# SETUP
# ============================================================

st.set_page_config(
    page_title="YOLO Plate Dashboard",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_model()


model, device = get_model()

INPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
INPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================

if "current_page" not in st.session_state:
    st.session_state.current_page = "menu"

if "input_video_path" not in st.session_state:
    st.session_state.input_video_path = None

if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None

if "uploaded_video_signature" not in st.session_state:
    st.session_state.uploaded_video_signature = None

if "processed_video_signature" not in st.session_state:
    st.session_state.processed_video_signature = None

if "processed_video_mode" not in st.session_state:
    st.session_state.processed_video_mode = None

if "input_image_path" not in st.session_state:
    st.session_state.input_image_path = None

if "processed_image_path" not in st.session_state:
    st.session_state.processed_image_path = None

if "uploaded_image_signature" not in st.session_state:
    st.session_state.uploaded_image_signature = None

if "processed_image_signature" not in st.session_state:
    st.session_state.processed_image_signature = None

if "processed_image_mode" not in st.session_state:
    st.session_state.processed_image_mode = None

if "use_sahi_image" not in st.session_state:
    st.session_state.use_sahi_image = bool(USE_SAHI_IMAGE)

if "use_sahi_video" not in st.session_state:
    st.session_state.use_sahi_video = bool(USE_SAHI_VIDEO)


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .main-subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #A0A0A0;
            margin-bottom: 2rem;
        }

        .menu-card {
            padding: 1.5rem;
            border-radius: 18px;
            background-color: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            min-height: 230px;
            text-align: center;
        }

        .menu-icon {
            font-size: 3rem;
            margin-bottom: 0.8rem;
        }

        .menu-card-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }

        .menu-card-text {
            font-size: 0.95rem;
            color: #B8B8B8;
            margin-bottom: 1rem;
        }

        .status-box {
            padding: 0.75rem 1rem;
            border-radius: 12px;
            background-color: rgba(0, 102, 255, 0.08);
            border: 1px solid rgba(0, 102, 255, 0.25);
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PÁGINA 0 - MENÚ PRINCIPAL
# ============================================================

def render_main_menu():
    st.markdown(
        """
        <div class="main-title">🚗 YOLOv8 - Detección de Placas Vehiculares</div>
        <div class="main-subtitle">
            Selecciona el modo de inferencia que deseas utilizar.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-box">
            <strong>Modelo cargado correctamente.</strong><br>
            Dispositivo actual: <code>{device}</code><br>
            Grid SAHI configurado: <code>{SAHI_GRID_ROWS}x{SAHI_GRID_COLS}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="menu-card">
                <div class="menu-icon">📹</div>
                <div class="menu-card-title">Video</div>
                <div class="menu-card-text">
                    Procesa segmentos de video usando YOLO + ByteTrack o SAHI grid por frame.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Entrar a Video", use_container_width=True, type="primary", key="go_video"):
            go_to_page("video")

    with col2:
        st.markdown(
            """
            <div class="menu-card">
                <div class="menu-icon">🖼️</div>
                <div class="menu-card-title">Foto</div>
                <div class="menu-card-text">
                    Procesa imágenes completas con YOLO normal o división proporcional por grid.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Entrar a Foto", use_container_width=True, key="go_photo"):
            go_to_page("foto")

    with col3:
        st.markdown(
            """
            <div class="menu-card">
                <div class="menu-icon">📷</div>
                <div class="menu-card-title">Cámara en tiempo real</div>
                <div class="menu-card-text">
                    Abre la cámara local y ejecuta detección de placas en tiempo real usando OpenCV.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Entrar a Cámara", use_container_width=True, key="go_camera"):
            go_to_page("camara")


# ============================================================
# PÁGINA 1 - VIDEO
# ============================================================

def render_video_page():
    col_back, col_title = st.columns([1, 5])

    with col_back:
        if st.button("⬅️ Menú", key="back_from_video"):
            go_to_page("menu")

    with col_title:
        st.header("📹 Procesamiento de video")

        use_sahi_video = st.checkbox(
            "Usar SAHI grid en video",
            value=st.session_state.use_sahi_video,
            key="use_sahi_video_checkbox",
            help=(
                "Divide cada frame en secciones proporcionales y ejecuta YOLO en cada sección. "
                "Puede mejorar placas pequeñas, pero es más lento y no genera IDs de tracking."
            ),
        )

        st.session_state.use_sahi_video = bool(use_sahi_video)

        if st.session_state.use_sahi_video:
            st.warning(
                f"Modo SAHI grid activo en video: {SAHI_GRID_ROWS}x{SAHI_GRID_COLS}. "
                "Este modo no muestra IDs de tracking."
            )
        else:
            st.info("Modo normal activo: YOLO + ByteTrack con IDs de tracking.")

    video_file = st.file_uploader(
        "Sube un video",
        type=["mp4", "avi", "mov"],
        key="video_uploader",
    )

    if video_file is None:
        st.info("Sube un video para iniciar el procesamiento.")
        return

    current_signature = f"{video_file.name}_{video_file.size}"

    if st.session_state.uploaded_video_signature != current_signature:
        save_path = save_uploaded_file(video_file, INPUT_VIDEO_DIR)

        st.session_state.input_video_path = str(save_path)
        st.session_state.processed_video_path = None
        st.session_state.uploaded_video_signature = current_signature
        st.session_state.processed_video_signature = None
        st.session_state.processed_video_mode = None

    input_video_path = Path(st.session_state.input_video_path)

    try:
        metadata = get_video_metadata(input_video_path)

        duration_seconds = float(metadata["duration_seconds"])
        duration_minutes = float(metadata["duration_minutes"])
        fps = float(metadata["fps"])
        total_frames = int(metadata["total_frames"])

    except Exception as e:
        st.error("No se pudieron leer los metadatos del video.")
        st.exception(e)
        return

    st.info(
        f"Duración del video: {duration_minutes:.2f} minutos "
        f"({duration_seconds:.2f} segundos) | FPS: {fps:.2f} | Frames: {total_frames}"
    )

    st.subheader("Selecciona el intervalo a procesar")

    col_start, col_end = st.columns(2)

    with col_start:
        start_minute = st.number_input(
            "Minuto inicial",
            min_value=0.0,
            max_value=max(duration_minutes, 0.01),
            value=0.0,
            step=0.1,
            format="%.2f",
            help="Ejemplo: 2.5 equivale a minuto 2 con 30 segundos.",
            key="video_start_minute",
        )

    with col_end:
        end_minute = st.number_input(
            "Minuto final",
            min_value=0.0,
            max_value=max(duration_minutes, 0.01),
            value=max(duration_minutes, 0.01),
            step=0.1,
            format="%.2f",
            help="Ejemplo: 3.0 equivale al minuto 3 exacto.",
            key="video_end_minute",
        )

    start_time_sec = start_minute * 60
    end_time_sec = end_minute * 60

    start_time_sec = max(0, min(start_time_sec, duration_seconds))
    end_time_sec = max(0, min(end_time_sec, duration_seconds))

    interval_is_valid = start_time_sec < end_time_sec

    mode_label = get_video_mode_label(st.session_state.use_sahi_video)
    mode_token = "sahi" if st.session_state.use_sahi_video else "tracking"

    current_video_process_signature = (
        f"{input_video_path.name}_"
        f"{start_time_sec:.2f}_"
        f"{end_time_sec:.2f}_"
        f"{mode_token}_"
        f"{SAHI_GRID_ROWS}x{SAHI_GRID_COLS}"
    )

    if not interval_is_valid:
        st.warning("El minuto inicial debe ser menor que el minuto final.")
    else:
        selected_duration = end_time_sec - start_time_sec

        st.success(
            f"Se procesará el segmento desde {seconds_to_mmss(start_time_sec)} "
            f"hasta {seconds_to_mmss(end_time_sec)} "
            f"({selected_duration:.2f} segundos). "
            f"Modo: {mode_label}."
        )

    st.divider()

    col_original, col_processed = st.columns(2)

    with col_original:
        st.subheader("Video original")

        if input_video_path.exists():
            st.video(str(input_video_path))
            st.caption(f"Entrada: {input_video_path.name}")
        else:
            st.error("El video original no se encontró en disco.")

    with col_processed:
        st.subheader("Video procesado")

        if st.session_state.processed_video_path is None:
            st.info("Presiona el botón de procesamiento para generar el resultado.")
        else:
            processed_path = Path(st.session_state.processed_video_path)

            if (
                st.session_state.processed_video_signature is not None
                and st.session_state.processed_video_signature != current_video_process_signature
            ):
                st.warning(
                    "El video procesado mostrado corresponde a otro intervalo o modo. "
                    "Procesa nuevamente para generar el resultado con la configuración actual."
                )

            if not processed_path.exists():
                st.error("El archivo procesado no se encontró en disco.")

            elif processed_path.suffix.lower() != ".mp4":
                st.error("El archivo procesado no está en formato MP4 compatible con navegador.")
                st.caption(f"Archivo generado: {processed_path}")

            elif processed_path.stat().st_size == 0:
                st.error("El archivo procesado existe, pero está vacío.")

            else:
                st.video(str(processed_path))
                caption = (
                    f"Salida: {processed_path.name} "
                    f"({processed_path.stat().st_size / (1024 * 1024):.2f} MB)"
                )

                if st.session_state.processed_video_mode:
                    caption += f" | {st.session_state.processed_video_mode}"

                st.caption(caption)

    st.divider()

    col_process, col_clear = st.columns([1, 1])

    with col_process:
        if st.button(
            "🔍 Procesar segmento seleccionado",
            type="primary",
            use_container_width=True,
            disabled=not interval_is_valid,
            key="process_video_button",
        ):
            with st.spinner("Procesando segmento de video..."):
                try:
                    output_path = process_video(
                        model,
                        st.session_state.input_video_path,
                        device,
                        start_time_sec=start_time_sec,
                        end_time_sec=end_time_sec,
                        use_sahi=st.session_state.use_sahi_video,
                    )

                    st.session_state.processed_video_path = output_path
                    st.session_state.processed_video_signature = current_video_process_signature
                    st.session_state.processed_video_mode = mode_label

                    st.success("Segmento de video procesado correctamente.")
                    st.rerun()

                except Exception as e:
                    st.session_state.processed_video_path = None
                    st.session_state.processed_video_signature = None
                    st.session_state.processed_video_mode = None
                    st.error("Ocurrió un error procesando el segmento de video.")
                    st.exception(e)

    with col_clear:
        if st.button("🧹 Limpiar video actual", use_container_width=True, key="clear_video_button"):
            reset_video_state()
            st.rerun()


# ============================================================
# PÁGINA 2 - FOTO
# ============================================================

def render_photo_page():
    col_back, col_title = st.columns([1, 5])

    with col_back:
        if st.button("⬅️ Menú", key="back_from_photo"):
            go_to_page("menu")

    with col_title:
        st.header("🖼️ Procesamiento de imagen")

        use_sahi_image = st.checkbox(
            "Usar SAHI grid en imagen",
            value=st.session_state.use_sahi_image,
            key="use_sahi_image_checkbox",
            help="Divide la imagen en secciones proporcionales y ejecuta YOLO en cada sección.",
        )

        st.session_state.use_sahi_image = bool(use_sahi_image)

        if st.session_state.use_sahi_image:
            st.info(f"Modo SAHI grid activo en imagen: {SAHI_GRID_ROWS}x{SAHI_GRID_COLS}.")
        else:
            st.info("Modo YOLO normal activo en imagen.")

    image_file = st.file_uploader(
        "Sube una imagen",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if image_file is None:
        st.info("Sube una imagen para detectar placas.")
        return

    current_signature = f"{image_file.name}_{image_file.size}"

    if st.session_state.uploaded_image_signature != current_signature:
        save_path = save_uploaded_file(image_file, INPUT_IMAGE_DIR)

        st.session_state.input_image_path = str(save_path)
        st.session_state.processed_image_path = None
        st.session_state.uploaded_image_signature = current_signature
        st.session_state.processed_image_signature = None
        st.session_state.processed_image_mode = None

    input_image_path = Path(st.session_state.input_image_path)

    image_mode_label = get_image_mode_label(st.session_state.use_sahi_image)
    image_mode_token = "sahi" if st.session_state.use_sahi_image else "yolo"

    current_image_process_signature = (
        f"{input_image_path.name}_"
        f"{image_mode_token}_"
        f"{SAHI_GRID_ROWS}x{SAHI_GRID_COLS}"
    )

    col_original, col_processed = st.columns(2)

    with col_original:
        st.subheader("Imagen original")

        if input_image_path.exists():
            st.image(str(input_image_path), use_container_width=True)
            st.caption(f"Entrada: {input_image_path.name}")
        else:
            st.error("La imagen original no se encontró en disco.")

    with col_processed:
        st.subheader("Imagen procesada")

        if st.session_state.processed_image_path is None:
            st.info("Presiona el botón de procesamiento para generar el resultado.")
        else:
            processed_image_path = Path(st.session_state.processed_image_path)

            if (
                st.session_state.processed_image_signature is not None
                and st.session_state.processed_image_signature != current_image_process_signature
            ):
                st.warning(
                    "La imagen procesada mostrada corresponde a otra configuración. "
                    "Procesa nuevamente para aplicar el modo actual."
                )

            if not processed_image_path.exists():
                st.error("La imagen procesada no se encontró en disco.")

            elif processed_image_path.stat().st_size == 0:
                st.error("La imagen procesada existe, pero está vacía.")

            else:
                st.image(str(processed_image_path), use_container_width=True)

                caption = (
                    f"Salida: {processed_image_path.name} "
                    f"({processed_image_path.stat().st_size / 1024:.2f} KB)"
                )

                if st.session_state.processed_image_mode:
                    caption += f" | {st.session_state.processed_image_mode}"

                st.caption(caption)

    st.divider()

    col_process, col_clear = st.columns([1, 1])

    with col_process:
        if st.button(
            "🔍 Procesar imagen",
            type="primary",
            use_container_width=True,
            key="process_image_button",
        ):
            with st.spinner("Procesando imagen..."):
                try:
                    output_path = process_image(
                        model,
                        st.session_state.input_image_path,
                        device,
                        use_sahi=st.session_state.use_sahi_image,
                    )

                    st.session_state.processed_image_path = output_path
                    st.session_state.processed_image_signature = current_image_process_signature
                    st.session_state.processed_image_mode = image_mode_label

                    st.success("Imagen procesada correctamente.")
                    st.rerun()

                except Exception as e:
                    st.session_state.processed_image_path = None
                    st.session_state.processed_image_signature = None
                    st.session_state.processed_image_mode = None
                    st.error("Ocurrió un error procesando la imagen.")
                    st.exception(e)

    with col_clear:
        if st.button("🧹 Limpiar imagen actual", use_container_width=True, key="clear_image_button"):
            reset_image_state()
            st.rerun()


# ============================================================
# PÁGINA 3 - CÁMARA
# ============================================================

def render_camera_page():
    col_back, col_title = st.columns([1, 5])

    with col_back:
        if st.button("⬅️ Menú", key="back_from_camera"):
            go_to_page("menu")

    with col_title:
        st.header("📷 Cámara en tiempo real")

    st.markdown(
        """
        Este modo abre la cámara local con OpenCV y ejecuta detección de placas en tiempo real.
        """
    )

    st.warning("Cuando la ventana de cámara esté abierta, presiona Q para cerrarla.")

    if st.button("🎥 Iniciar cámara", type="primary", key="start_camera_button"):
        run_camera(model, device)


# ============================================================
# ROUTER PRINCIPAL
# ============================================================

if st.session_state.current_page == "menu":
    render_main_menu()

elif st.session_state.current_page == "video":
    render_video_page()

elif st.session_state.current_page == "foto":
    render_photo_page()

elif st.session_state.current_page == "camara":
    render_camera_page()

else:
    st.session_state.current_page = "menu"
    st.rerun()