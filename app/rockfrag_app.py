"""
RockFrag App — Análisis de Fragmentación con SAM
"""

import streamlit as st
import cv2
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.segmentor import RockFragAnalyzer, RockFragVisualizer

st.set_page_config(
    page_title="RockFrag AI — SPCC Cuajone",
    page_icon="⛏️",
    layout="wide",
)

# CSS personalizado (industrial/minero)
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stApp { background-color: #0f0f1a; }
    h1 { color: #00d4ff !important; font-family: 'Courier New', monospace; }
    h2, h3 { color: #e0e0e0 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #00d4ff33;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #00d4ff; }
    .metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; }
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: #000;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 8px 24px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
# ⛏️ RockFrag AI
### Ingeniería Mina - SPCC Cuajone
*Análisis de Fragmentación de Roca con Segment Anything Model (SAM)*
""")

st.markdown("---")

# Panel lateral
with st.sidebar:
    st.header("⚙️ Parámetros")
    
    st.subheader("📏 Escala de referencia")
    scale_mode = st.radio("Modo de escala", ["Auto-detectar barra", "Ingresar manualmente"])
    scale_ref_cm = st.number_input("Longitud real de la referencia (cm)", min_value=1.0, max_value=500.0, value=30.0)
    manual_px_per_cm = None
    if scale_mode == "Ingresar manualmente":
        manual_px_per_cm = st.number_input("Píxeles por cm (px/cm)", min_value=1.0, max_value=1000.0, value=20.0)
    
    st.subheader("🔬 Segmentación con SAM")
    min_frag_px = st.slider("Tamaño mínimo de fragmento (px²)", min_value=50, max_value=2000, value=300)
    max_frag_ratio = st.slider("Tamaño máximo (% de imagen)", min_value=10, max_value=80, value=60)
    
    st.markdown("---")
    st.caption("RockFrag AI v2.0 | SPCC Cuajone")

# Área principal
col_upload, _ = st.columns([2, 1])
with col_upload:
    st.subheader("📸 Imagen de entrada")
    uploaded = st.file_uploader("Sube una foto de la pila de roca", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_to_analyze = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_to_analyze, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Imagen cargada", use_column_width=True)
    st.markdown("---")
    
    if st.button("🔍 ANALIZAR FRAGMENTACIÓN", use_container_width=True):
        with st.spinner("Analizando con SAM..."):
            try:
                tmp_path = Path("/tmp/rockfrag_input.jpg")
                cv2.imwrite(str(tmp_path), img_to_analyze)
                
                analyzer = RockFragAnalyzer(
                    scale_reference_cm=scale_ref_cm,
                    min_fragment_area_px=min_frag_px,
                    max_fragment_ratio=max_frag_ratio / 100,
                    sam_model_path="sam2_t.pt"
                )
                px_per_cm = manual_px_per_cm if scale_mode == "Ingresar manualmente" else None
                result = analyzer.analyze(str(tmp_path), scale_px_per_cm=px_per_cm)
                
                # Métricas
                st.subheader("📊 Resultados")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""<div class="metric-card"><div class="metric-value">{result.total_fragments}</div><div class="metric-label">Fragmentos detectados</div></div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="metric-card"><div class="metric-value">{result.p50:.1f} cm</div><div class="metric-label">D50 (mediana)</div></div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""<div class="metric-card"><div class="metric-value">{result.mean_diameter:.1f} cm</div><div class="metric-label">Diámetro promedio</div></div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""<div class="metric-card"><div class="metric-value">{result.max_diameter:.1f} cm</div><div class="metric-label">Fragmento máximo</div></div>""", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("📐 Distribución granulométrica")
                pc1, pc2, pc3 = st.columns(3)
                with pc1:
                    st.markdown(f"""<div style="background:#ff6b6b22;border:1px solid #ff6b6b;border-radius:8px;padding:16px;text-align:center"><div style="font-size:1.8rem;font-weight:bold;color:#ff6b6b">{result.p20:.1f} cm</div><div style="color:#aaa;">P20 — 20% del material pasa</div></div>""", unsafe_allow_html=True)
                with pc2:
                    st.markdown(f"""<div style="background:#ffd93d22;border:1px solid #ffd93d;border-radius:8px;padding:16px;text-align:center"><div style="font-size:1.8rem;font-weight:bold;color:#ffd93d">{result.p50:.1f} cm</div><div style="color:#aaa;">P50 — Mediana granulométrica</div></div>""", unsafe_allow_html=True)
                with pc3:
                    st.markdown(f"""<div style="background:#6bcb7722;border:1px solid #6bcb77;border-radius:8px;padding:16px;text-align:center"><div style="font-size:1.8rem;font-weight:bold;color:#6bcb77">{result.p80:.1f} cm</div><div style="color:#aaa;">P80 — 80% del material pasa</div></div>""", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Visualización
                col_seg, col_curve = st.columns(2)
                with col_seg:
                    st.subheader("🎨 Segmentación SAM")
                    seg_img = RockFragVisualizer.draw_segmentation(img_to_analyze, result)
                    seg_rgb = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
                    st.image(seg_rgb, use_column_width=True)
                with col_curve:
                    st.subheader("📈 Curva granulométrica")
                    curve_bytes = RockFragVisualizer.plot_grading_curve(result)
                    st.image(curve_bytes, use_column_width=True)
                
                # Descargas
                st.subheader("💾 Descargar resultados")
                result_json = json.dumps(RockFragVisualizer.result_to_dict(result), indent=2, ensure_ascii=False)
                st.download_button("⬇️ Datos JSON", result_json, "rockfrag_resultado.json", "application/json")
                _, seg_encoded = cv2.imencode('.png', seg_img)
                st.download_button("⬇️ Imagen segmentada", seg_encoded.tobytes(), "rockfrag_segmentacion.png", "image/png")
                st.download_button("⬇️ Curva granulométrica", curve_bytes, "rockfrag_curva.png", "image/png")
                st.success("✅ Análisis completado")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Ajusta los parámetros o verifica la imagen.")
else:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#555">
        <div style="font-size:4rem">⛏️</div>
        <h2 style="color:#444">Sube una foto para comenzar</h2>
        <p>Segmentación avanzada con SAM</p>
    </div>
    """, unsafe_allow_html=True)
