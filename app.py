import streamlit as st
import tempfile
import cv2
import numpy as np
from core.verification import verify_faces

st.set_page_config(
    page_title="FaceGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "selfie_path" not in st.session_state:
    st.session_state.selfie_path = None
if "result" not in st.session_state:
    st.session_state.result = None

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    background-color: #020408 !important;
    color: #c8d8e8;
}
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* HEADER */
.fg-header { padding: 2.5rem 0 2rem; text-align: center; }
.fg-logo-ring {
    width: 72px; height: 72px;
    border: 2px solid #00ffc8; border-radius: 50%;
    margin: 0 auto 1rem;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 24px rgba(0,255,200,0.3), inset 0 0 24px rgba(0,255,200,0.05);
    animation: pulse-ring 3s ease-in-out infinite;
}
.fg-logo-inner {
    width: 48px; height: 48px;
    border: 1px solid rgba(0,255,200,0.4); border-radius: 50%;
    display: flex; align-items: center; justify-content: center; font-size: 22px;
}
@keyframes pulse-ring {
    0%,100% { box-shadow: 0 0 18px rgba(0,255,200,0.25), inset 0 0 18px rgba(0,255,200,0.04); }
    50% { box-shadow: 0 0 40px rgba(0,255,200,0.5), inset 0 0 24px rgba(0,255,200,0.1); }
}
.fg-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem; font-weight: 700;
    letter-spacing: 8px; color: #e8f4f0;
    text-transform: uppercase; margin: 0; line-height: 1;
}
.fg-title span { color: #00ffc8; }
.fg-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem; color: #2a6a5a;
    letter-spacing: 3px; text-transform: uppercase; margin-top: 0.4rem;
}
.fg-divider {
    width: 200px; height: 1px;
    background: linear-gradient(90deg, transparent, #00ffc8, transparent);
    margin: 1.2rem auto 0;
}

/* SECTION LABEL */
.fg-section {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem; letter-spacing: 3px;
    color: #00ffc8; text-transform: uppercase;
    margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 8px;
}
.fg-section::before { content:''; display:inline-block; width:16px; height:1px; background:#00ffc8; }
.fg-section::after { content:''; flex:1; height:1px; background: linear-gradient(90deg, rgba(0,255,200,0.2), transparent); }

/* PANEL */
.fg-panel {
    background: rgba(0,20,16,0.7);
    border: 1px solid rgba(0,255,200,0.1);
    border-top: 1px solid rgba(0,255,200,0.35);
    border-radius: 3px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    position: relative;
}
.fg-panel::before {
    content:''; position:absolute; top:0; left:0;
    width:36px; height:2px; background:#00ffc8;
}

/* STEP */
.fg-step {
    display:inline-flex; align-items:center; gap:10px;
    font-family:'Share Tech Mono',monospace;
    font-size:0.68rem; color:#2a6a5a; letter-spacing:2px; margin-bottom:0.6rem;
}
.fg-step-num {
    width:24px; height:24px; border:1px solid #00ffc8; border-radius:2px;
    display:inline-flex; align-items:center; justify-content:center;
    font-size:0.72rem; color:#00ffc8; font-weight:700;
}

/* TIP */
.fg-tip { font-family:'Share Tech Mono',monospace; font-size:0.67rem; color:#2a5a4a; letter-spacing:1px; line-height:1.9; padding: 0.3rem 0; }

/* IDLE RESULT */
.fg-result-idle {
    min-height:300px; display:flex; flex-direction:column;
    align-items:center; justify-content:center; text-align:center; gap:1.2rem;
}
.fg-result-icon {
    width:60px; height:60px; border:1px solid rgba(0,255,200,0.18); border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:1.5rem; color:rgba(0,255,200,0.25);
    animation: idle-spin 14s linear infinite;
}
@keyframes idle-spin { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
.fg-idle-text { font-family:'Share Tech Mono',monospace; font-size:0.68rem; color:#1e4a3a; letter-spacing:2px; line-height:2.2; }

/* MATCH / NO MATCH */
.fg-match {
    background: rgba(0,255,100,0.04);
    border:1px solid rgba(0,255,100,0.22); border-top:2px solid #00ff64;
    border-radius:3px; padding:1.4rem; text-align:center;
}
.fg-nomatch {
    background: rgba(255,40,40,0.04);
    border:1px solid rgba(255,40,40,0.22); border-top:2px solid #ff2828;
    border-radius:3px; padding:1.4rem; text-align:center;
}
.fg-result-label {
    font-family:'Rajdhani',sans-serif; font-size:1.5rem; font-weight:700;
    letter-spacing:4px; text-transform:uppercase; margin-bottom:0.2rem;
}
.fg-result-sub { font-family:'Share Tech Mono',monospace; font-size:0.65rem; letter-spacing:2px; opacity:0.45; }

/* CONFIDENCE BAR */
.fg-conf-wrap { margin:1.3rem 0 0.8rem; }
.fg-conf-header {
    display:flex; justify-content:space-between;
    font-family:'Share Tech Mono',monospace; font-size:0.65rem;
    color:#2a6a5a; letter-spacing:2px; margin-bottom:6px;
}
.fg-conf-val { color:#00ffc8; font-weight:bold; }
.fg-bar-track {
    height:5px; background:rgba(0,255,200,0.07); border-radius:0;
    border:1px solid rgba(0,255,200,0.1); overflow:hidden; position:relative;
}
.fg-bar-fill { height:100%; position:relative; }
.fg-bar-fill::after {
    content:''; position:absolute; right:0; top:0; bottom:0;
    width:5px; background:white; opacity:0.6;
}

/* METRICS */
.fg-metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:7px; margin-top:0.9rem; }
.fg-metric {
    background:rgba(0,255,200,0.02); border:1px solid rgba(0,255,200,0.09);
    border-radius:2px; padding:0.75rem; text-align:center;
}
.fg-metric-val { font-family:'Share Tech Mono',monospace; font-size:0.95rem; color:#c8e8d8; display:block; }
.fg-metric-key { font-family:'Share Tech Mono',monospace; font-size:0.57rem; color:#2a5a4a; letter-spacing:2px; text-transform:uppercase; display:block; margin-top:3px; }

/* STREAMLIT OVERRIDES */
.stButton > button {
    width:100%; background:transparent !important;
    border:1px solid #00ffc8 !important; border-radius:2px !important;
    color:#00ffc8 !important;
    font-family:'Share Tech Mono',monospace !important;
    font-size:0.75rem !important; letter-spacing:3px !important;
    height:3rem !important; text-transform:uppercase !important;
    box-shadow:0 0 14px rgba(0,255,200,0.1) !important;
    transition:all 0.2s !important;
}
.stButton > button:hover {
    background:rgba(0,255,200,0.07) !important;
    box-shadow:0 0 28px rgba(0,255,200,0.28) !important;
}
.stRadio label { font-family:'Share Tech Mono',monospace !important; font-size:0.73rem !important; }
div[data-testid="stImage"] img { border-radius:2px !important; border:1px solid rgba(0,255,200,0.18) !important; }
section[data-testid="stFileUploadDropzone"] {
    background:rgba(0,20,16,0.5) !important;
    border:1px dashed rgba(0,255,200,0.18) !important; border-radius:2px !important;
}
.stAlert { font-family:'Share Tech Mono',monospace !important; font-size:0.7rem !important; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────
st.markdown("""
<div class="fg-header">
    <div class="fg-logo-ring"><div class="fg-logo-inner">🛡️</div></div>
    <h1 class="fg-title">FACE<span>GUARD</span></h1>
    <p class="fg-subtitle">// identity verification system v2.0 //</p>
    <div class="fg-divider"></div>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

# ── LEFT — INPUTS ────────────────────────
with left_col:

    st.markdown('<div class="fg-section">Input — ID document</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fg-panel">
        <div class="fg-step"><span class="fg-step-num">01</span>UPLOAD ID PROOF</div>
    </div>
    """, unsafe_allow_html=True)

    id_image = st.file_uploader("Upload ID", type=["jpg","jpeg","png"], label_visibility="collapsed")

    if id_image:
        st.image(id_image, caption="ID Document", use_container_width=True)
    else:
        st.markdown('<p class="fg-tip">› Aadhaar &nbsp;› Passport &nbsp;› PAN Card &nbsp;› Driving License</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="fg-section">Input — live biometric</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="fg-panel">
        <div class="fg-step"><span class="fg-step-num">02</span>CAPTURE SELFIE</div>
    </div>
    """, unsafe_allow_html=True)

    method = st.radio("Method", ["📷 Webcam", "🖼️ Upload"], horizontal=True, label_visibility="collapsed")

    if method == "🖼️ Upload":
        selfie_file = st.file_uploader("Upload selfie", type=["jpg","jpeg","png"], key="selfie_up", label_visibility="collapsed")
        if selfie_file:
            st.image(selfie_file, caption="Selfie", use_container_width=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                f.write(selfie_file.read())
                st.session_state.selfie_path = f.name
    else:
        st.markdown('<p class="fg-tip">› Face camera directly &nbsp;› Even lighting &nbsp;› Remove glasses if possible</p>', unsafe_allow_html=True)
        cam = st.camera_input("Capture", label_visibility="collapsed")
        if cam:
            st.image(cam, caption="Live Capture", use_container_width=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                f.write(cam.read())
                st.session_state.selfie_path = f.name
            img_chk = cv2.imread(st.session_state.selfie_path, cv2.IMREAD_GRAYSCALE)
            if img_chk is None or np.var(img_chk) < 20:
                st.error("[ ERR ] Image quality too low. Retake in better lighting.")
                st.session_state.selfie_path = None

    st.markdown("<br>", unsafe_allow_html=True)
    verify_btn = st.button("⬡  INITIATE VERIFICATION", use_container_width=True)

# ── RIGHT — RESULT ───────────────────────
with right_col:
    st.markdown('<div class="fg-section">Analysis output</div>', unsafe_allow_html=True)

    if verify_btn:
        if not id_image:
            st.warning("[ WARN ] No ID document provided.")
        elif not st.session_state.selfie_path:
            st.warning("[ WARN ] No selfie captured.")
        else:
            with st.spinner("[ PROCESSING ] Running biometric analysis..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                    id_image.seek(0)
                    f.write(id_image.read())
                    id_path = f.name
                result = verify_faces(id_path, st.session_state.selfie_path)
                st.session_state.result = result

    result = st.session_state.result

    if result is None:
        st.markdown("""
        <div class="fg-panel fg-result-idle">
            <div class="fg-result-icon">◎</div>
            <div class="fg-idle-text">
                SYSTEM READY<br>── ── ── ── ──<br>
                UPLOAD ID + SELFIE<br>THEN INITIATE SCAN
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif result.get("error"):
        st.markdown(f"""
        <div class="fg-panel">
            <div class="fg-section" style="color:#ff4040;">System alert</div>
            <p style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#ff5050;line-height:1.9;">
                [ ERR ] {result['error']}
            </p>
            <p class="fg-tip">
                › Face must be clearly visible in both images<br>
                › Use even frontal lighting — avoid shadows<br>
                › Remove sunglasses or face obstructions
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        verified   = result["verified"]
        confidence = result["confidence"]
        distance   = result["distance"]
        threshold  = result["threshold"]

        if verified:
            bar_color = "#00ff64"
            st.markdown("""
            <div class="fg-match">
                <div class="fg-result-label" style="color:#00ff64;">✓ Identity Confirmed</div>
                <div class="fg-result-sub">BIOMETRIC MATCH DETECTED</div>
            </div>""", unsafe_allow_html=True)
        else:
            bar_color = "#ff2828"
            st.markdown("""
            <div class="fg-nomatch">
                <div class="fg-result-label" style="color:#ff2828;">✗ Identity Mismatch</div>
                <div class="fg-result-sub">NO BIOMETRIC MATCH FOUND</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="fg-conf-wrap">
            <div class="fg-conf-header">
                <span>MATCH CONFIDENCE</span>
                <span class="fg-conf-val">{confidence}%</span>
            </div>
            <div class="fg-bar-track">
                <div class="fg-bar-fill" style="width:{confidence}%;background:{bar_color};"></div>
            </div>
        </div>
        <div class="fg-metrics">
            <div class="fg-metric">
                <span class="fg-metric-val">{round(distance,4)}</span>
                <span class="fg-metric-key">Distance</span>
            </div>
            <div class="fg-metric">
                <span class="fg-metric-val">{round(threshold,4)}</span>
                <span class="fg-metric-key">Threshold</span>
            </div>
            <div class="fg-metric">
                <span class="fg-metric-val">{confidence}%</span>
                <span class="fg-metric-key">Confidence</span>
            </div>
        </div>
        <p class="fg-tip" style="margin-top:1rem;">
            › MODEL: ArcFace &nbsp;&nbsp;› DETECTOR: MTCNN<br>
            › Distance &lt; threshold = match confirmed
        </p>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬡  RESET — NEW VERIFICATION"):
            st.session_state.result = None
            st.session_state.selfie_path = None
            st.rerun()
