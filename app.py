import streamlit as st
import tempfile
import cv2
import numpy as np
from core.verification import verify_faces

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Dwarapala Yantra",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- SESSION STATE INIT ----------------
if "verification_status" not in st.session_state:
    st.session_state.verification_status = "Waiting for user input..."

if "selfie_path" not in st.session_state:
    st.session_state.selfie_path = None

# ---------------- STYLES ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.stImage {
    width: 100%;
    border-radius: 16px;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-size: 1.1em;
    font-weight: 600;
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    color: white;
    border: none;
}

h1, h2, h3 {
    text-align: center;
    font-weight: 700;
}

.success-box {
    background-color: #0f5132;
    padding: 1rem;
    border-radius: 12px;
    color: white;
    text-align: center;
}

.error-box {
    background-color: #842029;
    padding: 1rem;
    border-radius: 12px;
    color: white;
    text-align: center;
}

.hover-card {
    background: #111827;
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.3s ease;
}

.hover-card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0 20px 40px rgba(0,0,0,0.35);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🔐 Identity Verification")
st.subheader("AI-powered Multi-Frame Inspired Face Verification")

left, right = st.columns([1.2, 1])

with left:
    st.markdown("""
    <div class="hover-card">
        <h3>📷 Verification Instructions</h3>
        <p>Upload a valid ID and verify using a live selfie.</p>
    </div>
    """, unsafe_allow_html=True)

with right:
    status_box = st.empty()
    status_box.markdown(f"""
    <div class="hover-card">
        <h3>📡 Verification Status</h3>
        <p>{st.session_state.verification_status}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- STEP 1: ID UPLOAD ----------------
st.markdown("## Step 1: Upload ID Proof")

id_image = st.file_uploader(
    "Government ID / Official Photo",
    type=["jpg", "jpeg", "png"]
)

# ---- ID PREVIEW ----
if id_image:
    st.markdown("### 🪪 ID Preview")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(id_image, width=250)

# ---------------- STEP 2: SELFIE METHOD ----------------
st.markdown("## Step 2: Selfie Input Method")

method = st.radio(
    "Choose how you want to provide selfie",
    ["Upload Photo", "Use Webcam"]
)


# -------- UPLOAD SELFIE --------
if method == "Upload Photo":
    selfie_image = st.file_uploader(
        "Upload Selfie",
        type=["jpg", "jpeg", "png"]
    )

    if selfie_image:
        st.markdown("### 🤳 Selfie Preview")
        st.image(selfie_image, width=250)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(selfie_image.read())
            st.session_state.selfie_path = f.name

# -------- WEBCAM SELFIE (ONLY CAMERA) --------
if method == "Use Webcam":
    st.info("Move your head slightly. Ensure good lighting.")

    cam_image = st.camera_input("📸 Capture Live Selfie")

    if cam_image:
       st.markdown("### 🤳 Captured Selfie")
       st.image(cam_image, width=300)

       with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(cam_image.read())
            st.session_state.selfie_path = f.name

       img = cv2.imread(st.session_state.selfie_path, cv2.IMREAD_GRAYSCALE)
       if img is None or np.var(img) < 20:
            st.error("❌ Image quality too low. Try again.")
            st.stop()


# ---------------- VERIFY ----------------
st.markdown("---")
verify_btn = st.button("🔐 Verify Identity")

if verify_btn:
    if not id_image:
        st.warning("Please upload ID proof.")
        st.stop()

    if not st.session_state.selfie_path:
        st.warning("Please provide a selfie.")
        st.stop()

    with st.spinner("🧠 Verifying identity..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(id_image.read())
            id_path = f.name

        result = verify_faces(id_path, st.session_state.selfie_path)

    if result.get("verified"):
        st.session_state.verification_status = "✅ Identity Matched"
        status_box.markdown(f"""
        <div class="hover-card">
            <h3>📡 Verification Status</h3>
            <p>{st.session_state.verification_status}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<div class='success-box'>✅ Identity Verified Successfully</div>",
            unsafe_allow_html=True
        )
    else:
        st.session_state.verification_status = "❌ Identity Not Matched"
        status_box.markdown(f"""
        <div class="hover-card">
            <h3>📡 Verification Status</h3>
            <p>{st.session_state.verification_status}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<div class='error-box'>❌ Identity Verification Failed</div>",
            unsafe_allow_html=True
        )

    st.metric("Distance", round(result["distance"], 4))
    st.metric("Threshold", round(result["threshold"], 4))