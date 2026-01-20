import streamlit as st
import os
from PIL import Image
from core.verification import verify_faces
st.set_page_config(
    page_title="Dwarapala Yantra",
    page_icon="🛡️",
    layout="centered"
)

st.markdown("""
# 🛡️ Dwarapala Yantra
### AI-powered Face Verification System

> *Inspired by the ancient guardians of temples,  
Dwarapala Yantra verifies identity using deep learning.*
""")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload ID Image")
    id_image = st.file_uploader(
        "Government ID / Official Photo",
        type=["jpg", "jpeg", "png"],
        key="id"
    )

with col2:
    st.subheader("🤳 Upload Selfie")
    selfie_image = st.file_uploader(
        "Live Selfie",
        type=["jpg", "jpeg", "png"],
        key="selfie"
    )
if id_image and selfie_image:
    st.markdown("### 🔍 Image Preview")

    p1, p2 = st.columns(2)
    with p1:
        st.image(id_image, caption="ID Image", use_container_width=True)
    with p2:
        st.image(selfie_image, caption="Selfie Image", use_container_width=True)
st.markdown("---")
verify_btn = st.button("🔐 Verify Identity", use_container_width=True)
if verify_btn and id_image and selfie_image:
    with st.spinner("🧠 Verifying identity..."):
        # save uploaded images temporarily
        os.makedirs("temp", exist_ok=True)

        id_path = "temp/id.jpg"
        selfie_path = "temp/selfie.jpg"

        with open(id_path, "wb") as f:
            f.write(id_image.getbuffer())

        with open(selfie_path, "wb") as f:
            f.write(selfie_image.getbuffer())

        result = verify_faces(id_path, selfie_path)
    st.markdown("## 📊 Result")

    if result["verified"]:
        st.success("✅ Identity Verified")
    else:
        st.error("❌ Identity Mismatch")

    st.metric("Distance", round(result["distance"], 4))
    st.metric("Threshold", round(result["threshold"], 4))


