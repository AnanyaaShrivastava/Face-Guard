from deepface import DeepFace
import cv2
import numpy as np
import streamlit as st

MODEL_NAME = "ArcFace"
DETECTOR = "mtcnn"


def check_image_quality(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, "Could not read image."
    if np.var(img) < 20:
        return False, "Image is too dark or blurry."
    return True, "OK"


def get_confidence(distance, threshold):
    if distance is None:
        return 0.0
    confidence = max(0.0, min(100.0, (1 - distance / (threshold * 2)) * 100))
    return round(confidence, 1)


@st.cache_resource(show_spinner="⏳ Loading AI model (first time only)...")
def load_model():
    """
    Pre-warm DeepFace model once and cache it in memory.
    This prevents the app from crashing on every verification call.
    """
    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    dummy[60:100, 60:100] = 200

    try:
        DeepFace.represent(
            img_path=dummy,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False
        )
    except Exception:
        pass

    return True


def verify_faces(id_path, selfie_path):
    # Ensure model is loaded/cached before verifying
    load_model()

    for path, label in [(id_path, "ID"), (selfie_path, "Selfie")]:
        ok, msg = check_image_quality(path)
        if not ok:
            return {
                "verified": False,
                "distance": None,
                "threshold": None,
                "confidence": 0.0,
                "error": f"{label} image issue: {msg}"
            }

    try:
        result = DeepFace.verify(
            img1_path=id_path,
            img2_path=selfie_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True
        )

        distance = float(result["distance"])
        threshold = float(result["threshold"])
        verified = bool(result["verified"])
        confidence = get_confidence(distance, threshold)

        return {
            "verified": verified,
            "distance": distance,
            "threshold": threshold,
            "confidence": confidence,
            "error": None
        }

    except ValueError:
        return {
            "verified": False,
            "distance": None,
            "threshold": None,
            "confidence": 0.0,
            "error": "No face detected in one or both images. Please use a clear, well-lit photo."
        }

    except Exception as e:
        return {
            "verified": False,
            "distance": None,
            "threshold": None,
            "confidence": 0.0,
            "error": f"Verification error: {str(e)}"
        }
