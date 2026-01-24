from deepface import DeepFace

MODEL_NAME = "ArcFace"
DETECTOR = "mtcnn"
THRESHOLD = 0.6

def verify_faces(id_path, selfie_path):
    try:
        result = DeepFace.verify(
            img1_path=id_path,
            img2_path=selfie_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False   # IMPORTANT
        )

        return {
            "verified": bool(result["verified"]),
            "distance": float(result["distance"]),
            "threshold": float(result["threshold"])
        }

    except Exception as e:
        return {
            "verified": False,
            "distance": None,
            "threshold": THRESHOLD,
            "error": str(e)
        }
