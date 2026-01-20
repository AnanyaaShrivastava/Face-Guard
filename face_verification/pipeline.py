from deepface import DeepFace
from face_verification.liveness import check_liveness


def run_pipeline(id_img_path, selfie_img_path):
    """
    Full verification pipeline:
    1. Static liveness detection
    2. Face verification (ArcFace)
    """

    # Step 1: Liveness check
    is_live, texture_score = check_liveness(selfie_img_path)

    # ---- DEFAULT RESULT (IMPORTANT) ----
    result_dict = {
        "liveness": False,
        "texture_score": texture_score,
        "verified": False,
        "distance": None,
        "threshold": None,
        "model": "ArcFace",
        "reason": None
    }

    # If liveness fails, return early (BUT with full schema)
    if not is_live:
        result_dict["reason"] = "Liveness check failed"
        return result_dict

    # Step 2: Face verification
    df_result = DeepFace.verify(
    img1_path=id_img_path,
    img2_path=selfie_img_path,
    model_name="ArcFace",
    enforce_detection=False,
    distance_metric="cosine"
)


    # Fill verification results
    result_dict["liveness"] = True
    result_dict["verified"] = df_result.get("verified", False)
    result_dict["distance"] = df_result.get("distance", None)
    result_dict["threshold"] = df_result.get("threshold", None)

    return result_dict

