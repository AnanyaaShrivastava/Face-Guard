from deepface import DeepFace


def verify_faces(id_img_path, selfie_img_path):
    result = DeepFace.verify(
        img1_path=id_img_path,
        img2_path=selfie_img_path,
        model_name="ArcFace",
        detector_backend="retinaface",
        enforce_detection=True
    )

    return {
        "verified": result["verified"],
        "distance": result["distance"],
        "threshold": result["threshold"],
        "model": "ArcFace"
    }


if __name__ == "__main__":
    output = verify_faces(
        "data/id_images/id.jpg",
        "data/selfie_images/selfie.jpg"
    )

    print(output)
