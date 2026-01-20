import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not found")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

if __name__ == "__main__":
    # Initialize face model
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load images
    id_img = load_image("data/id_images/id.jpg")
    selfie_img = load_image("data/selfie_images/selfie.jpg")

    # Detect faces and extract embeddings
    id_faces = app.get(id_img)
    selfie_faces = app.get(selfie_img)

    if len(id_faces) == 0 or len(selfie_faces) == 0:
        raise ValueError("No face detected in one of the images")

    id_embedding = id_faces[0].embedding
    selfie_embedding = selfie_faces[0].embedding

    similarity = cosine_similarity(id_embedding, selfie_embedding)

    print(f"Cosine Similarity Score: {similarity:.4f}")

    if similarity > 0.5:
        print("✅ Identity Match")
    else:
        print("❌ Identity Mismatch")
