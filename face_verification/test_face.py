import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from face_verification.pipeline import run_pipeline

ID_IMAGE = "data/id.jpg"
SELFIE_IMAGE = "data/selfie.jpg"

print("\n========== DWARAPALA YANTRA ==========\n")

result = run_pipeline(ID_IMAGE, SELFIE_IMAGE)

if not result["liveness"]:
    print("❌ LIVENESS CHECK FAILED")
    print(f"Texture Score : {round(result['texture_score'], 2)}")
    print("Verification aborted.\n")

else:
    print("✅ LIVENESS CHECK PASSED")
    print(f"Texture Score : {round(result['texture_score'], 2)}\n")

    print("========== FACE VERIFICATION ==========\n")
    print(f"Model Used        : {result['model']}")
    print(f"Same Person?      : {result['verified']}")
    print(f"Similarity Score  : {round(result['distance'], 4)}")
    print(f"Decision Threshold: {result['threshold']}")
    print("\n=====================================\n")
