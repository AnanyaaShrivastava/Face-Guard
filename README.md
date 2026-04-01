# 🛡️ Dwarapala Yantra — Identity Verification

AI-powered face verification that compares an ID photo with a live selfie using **ArcFace** deep learning model via DeepFace.

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py                  # Main Streamlit app
├── core/
│   └── verification.py     # Face verification logic (ArcFace + MTCNN)
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level dependencies (for Streamlit Cloud)
├── .streamlit/
│   └── config.toml         # Streamlit theme config
└── README.md
```

## ☁️ Deploy to Streamlit Cloud

1. Push this project to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your GitHub repo, branch (`main`), and set main file as `app.py`
5. Click **Deploy** — that's it!

> ⚠️ First deploy takes ~5–10 minutes as DeepFace models are downloaded.

## 🧠 How It Works

| Step | Detail |
|------|--------|
| Face Detection | MTCNN — accurate, handles varied angles |
| Face Recognition | ArcFace — state-of-the-art accuracy |
| Match Decision | Cosine distance compared against ArcFace threshold |
| Confidence Score | Derived from distance vs threshold ratio |

## ⚙️ Tech Stack

- **Frontend**: Streamlit
- **Face Verification**: DeepFace (ArcFace + MTCNN)
- **Image Processing**: OpenCV, NumPy
