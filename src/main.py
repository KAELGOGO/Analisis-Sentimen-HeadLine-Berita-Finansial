from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# 1. Inisialisasi Aplikasi FastAPI
app = FastAPI(title="API Sentimen Berita", version="1.0")

# 2. Konfigurasi CORS (Sangat penting agar Next.js bisa connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Saat di-deploy nanti, ganti "*" dengan link web Vercel temanmu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load Model AI dan Kamus TF-IDF (Hanya dilakukan sekali saat server menyala)
print("Memuat model SVM dan Vectorizer...")
try:
    model = joblib.load('model_sentimen_svm.pkl')
    vectorizer = joblib.load('vectorizer_tfidf.pkl')
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model: {e}")

# 4. Membuat Struktur Data Input (Format JSON yang diharapkan)
class BeritaInput(BaseModel):
    teks: str

# 5. Endpoint Utama untuk Prediksi
@app.post("/api/analisis")
def analisis_sentimen(berita: BeritaInput):
    # Cek jika input kosong
    if not berita.teks.strip():
        raise HTTPException(status_code=400, detail="Teks berita tidak boleh kosong")
    
    try:
        # Ubah teks menjadi angka menggunakan TF-IDF
        teks_vektor = vectorizer.transform([berita.teks])
        
        # Model SVM menebak sentimen
        tebakan = model.predict(teks_vektor)[0]
        
        # Kembalikan hasil dalam format JSON
        return {
            "status": "success",
            "teks_asli": berita.teks,
            "sentimen": tebakan
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint bonus untuk ngecek apakah server hidup
@app.get("/")
def read_root():
    return {"message": "Server API Sentimen Berita Aktif! Kirim POST request ke /api/analisis"}