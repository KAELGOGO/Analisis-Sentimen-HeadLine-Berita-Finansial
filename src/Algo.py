import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ==========================================
# FASE 1: LOAD DATA
# ==========================================
print("Membaca dataset...")
# Ganti 'dataset_berita.csv' dengan nama file kamu
df = pd.read_csv('dataset_labeled.csv') 

# Asumsi nama kolom teksnya 'Judul' dan labelnya 'Sentimen'
X = df['Judul']      # Data teks (Fitur)
y = df['Sentimen']   # Label/Kunci Jawaban (Positif, Negatif, Netral)

# ==========================================
# FASE 2: SPLITTING DATA (Sesuai PPT Slide 9)
# ==========================================
# Kita bagi 80% untuk Training (Belajar), 20% untuk Testing (Ujian)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Jumlah Data Training: {len(X_train)}")
print(f"Jumlah Data Testing: {len(X_test)}")

# ==========================================
# FASE 3: FEATURE EXTRACTION (Ubah Teks -> Angka)
# ==========================================
print("\nMengekstraksi fitur dengan TF-IDF...")
vectorizer = TfidfVectorizer(lowercase=True, stop_words=None) # Stop words bisa disesuaikan nanti
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================================
# FASE 4: TRAINING MODEL 
# ==========================================
print("\nMelatih Model AI...")
# Pilihan 1: Naive Bayes (Super Cepat)
# model = MultinomialNB()

# Pilihan 2: Support Vector Machine / SVM (Biasanya lebih akurat untuk teks)
model = SVC(kernel='linear', random_state=42)

# Mulai proses belajar!
model.fit(X_train_vec, y_train)
print("Training Selesai!")

# ==========================================
# FASE 5: EVALUATION (Sesuai PPT Slide 8 & 9)
# ==========================================
# AI mencoba menebak data Testing
y_pred = model.predict(X_test_vec)

# Menghitung Metrik Evaluasi
print("\n=== HASIL EVALUASI MODEL ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
print(f"Recall   : {recall_score(y_test, y_pred, average='weighted'):.2f}")
print(f"F1-Score : {f1_score(y_test, y_pred, average='weighted'):.2f}") # Metrik andalanmu di presentasi!

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# Membuat Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)
plt.title('Confusion Matrix - Prediksi Sentimen Berita')
plt.ylabel('Sentimen Asli (Aktual)')
plt.xlabel('Tebakan AI (Prediksi)')
plt.show()


# ==========================================
# FASE 6: MENYIMPAN MODEL & VECTORIZER (Siap Deploy)
# ==========================================
print("\nMenyimpan model ke dalam file...")

# Menyimpan model SVM
joblib.dump(model, 'model_sentimen_svm.pkl')

# Menyimpan TF-IDF Vectorizer
joblib.dump(vectorizer, 'vectorizer_tfidf.pkl')

print("Berhasil! Model dan Vectorizer sudah tersimpan (Cek folder projek kamu).")