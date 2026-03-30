import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ==========================================
# 1. PERSIAPAN DATA & EKSTRAKSI FITUR
# ==========================================
print("Membaca dataset dan memproses TF-IDF...")
df = pd.read_csv('dataset_labeled.csv') # Pastikan nama file sesuai

X = df['Judul']
y = df['Sentimen']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================================
# 2. MODEL 1: NAIVE BAYES (BASELINE)
# ==========================================
print("\n[1/2] Melatih Model Naive Bayes (Baseline)...")
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Evaluasi Naive Bayes
y_pred_nb = nb_model.predict(X_test_vec)
acc_nb = accuracy_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb, average='weighted')

print(f"-> Akurasi Naive Bayes : {acc_nb:.4f}")
print(f"-> F1-Score Naive Bayes: {f1_nb:.4f}")

# ==========================================
# 3. MODEL 2: SVM dengan GridSearchCV (TUNING)
# ==========================================
print("\n[2/2] Melatih Model SVM & Mencari Parameter Terbaik (GridSearchCV)...")
# Kita suruh komputer mencoba nilai C: 0.1, 1, 10, dan 100
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear']}

svm_base = SVC(random_state=42)
grid_search = GridSearchCV(svm_base, param_grid, cv=5, scoring='f1_weighted', verbose=1)

# Proses pencarian dimulai (mungkin butuh waktu beberapa detik/menit)
grid_search.fit(X_train_vec, y_train)

# Ambil model SVM yang paling sempurna
best_svm_model = grid_search.best_estimator_

# Evaluasi SVM Terbaik
y_pred_svm = best_svm_model.predict(X_test_vec)
acc_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

print(f"-> Parameter C Terbaik : {grid_search.best_params_['C']}")
print(f"-> Akurasi SVM Terbaik : {acc_svm:.4f}")
print(f"-> F1-Score SVM Terbaik: {f1_svm:.4f}")

# ==========================================
# 4. KESIMPULAN & VISUALISASI
# ==========================================
print("\n=== KESIMPULAN PERBANDINGAN ===")
if f1_svm > f1_nb:
    print(f"SVM MENGALAHKAN Naive Bayes dengan selisih F1-Score: {f1_svm - f1_nb:.4f}")
else:
    print("Wah, Naive Bayes ternyata lebih unggul atau seimbang!")

# Bikin 2 Grafik Confusion Matrix jejeran biar keren pas presentasi
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grafik Naive Bayes
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Reds', 
            xticklabels=nb_model.classes_, yticklabels=nb_model.classes_, ax=axes[0])
axes[0].set_title('Confusion Matrix - Naive Bayes (Baseline)')
axes[0].set_ylabel('Asli (Aktual)')
axes[0].set_xlabel('Prediksi')

# Grafik SVM
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_svm_model.classes_, yticklabels=best_svm_model.classes_, ax=axes[1])
axes[1].set_title(f'Confusion Matrix - Tuned SVM (C={grid_search.best_params_["C"]})')
axes[1].set_ylabel('Asli (Aktual)')
axes[1].set_xlabel('Prediksi')

plt.tight_layout()
plt.show()