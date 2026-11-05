"""
Pipeline Lengkap Analisis & Pemodelan Machine Learning
Dataset: Credit Approval (CRX) - Sains Data Minggu ke-7
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Setup style visualisasi
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

# 1. Pemuatan Data
print("=" * 60)
print("1. MEMUAT DATASET CREDIT APPROVAL (CRX)")
print("=" * 60)

col_names = [f'A{i}' for i in range(1, 16)] + ['Approved']
df = pd.read_csv('crx.data', header=None, names=col_names, na_values='?')

print(f"Total Baris: {df.shape[0]}")
print(f"Total Kolom: {df.shape[1]}")
print(f"Kolom: {list(df.columns)}")

# Identifikasi Tipe Kolom
num_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
cat_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
target_col = 'Approved'

# Binarisasi target (+ -> 1, - -> 0)
df['Approved_Num'] = (df[target_col] == '+').astype(int)

# 2. Visualisasi EDA
print("\n" + "=" * 60)
print("2. MEMBUAT VISUALISASI EDA")
print("=" * 60)

# Visualisasi 1: Distribusi Target
fig, ax = plt.subplots(figsize=(7, 4.5))
palette = ['#e74c3c', '#2ecc71']
counts = df['Approved'].value_counts()
bars = ax.bar(['Ditolak (-)', 'Disetujui (+)'], counts.values, color=palette, width=0.55, edgecolor='black', linewidth=1.2)
ax.set_title('Distribusi Keputusan Persetujuan Kredit (Target: Approved)', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Jumlah Pemohon', fontsize=11)
ax.set_ylim(0, max(counts.values) * 1.18)

for bar in bars:
    yval = bar.get_height()
    pct = (yval / len(df)) * 100
    ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 10, f'{yval:,} ({pct:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_distribusi_target.png')
plt.close()
print("- Saved: eda_distribusi_target.png")

# Visualisasi 2: Hubungan Fitur A9 (PriorDefault) dan A10 (Employed) terhadap Keputusan Kredit
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot A9
ct_a9 = pd.crosstab(df['A9'], df['Approved'], normalize='index') * 100
ct_a9.plot(kind='bar', stacked=True, ax=axes[0], color=['#e74c3c', '#2ecc71'], edgecolor='black', width=0.5)
axes[0].set_title('Pengaruh Prior Default (A9) terhadap Keputusan Kredit', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Riwayat Gagal Bayar (A9: t = Ya, f = Tidak)', fontsize=10)
axes[0].set_ylabel('Persentase (%)', fontsize=10)
axes[0].legend(['Ditolak (-)', 'Disetujui (+)'], title='Keputusan', loc='upper right')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Plot A10
ct_a10 = pd.crosstab(df['A10'], df['Approved'], normalize='index') * 100
ct_a10.plot(kind='bar', stacked=True, ax=axes[1], color=['#e74c3c', '#2ecc71'], edgecolor='black', width=0.5)
axes[1].set_title('Pengaruh Status Bekerja (A10) terhadap Keputusan Kredit', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Status Pekerjaan (A10: t = Bekerja, f = Tidak)', fontsize=10)
axes[1].set_ylabel('Persentase (%)', fontsize=10)
axes[1].legend(['Ditolak (-)', 'Disetujui (+)'], title='Keputusan', loc='upper right')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('eda_fitur_kunci.png')
plt.close()
print("- Saved: eda_fitur_kunci.png")

# Visualisasi 3: Heatmap Korelasi Numerik
fig, ax = plt.subplots(figsize=(8, 6))
corr_df = df[num_cols + ['Approved_Num']].corr()
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True,
            linewidths=1, linecolor='white', ax=ax, vmin=-1, vmax=1)
ax.set_title('Heatmap Korelasi Pearson Antara Fitur Numerik & Target Kredit', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('eda_korelasi_numerik.png')
plt.close()
print("- Saved: eda_korelasi_numerik.png")

# 3. Preprocessing Data & Pipeline
print("\n" + "=" * 60)
print("3. MEMBANGUN PIPELINE PREPROCESSING & SPLIT DATA")
print("=" * 60)

X = df[num_cols + cat_cols].copy()
y = df['Approved_Num'].copy()

# Split Data (Stratified 80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Data Latih (Training): {X_train.shape[0]} baris")
print(f"Data Uji (Testing)   : {X_test.shape[0]} baris")

# Preprocessor:
# - Numerik: SimpleImputer (median) + StandardScaler
# - Kategorikal: SimpleImputer (most_frequent) + OneHotEncoder (handle_unknown='ignore')
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# 4. Pelatihan Model & Evaluasi Multi-Model
print("\n" + "=" * 60)
print("4. PELATIHAN & EVALUASI MODEL KLASIFIKASI")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    'Support Vector Machine': SVC(probability=True, kernel='rbf', random_state=42),
    'Naive Bayes': GaussianNB()
}

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model_name = None
best_f1 = 0
best_pipeline = None

for name, model in models.items():
    pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', model)
    ])
    
    # Cross Validation Score (5-fold)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Fit pada training set
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
    
    results.append({
        'Model': name,
        'CV Accuracy': cv_scores.mean(),
        'Test Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    })
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_pipeline = pipe

res_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
print(res_df.to_string(index=False))

# 5. Visualisasi Perbandingan Model
fig, ax = plt.subplots(figsize=(11, 5.5))
metrics_to_plot = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
res_plot = res_df.set_index('Model')[metrics_to_plot]

res_plot.plot(kind='bar', ax=ax, width=0.75, colormap='viridis', edgecolor='black', linewidth=0.8)
ax.set_title('Perbandingan Performa Model Klasifikasi Persetujuan Kredit', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Skor Evaluasi (0 - 1.0)', fontsize=11)
ax.set_ylim(0.65, 1.0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
ax.legend(title='Metrik Evaluasi', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('perbandingan_performa_model.png')
plt.close()
print("- Saved: perbandingan_performa_model.png")

# 6. Confusion Matrix & ROC Curve untuk Model Terbaik
print("\n" + "=" * 60)
print(f"5. EVALUASI MODEL TERBAIK: {best_model_name}")
print("=" * 60)

y_pred_best = best_pipeline.predict(X_test)
y_prob_best = best_pipeline.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred_best)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            annot_kws={'size': 14, 'weight': 'bold'},
            xticklabels=['Ditolak (-)', 'Disetujui (+)'],
            yticklabels=['Ditolak (-)', 'Disetujui (+)'])
axes[0].set_title(f'Confusion Matrix: {best_model_name}', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Prediksi Model', fontsize=11)
axes[0].set_ylabel('Label Sebenarnya (Ground Truth)', fontsize=11)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
auc_val = roc_auc_score(y_test, y_prob_best)
axes[1].plot(fpr, tpr, color='#2980b9', lw=2.5, label=f'{best_model_name} (AUC = {auc_val:.3f})')
axes[1].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Baseline (AUC = 0.500)')
axes[1].set_title('Kurva ROC (Receiver Operating Characteristic)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('False Positive Rate (FPR)', fontsize=11)
axes[1].set_ylabel('True Positive Rate (TPR)', fontsize=11)
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('confusion_matrix_terbaik.png')
plt.close()
print("- Saved: confusion_matrix_terbaik.png")

# 7. Analisis Kepentingan Fitur (Feature Importance dari Random Forest)
rf_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
])
rf_pipe.fit(X_train, y_train)

# Ambil nama fitur setelah one-hot encoding
cat_encoded_names = list(rf_pipe.named_steps['prep'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols))
all_feature_names = num_cols + cat_encoded_names
importances = rf_pipe.named_steps['clf'].feature_importances_

feat_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', hue='Feature', palette='crest', ax=ax, edgecolor='black', legend=False)
ax.set_title('Top 10 Fitur Paling Berpengaruh (Random Forest)', fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Tingkat Kepentingan (Gini Importance)', fontsize=11)
ax.set_ylabel('Nama Fitur', fontsize=11)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("- Saved: feature_importance.png")

print("\n" + "=" * 60)
print("PIPELINE SELESAI DENGAN SUKSES!")
print("=" * 60)

