import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# CONFIGURARE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../"))
data_path = os.path.join(base_dir, "data/train/train.csv")
results_dir = os.path.join(base_dir, "results")
docs_optimization_dir = os.path.join(base_dir, "docs/optimization")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(docs_optimization_dir, exist_ok=True)

print("--- ETAPA 6: Rulare 5 Experimente de Optimizare ---")

# Incarcare Date
if not os.path.exists(data_path):
    print("❌ EROARE: Nu gasesc train.csv!")
    exit()

df = pd.read_csv(data_path).sample(frac=0.3, random_state=42) 
X = df[['rpm', 'speed', 'acceleration', 'throttle', 'brake', 'tilt', 'gear']].values
y = df['style_label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definire modele
def get_model(arch_type, input_shape):
    model = Sequential()
    if arch_type == "Small (Rapid)":
        model.add(Dense(16, input_shape=input_shape, activation='relu'))
        model.add(Dense(3, activation='softmax'))
    elif arch_type == "Baseline (Tanh)": 
        model.add(Dense(32, input_shape=input_shape, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(16, activation='tanh'))
        model.add(Dense(3, activation='softmax'))
    elif arch_type == "Baseline (ReLU)":
        model.add(Dense(32, input_shape=input_shape, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(3, activation='softmax'))
    elif arch_type == "Pyramid (Deep)":
        model.add(Dense(64, input_shape=input_shape, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(3, activation='softmax'))
    elif arch_type == "Large (Dropout)":
        model.add(Dense(128, input_shape=input_shape, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
# Antrenament si evaluare
results = []
architectures = ["Small (Rapid)", "Baseline (Tanh)", "Baseline (ReLU)", "Pyramid (Deep)", "Large (Dropout)"]

print("\n--- Start Antrenament Comparativ (5 Modele) ---")
for arch in architectures:
    print(f"   -> Testare: {arch} ...")
    model = get_model(arch, (X_train.shape[1],))
    
    start_train = time.time()
    model.fit(X_train, y_train, epochs=6, batch_size=32, verbose=0) 
    train_time = time.time() - start_train
    
    start_inf = time.time()
    preds = model.predict(X_test, verbose=0)
    end_inf = time.time()
    latency_ms = ((end_inf - start_inf) / len(X_test)) * 1000 
    
    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"      [Rezultat] F1: {f1:.4f} | Latency: {latency_ms:.4f} ms")
    
    results.append({
        "Experiment ID": f"EXP-{len(results)+1:02d}",
        "Arhitectura": arch,
        "Accuracy": acc,
        "F1 Score": f1,
        "Latency": latency_ms,
        "Time": train_time
    })
# Salvare rezultate si grafice
df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(results_dir, "optimization_experiments.csv"), index=False)

plt.figure(figsize=(10, 6))
colors_f1 = ['#95a5a6', '#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
sns.barplot(x="F1 Score", y="Arhitectura", data=df_res, palette=colors_f1)
plt.title("Comparatie F1 Score (Stabilitate)", fontsize=14)
plt.xlabel("F1 Score", fontsize=12)
plt.xlim(0.8, 1.0) 
plt.grid(axis='x', linestyle='--', alpha=0.7)
for index, row in df_res.iterrows():
    plt.text(row['F1 Score'], index, f'{row["F1 Score"]:.4f}', color='black', ha="left", va="center")

plt.savefig(os.path.join(docs_optimization_dir, "f1_comparison.png"), bbox_inches='tight')

plt.figure(figsize=(10, 6))
sns.barplot(x="Accuracy", y="Arhitectura", data=df_res, palette="magma")
plt.title("Comparatie Acuratețe (Precizie Globală)", fontsize=14)
plt.xlabel("Accuracy", fontsize=12)
plt.xlim(0.8, 1.0) # Zoom 
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Adaugam etichetele 
for index, row in df_res.iterrows():
    plt.text(row.Accuracy, index, f'{row.Accuracy:.4f}', color='black', ha="left", va="center")

plt.savefig(os.path.join(docs_optimization_dir, "accuracy_comparison.png"), bbox_inches='tight')

print(f"\n✅ Grafic F1 salvat: docs/optimization/f1_comparison.png")
print(f"✅ Grafic Accuracy salvat: docs/optimization/accuracy_comparison.png")
print("✅ CSV salvat.")