import pandas as pd
import numpy as np
import os
import time
import json
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
os.makedirs(results_dir, exist_ok=True)

print("--- ETAPA 6: Rulare Experimente Optimizare (MINIM 4) ---")

# 1. Încărcare Date
print(f"1. Incarcare date din: {data_path}")
if not os.path.exists(data_path):
    print("❌ EROARE: Nu gasesc train.csv. Ruleaza intai generate_data.py!")
    exit()

# Folosim 30% din date pentru viteză
df = pd.read_csv(data_path).sample(frac=0.3) 
X = df[['rpm', 'speed', 'acceleration', 'throttle', 'brake', 'tilt', 'gear']].values
y = df['style_label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Definire Arhitecturi Experimentale
def get_model(arch_type, input_shape):
    model = Sequential()
    
    if arch_type == "Small (Rapid)":
        # Model mic
        model.add(Dense(16, input_shape=input_shape, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        
    elif arch_type == "Baseline (Tanh)":
        # Modelul curent
        model.add(Dense(32, input_shape=input_shape, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(16, activation='tanh'))
        model.add(Dense(3, activation='softmax'))
        
    elif arch_type == "Baseline (ReLU)":
        # Baseline, dar cu ReLU
        model.add(Dense(32, input_shape=input_shape, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        
    elif arch_type == "Large (Dropout)":
        # Model mare cu Dropout
        model.add(Dense(128, input_shape=input_shape, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. Rulare Experimente
results = []
# Lista cu cele 4 experimente
architectures = ["Small (Rapid)", "Baseline (Tanh)", "Baseline (ReLU)", "Large (Dropout)"]

print("\n--- Start Antrenament Comparativ (4 Modele) ---")
for arch in architectures:
    print(f"   -> Testare: {arch} ...")
    model = get_model(arch, (X_train.shape[1],))
    
    # Antrenare scurta
    start_train = time.time()
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0) 
    train_time = time.time() - start_train
    
    # Predictie & Timp
    start_inf = time.time()
    preds = model.predict(X_test, verbose=0)
    end_inf = time.time()
    latency_ms = ((end_inf - start_inf) / len(X_test)) * 1000 
    
    # Metrici
    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"      [Rezultat] Acc: {acc:.4f} | Latency: {latency_ms:.4f} ms")
    
    # Observatii pentru tabel
    obs = "Optim"
    if acc < 0.90: obs = "Sub-fit (Prea simplu)"
    elif "Large" in arch: obs = "Complex (Latență mare)"
    elif "ReLU" in arch: obs = "Alternativă validă"
    
    results.append({
        "Experiment ID": f"EXP-{len(results)+1:02d}",
        "Arhitectura": arch,
        "Accuracy": round(acc, 4),
        "F1 Score": round(f1, 4),
        "Inference Latency (ms)": round(latency_ms, 4),
        "Training Time (s)": round(train_time, 2),
        "Observatii": obs
    })

# 4. Salvare
csv_path = os.path.join(results_dir, "optimization_experiments.csv")
res_df = pd.DataFrame(results)
res_df.to_csv(csv_path, index=False)
print(f"\n✅ Tabel experimente salvat in: {csv_path}")
print(res_df.to_string())