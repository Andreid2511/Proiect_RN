import pandas as pd
import numpy as np
import os
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- 1. SETUP CAI (PATHS) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Data e in ../../data
data_dir = os.path.abspath(os.path.join(current_dir, "../../data"))
# Config e in ../../config (aici salvam modelul)
config_dir = os.path.abspath(os.path.join(current_dir, "../../config"))
os.makedirs(config_dir, exist_ok=True)

print("--- INCEPERE ANTRENARE ---")

# --- 2. INCARCARE DATE ---
print("1. Incarc datele din CSV...")
try:
    train_df = pd.read_csv(os.path.join(data_dir, "train", "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "validation", "validation.csv"))
except FileNotFoundError:
    print("EROARE: Nu gasesc CSV-urile! Ruleaza intai generate_data.py")
    exit()

# Definim intrarile (Senzorii) si Iesirea (Eticheta)
# Folosim RPM, Speed, Throttle, Brake, Tilt. 
# Nu folosim 'gear' neaparat, dar ajuta. Hai sa-l includem pentru precizie maxima.
features = ['rpm', 'speed', 'throttle', 'brake', 'tilt', 'gear']
target = 'style_label'

X_train = train_df[features]
y_train = train_df[target]

X_val = val_df[features]
y_val = val_df[target]

print(f"   Date antrenare: {len(X_train)} linii")
print(f"   Date validare: {len(X_val)} linii")

# --- 3. PREPROCESARE (SCALARE) ---
print("2. Standardizez datele (Scalare)...")
# Retelele Neuronale au nevoie ca toate intrarile sa aiba aceeasi scara (aprox -1...1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- 4. CONFIGURARE SI ANTRENARE RETEA ---
print("3. Construiesc Reteaua Neuronala (MLP)...")
# Am crescut putin capacitatea pentru a intelege modelele complexe de trafic
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16), # 2 straturi ascunse
    activation='relu',       # Functia de activare standard
    solver='adam',           # Algoritmul de optimizare
    max_iter=1000,           # Numar maxim de epoci (mai multe pt convergenta)
    random_state=42,
    verbose=True             # Ne arata progresul (Loss-ul)
)

print("   Start Antrenare (poate dura cateva secunde)...")
mlp.fit(X_train_scaled, y_train)

# --- 5. EVALUARE ---
print("4. Evaluare Model...")
predictions = mlp.predict(X_val_scaled)
acc = accuracy_score(y_val, predictions)
print(f"   ACURATETE PE VALIDARE: {acc * 100:.2f}%")

print("\nRaport Detaliat:")
print(classification_report(y_val, predictions, target_names=['Urban (0)', 'Extra (1)', 'Auto (2)']))

# Matricea de confuzie ne arata unde greseste
print("\nMatrice de Confuzie:")
print(confusion_matrix(y_val, predictions))

# --- 6. SALVARE ---
print("5. Salvare Model si Scaler...")
joblib.dump(mlp, os.path.join(config_dir, 'driver_model.pkl'))
joblib.dump(scaler, os.path.join(config_dir, 'scaler.pkl'))

print(f"GATA! Modelul a fost salvat in: {config_dir}")