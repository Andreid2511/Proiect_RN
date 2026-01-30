import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# CONFIGURARE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../"))

# Cai Catre Fisiere
test_data_path = os.path.join(base_dir, "data/test/test.csv")
model_path = os.path.join(base_dir, "models/trained_model.h5") 
scaler_path = os.path.join(base_dir, "config/preprocessing_params.pkl")
results_dir = os.path.join(base_dir, "results")
docs_dir = os.path.join(base_dir, "docs")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(docs_dir, exist_ok=True)

print("--- ETAPA 6: Evaluare Finală Model (Test Set) ---")

# INCARCARE RESURSE
print("1. Incarcare resurse...")

# Date de Test
if not os.path.exists(test_data_path):
    print(f"❌ EROARE: Nu gasesc {test_data_path}")
    exit()
df_test = pd.read_csv(test_data_path)
X_test = df_test[['rpm', 'speed', 'acceleration', 'throttle', 'brake', 'tilt', 'gear']].values
y_test = df_test['style_label'].values

# Scaler
if not os.path.exists(scaler_path):
    print("❌ EROARE: Nu gasesc preprocessing_params.pkl. Ruleaza train_model.py intai!")
    exit()
scaler = joblib.load(scaler_path)
X_test_scaled = scaler.transform(X_test)

# Model
if not os.path.exists(model_path):
    print("❌ EROARE: Nu gasesc modelul antrenat (.h5)!")
    exit()
model = load_model(model_path)
print("✅ Model si date incarcate cu succes.")

# RULARE PREDICTII
print("2. Generare predictii...")
start_time = tf.timestamp()
probs = model.predict(X_test_scaled, verbose=0)
end_time = tf.timestamp()

y_pred = np.argmax(probs, axis=1)
inference_time_ms = (float(end_time - start_time) * 1000) / len(y_test)

# CALCUL METRICI
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Eco', 'Normal', 'Sport'], output_dict=True)

print("\n" + "="*40)
print(f"📊 REZULTATE FINALE:")
print(f"   Accuracy:  {acc*100:.2f}%")
print(f"   F1-Score:  {f1*100:.2f}%")
print(f"   Inference: {inference_time_ms:.4f} ms/sample")
print("="*40 + "\n")

# Salvam cazurile gresite intr-un CSV pentru a le analiza manual
misclassified_indices = np.where(y_test != y_pred)[0]
if len(misclassified_indices) > 0:
    errors_df = df_test.iloc[misclassified_indices].copy()
    errors_df['Predicted'] = y_pred[misclassified_indices]
    errors_path = os.path.join(results_dir, "errors_found.csv")
    errors_df.to_csv(errors_path, index=False)
    print(f"⚠️  Au fost gasite {len(misclassified_indices)} erori. Detalii salvate in: {errors_path}")
else:
    print("🌟 Modelul a prezis perfect tot setul de test!")

print("\n--- Evaluare Completa! ---")