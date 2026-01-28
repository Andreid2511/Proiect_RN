import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# --- 1. CONFIGURARE & HIPERPARAMETRI ---
print(f"--- ANTRENAMENT KERAS FINAL (GPU Disponibil: {len(tf.config.list_physical_devices('GPU')) > 0}) ---")

PARAMS = {
    "architecture": "Dense(32, tanh) -> Dense(32, tanh) -> Dense(16, tanh) -> Dense(3, softmax)",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 150,
    "optimizer": "Adam",
    "loss": "categorical_crossentropy",
    "patience": 15
}

# Cai foldere
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../"))
data_dir = os.path.join(base_dir, "data")
config_dir = os.path.join(base_dir, "config")
models_dir = os.path.join(base_dir, "models")
docs_dir = os.path.join(base_dir, "docs")
results_dir = os.path.join(base_dir, "results")

# Creare foldere daca nu exista
for d in [config_dir, models_dir, docs_dir, results_dir]:
    os.makedirs(d, exist_ok=True)

# --- 2. INCARCARE DATE ---
print("1. Incarc datele...")
try:
    train_df = pd.read_csv(os.path.join(data_dir, "train", "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "validation", "validation.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test", "test.csv"))
except FileNotFoundError:
    print("EROARE: Nu gasesc CSV-urile! Ruleaza generate_data.py")
    exit()

features = ['rpm', 'speed', 'acceleration', 'throttle', 'brake', 'tilt', 'gear']
target = 'style_label'

X_train, y_train = train_df[features].values, train_df[target].values
X_val, y_val = val_df[features].values, val_df[target].values
X_test, y_test = test_df[features].values, test_df[target].values

# --- 3. PREPROCESARE ---
print("2. Preprocesare (Scalare + OneHot)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

y_train_cat = to_categorical(y_train, 3)
y_val_cat = to_categorical(y_val, 3)
y_test_cat = to_categorical(y_test, 3)

# --- 4. DEFINIRE MODEL ---
print("3. Construiesc modelul...")

def build_model():
    model = Sequential([
        Input(shape=(7,)),
        Dense(32, activation='tanh'),
        Dense(32, activation='tanh'),
        Dense(16, activation='tanh'),
        Dense(3, activation='softmax')
    ])
    opt = Adam(learning_rate=PARAMS["learning_rate"])
    model.compile(loss=PARAMS["loss"], optimizer=opt, metrics=['accuracy'])
    return model

model = build_model()

# [CERINTA README] Salvare model neantrenat (untrained)
untrained_path = os.path.join(models_dir, 'untrained_model.h5')
model.save(untrained_path)
print(f"   -> Model neantrenat salvat in: {untrained_path}")

# --- 5. ANTRENARE ---
print("4. Start Antrenare...")

early_stop = EarlyStopping(monitor='val_loss', patience=PARAMS["patience"], restore_best_weights=True, verbose=1)
# Salvam modelul antrenat
trained_path = os.path.join(models_dir, 'trained_model.h5')
checkpoint = ModelCheckpoint(trained_path, monitor='val_loss', save_best_only=True)

history = model.fit(
    X_train_scaled, y_train_cat,
    validation_data=(X_val_scaled, y_val_cat),
    epochs=PARAMS["epochs"],
    batch_size=PARAMS["batch_size"],
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# --- 6. SALVARE REZULTATE (RESULTS FOLDER) ---
print("5. Salvez rezultatele cerute in README...")

# A. Training History CSV
history_df = pd.DataFrame(history.history)
history_csv_path = os.path.join(results_dir, "training_history.csv")
history_df.to_csv(history_csv_path, index_label="epoch")
print(f"   -> Training history salvat in: {history_csv_path}")

# B. Hyperparameters YAML
yaml_content = f"""
model_name: "SIA_Driver_Classifier"
framework: "Keras"
parameters:
  learning_rate: {PARAMS['learning_rate']}
  batch_size: {PARAMS['batch_size']}
  architecture: "{PARAMS['architecture']}"
  optimizer: "{PARAMS['optimizer']}"
  loss_function: "{PARAMS['loss']}"
"""
with open(os.path.join(results_dir, "hyperparameters.yaml"), "w") as f:
    f.write(yaml_content.strip())
print(f"   -> Hyperparameters salvati in results/hyperparameters.yaml")

# --- 7. GRAFICE & EVALUARE ---
print("6. Generez grafice si metrici finale...")

# Grafic Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(docs_dir, "loss_curve.png"))

# Metrici Test
predictions_prob = model.predict(X_test_scaled)
predictions = np.argmax(predictions_prob, axis=1)

# C. Test Metrics JSON
test_acc = accuracy_score(y_test, predictions)
test_f1 = f1_score(y_test, predictions, average='macro')

metrics_dict = {
    "test_accuracy": float(test_acc),
    "test_f1_macro": float(test_f1),
    "report": classification_report(y_test, predictions, target_names=['Eco', 'Normal', 'Agresiv'], output_dict=True)
}

with open(os.path.join(results_dir, "test_metrics.json"), "w") as f:
    json.dump(metrics_dict, f, indent=4)
print(f"   -> Metrici JSON salvate in results/test_metrics.json")

print("\n--- REPORT ---")
print(classification_report(y_test, predictions, target_names=['Eco', 'Normal', 'Agresiv']))

# Matrice Confuzie
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Eco', 'Normal', 'Agresiv'], yticklabels=['Eco', 'Normal', 'Agresiv'])
plt.title('Matrice de Confuzie')
plt.savefig(os.path.join(docs_dir, "confusion_matrix.png"))

# --- 8. SALVARE SCALER ---
joblib.dump(scaler, os.path.join(config_dir, 'scaler.pkl'))

# --- BONUS: EXPORT ONNX (Nivel 3) ---
try:
    import tf2onnx
    import onnx
    print("7. [BONUS] Exporting to ONNX...")
    spec = (tf.TensorSpec((None, 7), tf.float32, name="input"),)
    output_path = os.path.join(models_dir, "final_model.onnx")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(model_proto, output_path)
    print(f"   -> Model ONNX salvat: {output_path}")
except ImportError:
    print("   [INFO] tf2onnx nu este instalat. Skip bonus ONNX.")
except Exception as e:
    print(f"   [INFO] Eroare la export ONNX: {e}")

print("✅ TOATE FISIERELE GENERATE CU SUCCES!")