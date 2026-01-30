import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import tensorflow as tf
from keras.models import load_model

plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = '#121212'
plt.rcParams['savefig.facecolor'] = '#121212'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Cai Foldere
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../"))
docs_results_dir = os.path.join(base_dir, "docs/results")
os.makedirs(docs_results_dir, exist_ok=True)

print("--- RULARE VISUALIZE.PY (GENERARE GRAFICE FINALE) ---")

#metrics_evolution.png
def generate_metrics_evolution():
    print("1. Generare 'metrics_evolution.png'...")
    
    # DATELE TALE REALE
    stages = ["Etapa 4\n(Untrained)", "Etapa 5\n(Baseline)", "Etapa 6\n(Optimized)"]
    accuracies = [17.77, 98.13, 98.27] 
    plt.figure(figsize=(10, 6))
    
    # Plotare linie
    plt.plot(stages, accuracies, marker='o', markersize=12, linestyle='-', linewidth=3, color='#3498db') 
    
    # Zone de context
    plt.axhspan(0, 20, facecolor='#e74c3c', alpha=0.2, label='Zona Random (<20%)')
    plt.axhspan(70, 100, facecolor='#2ecc71', alpha=0.2, label='Zona Target (>70%)')

    # Etichete valori
    for i, acc in enumerate(accuracies):
        offset = 5 if i > 0 else -8
        va = 'bottom' if i > 0 else 'top'
        color = 'white' # Text alb pe fundal negru
        
        plt.text(i, acc + offset, f"{acc:.2f}%", 
                 ha='center', va=va, fontsize=12, fontweight='bold', color=color,
                 bbox=dict(facecolor='#2c3e50', edgecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

    plt.title("Evoluția Acurateței Modelului (Etapele 4 → 6)", fontsize=16, fontweight='bold', color='white', pad=20)
    plt.ylabel("Acuratețe (%)", fontsize=12, color='white')
    plt.ylim(0, 110)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Styling axa X
    plt.tick_params(axis='x', colors='white', labelsize=11)
    plt.tick_params(axis='y', colors='white')
    
    # Chenar alb
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('white')

    save_path = os.path.join(docs_results_dir, "metrics_evolution.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Salvat: {save_path}")


def draw_modern_cell(ax, row, true_lbl, pred_lbl, conf, is_correct):
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Culori Status
    status_color = '#00ff7f' if is_correct else '#ff4757' 
    bg_bar_color = '#2f3640'
    
    # HEADER
    ax.text(0.05, 0.92, f"Real: {true_lbl}", color='white', fontsize=10, fontweight='bold', transform=ax.transAxes)
    ax.text(0.05, 0.84, f"AI: {pred_lbl}", color=status_color, fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.95, 0.84, f"{conf*100:.0f}%", color=status_color, fontsize=11, ha='right', transform=ax.transAxes)
    
    # Separator
    rect = patches.Rectangle((0.02, 0.78), 0.96, 0.02, linewidth=0, facecolor=status_color, transform=ax.transAxes, alpha=0.7)
    ax.add_patch(rect)

    # BARE PROGRES (RPM, SPEED)
    norm_rpm = np.clip(row['rpm'] / 7000, 0, 1)
    norm_speed = np.clip(row['speed'] / 180, 0, 1)
    
    # RPM
    ax.text(0.05, 0.68, f"RPM: {row['rpm']:.0f}", color='white', fontsize=9, transform=ax.transAxes)
    rect_bg = patches.Rectangle((0.05, 0.60), 0.9, 0.06, facecolor=bg_bar_color, transform=ax.transAxes)
    ax.add_patch(rect_bg)
    rpm_color = plt.cm.RdYlGn_r(norm_rpm*0.8 + 0.1) 
    rect_fg = patches.Rectangle((0.05, 0.60), 0.9 * norm_rpm, 0.06, facecolor=rpm_color, transform=ax.transAxes)
    ax.add_patch(rect_fg)

    # Speed
    ax.text(0.05, 0.50, f"Speed: {row['speed']:.0f} km/h", color='white', fontsize=9, transform=ax.transAxes)
    rect_bg = patches.Rectangle((0.05, 0.42), 0.9, 0.06, facecolor=bg_bar_color, transform=ax.transAxes)
    ax.add_patch(rect_bg)
    rect_fg = patches.Rectangle((0.05, 0.42), 0.9 * norm_speed, 0.06, facecolor='#3498db', transform=ax.transAxes)
    ax.add_patch(rect_fg)

    # PEDALE TEXT
    thr_val = row['throttle']
    brk_val = row['brake']
    ax.text(0.05, 0.28, f"Throttle: {thr_val:.0f}%", color='#2ecc71', fontsize=10, transform=ax.transAxes, fontweight='bold')
    ax.text(0.55, 0.28, f"Brake: {brk_val:.0f}%", color='#e74c3c', fontsize=10, transform=ax.transAxes, fontweight='bold')

    # FOOTER
    footer_text = f"Gear: {row['gear']:.0f} | Accel: {row['acceleration']:.2f} | Tilt: {row['tilt']:.1f}°"
    ax.text(0.5, 0.05, footer_text, color='#bdc3c7', fontsize=8, ha='center', transform=ax.transAxes)
    
    # Chenar
    for spine in ax.spines.values():
        spine.set_edgecolor(status_color)
        spine.set_linewidth(1.5)
        spine.set_alpha(0.5)

#example_predictions.png
def generate_dashboard_grid():
    print("2. Generare 'example_predictions.png' (Dashboard Style)...")
    
    # Cai fisiere necesare
    test_data_path = os.path.join(base_dir, "data/test/test.csv")
    model_path = os.path.join(base_dir, "models/optimized_model.h5")
    if not os.path.exists(model_path):
         model_path = os.path.join(base_dir, "models/trained_model.h5")
    scaler_path = os.path.join(base_dir, "config/preprocessing_params.pkl") 
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print("⚠️ Nu gasesc modelul sau scaler-ul. Asigura-te ca ai rulat train.py!")
        return

    # Incarcare
    df_test = pd.read_csv(test_data_path)
    X_test = df_test[['rpm', 'speed', 'acceleration', 'throttle', 'brake', 'tilt', 'gear']].values
    y_test = df_test['style_label'].values
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    model = load_model(model_path)

    # Predictii
    probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # Selectare (Random)
    correct_indices = np.where(y_test == y_pred)[0]
    incorrect_indices = np.where(y_test != y_pred)[0]

    np.random.seed(101)
    num_wrong = min(len(incorrect_indices), 3)
    num_correct = 9 - num_wrong

    selected_correct = np.random.choice(correct_indices, num_correct, replace=False)
    selected_incorrect = np.random.choice(incorrect_indices, num_wrong, replace=False)
    selected_indices = np.concatenate([selected_correct, selected_incorrect])
    np.random.shuffle(selected_indices)

    class_names = ['Eco', 'Normal', 'Sport']

    # Desenare
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle('Analiză Predicții Model Final (Dashboard View)', fontsize=18, fontweight='bold', color='white', y=0.98)
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92, bottom=0.05, left=0.05, right=0.95)

    axes = axes.flatten()
    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        row = df_test.iloc[idx]
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        confidence = np.max(probs[idx])
        is_correct = (true_label == pred_label)
        
        draw_modern_cell(ax, row, true_label, pred_label, confidence, is_correct)

    save_path = os.path.join(docs_results_dir, "example_predictions.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ Salvat: {save_path}")

if __name__ == "__main__":
    generate_metrics_evolution()
    generate_dashboard_grid()
    print("\n🎉 Toate vizualizarile finale au fost generate!")