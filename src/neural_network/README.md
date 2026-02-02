### 📂 README pentru `src/neural_network/` (Rețeaua)
**Fișier:** `src/neural_network/README.md`

```markdown
# 🧠 Modulul 2: Rețea Neuronală (Antrenare & Optimizare)

Aici se află "creierul" sistemului SIA. Acest modul definește, antrenează și evaluează modelul de Deep Learning.

## 🏗️ Arhitectura Modelului (DNN)
Modelul este un **Multi-Layer Perceptron (MLP)** adânc, optimizat pentru clasificarea datelor tabulare rapide.

* **Input Layer:** 7 neuroni (corespunzător celor 7 senzori).
* **Hidden Layers:**
    * Dense (32 neuroni, activare `ReLU`)
    * Dense (32 neuroni, activare `ReLU`)
    * Dense (16 neuroni, activare `ReLU`)
* **Output Layer:** 3 neuroni (activare `Softmax`) -> Probabilități pentru Eco/Normal/Sport.

## 🛠️ Scripturi
1.  **`train_model.py`**:
    * Încarcă datele din `data/`.
    * Normalizează datele folosind `StandardScaler`.
    * Antrenează modelul folosind optimizatorul **Adam**.
    * Salvează modelul antrenat în `models/optimized_model.h5`.
    * Salvează metricile și scaler-ul.

2.  **`optimize.py`**:
    * Script utilizat în Etapa 6 pentru a testa diferiți hiperparametri (Learning Rate, Batch Size, Arhitectură) și a găsi configurația optimă.

## 📈 Performanță
* **Acuratețe Finală:** ~98.28%
* **Loss:** ~0.08
* **Latență:** ~0.029 ms / inferență

## ⚙️ Execuție Antrenament
```bash
python src/neural_network/train_model.py