# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Boata Andrei-Darius  
**Link Repository GitHub:** https://github.com/Andreid2511/Proiect_RN.git  
**Data predării:** 19.12.2025

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN**.

**Obiectiv principal:** Antrenarea unui model Deep Learning (Keras) capabil să identifice stilul de condus în timp real pentru a **optimiza consumul de combustibil**. 
Sistemul (SIA) ajustează strategia cutiei de viteze (Shift Points) pentru a preveni risipa de energie în regim urban (Stop & Go) și pentru a proteja motorul la urcarea pantelor.

**Pornire obligatorie:** Arhitectura completă din Etapa 4:
- State Machine definit (cu stări specifice: *Forced Eco, Hill Descent*).
- Cele 3 module funcționale (Data Logging, RN, UI).
- Dataset generat 100% original prin simulare fizică (180.000 eșantioane).

---

## PREREQUISITE – Verificare Etapa 4 (REALIZAT)

- [x] **State Machine** documentat în `docs/state_machine.png` (Logică axată pe eficiență).
- [x] **Contribuție 100% date originale** în `data/`.
- [x] **Modul 1 (Data Logging)** funcțional - generează CSV-uri compatibile.
- [x] **Modul 2 (RN)** pipeline unificat de antrenare și export (`train_model.py`).
- [x] **Modul 3 (UI/Web Service)** funcțional, optimizat pentru latență mică.

---

## 1. Configurația Modelului și Hiperparametrii

Am ales o arhitectură **Deep Feed-Forward (DNN)**, optimizată pentru a corela cei 7 parametri fizici (RPM, Speed, Acceleration, Throttle, Brake, Tilt, Gear) cu intenția șoferului.

### Tabel Hiperparametri Finali

| Parametru | Valoare | Justificare |
| :--- | :--- | :--- |
| **Framework** | TensorFlow / Keras | Standard industrial, permite salvarea modelului portabil `.h5`. |
| **Arhitectură** | `Dense(32) -> Dense(32) -> Dense(16) -> Dense(3)` | 3 straturi ascunse sunt necesare pentru a modela relația non-liniară dintre Pantă (Tilt) și Pedală (Throttle). O rețea mai simplă ar confunda urcarea unui deal (pedală mare) cu stilul Agresiv. |
| **Funcție Activare** | `ReLU` (hidden), `softmax` (output) | `ReLU` accelerează antrenarea și previne "vanishing gradient". `softmax` este obligatoriu pentru clasificarea probabilistică (Eco/Normal/Sport). |
| **Optimizer** | `Adam (lr=0.001)` | Cel mai stabil optimizator pentru date cu zgomot inerent (simulat prin variații gaussiene). |
| **Batch Size** | `32` | Oferă un gradient stabil și previne blocarea în minime locale. |
| **Epoci** | `150` (cu Early Stopping) | Antrenarea se oprește automat dacă eroarea pe validare nu scade timp de 15 epoci (patience), prevenind Overfitting-ul. |

---

## 2. Rezultate și Performanță

Antrenarea a rulat timp de 150 de epoci, modelul final având o performanță excelentă pe setul de testare (date nevăzute).

### A. Grafice de Performanță (Cerință Nivel 2)

**1. Curba de Învățare (Loss vs. Val Loss):**
Graficul `docs/results/learning_curves_final.png` arată evoluția erorii:
* **Convergență:** Atât *Train Loss* cât și *Validation Loss* scad rapid în primele 20 de epoci.
* **Lipsa Overfitting-ului:** Linia de validare (portocalie) rămâne apropiată de cea de antrenare, demonstrând generalizarea corectă.

**2. Matricea de Confuzie:**
Graficul `docs/confusion_matrix_optimized.png` arată precizia pe clase:
* **Precision Agresiv:** >98%. Aceasta este metrica cheie pentru detectarea situațiilor care necesită putere maximă (depășiri).

### B. Metrici Finale (Test Set)

Conform fișierului generat `results/final_metrics.json`:

| Metrică | Valoare Obținută | Obiectiv Îndeplinit |
| :--- | :--- | :--- |
| **Acuratețe** | **~98.28%** | ✅ (> 65%) |
| **F1-Score** | **~0.98** | ✅ (> 0.60) |
| **Recall (Eco)** | **>0.98** | ✅ (Excelent) |

---

## 3. Analiză Erori în Contextul Eficienței (OBLIGATORIU)

Performanța modelului este analizată din perspectiva **reducerii consumului de combustibil**:

### 1. Pe ce clase greșește modelul?
Confuziile minore (sub 2%) apar între clasele **Eco** și **Normal**.
*Cauză:* În regim de croazieră (viteză constantă pe autostradă), amprenta senzorială a unui șofer Eco este matematic identică cu a unui șofer Normal (accelerație ~0, viteză constantă). Diferențierea se poate face doar contextual.

### 2. Ce implicații are pentru aplicație (Consum)?
* **False Positive (Normal clasificat ca Agresiv):** Ar fi o eroare costisitoare, deoarece ar tura motorul inutil. Modelul nostru are o precizie excelentă pe Agresiv, deci această eroare este minimizată.
* **False Negative (Agresiv clasificat ca Normal):** Ar duce la o întârziere în retrogradarea vitezei la depășire (Kickdown).

### 3. Ce măsuri corective propuneți?
1. **Integrare Logică "Forced Eco":** În UI (`main.py`), dacă modelul detectează "Agresiv" la viteze de oraș (<65 km/h), sistemul ignoră parțial dorința șoferului de putere și schimbă vitezele devreme (2500 RPM) pentru a salva combustibil.
2. **Smoothing:** Pentru a evita schimbarea haotică a strategiei între Eco și Normal, am implementat un buffer de 5 cadre în inferență (istoric predicții).
3. **Override pentru Pantă:** Dacă senzorul de înclinație detectează o coborâre abruptă, sistemul forțează modul Eco indiferent de turația mare a motorului (frână de motor).

---

## 4. Fișiere Generate și Structură

Repository-ul este organizat conform cerințelor, cu scripturile de antrenare consolidate pentru consistență:

```text
proiect-rn-[prenume-nume]/
├── README.md                           # Overview
├── etapa3_analiza_date.md              # Documentație Etapa 3
├── etapa4_arhitectura_sia.md           # Documentație Etapa 4
├── etapa5_antrenare_model.md           # ← ACEST FIȘIER
│
├── docs/
│   ├── state_machine.png               # Diagrama Logicii (Coasting/Forced Eco)
│   ├── confusion_matrix_optimized.png  # Performanță pe clase (Generat)
│   ├── results/
│   │   └── learning_curves_final.png   # Grafic Learning Curve (Generat)
│   └── screenshots/
│       ├── inference_real.png          # Dovada UI funcțional
│       └── ui_demo.png                 # Actualizat
│
├── data/                               # Dataset
│   ├── train/ ...                      # CSV-uri formatate pentru AI
│   ├── validation/ ...
│   └── test/ ...
│
├── src/
│   ├── data_acquisition/
│   │   └── generate_data.py            # Generator Fizic (180k samples)
│   ├── neural_network/
│   │   ├── train_model.py              # Pipeline Unificat: Config -> Train -> Evaluate
│   │   └── optimize.py                 # Script căutare hiperparametri
│   └── app/
│       └── main.py                     # Dashboard UI (Optimizat Low-Latency)
│
├── models/
│   ├── untrained_model.h5              # Model inițial (neantrenat)
│   ├── trained_model.h5                # Model antrenat (Etapa 5)
│   ├── optimized_model.h5              # Model final optimizat (Etapa 6)
│   └── final_model.onnx                # (Bonus: Export interoperabil)
│
├── results/                            # Rezultate Antrenare (Dovezi)
│   ├── training_history.csv            # Log detaliat epoci
│   ├── final_metrics.json              # Scoruri finale
│   └── hyperparameters.yaml            # Configurație
│
├── config/
│   ├── scaler.pkl                      # Obiect standardizare
│   └── preprocessing_params.pkl        # (Alias pentru scaler)
│
└── requirements.txt