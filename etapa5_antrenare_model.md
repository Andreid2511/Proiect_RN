# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Boata Andrei-Darius  
**Link Repository GitHub:** [https://github.com/Andreid2511/Proiect_RN.git]  
**Data predării:** 19.12.2025

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN**.

**Obiectiv principal:** Antrenarea unui model Deep Learning (Keras) capabil să identifice stilul de condus în timp real pentru a **optimiza consumul de combustibil**. 
Sistemul (SIA) ajustează strategia cutiei de viteze (Shift Points) pentru a preveni risipa de energie în regim urban (Stop & Go) și pentru a proteja motorul la urcarea pantelor.

**Pornire obligatorie:** Arhitectura completă din Etapa 4:
- State Machine definit (cu stări specifice: *Forced Eco, Hill Descent*).
- Cele 3 module funcționale (Data Logging, RN, UI).
- Dataset generat 100% original prin simulare fizică.

---

## PREREQUISITE – Verificare Etapa 4 (REALIZAT)

- [x] **State Machine** documentat în `docs/state_machine.png` (Logică axată pe eficiență).
- [x] **Contribuție 100% date originale** în `data/generated/` (Simulare fizică VW Polo).
- [x] **Modul 1 (Data Logging)** funcțional - generează raport detaliat cu unități de măsură.
- [x] **Modul 2 (RN)** pipeline unificat de antrenare și export.
- [x] **Modul 3 (UI/Web Service)** funcțional, optimizat pentru latență mică.

---

## 1. Configurația Modelului și Hiperparametrii

Am ales o arhitectură **Deep Feed-Forward (DNN)**, optimizată pentru a corela cei 7 parametri fizici (inclusiv Panta) cu intenția șoferului.

### Tabel Hiperparametri Finali

| Parametru | Valoare | Justificare |
| :--- | :--- | :--- |
| **Framework** | TensorFlow / Keras | Standard industrial, permite salvarea modelului portabil `.keras`. |
| **Arhitectură** | `Dense(32) -> Dense(32) -> Dense(16) -> Dense(3)` | 3 straturi ascunse sunt necesare pentru a modela relația non-liniară dintre Pantă (Tilt) și Pedală (Throttle). O rețea mai simplă ar confunda urcarea unui deal (pedală mare) cu stilul Agresiv. |
| **Funcție Activare** | `tanh` (hidden), `softmax` (output) | `tanh` centrează datele în 0, accelerând convergența. `softmax` este obligatoriu pentru clasificarea probabilistică (Eco/Normal/Sport). |
| **Optimizer** | `Adam (lr=0.001)` | Cel mai stabil optimizator pentru date cu zgomot inerent (simulat prin variații gaussiene). |
| **Batch Size** | `32` | Oferă un gradient stabil și previne blocarea în minime locale. |
| **Epoci** | `150` (cu Early Stopping) | Antrenarea se oprește automat dacă eroarea pe validare nu scade, prevenind Overfitting-ul. |

---

## 2. Rezultate și Performanță

Antrenarea a rulat timp de 150 de epoci, modelul final având o performanță excelentă pe setul de testare (date nevăzute).

### A. Grafice de Performanță (Cerință Nivel 2)

**1. Curba de Învățare (Loss vs. Val Loss):**
Graficul `docs/loss_curve.png` arată evoluția erorii:
* **Convergență:** Atât *Train Loss* cât și *Validation Loss* scad constant și se stabilizează în jurul valorii de 0.08.
* **Lipsa Overfitting-ului:** Linia de validare (portocalie) rămâne lipită de cea de antrenare, ceea ce demonstrează că modelul a învățat "fizica" din spatele datelor, nu a memorat exemplele.

**2. Matricea de Confuzie:**
Graficul `docs/confusion_matrix.png` arată precizia pe clase:
* **Precision Agresiv:** 99%. Aceasta este metrica cheie pentru reducerea consumului (vezi Analiza Erori).

### B. Metrici Finale (Test Set)

Conform fișierului generat `results/test_metrics.json`:

| Metrică | Valoare Obținută | Obiectiv Îndeplinit |
| :--- | :--- | :--- |
| **Acuratețe** | **97%** | ✅ (> 65%) |
| **F1-Score** | **0.97** | ✅ (> 0.60) |
| **Recall (Eco)** | **0.97** | ✅ (Excelent) |

---

## 3. Analiză Erori în Contextul Eficienței (OBLIGATORIU)

Performanța modelului este analizată din perspectiva **reducerii consumului de combustibil**:

### 1. Pe ce clase greșește modelul?
Confuziile minore (aprox 3%) apar între clasele **Eco** și **Normal**.
*Cauză:* În regim de croazieră (viteză constantă pe autostradă), amprenta senzorială a unui șofer Eco este matematic identică cu a unui șofer Normal. Diferențierea se poate face doar la schimbarea accelerației.

### 2. Ce implicații are pentru aplicație (Consum)?
* **False Positive (Normal clasificat ca Agresiv):** Ar fi o eroare costisitoare, deoarece ar tura motorul inutil. Modelul nostru are o precizie de 99% pe Agresiv, deci această eroare este aproape eliminată.
* **False Negative (Agresiv clasificat ca Normal):** Ar duce la ratarea oportunității de a activa "Forced Eco". Cu un Recall de 99%, sistemul prinde aproape toate momentele de risipă.

### 3. Ce măsuri corective propuneți?
1.  **Integrare Logică "Forced Eco":** În UI (`app_gui.py`), dacă modelul detectează "Agresiv" la viteze de oraș (<60 km/h), sistemul ignoră dorința șoferului de putere și schimbă vitezele devreme (2500 RPM) pentru a salva combustibil.
2.  **Smoothing:** Pentru a evita schimbarea haotică a strategiei între Eco și Normal, am implementat un buffer de 5 cadre în inferență.
3.  **Coasting Override:** Dacă pedalele nu sunt apăsate, sistemul intră automat în mod "Coasting" (consum 0), indiferent de predicția rețelei neuronale.

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
│   ├── Diagrama_flux.drawio.png        # Diagrama Fluxului aplicatiei
│   ├── loss_curve.png                  # Grafic Learning Curve (Generat)
│   ├── confusion_matrix.png            # Performanță pe clase (Generat)
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
│   │   └── generate_data.py            # Generator Fizic V8.0
│   ├── neural_network/
│   │   └── train_model.py              # Pipeline Unificat: Config -> Train -> Evaluate -> Export
│   └── app/
│       └── app_gui.py                  # Dashboard UI (Optimizat Low-Latency)
│
├── models/
│   ├── untrained_model.keras           # Model inițial (neantrenat)
│   ├── trained_model.keras             # Model final (Loss 0.08)
│   └── final_model.onnx                # (Bonus: Încercare export interoperabil)
│
├── results/                            # Rezultate Antrenare (Dovezi)
│   ├── training_history.csv            # Log detaliat epoci
│   ├── test_metrics.json               # Scoruri finale
│   └── hyperparameters.yaml            # Configurație
│
├── config/
│   └── scaler.pkl                      # Obiect standardizare
│
├── RAPORT_DATE_PREZENTARE.csv          # Exemplu date cu unități de măsură
└── requirements.txt