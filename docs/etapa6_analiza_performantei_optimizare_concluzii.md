# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Boata Andrei-Darius  
**Link Repository GitHub:** [https://github.com/Andreid2511/Proiect_RN.git]  
**Data predării:** 31.01.2025

---
## Scopul Etapei 6

Această etapă corespunde punctelor **7. Analiza performanței și optimizarea parametrilor**, **8. Analiza și agregarea rezultatelor** și **9. Formularea concluziilor finale** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Maturizarea completă a Sistemului cu Inteligență Artificială (SIA) prin optimizarea modelului RN, analiza detaliată a performanței și integrarea îmbunătățirilor în aplicația software completă.

**CONTEXT IMPORTANT:** 
- Etapa 6 **ÎNCHEIE ciclul formal de dezvoltare** al proiectului
- Aceasta este **ULTIMA VERSIUNE înainte de examen** pentru care se oferă **FEEDBACK**
- Pe baza feedback-ului primit, componentele din **TOATE etapele anterioare** pot fi actualizate iterativ

**Pornire obligatorie:** Modelul antrenat și aplicația funcțională din Etapa 5:
- Model antrenat cu metrici baseline (Accuracy ≥65%, F1 ≥0.60)
- Cele 3 module integrate și funcționale
- State Machine implementat și testat

---


## PREREQUISITE – Verificare Etapa 5 (OBLIGATORIU)

**Înainte de a începe Etapa 6, verificați că aveți din Etapa 5:**

- [x] **Model antrenat** salvat în `models/trained_model.h5` (sau `.pt`, `.lvmodel`)
- [x] **Metrici baseline** raportate: Accuracy ≥65%, F1-score ≥0.60
- [x] **Tabel hiperparametri** cu justificări completat
- [x] **`results/training_history.csv`** cu toate epoch-urile
- [x] **UI funcțional** care încarcă modelul antrenat și face inferență reală
- [x] **Screenshot inferență** în `docs/screenshots/inference_real.png`
- [x] **State Machine** implementat conform definiției din Etapa 4

**Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 5 înainte de a continua.**

---

## 1. Actualizarea Aplicației Software în Etapa 6 

**CERINȚĂ CENTRALĂ:** Modificările aduse aplicației software ca urmare a optimizării modelului și a analizei erorilor.

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| **Model încărcat** | `trained_model.h5` (Baseline Tanh) | `optimized_model.h5` (ReLU) | Eficiență computațională superioară și robustețe. |
| **Logic Override** | Niciunul | **Hill Descent Override** | Corecție pentru confuzia AI la coborârea pantelor (RPM mare, pedala 0). |
| **Arhitectură Fizică** | Parametri standard | Parametri calibrați | Putere (0.14) și Gravitație (0.018) ajustate pentru realism. |
| **Management Erori** | Implicit | `try-except` pe inferență | Prevenirea crash-ului aplicației dacă modelul nu încarcă datele. |
| **Statistici** | N/A | Cumulative pe sesiune | Feedback vizual pentru stilul dominant de condus. |

### Modificări concrete aduse în Etapa 6:

1. **Model înlocuit:** `models/trained_model.h5` → `models/optimized_model.h5`
   - Am trecut de la activare **Tanh** la **ReLU**.
   - Deși acuratețea este similară, ReLU este mai eficient pentru inferența în timp real (calcule liniare vs exponențiale).

2. **Implementare "Rule-Based Override" în `main.py`:**
   - **Problemă identificată:** La coborârea pantelor abrupte (Hill Descent), motorul are RPM mare (frână de motor), iar AI-ul clasifica greșit acest comportament ca "Sport" sau "Normal", deși consumul este 0.
   - **Soluție:** Am adăugat o regulă logică care suprascrie predicția AI:
     ```python
     if self.tilt < -2 and self.throttle < 5:
         prediction = 0 # FORCED ECO
     ```

3. **UI îmbunătățit:**
   - Adăugarea statisticilor de sesiune ("Dominant Style").
   - Integrarea logicii de cutie de viteze direct în bucla de fizică pentru o latență minimă.
   - Adăugarea unei căsuțe cu indicații/explicații
4. **Logica noua adaugata:**
   - Kickdown
      ```python
     kickdown_enabled = False # Variabila kickdown

     if self.throttle > 90: # Activare kickdown
            kickdown_enabled = True
        else:
            kickdown_enabled = False
     ```

---

## 2. Analiza Detaliată a Performanței

### 2.1 Confusion Matrix și Interpretare

**Locație:** `docs/confusion_matrix_optimized.png`

**Analiză obligatorie (completați):**

### Interpretare Confusion Matrix:

**Clasa cu cea mai bună performanță:** **Sport**
- **Precision:** 99.48%
- **Recall:** 99.10%
- **Explicație:** Clasa Sport este caracterizată de valori extreme (RPM mare, accelerație bruscă), ceea ce o face foarte distinctă în spațiul vectorial al datelor față de celelalte clase.

**Clasa cu cea mai slabă performanță:** **Normal**
- **Precision:** 97.27%
- **Recall:** 97.54%
- **Explicație:** Clasa "Normal" reprezintă zona de mijloc. Se suprapune parțial cu "Eco" (la viteze de croazieră) și cu "Sport" (în rampe ușoare), generând cele mai multe confuzii marginale.

**Confuzii principale:**
1. **Clasa [Eco] confundată cu clasa [Normal]**
   - **Cauză:** Suprapunere de features la viteze constante și accelerații mici. Diferența dintre un condus Eco agresiv și unul Normal relaxat este subtilă.
   - **Impact industrial:** Redus. Transmisia ar putea menține o treaptă superioară puțin mai mult timp, fără impact asupra siguranței.
   
2. **Clasa [Normal] confundată cu clasa [Sport]**
   - **Cauză:** În situații de sarcină (pante, greutate), motorul se turează pentru a menține viteza, ceea ce modelul interpretează uneori ca stil agresiv (Sport).
   - **Impact industrial:** Creșterea ușoară a consumului de combustibil prin menținerea turației ridicate nejustificat.

### 2.2 Analiza Detaliată a 5 Exemple Greșite

Am analizat cazurile unde modelul a greșit pentru a înțelege limitele sistemului (error_analysis.json).

| **Index** | **True Label** | **Predicted** | **Confidence** | **Input Features (Rezumat)** | **Cauză probabilă & Soluție** |
|-----------|----------------|---------------|----------------|------------------------------|-------------------------------|
| #10437 | **Normal** | **Sport** | 100% | RPM: 3798, Throttle: 84%, Speed: 58 | **Cauză:** Turație și pedală mari. Modelul asociază agresivitatea cu Sport. Eticheta "Normal" este discutabilă aici. |
| #12187 | **Sport** | **Normal** | 99.9% | RPM: 4134, Throttle: 85%, Speed: 52 | **Cauză:** Deși parametrii sunt de Sport, accelerația era mică (0.04). Modelul a crezut că e un regim de croazieră turată. |
| #704 | **Eco** | **Normal** | 99.8% | RPM: 2016, **Tilt: -14.5**, Throttle: 4.5% | **Cauză:** Coborâre abruptă (Hill Descent). Modelul a văzut RPM 2000 (frână motor) și a crezut că e Normal. **Soluție:** Am implementat Override-ul logic în `main.py`. |
| #3859 | **Normal** | **Eco** | 99.7% | Speed: 15, Throttle: 3% | **Cauză:** Viteza și pedala foarte mici (rulare la pas). E greu de distins Normal de Eco aici fără un context temporal mai larg. |
| #12022 | **Sport** | **Normal** | 99.7% | RPM: 3876, Tilt: 11.5 (Urcare) | **Cauză:** Urcare în rampă. Motorul e turat din cauza sarcinii, nu a stilului sportiv. Modelul a confundat sarcina cu stilul normal. |

**Concluzie Analiză Erori:**
Majoritatea erorilor provin din situații ambigue (pante abrupte sau zone de graniță între stiluri). Implementarea regulilor hard-codate (Override) pentru pante a rezolvat erorile critice de tipul #704.

---

## 3. Optimizarea Parametrilor și Experimentare

### 3.1 Strategia de Optimizare

Pentru a justifica alegerea arhitecturii finale, am rulat un script de optimizare care a comparat 5 arhitecturi distincte.

### Tabel Rezultate Experimentale (Generat în `results/optimization_experiments.csv`)

| Experiment ID | Arhitectură | Accuracy | F1 Score | Latency (ms) | Time (s) | Observații |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Small (Rapid) | 0.8786 | 0.8787 | 0.0356 | **5.14** | **Sub-fit.** Modelul este prea simplu, pierzând ~7% precizie față de baseline. |
| **EXP-02** | Baseline (Tanh) | 0.9270 | 0.9271 | 0.0358 | 5.78 | Performanță decentă, dar funcția Tanh limitează convergența rapidă. |
| **EXP-03** | **Baseline (ReLU)** | **0.9353** | **0.9350** | **0.0346** | 5.53 | **ALES FINAL.** Balanța perfectă între precizie și eficiență computațională. |
| **EXP-04** | Pyramid (Deep) | 0.9459 | 0.9457 | 0.0385 | 5.50 | Cea mai mare acuratețe brută, dar latența este ușor mai mare și riscă overfitting. |
| **EXP-05** | Large (Dropout) | 0.9377 | 0.9376 | 0.0343 | 5.92 | Performanță similară, dar complexitate inutilă. |


### 3.2 Grafice Comparative

Generați și salvați în `docs/optimization/`:
- `accuracy_comparison.png` - Accuracy per experiment
- `f1_comparison.png` - F1-score per experiment
- `learning_curves_best.png` - Loss și Accuracy pentru modelul final

### 3.3 Raport Final Optimizare

**Configurație finală aleasă (Optimized ReLU - bazată pe EXP-03):**
- **Arhitectură:** `Input(7) -> Dense(32, ReLU) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Output(3, Softmax)`

**Analiză Comparativă (Baseline Tanh vs. Optimized ReLU):**
În urma experimentelor, am observat că ambele modele converg către o acuratețe ridicată (~98% la antrenare completă), însă am selectat **ReLU** pentru etapa de producție din considerente de **eficiență**:

1. **Performanță:** Modelul ReLU a atins **98.28% Accuracy** și **0.98 F1-Score**, menținând standardul ridicat impus de Baseline.
2. **Viteză de Învățare:** Graficele de învățare (`learning_curves_final.png`) demonstrează că ReLU reduce eroarea (Loss) mai agresiv în primele 20 de epoci comparativ cu Tanh.
3. **Eficiență Hardware:** Deoarece aplicația trebuie să ruleze în timp real, funcția ReLU (fiind liniară pe porțiuni) este mai puțin costisitoare pentru procesor decât funcția Tanh (exponențială), reducând încărcarea CPU fără a sacrifica precizia.

**Concluzie:** Optimizarea nu a vizat doar creșterea procentelor (deja foarte mari), ci **robustizarea** modelului și reducerea complexității matematice pentru inferență.

---

## 4. Agregarea Rezultatelor și Vizualizări

### 4.1 Tabel Sumar Rezultate Finale

Datele de mai jos compară evoluția sistemului de la stadiul de model neantrenat (Etapa 4) la baseline (Etapa 5) și varianta finală optimizată (Etapa 6).

| **Metrică** | **Etapa 4** (Untrained) | **Etapa 5** (Baseline) | **Etapa 6** (Optimizat) | **Target Industrial** | **Status** |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Accuracy** | 17.77% | 98.13% | **98.28%** | ≥ 95% | ✅Depășit |
| **F1-score (macro)** | 0.1718 | 0.9812 | **0.9827** | ≥ 0.95 | ✅Depășit |
| **Precision (macro)** | 0.4190 | 0.9812 | **0.9827** | ≥ 0.95 | ✅Depășit |
| **Recall (macro)** | 0.3602 | 0.9812 | **0.9827** | ≥ 0.95 | ✅Depășit |
| **False Negative Rate** | 63.98% | 1.88% | **1.73%** | ≤ 2.0% | ✅Depășit |
| **Latență inferență** | 0.0292 ms | 0.0275 ms | **0.0287 ms** | ≤ 1.0 ms | ✅Depășit |
| **Throughput** | ~34k inf/s | ~36k inf/s | **~34.8k inf/s** | ≥ 1000 inf/s | ✅Depășit |

**Throughput calculat teoretic pe baza latenței (1000 / latency_ms).**

**Observații privind evoluția:**
1. **Salt major de performanță:** Trecerea de la modelul neantrenat (17.77%) la cel antrenat a adus un câștig masiv, confirmând că datele conțin tipare clare pe care rețeaua le poate învăța.
2. **Rafinare în Etapa 6:** Deși Baseline-ul (Etapa 5) era deja performant, optimizarea din Etapa 6 (activare ReLU + ajustări fine) a reușit să reducă rata de **False Negatives** de la 1.88% la **1.73%**, ceea ce înseamnă o siguranță sporită în decizii.
3. **Latență neglijabilă:** Latența de ~0.029ms este de sute de ori mai mică decât bugetul de timp pentru 30 FPS (33ms), permițând rularea modelului pe procesoare slabe fără a afecta fluiditatea aplicației.

### 4.2 Vizualizări Obligatorii

Salvați în `docs/results/`:

- [x] `confusion_matrix_optimized.png` - Confusion matrix model final
- [x] `learning_curves_final.png` - Loss și accuracy vs. epochs
- [x] `metrics_evolution.png` - Evoluție metrici Etapa 4 → 5 → 6
- [x] `example_predictions.png` - Grid cu 9+ exemple (correct + greșite)

---

## 5. Concluzii Finale și Lecții Învățate

**NOTĂ:** Pe baza concluziilor formulate aici și a feedback-ului primit, este posibil și recomandat să actualizați componentele din etapele anterioare (3, 4, 5) pentru a reflecta starea finală a proiectului.

### 5.1 Evaluarea Performanței Finale

### Evaluare sintetică a proiectului

**Obiective atinse:**
- [x] Model RN funcțional cu accuracy [98]% pe test set
- [x] Integrare completă în aplicație software (3 module)
- [x] State Machine implementat și actualizat
- [x] Pipeline end-to-end testat și documentat
- [x] UI demonstrativ cu inferență reală
- [x] Documentație completă pe toate etapele

Proiectul a atins obiectivele propuse. Sistemul SIA este capabil să clasifice stilul de condus cu o precizie de peste 98% și să adapteze transmisia automată în timp real (30 FPS).

**Puncte Forte:**
1. **Latență extrem de mică (0.029ms):** Modelul optimizat ReLU este extrem de rapid, permițând rularea pe hardware modest.
2. **Sistem Hibrid:** Combinarea Rețelei Neuronale cu reguli logice (Rule-Based Override pentru pante) a eliminat erorile de "bun simț" pe care AI-ul pur le făcea.
3. **Interfață Robustă:** UI-ul reflectă clar deciziile sistemului.


### 5.2 Limitări Identificate

### Limitări tehnice ale sistemului

1. **Dependența de Date:** Modelul este antrenat pe date sintetice/simulate. Comportamentul în lumea reală ar putea varia din cauza zgomotului senzorilor (vibrații).
2. **Confuzii la Limită:** Tranzițiile foarte fine între "Eco" și "Normal" sunt uneori etichetate subiectiv, ceea ce se reflectă în o parte dintre cele 1.7% erori.


### 5.3 Direcții de Cercetare și Dezvoltare

**Direcții viitoare de dezvoltare:**

**Pe termen scurt (1-3 luni):**
1. **Colectare date reale:** Instalarea unui logger OBD-II pe un vehicul real pentru a colecta date de telemetrie și validarea modelului în condiții de trafic real.
2. **Rafinare State Machine:** Implementarea unei logici de "histerezis" mai avansate pentru a preveni schimbările prea frecvente de viteze (gear hunting) în zonele de graniță.
3. **Optimizare ONNX:** Exportarea modelului în format ONNX pentru a permite integrarea pe microcontrollere auto (ex: Infineon Aurix).

**Pe termen mediu (3-6 luni):**
1. **Integrare cu navigația:** Utilizarea datelor GPS (pante viitoare, curbe) ca input suplimentar pentru a anticipa schimbarea vitezelor (Predictive Powertrain Control).
2. **Personalizare:** Implementarea unui mod de "învățare continuă" care să adapteze modelul la stilul specific al unui șofer individual.

### 5.4 Lecții Învățate

1. **Complexitatea nu înseamnă Performanță:** Arhitectura `Pyramid Deep` (mai complexă) a avut acuratețe mai mare, dar am ales `Baseline ReLU` pentru simplitate și viteză, diferența de precizie fiind neglijabilă.
2. **Impactul setului de date:** Calitatea și echilibrarea datelor au fost decisive.
3. **Importanța Analizei Erorilor:** Doar uitându-ne la exemplele greșite (JSON) am realizat că modelul nu înțelegea fizica coborârii pantelor, ceea ce a dus la implementarea fix-ului logic.
4. **Iterația este cheia:** Proiectul a evoluat de la 17% acuratețe la 98% prin ajustări succesive ale datelor și parametrilor.
5. **Testarea end-to-end:** A dus la descoperirea a multor probleme ascunse ale rețelei neuronale și ale logicii ui-ului (acțiunile pe baza predicției) care nu erau evidente doar din metricile de antrenare.
---
## Structura Repository-ului la Finalul Etapei 6

**Structură COMPLETĂ și FINALĂ:**

```
proiect-rn-[prenume-nume]/
├── README.md                               # Overview general proiect (FINAL)
├── etapa3_analiza_date.md                  # Din Etapa 3
├── etapa4_arhitectura_sia.md               # Din Etapa 4
├── etapa5_antrenare_model.md               # Din Etapa 5
├── etapa6_optimizare_concluzii.md          # ← ACEST FIȘIER (completat)
│
├── docs/
│   ├── state_machine.png                   # Din Etapa 4
│   ├── state_machine_v2.png                # NOU - Actualizat (dacă modificat)
│   ├── loss_curve.png                      # Din Etapa 5
│   ├── confusion_matrix_optimized.png      # NOU - OBLIGATORIU
│   ├── results/                            # NOU - Folder vizualizări
│   │   ├── metrics_evolution.png           # NOU - Evoluție Etapa 4→5→6
│   │   ├── learning_curves_final.png       # NOU - Model optimizat
│   │   └── example_predictions.png         # NOU - Grid exemple
│   ├── optimization/                       # NOU - Grafice optimizare
│   │   ├── accuracy_comparison.png
│   │   └── f1_comparison.png
│   └── screenshots/
│       ├── ui_demo.png                     # Din Etapa 4
│       ├── inference_real.png              # Din Etapa 5
│       └── inference_optimized.png         # NOU - OBLIGATORIU
│
├── data/                                   # Din Etapa 3-5 (NESCHIMBAT)
│   ├── raw/
│   ├── generated/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data_acquisition/                   # Din Etapa 4
│   ├── preprocessing/                      # Din Etapa 3
│   ├── neural_network/
│   │   ├── visualize.py                    # NOU - generare grafice si grid exemple
│   │   ├── train_model.py                  # Din Etapa 5
│   │   ├── evaluate.py                     # Din Etapa 5
│   │   └── optimize.py                     # NOU - Script optimizare/tuning
│   └── app/
│       └── main.py                         # ACTUALIZAT - încarcă model OPTIMIZAT
│
├── models/
│   ├── untrained_model.h5                  # Din Etapa 4
│   ├── trained_model.h5                    # Din Etapa 5
│   ├── optimized_model.h5                  # NOU - OBLIGATORIU
│
├── results/
│   ├── training_history.csv                # Din Etapa 5
│   ├── test_metrics.json                   # Din Etapa 5
│   ├── optimization_experiments.csv        # NOU - OBLIGATORIU
│   ├── final_metrics.json                  # NOU - Metrici model optimizat
│
├── config/
│   ├── preprocessing_params.pkl            # Din Etapa 3
│   └── optimized_config.yaml               # NOU - Config model final
│
├── requirements.txt                        # Actualizat
└── .gitignore
```
---
## Predare și Contact

**Predarea se face prin:**
1. Commit pe GitHub: `"Etapa 6 completă – Accuracy=X.XX, F1=X.XX (optimizat)"`
2. Tag: `git tag -a v0.6-optimized-final -m "Etapa 6 - Model optimizat + Concluzii"`
3. Push: `git push origin main --tags`

---
