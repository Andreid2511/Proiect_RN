# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Boata Andrei-Darius  
**Link Repository GitHub:** [https://github.com/Andreid2511/Proiect_RN.git]  
**Data predării:** 23.01.2026

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
2. Tag: `# 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Boata Andrei-Darius |
| **Grupa / Specializare** | 633AB / SIA |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | https://github.com/Andreid2511/Proiect_RN.git |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python (Keras, TensorFlow, Tkinter) |
| **Domeniul Industrial de Interes (DII)** | Automotive (Powertrain Control) |
| **Tip Rețea Neuronală** | MLP (Multi-Layer Perceptron) |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 5 (Baseline) | Rezultat Final (Optimizat) | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | 98.13% | **98.28%** | +0.15% | ✅ |
| F1-Score (Macro) | ≥0.65 | 0.9812 | **0.9827** | +0.0015 | ✅ |
| Latență Inferență | < 50ms | 0.0275 ms | **0.0287 ms** | Neglijabilă | ✅ |
| Contribuție Date Originale | ≥40% | 100% | **100%** | - | ✅ |
| Nr. Experimente Optimizare | ≥4 | - | **5** | - | ✅ |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1 | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [x] DA |
| 2 | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [x] DA |
| 3 | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [x] DA |
| 4 | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [x] DA |
| 5 | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [x] DA |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

În industria auto modernă, optimizarea consumului de combustibil și reducerea emisiilor sunt prioritare. Transmisiile automate convenționale folosesc hărți statice de schimbare a vitezelor, care nu se adaptează stilului șoferului sau condițiilor de drum (pante). Acest lucru duce la situații ineficiente: turarea excesivă a motorului în oraș (mod "Sport" uitat activat) sau lipsa de putere în depășiri (mod "Eco" prea lent).

Proiectul propune un Sistem de Inteligență Artificială (SIA) care analizează în timp real telemetria vehiculului (RPM, viteză, pedală, accelerație, înclinație) pentru a detecta intenția șoferului (Eco/Normal/Sport) și a adapta dinamic strategia de schimbare a vitezelor.

### 2.2 Beneficii Măsurabile Urmărite

1. **Reducerea consumului:** Prin detectarea stilului "Eco" și forțarea schimbării timpurii a treptelor (<2200 RPM).
2. **Siguranță în depășiri:** Activarea automată a modului "Sport" (schimbare >5000 RPM) la detecția unei accelerații bruște.
3. **Protecția motorului:** Evitarea subturării la urcarea pantelor prin detecția înclinației și menținerea unei trepte inferioare.
4. **Confort:** Eliminarea necesității comutării manuale între modurile de condus.

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Optimizare consum urban | Clasificare stil Eco → Shift Point 2200 RPM | RN (Inference) + Main App | Acuratețe > 95% |
| Putere la cerere (Kickdown) | Detecție pedală > 90% → Retrogradare | Main App (Logic Override) | Timp răspuns < 50ms |
| Evitare alarmă falsă la vale | Detecție Hill Descent → Ignorare RPM mare | Main App (Physics Logic) | 0 False Positives |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Simulare |
| **Sursa concretă** | Generator fizic propriu (Python) |
| **Număr total observații finale (N)** | 180,000 |
| **Număr features** | 7 (RPM, Speed, Acceleration, Throttle, Brake, Tilt, Gear) |
| **Tipuri de date** | Numerice (senzori) + Categoriale (etichete) |
| **Format fișiere** | CSV |
| **Perioada colectării/generării** | Ianuarie 2026 |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 180,000 |
| **Observații originale (M)** | 180,000 |
| **Procent contribuție originală** | **100%** |
| **Tip contribuție** | Simulare fizică |
| **Locație cod generare** | `src/data_acquisition/generate_data.py` |
| **Locație date originale** | `data/train/train.csv`, `data/validation/validation.csv`, `data/test/test.csv`  |

**Descriere metodă generare/achiziție:**

Am dezvoltat un simulator fizic complet în Python care modelează dinamica longitudinală a unui vehicul (tracțiune, rezistență la aer, rezistență la rulare, gravitație pe pantă). Simulatorul rulează scenarii de condus (urban, extra-urban, autostradă) cu parametri aleatori (accelerație, frânare) specifici celor 3 stiluri de condus (Eco, Normal, Sport). Frecvența de eșantionare este de 30Hz pentru a simula senzorii reali.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | 126,000 |
| Validation | 15% | 27,000 |
| Test | 15% | 27,000 |

**Preprocesări aplicate:**
- Normalizare Standard (Z-score) pe features numerice (`StandardScaler`).
- One-Hot Encoding pentru etichetele categoriale (0, 1, 2).
- Shuffling pentru eliminarea dependenței temporale în antrenament.

**Referințe fișiere:** `src/neural_network/train_model.py`, `config/preprocessing_params.pkl`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging / Acquisition** | Python (NumPy) | Simulare fizică vehicul și generare CSV | `src/data_acquisition/` |
| **Neural Network** | Keras / TensorFlow | Clasificare stil condus (MLP) | `src/neural_network/` |
| **Web Service / UI** | Python (Tkinter) | Dashboard Auto (Cockpit) Interactiv | `src/app/` |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine_v2.png`

**Stări principale și descriere:**

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `INIT` | Inițializare senzori și încărcare model | Start aplicație | Model Loaded |
| `UPDATE_PHYSICS` | Calcul forțe și viteze (30Hz) | Timer tick (33ms) | Date noi disponibile |
| `INFERENCE` | Predicție stil condus (AI) | Date noi | Rezultat (0/1/2) |
| `LOGIC_OVERRIDE` | Verificare reguli siguranță (Pante) | Rezultat AI | Stil Final |
| `UPDATE_GEARBOX` | Decizie schimbare treaptă | Stil Final | RPM > prag |
| `UI_UPDATE` | Redesenare ace și text | Valori noi | Așteptare tick |

**Justificare alegere arhitectură State Machine:**

Am ales o arhitectură hibridă (AI + Rule-Based) deoarece un sistem de siguranță critică (cum este transmisia auto) nu se poate baza exclusiv pe probabilități neuronale. Stările de "Override" (Hill Descent, Kickdown) asigură comportamentul determinist în situații limită, în timp ce AI-ul optimizează eficiența în regim normal.

### 4.3 Actualizări State Machine în Etapa 6

| Componentă Modificată | Valoare Etapa 5 | Valoare Etapa 6 | Justificare Modificare |
|----------------------|-----------------|-----------------|------------------------|
| Logică Pantă | N/A | `HILL_DESCENT` Override | Corecție eroare AI la vale (RPM mare) |
| Logică Pedală | N/A | `KICKDOWN` (>90%) | Prioritate maximă pentru putere |

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

```
Input (shape: [7])  # 7 senzori: RPM, Speed, Accel, Throttle, Brake, Tilt, Gear
  → Dense(32, ReLU)
  → Dense(32, ReLU)
  → Dense(16, ReLU)
  → Dense(3, Softmax)
Output: 3 probabilități (Eco, Normal, Sport) - suma = 1.0
```

**Justificare alegere arhitectură:**
Am ales o rețea MLP (Multi-Layer Perceptron) deoarece datele sunt tabulare și relațiile dintre senzori sunt non-liniare dar nu necesită convoluții (nu sunt imagini) sau recurență complexă (starea curentă conține suficientă informație prin feature-ul de accelerație). Arhitectura este suficient de adâncă pentru a învăța, dar suficient de mică pentru latență minimă. ReLU accelerează antrenarea vs Tanh.

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | 0.001 | Convergență stabilă și rapidă cu Adam |
| Batch Size | 32 | Gradient stabil, previne blocarea în minime locale |
| Epochs | 150 | Suficiente pentru convergență completă |
| Optimizer | Adam | Standardul pentru date zgomotoase |
| Loss Function | Categorical Crossentropy | Clasificare multi-clasă |
| Activare Hidden | ReLU | Eficiență computațională, fără vanishing gradient |
| Early Stopping | patience=15 | Prevenire overfitting, oprire la optim |

### 5.3 Experimente de Optimizare

| Exp# | Modificare față de Baseline | Accuracy | F1-Score | Timp Antrenare | Observații |
|------|----------------------------|----------|----------|----------------|------------|
| Exp 1 | Small (Rapid) | 87.86% | 0.8787 | 5.14s | Sub-fit, prea simplu |
| Exp 2 | Baseline (Tanh) | 92.70% | 0.9271 | 5.78s | Stabil, dar Tanh e lent |
| Exp 3 | **Baseline (ReLU)** | **93.53%** | **0.9350** | **5.53s** | **Optim (Viteză/Precizie)** |
| Exp 4 | Pyramid (Deep) | 94.59% | 0.9457 | 5.50s | Bun, dar risc overfitting |
| Exp 5 | Large (Dropout) | 93.77% | 0.9376 | 5.92s | Complexitate inutilă |
| **FINAL** | **Baseline ReLU** | **98.28%** | **0.9827** | **~6 min** | **Antrenat complet (150 epoci)** |

**Justificare alegere model final:**
Am ales configurația **Baseline ReLU** (Exp 3) deoarece oferă cel mai bun echilibru între performanță (98% accuracy final) și eficiență computațională (latență minimă pe CPU), esențială pentru o aplicație real-time la 30 FPS.

**Referințe fișiere:** `results/optimization_experiments.csv`, `models/optimized_model.h5`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | 98.28% | ≥70% | ✅ |
| **F1-Score (Macro)** | 0.9827 | ≥0.65 | ✅ |
| **Precision (Macro)** | 98.27% | - | - |
| **Recall (Macro)** | 98.27% | - | - |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline) | Etapa 6 (Optimizat) | Îmbunătățire |
|--------|-------------------|---------------------|--------------|
| Accuracy | 98.13% | 98.28% | +0.15% |
| F1-Score | 0.9812 | 0.9827 | +0.0015 |

**Referință fișier:** `results/final_metrics.json`

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**

| Aspect | Observație |
|--------|------------|
| **Clasa cu cea mai bună performanță** | Sport - Precision >99%, Recall >99% |
| **Clasa cu cea mai slabă performanță** | Normal - Mici confuzii cu Eco și Sport |
| **Confuzii frecvente** | Eco confundat cu Normal (1.2%) la viteze constante |
| **Dezechilibru clase** | Clasele sunt echilibrate perfect prin generare (60k fiecare) |

### 6.3 Analiza Top 5 Erori

| # | Input (descriere scurtă) | Predicție RN | Clasă Reală | Cauză Probabilă | Implicație Industrială |
|---|--------------------------|--------------|-------------|-----------------|------------------------|
| 1 | RPM mare, Pedala 0, Pantă negativă | Sport | Eco | Frână de motor la vale | Schimbare inutilă a vitezei în sus (risc siguranță) |
| 2 | Viteză mică, Pedală mică | Eco | Normal | Ambiguitate date (overlap) | Schimbare prea devreme a vitezei (subturare) |
| 3 | Urcare rampă, Pedală mare | Sport | Normal | Efort motor interpretat ca agresivitate | Turare excesivă, consum crescut |
| 4 | Kickdown brusc | Normal | Sport | Latență în detecția accelerației (derivată) | Întârziere răspuns depășire |
| 5 | Croazieră autostradă | Normal | Eco | Lipsă variație pedală | Niciuna (comportament identic) |

### 6.4 Validare în Context Industrial

Rezultatele indică o robustețe de 98%, ceea ce este excelent pentru un sistem de asistență (Driver Support). Erorile critice (de tipul #1 - Hill Descent) au fost rezolvate prin logica de "Override" din `main.py`, asigurând siguranța vehiculului indiferent de predicția AI. Sistemul este gata de testare pe un vehicul real.

**Pragul de acceptabilitate pentru domeniu:** Acuratețe > 90%
**Status:** Atins (98.28%)

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| **Model încărcat** | `trained_model.h5` | `optimized_model.h5` | Performanță și eficiență maximizate |
| **Logică Pantă** | Absentă | Implementată | Rezolvare eroare critică Hill Descent |
| **UI - feedback vizual** | Text simplu | Justification Text | Explicabilitate decizie AI pentru șofer |
| **UI - indicații folosire** | Absente | Implementate | Explicare folosire și capabilități UI |
| **Logging** | Implicit | Statistici Sesiune | Feedback asupra stilului dominant |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`

*Screenshot-ul arată interfața în timpul unei simulări de condus normal, predicția "NORMAL" activă și textul justificativ "Regim de condus mixt".*

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/`

**Fluxul demonstrat:**
1. Utilizatorul mișcă slider-ul de accelerație.
2. Fizica vehiculului actualizează RPM și Viteza.
3. Rețeaua Neuronală prezice stilul.
4. Cutia de viteze schimbă treapta automat.
5. UI-ul se actualizează instantaneu (30 FPS).

**Latență măsurată end-to-end:** ~30 ms (timp de cadru)

---

## 8. Structura Repository-ului Final

```
proiect-rn-[nume-prenume]/
│
├── README.md                               # ← ACEST FIȘIER (Overview Final Proiect - Pe moodle la Evaluare Finala RN > Upload Livrabil 1 - Proiect RN (Aplicatie Sofware) - trebuie incarcat cu numele: NUME_Prenume_Grupa_README_Proiect_RN.md)
│
├── docs/
│   ├── etapa3_analiza_date.md              # Documentație Etapa 3
│   ├── etapa4_arhitectura_SIA.md           # Documentație Etapa 4
│   ├── etapa5_antrenare_model.md           # Documentație Etapa 5
│   ├── etapa6_optimizare_concluzii.md      # Documentație Etapa 6
│   │
│   ├── state_machine.png                   # Diagrama State Machine inițială
│   ├── state_machine_v2.png                # Versiune actualizată Etapa 6
│   ├── confusion_matrix_optimized.png      # Confusion matrix model final
│   │
│   ├── screenshots/
│   │   ├── ui_demo.png                     # Screenshot UI schelet (Etapa 4)
│   │   ├── inference_real.png              # Inferență model antrenat (Etapa 5)
│   │   └── inference_optimized.png         # Inferență model optimizat (Etapa 6)
│   │
│   ├── demo/                               # Demonstrație funcțională end-to-end
│   │   └── demo_end_to_end.gif             # (sau .mp4 / secvență screenshots)
│   │
│   ├── results/                            # Vizualizări finale
│   │   ├── loss_curve.png                  # Grafic loss/val_loss (Etapa 5)
│   │   ├── metrics_evolution.png           # Evoluție metrici (Etapa 6)
│   │   └── learning_curves_final.png       # Curbe învățare finale
│   │
│   └── optimization/                       # Grafice comparative optimizare
│       ├── accuracy_comparison.png         # Comparație accuracy experimente
│       └── f1_comparison.png               # Comparație F1 experimente
│
├── data/
│   ├── README.md                           # Descriere detaliată dataset
│   ├── raw/                                # Date brute originale
│   ├── processed/                          # Date curățate și transformate
│   ├── generated/                          # Date originale (contribuția ≥40%)
│   ├── train/                              # Set antrenare (70%)
│   ├── validation/                         # Set validare (15%)
│   └── test/                               # Set testare (15%)
│
├── src/
│   ├── data_acquisition/                   # MODUL 1: Generare/Achiziție date
│   │   ├── README.md                       # Documentație modul
│   │   ├── generate.py                     # Script generare date originale
│   │   └── [alte scripturi achiziție]
│   │
│   ├── preprocessing/                      # Preprocesare date (Etapa 3+)
│   │   ├── data_cleaner.py                 # Curățare date
│   │   ├── feature_engineering.py          # Extragere/transformare features
│   │   ├── data_splitter.py                # Împărțire train/val/test
│   │   └── combine_datasets.py             # Combinare date originale + externe
│   │
│   ├── neural_network/                     # MODUL 2: Model RN
│   │   ├── README.md                       # Documentație arhitectură RN
│   │   ├── model.py                        # Definire arhitectură (Etapa 4)
│   │   ├── train.py                        # Script antrenare (Etapa 5)
│   │   ├── evaluate.py                     # Script evaluare metrici (Etapa 5)
│   │   ├── optimize.py                     # Script experimente optimizare (Etapa 6)
│   │   └── visualize.py                    # Generare grafice și vizualizări
│   │
│   └── app/                                # MODUL 3: UI/Web Service
│       ├── README.md                       # Instrucțiuni lansare aplicație
│       └── main.py                         # Aplicație principală
│
├── models/
│   ├── untrained_model.h5                  # Model schelet neantrenat (Etapa 4)
│   ├── trained_model.h5                    # Model antrenat baseline (Etapa 5)
│   ├── optimized_model.h5                  # Model FINAL optimizat (Etapa 6) ← FOLOSIT
│   └── final_model.onnx                    # (opțional) Export ONNX pentru deployment
│
├── results/
│   ├── training_history.csv                # Istoric antrenare - toate epocile (Etapa 5)
│   ├── test_metrics.json                   # Metrici baseline test set (Etapa 5)
│   ├── optimization_experiments.csv        # Toate experimentele optimizare (Etapa 6)
│   ├── final_metrics.json                  # Metrici finale model optimizat (Etapa 6)
│   └── error_analysis.json                 # Analiza detaliată erori (Etapa 6)
│
├── config/
│   ├── preprocessing_params.pkl            # Parametri preprocesare salvați (Etapa 3)
│   └── optimized_config.yaml               # Configurație finală model (Etapa 6)
│
├── requirements.txt                        # Dependențe Python (actualizat la fiecare etapă)
└── .gitignore                              # Fișiere excluse din versionare
```

### Legendă Progresie pe Etape

| Folder / Fișier | Etapa 3 | Etapa 4 | Etapa 5 | Etapa 6 |
|-----------------|:-------:|:-------:|:-------:|:-------:|
| `data/raw/`, `processed/`, `train/`, `val/`, `test/` | ✓ Creat | - | Actualizat* | - |
| `data/generated/` | - | ✓ Creat | - | - |
| `src/preprocessing/` | ✓ Creat | - | Actualizat* | - |
| `src/data_acquisition/` | - | ✓ Creat | - | - |
| `src/neural_network/model.py` | - | ✓ Creat | - | - |
| `src/neural_network/train.py`, `evaluate.py` | - | - | ✓ Creat | - |
| `src/neural_network/optimize.py`, `visualize.py` | - | - | - | ✓ Creat |
| `src/app/` | - | ✓ Creat | Actualizat | Actualizat |
| `models/untrained_model.*` | - | ✓ Creat | - | - |
| `models/trained_model.*` | - | - | ✓ Creat | - |
| `models/optimized_model.*` | - | - | - | ✓ Creat |
| `docs/state_machine.*` | - | ✓ Creat | - | (v2 opțional) |
| `docs/etapa3_analiza_date.md` | ✓ Creat | - | - | - |
| `docs/etapa4_arhitectura_SIA.md` | - | ✓ Creat | - | - |
| `docs/etapa5_antrenare_model.md` | - | - | ✓ Creat | - |
| `docs/etapa6_optimizare_concluzii.md` | - | - | - | ✓ Creat |
| `docs/confusion_matrix_optimized.png` | - | - | - | ✓ Creat |
| `docs/screenshots/` | - | ✓ Creat | Actualizat | Actualizat |
| `results/training_history.csv` | - | - | ✓ Creat | - |
| `results/optimization_experiments.csv` | - | - | - | ✓ Creat |
| `results/final_metrics.json` | - | - | - | ✓ Creat |
| **README.md** (acest fișier) | Draft | Actualizat | Actualizat | **FINAL** |

*\* Actualizat dacă s-au adăugat date noi în Etapa 4*

### Convenție Tag-uri Git

| Tag | Etapa | Commit Message Recomandat |
|-----|-------|---------------------------|
| `v0.3-data-ready` | Etapa 3 | "Etapa 3 completă - Dataset analizat și preprocesat" |
| `v0.4-architecture` | Etapa 4 | "Etapa 4 completă - Arhitectură SIA funcțională" |
| `v0.5-model-trained` | Etapa 5 | "Etapa 5 completă - Accuracy=X.XX, F1=X.XX" |
| `v0.6-optimized-final` | Etapa 6 | "Etapa 6 completă - Accuracy=X.XX, F1=X.XX (optimizat)" |

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare
Python >= 3.8

### 9.2 Instalare
```bash
git clone [https://github.com/Andreid2511/Proiect_RN.git](https://github.com/Andreid2511/Proiect_RN.git)
cd Proiect_RN

# 2. Creare mediu virtual (recomandat)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# sau: venv\Scripts\activate    # Windows

# 3. Instalare dependențe
pip install -r requirements.txt
```

### 9.3 Rulare Pipeline Complet

```bash
# 1. Generare date
python src/data_acquisition/generate_data.py

# 2. Antrenare model
python src/neural_network/train_model.py

# 3. Rulare Aplicatie
python src/app/main.py
```

### 9.4 Verificare Rapidă 

```bash
# Verificare că modelul se încarcă corect
python -c "from keras.models import load_model; m = load_model('models/optimized_model.h5'); print('✓ Model încărcat cu succes')"

# Verificare inferență pe un exemplu
python src/neural_network/evaluate.py --model models/optimized_model.h5 --quick-test
```

### 9.5 Structură Comenzi LabVIEW (dacă aplicabil)

```
[Completați dacă proiectul folosește LabVIEW]
1. Deschideți [nume_proiect].lvproj
2. Rulați Main.vi
3. ...
```

---

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit | Target | Realizat | Status |
|------------------|--------|----------|--------|
| Optimizare consum urban (Eco) | Acc >95% | **98.28%** | ✅ |
| Putere la cerere (Kickdown) | < 50ms | **~29ms** | ✅ |
| Accuracy pe test set | ≥70% | **98.28%** | ✅ |
| F1-Score pe test set | ≥0.65 | **0.9827** | ✅ |
| Evitare alarmă falsă la vale | 0 FP | **0 (prin Override)** | ✅ |

### 10.2 Ce NU Funcționează – Limitări Cunoscute

1. **Limitare 1 (Date Sintetice):** Modelul este antrenat exclusiv pe date generate prin simulare fizică. Deși ecuațiile sunt corecte, zgomotul real al senzorilor (vibrații motor, interferențe electrice) nu este perfect replicat, ceea ce ar putea scădea precizia într-un test pe șosea.
2. **Limitare 2 (Dependență CPU):** Latența excelentă (0.029ms) este măsurată pe un procesor de PC. Pe un microcontroller auto (ex: Infineon Aurix), inferența Python ar fi imposibilă. Este necesar un export în C++ sau TensorFlow Lite Micro.
3. **Limitare 3 (Memorie Scurtă):** Arhitectura MLP clasifică instantaneu. Nu are "memorie" (ca un LSTM) pentru a înțelege contextul ultimelor 10 secunde (ex: "șoferul a fost agresiv acum 5 secunde, deci probabil va mai fi").
4. **Funcționalități neimplementate:** Exportul automat în format ONNX pentru integrare embedded nu este complet funcțional (necesită librării suplimentare).

### 10.3 Lecții Învățate (Top 5)

1. **[Lecție 1]:** Importanța înțelegerii fizicii din spatele datelor. Fără a analiza fenomenul "Hill Descent" (turație mare fără pedală), nu aș fi putut corecta erorile critice ale modelului doar prin antrenare.
2. **[Lecție 2]:** Sistemele hibride (AI + Reguli) sunt superioare AI-ului pur în inginerie. Un simplu `if` (Override) este mai sigur și mai rapid decât 1000 de ore de antrenament pentru situații de siguranță critică.
3. **[Lecție 3]:** Optimizarea arhitecturii (ReLU vs Tanh) a adus stabilitate și viteză, demonstrând că funcțiile de activare moderne sunt esențiale pentru convergență rapidă.
4. **[Lecție 4]:** Generarea propriilor date (Simulare) oferă un control mult mai bun asupra cazurilor limită (ex: pante de 15 grade) decât utilizarea unui dataset public generic.
5. **[Lecție 5]:** Documentarea incrementală (README pe etape) a transformat integrarea finală dintr-un coșmar într-un proces simplu de asamblare.

### 10.4 Retrospectivă

**Ce ați schimba dacă ați reîncepe proiectul?**

1. **Implementarea LSTM (Long Short-Term Memory):** Dacă aș lua proiectul de la zero, aș înlocui arhitectura MLP (Feed-Forward) cu o rețea recurentă **LSTM** sau **GRU**. Deși MLP-ul curent este rapid și precis, un LSTM ar putea lua decizii bazate pe *istoricul* ultimelor 3-5 secunde de condus, eliminând necesitatea unor buffere externe de "smoothing" în codul aplicației și oferind o predicție mai naturală a intenției șoferului.
2. **Ar implementa mai devreme integrare GPS:** Context geographic ar rezolva ambiguitatea Eco/Normal pe autostradă. Curent, doar din senzori e insuficient → ar trebui API externa.

3. **Ar colecta date reale paralel cu simularea:** 70% date simulate + 30% reale (din vehicule pilot) ar fi mai reprezentativ decât 100% simulat. Simularea are bias - nu captează toți edge cases reali.

4. **Ar folosi model interpretation tools (SHAP/LIME):** Pentru a înțelege care senzori sunt mai importanți pentru fiecare predicție și a evita "black box" output. Operatorul ar putea mai bine debugga erori.

5. **Ar implementa A/B testing cu șoferi reali mai devreme:** Doar simulation ne-a dat 98% accuracy, dar posibil alți șoferi (stile noi) să ne surprindă. Recolectare și reantrenare incrementală pe date reale din ziua 1.

6. **Ar structura codul mai modular de la început:** Curent, state machine logic e în main.py - ar trebui extins în classe separate pentru ușura testing și deployment.

### 10.5 Direcții de Dezvoltare Ulterioară

| Termen | Îmbunătățire Propusă | Beneficiu Estimat |
|--------|---------------------|-------------------|
| **Short-term** (1-2 săptămâni) | Colectare date OBD-II reale | Validare model pe mașină reală (Proof of Concept) |
| **Medium-term** (1-2 luni) | Export model în ONNX/TFLite | Rulare pe Raspberry Pi (Edge AI) cu latență <5ms |
| **Long-term** | Integrare GPS (pante viitoare) | Predicție proactivă (Predictive Cruise Control) |

---

## 11. Bibliografie

1. Keras Documentation, 2024. *The Sequential Model*. Disponibil la: https://keras.io/guides/sequential_model/

2. Sensor Technology Automotive Sensor Mechatronics, 2024, SENSORS TECHNOLOGY
AUTOMOTIVE SENSOR.  https://kanchiuniv.ac.in/wp-content/uploads/2024/12/HONS.-COURSE-–-Sensor-Technology-Automotive-Sensor-Mechatronics_compressed.pdf
3. Chollet, F., 2024. Keras: Deep Learning for humans. https://keras.io/

4. TensorFlow Documentation, 2024. TensorFlow 2.13 API Reference. https://www.tensorflow.org/api_docs

6. Goodfellow, I., Bengio, Y., Courville, A., 2016. Deep Learning. MIT Press. https://www.deeplearningbook.org/

6. Abaza, B., 2025. AI-Driven Dynamic Covariance for ROS 2 Mobile Robot Localization. Sensors, 25, 3026. https://doi.org/10.3390/s25103026

7. Scikit-learn Developers, 2024. scikit-learn documentation: Preprocessing and feature engineering. https://scikit-learn.org/stable/modules/preprocessing.html

---

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [x] **Accuracy ≥70%** pe test set (realizat: 98.28%)
- [x] **F1-Score ≥0.65** pe test set (realizat: 0.98)
- [x] **Contribuție ≥40% date originale** (realizat: 100% Simulare)
- [x] **Model antrenat de la zero** (fără pre-trained weights)
- [x] **Minimum 4 experimente** de optimizare documentate (în Secțiunea 5.3)
- [x] **Confusion matrix** generată și interpretată (în Secțiunea 6.2)
- [x] **State Machine** definit și implementat (inclusiv Hill Descent)
- [x] **Cele 3 module funcționale:** Data Logging, RN, UI
- [x] **Demonstrație end-to-end** (aplicația `main.py` funcțională)

### Repository și Documentație

- [x] **README.md** complet (toate secțiunile completate)
- [x] **4 README-uri etape** prezente în `docs/`
- [x] **Screenshots** prezente în `docs/screenshots/`
- [x] **Structura repository** conformă cu cerințele
- [x] **requirements.txt** actualizat
- [x] **Cod comentat** (explicații clare în `main.py` și `train_model.py`)
- [x] **Toate path-urile relative** (compatibil pe orice PC)

### Acces și Versionare

- [x] **Repository accesibil** (Public)
- [x] **Tag `v0.6-optimized-final`** creat
- [x] **Commit-uri incrementale** (istoric vizibil)
- [x] **Fișiere mari** excluse (dataset-urile mari, modelele uriașe dacă e cazul)

### Verificare Anti-Plagiat

- [x] Model antrenat **de la zero**
- [x] **Minimum 40% date originale**
- [x] Cod propriu sau clar atribuit

---

## Note Finale

**Versiune document:** FINAL pentru examen  
**Ultima actualizare:** 31.01.2025  
**Tag Git:** `v0.6-optimized-final`"`
3. Push: `git push origin main --tags`

---
