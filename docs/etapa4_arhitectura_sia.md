# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Boata Andrei-Darius  
**Link Repository GitHub:** https://github.com/Andreid2511/Proiect_RN.git  
**Data:** 15.01.2025

---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN**.  
**Obiectiv:** Livrarea unui schelet complet și funcțional al Sistemului cu Inteligență Artificială (SIA), în care toate modulele comunică între ele, iar modelul RN este definit și integrat (chiar dacă neantrenat la performanță maximă).

---

## Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| **Reducerea consumului** în traficul urban aglomerat ("Stop & Go") | Detectează stilul "Eco" sau "Normal" la viteze mici și **forțează schimbarea treptelor la <2200 RPM** pentru a preveni turarea inutilă. | `app/main.py` (Gearbox Logic) + `neural_network/train_model.py` |
| **Siguranță și putere** la depășiri pe autostradă | Detectează stilul "Sport" (accelerație bruscă) și **permite motorului să urce la 5800 RPM** înainte de schimbare, oferind cuplu maxim. | `app/main.py` (Gearbox Logic) + `neural_network/train_model.py` |
| **Evitarea alarmelor false** la coborârea pantelor abrupte (Hill Descent) | Folosește senzorul de înclinație (Tilt < -2°) pentru a detecta coborârea și a ignora turația mare cauzată de frâna de motor, clasificând corect situația ca **FORCED_ECO**. | `app/main.py` (Override Logic) + `data_acquisition/generate_data.py` |

---

### 2. Contribuția Originală la Setul de Date – 100% Original

Deoarece datele publice nu conțin informații specifice despre înclinația drumului (`tilt`) corelate cu turația și treapta de viteză pentru o cutie automată specifică, am ales să generez **întregul set de date** prin simulare fizică.

**Total observații finale:** 180,000  
**Observații originale:** 180,000 (100%)

**Tipul contribuției:**
[X] Date generate prin simulare fizică  
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  

**Descriere detaliată:**
Am implementat un simulator fizic în Python (`src/data_acquisition/generate_data.py`) care modelează dinamica longitudinală a unui vehicul. Simulatorul ia în calcul forțele de tracțiune (bazate pe curba de cuplu a unui motor aspirat), rezistența la rulare, gravitația (în funcție de pantă) și rezistența aerodinamică.
Datele sunt generate la o frecvență de 30Hz (dt=0.033s) pentru a imita perfect ciclul de execuție al aplicației finale. Au fost simulate 3 scenarii distincte de condus (Eco, Normal, Sport) prin variația agresivității apăsării pedalelor și a momentelor de schimbare a treptelor.

**Dovezi:**
- Codul sursă: `src/data_acquisition/generate_data.py`
- Datele generate: `data/train/train.csv` (conține coloanele `rpm`, `speed`, `acceleration`, `throttle`, `brake`, `tilt`, `gear`, `style_label`)

---

### 3. Diagrama State Machine a Întregului Sistem

Diagrama de stări descrie logica decizională a cutiei de viteze automate, care integrează predicția Rețelei Neuronale cu reguli de siguranță fizică.



**Legendă și Justificare:**

Am ales o arhitectură hibridă **RN + Rule-Based** (State Machine) pentru că o cutie de viteze trebuie să fie deterministă în situații critice, dar adaptabilă în rest.

**Stările principale:**
1. **PREPROCESS & INFERENCE:** Sistemul preia datele brute (senzori), le scalează și interoghează Rețeaua Neuronală pentru a afla intenția șoferului (Eco/Normal/Sport).
2. **HILL_DESC / HILL_CLIMB:** Stări activate prioritar de senzorul de înclinație (`tilt`). Dacă panta este abruptă, fizica dictează comportamentul (ex: frână de motor la vale), ignorând parțial stilul șoferului pentru siguranță.
3. **KICKDOWN:** O stare critică de "urgență". Dacă pedala este apăsată >90%, se ignoră orice mod Eco și se retrogradează imediat pentru putere maximă.
4. **ECO / SPORT / NORMAL MODE:** Stările standard de funcționare, unde pragurile de schimbare a vitezelor sunt ajustate dinamic de predicția AI.

**Tranziții critice:**
- `INFERENCE` → `KICKDOWN`: Are prioritate maximă (siguranță în depășiri).
- `INFERENCE` → `HILL_DESC`: Previne interpretarea greșită a turației mari la vale ca fiind "Sport".

---

### 4. Scheletul Complet al celor 3 Module

Toate modulele sunt implementate și funcționale în repository.

| **Modul** | **Implementare** | **Status Funcțional** |
|-----------|------------------|-----------------------|
| **1. Data Logging / Acquisition** | `src/data_acquisition/generate_data.py` | ✅ Rulează fără erori, generează 180k samples, exportă în `data/train/` |
| **2. Neural Network Module** | `src/neural_network/train_model.py` | ✅ Definește arhitectura MLP, antrenează pe datele generate și salvează modelul `.h5` |
| **3. Web Service / UI** | `src/app/main.py` (Tkinter) | ✅ Interfața grafică pornește, afișează ceasurile de bord, preia input de la slidere și afișează predicția modelului în timp real. |

#### Detalii per modul:

**Modul 1: Data Acquisition**
- Script Python care simulează fizica vehiculului.
- Rulează automat la execuție și populează folderele `data/` cu fișiere CSV gata de antrenare.

**Modul 2: Neural Network**
- Folosește TensorFlow/Keras.
- Arhitectura: MLP (Multi-Layer Perceptron) cu 3 straturi Dense și activare ReLU/Softmax.
- Scriptul încarcă datele, le normalizează cu `StandardScaler` și antrenează modelul.
- Output: `models/untrained_model.h5` (sau trained, în funcție de stadiu) și `config/preprocessing_params.pkl`.

**Modul 3: User Interface (App)**
- Aplicație Desktop construită cu `tkinter`.
- Simulează un bord digital de mașină (Cockpit).
- **Input:** Slidere pentru Accelerație, Frână, Pantă.
- **Procesare:** Rulează bucla fizică la 30 FPS + Inferență AI.
- **Output:** Turometru, Vitezometru, Treapta de viteză curentă și Modul detectat (Eco/Sport).

---

## Structura Repository-ului la Finalul Etapei 4

## Structura Repository-ului la Finalul Etapei 4 (OBLIGATORIE)

**Verificare consistență cu Etapa 3:**

```
proiect-rn-[nume-prenume]/
├── config/ 
│   └── preprocessing_params.pkl
├── data/   # CSV-uri generate
│   ├── train/
│   ├── validation/ 
│   └── test/
├── src/
│   ├── data_acquisition/
│       └── generate_data.py # Generatorul de date  (Modulul 1)
│   ├── neural_network/
│   └── app/  # UI schelet
├── docs/
│   ├── screenshots/ 
│   │   └── ui_demo.png              # Screenshot aplicație rulând  
│   └──  state_machine.png           #(state_machine.png sau state_machine.pptx sau state_machine.drawio)
├── models
|   └──untrained_model.h5            # Modelul compilat
├── config/
├── README.md
├── README_Etapa3.md                 # (deja existent)
├── README_Etapa4_Arhitectura_SIA.md # ← acest fișier completat (în rădăcină)
└── requirements.txt  # Sau .lvproj
```
---

## Checklist Final – Predare Etapa 4

### Documentație și Structură
- [x] Tabelul Nevoie → Soluție → Modul completat.
- [x] Declarație contribuție 100% date originale (Simulare Fizică).
- [x] Diagrama State Machine explicată.

### Modul 1: Data Logging / Acquisition
- [x] Cod `generate_data.py` rulează și produce date valide.

### Modul 2: Neural Network
- [x] Modelul este definit, compilat și salvat (`models/*.h5`).

### Modul 3: Web Service / UI
- [x] Aplicația `main.py` pornește și reacționează la input-ul utilizatorului.
- [x] Screenshot `ui_demo.png` existent în docs.

---