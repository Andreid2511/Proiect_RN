# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Boata Andrei-Darius 
**Data:** 21.11.2025 

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Date generate programatic pe baza unui mix de traseu (urban + extraurban + autostradă), care modelează comportamentul vehiculului în accelerare, decelerare și viteze constante.
* **Modul de achiziție:** ☑ Generare programatică (simulare)
* **Perioada / condițiile colectării:** Setul de date a fost generat în cadrul proiectului, folosind parametrii de condus definiți de mine (accelerații, frânări, timpi, distanțe, viteze pe segmente).

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 1000
* **Număr de caracteristici (features):** 6
* **Tipuri de date:** ☑ Numerice / ☑ Temporale
* **Format fișiere:** ☑ CSV

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| crankshaft_position_sensor | numeric | rpm | [Masurarea turatiilor] | 0–6000 |
| speed | categorial | numeric | km/h | [senzor ABS sau senzor de viteza VVS sau WWS]  | 0-250 |
| throttle_position | percentage | % | [TPS pozitia pedalei de acceleratie] | 0–100 |
| braking | percentage | % | [BPPS pozitia pedalei de frana] | 0–100 |
| tilt_sensor | degrees | ° | [Senzor cu plaja completa de masurare pentru toate automobilele] | ±90° |
| time | numeric | s | [Timpul de la pornire] | 0–100 |

**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

Asupra celor 6 caracteristici (crankshaft position, speed, throttle, braking, tilt, time) au fost calculate:

* **Media** – pentru a observa nivelul mediu al fiecărui senzor
* **Mediana** – pentru a reduce influența valorilor extreme
* **Deviația standard** – pentru a măsura variabilitatea fiecărei măsurători
* **Min–Max** – pentru a identifica domeniul real folosit
* **Distribuții pe caracteristici** (histograme)
* **Identificarea outlierilor** (IQR / percentile)

### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă** (% pe coloană)
* **Detectarea valorilor inconsistente sau eronate**
* **Identificarea caracteristicilor redundante sau puternic corelate**

### 3.3 Probleme identificate

* [exemplu] Feature X are 8% valori lipsă
* [exemplu] Distribuția feature Y este puternic neuniformă
* [exemplu] Variabilitate ridicată în clase (class imbalance)

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminare duplicatelor**
* **Tratarea valorilor lipsă:**
  * Feature A: imputare cu mediană
  * Feature B: eliminare (30% valori lipsă)
* **Tratarea outlierilor:** IQR / limitare percentile

### 4.2 Transformarea caracteristicilor

* **Normalizare:** Min–Max / Standardizare
* **Encoding pentru variabile categoriale**
* **Ajustarea dezechilibrului de clasă** (dacă este cazul)

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70–80% – train
* 10–15% – validation
* 10–15% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate
* Parametrii de preprocesare în `config/preprocessing_config.*` (opțional)

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [ ] Structură repository configurată
- [ ] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train/val/test generate
- [ ] Documentație actualizată în README + `data/README.md`

---
