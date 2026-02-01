# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Boață Andrei-Darius
**Data:** 10/12/2025  

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3** pentru proiectul **"Sistem de Recunoaștere a Stilului de Condus și Adaptare Inteligentă a Transmisiei"**. Scopul etapei este pregătirea unui set de date sintetic, dar realist fizic, care să permită antrenarea unui model RN capabil să clasifice stilul de condus (Eco, Normal, Agresiv) indiferent de condițiile de drum (pantă, limită de viteză).

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
├── README.md
├── docs/
│   └── datasets/          # grafice distribuție RPM vs Speed
├── data/
│   ├── raw/               # date brute (generate de simulator)
│   ├── processed/         # date curățate (dacă este cazul)
│   ├── train/             # set de instruire (70%)
│   ├── validation/        # set de validare (15%)
│   └── test/              # set de testare (15%)
├── src/
│   ├── preprocessing/     # scalare date (StandardScaler)
│   ├── data_acquisition/  # script generator (generate_data.py)
│   └── neural_network/    # implementarea RN (train_model.py)
├── config/                # fișiere model salvat (.pkl)
└── requirements.txt       # pandas, numpy, scikit-learn, joblib, tkinter
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Date generate programatic prin simulare fizică avansată.
* **Modul de achiziție:** ☐ Senzori reali / ☑ Simulare / ☐ Fișier extern / ☑ Generare programatică
* **Perioada / condițiile colectării:** Datele simulează comportamentul unui vehicul clasa B (ex: VW Polo 70-90CP) cu cutie automată ZF 8HP, în scenarii variate: Urban (Stop&Go), Extra-urban (serpentine) și Autostradă.
* **Frecvență eșantionare:** 30 Hz (DT = 0.033s), sincronizat cu rata de refresh a aplicației finale.

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** **180,000** (3 stiluri x 60,000 eșantioane).
* **Număr de caracteristici (features):** **7 Features de intrare** + 1 Target.
* **Tipuri de date:** ☑ Numerice / ☑ Categoriale (Target) / ☑ Temporale / ☐ Imagini
* **Format fișiere:** ☑ CSV / ☐ TXT / ☐ JSON / ☐ PNG / ☐ Altele: [...]

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| rpm | numeric | rot/min | Turația motorului | 800 – 7000 |
| speed | numeric | km/h | Viteza vehiculului | 0 – 260 |
| acceleration | numeric | m/s² | Accelerația vehiculului (derivata vitezei) | -5 ... +5 |
| throttle | numeric | % | Poziția pedalei de accelerație | 0 – 100 |
| brake | numeric | % | Poziția pedalei de frână | 0 – 100 |
| tilt | numeric | grade | Înclinația drumului (rampă/pantă) | -15 ... +15 |
| gear | numeric | - | Treapta de viteză curentă | 1 – 8 |
| style_label | categorial | - | Eticheta stilului (Eco/Normal/Sport) | {0, 1, 2} |

**Fișier recomandat:** `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Medie și deviație standard:** Calculate pentru RPM și Speed pentru a verifica realismul fizic (ex: RPM mediu ~2000 pentru Eco, ~4000 pentru Sport).
* **Distribuții:** Histogramele arată o distribuție bimodală a vitezei (opriri dese în urban vs viteză constantă pe autostradă).
* **Identificarea outlierilor:** Valori extreme de accelerație pe pante abrupte au fost verificate pentru consistență fizică.

### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă:** 0% (datele sunt generate controlat).
* **Consistență:** S-a verificat corelația RPM-Viteză-Treaptă (rapoartele de transmisie fixe).
* **Corelații:** Corelație puternică între `throttle` și `style_label`, dar moderată de `tilt` (panta).

### 3.3 Probleme identificate

* **Provocare:** Inițial, urcarea unui deal cu accelerația la maxim era clasificată greșit ca "Agresiv".
* **Soluție:** S-a introdus variabila `tilt` în setul de date și logica de compensare în generator, astfel încât "Pedală mare + Viteză mică + Pantă mare" = Normal, nu Agresiv.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminare duplicatelor:** Nu a fost necesar.
* **Tratarea zgomotului:** S-a introdus o funcție de "Smoothing" (inerție) la pedale în generator pentru a evita mișcările bruște nerealiste (jitter).

### 4.2 Transformarea caracteristicilor

* **Normalizare (StandardScaler):** Aplicată tuturor caracteristicilor numerice (`rpm`, `speed`, `acceleration`, `throttle`, `brake`, `tilt`, `gear`) pentru a aduce valorile la o scară comună (medie 0, deviație 1), esențială pentru convergența Rețelei Neuronale MLP.
* **Encoding:** Target-ul `style_label` este deja numeric (0, 1, 2).

### 4.3 Structurarea seturilor de date

**Împărțire realizată (Random Sample):**
* 70% – train (~126,000 samples)
* 15% – validation (~27,000 samples)
* 15% – test (~27,000 samples)

**Principii respectate:**
* **Shuffling:** Datele au fost amestecate complet (`df.sample(frac=1)`) înainte de salvare pentru a elimina dependența temporală, obligând rețeaua să învețe corelațiile instantanee dintre senzori, nu ordinea secvențială.
* **Stratificare:** S-a asigurat prezența echilibrată a tuturor celor 3 stiluri (câte 60k sample-uri fiecare inițial).

### 4.4 Salvarea rezultatelor preprocesării

* Datele sunt salvate în format CSV în folderele `data/train`, `data/validation`, `data/test`.
* Obiectul de scalare (`scaler.pkl`) este salvat în `config/` pentru a fi folosit ulterior în aplicația live.

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – (Nu se aplică, datele sunt generate direct procesate)
* `data/train/train.csv` – Set antrenare
* `data/validation/validation.csv` – Set validare
* `data/test/test.csv` – Set testare
* `src/data_acquisition/generate_data.py` – Codul generatorului fizic
* `src/neural_network/train_model.py` – Codul de preprocesare și antrenare

---

##  6. Stare Etapă (de completat de student)

- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [x] Date preprocesate (Generate cu logică Smooth & Shuffled)
- [x] Seturi train/val/test generate
- [x] Documentație actualizată în README + `data/README.md`

---