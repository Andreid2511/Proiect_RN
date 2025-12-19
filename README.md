# SISTEM INTELIGENT DE PREDICȚIE ȘI OPTIMIZARE A SCHIMBĂRII TREPTELOR DE VITEZĂ

**Student:** Boață Andrei-Darius  
**Grupa:** 633AB
**Facultatea:** Ingineria Industrială și Robotică (FIIR) - UPB  
**Disciplina:** Rețele Neuronale

---

## 📌 Descrierea Proiectului (Overview)

Acest proiect propune o soluție software avansată (**SIA - Sistem de Inteligență Artificială**) destinată optimizării transmisiei automate a unui autovehicul. 

Spre deosebire de cutiile automate clasice, care schimbă vitezele după hărți statice, acest sistem utilizează o **Rețea Neuronală Artificială (Deep Neural Network)** antrenată pe date fizice reale pentru a "înțelege" contextul drumului și intenția șoferului.

### 🎯 Obiectiv Principal: Eficiența Energetică
Scopul central nu este doar confortul, ci **reducerea consumului de combustibil** și a emisiilor în regim urban și extra-urban, prin strategii adaptive:
1.  **Forced ECO:** Detectează traficul urban și forțează schimbarea treptelor la turații joase (2000-2500 RPM), chiar dacă șoferul are un stil ușor agresiv.
2.  **Hill Logic:** Diferențiază corect între un șofer agresiv și nevoia de cuplu pentru urcarea unei pante, evitând subturarea motorului.
3.  **Coasting:** Recunoaște momentele de mers liber și decuplează sarcina pentru a maximiza inerția.

---

## ⚙️ Arhitectura Sistemului

Sistemul este modularizat în 3 componente interconectate, simulate într-un mediu virtual Python:

1.  **Modulul de Achiziție Date & Simulare Fizică:**
    * Simulează fizica unui vehicul clasa C (ex: Vehicul clasa compacta).
    * Generează date sintetice complexe (Pante sinusoidale, Frânări bruște, Accelerații variabile).
    * Include zgomot realist al senzorilor pentru robustete.

2.  **Modulul de Inteligență Artificială (Neural Network):**
    * **Tehnologie:** TensorFlow / Keras.
    * **Arhitectură:** Rețea Deep Feed-Forward (DNN) cu 3 straturi ascunse.
    * **Performanță:** Acuratețe >96% în clasificarea stilurilor (Eco / Normal / Sport).

3.  **Interfața Grafică (Virtual Cockpit):**
    * Dashboard digital în timp real (optimizat pentru latență minimă).
    * Afișează telemetria (Viteză, RPM, Pantă) și decizia AI-ului.
    * Execută schimbarea treptelor pe baza logicii hibride (AI + Fizică).

---

## 📂 Structura și Progresul Proiectului

Proiectul a fost dezvoltat incremental, fiecare etapă fiind documentată separat:

| Etapa | Descriere | Documentație |
| :--- | :--- | :--- |
| **Etapa 3** | Analiza datelor, generarea fizică și preprocesarea. | [Vezi README Etapa 3](./etapa3_analiza_date.md) |
| **Etapa 4** | Definirea arhitecturii software și a Diagramelor de Stare. | [Vezi README Etapa 4](./etapa4_arhitectura_sia.md) |
| **Etapa 5** | Antrenarea modelului Keras, optimizare și validare finală. | [Vezi README Etapa 5](./etapa5_antrenare_model.md) |

---

## 🚀 Cum se rulează proiectul (Quick Start)

### 1. Cerințe de sistem
* Python 3.8+
* Librării: `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `tkinter`, `matplotlib`, `seaborn`,`joblib`.

### 2. Instalare
```bash
pip install -r requirements.txt
```

### 3. Rulare Aplicație (Demo)
* Pentru a vedea bordul digital și a testa AI-ul în timp real:
```bash
python src/app/app_gui.py
```

### 4. Generare date si Re-antrenare Model
* Dacă doriți să regenerați datele și să antrenați un model nou:
```bash
# 1. Generare date noi
python src/data_acquisition/generate_data.py

# 2. Antrenare rețea neuronală
python src/neural_network/train_model.py
```

## 📊 Rezultate Cheie
    * Acuratețe Detecție: 97% pe setul de testare.
    * Timp de Răspuns: Sub 10ms (Inferență CPU optimizată).
    * Impact: Eliminarea schimbărilor inutile de viteze în regim "Stop & Go", reducând uzura și consumul.