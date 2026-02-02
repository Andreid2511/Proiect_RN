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
`
Sistemul este modularizat în 3 componente interconectate:

1.  **Modulul de Achiziție Date & Simulare Fizică (`src/data_acquisition`):**
    * Simulează fizica unui vehicul clasa C.
    * Generează date sintetice complexe (Pante, Frânări, Accelerații).
    * [Detalii complete aici](./src/data_acquisition/README.md)

2.  **Modulul de Inteligență Artificială (`src/neural_network`):**
    * **Tehnologie:** TensorFlow / Keras.
    * **Arhitectură:** Rețea Deep Feed-Forward (DNN) cu 3 straturi ascunse.
    * **Performanță:** Acuratețe >98% în clasificarea stilurilor.
    * [Detalii complete aici](./src/neural_network/README.md)

3.  **Interfața Grafică - Virtual Cockpit (`src/app`):**
    * Dashboard digital în timp real.
    * Afișează telemetria și decizia AI-ului.
    * [Detalii complete aici](./src/app/README.md)

---

## 📂 Structura și Progresul Proiectului

Proiectul a fost dezvoltat incremental, fiecare etapă fiind documentată separat:

| Etapa | Descriere | Documentație |
| :--- | :--- | :--- |
| **Etapa 3** | Analiza datelor, generarea fizică și preprocesarea. | [Vezi README Etapa 3](./docs/etapa3_analiza_date.md) |
| **Etapa 4** | Definirea arhitecturii software și a Diagramelor de Stare. | [Vezi README Etapa 4](./docs/etapa4_arhitectura_sia.md) |
| **Etapa 5** | Antrenarea modelului Keras, optimizare și validare finală. | [Vezi README Etapa 5](./docs/etapa5_antrenare_model.md) |
| **Etapa 6** | Analiza performanței, optimizare finală și concluzii. | [Vezi README Etapa 6](./docs/etapa6_optimizare_concluzii.md) |

---

## 🚀 Cum se rulează proiectul (Quick Start)

### 1. Cerințe de sistem
* Python 3.8+
* Dependențe: Vezi `requirements.txt`

### 2. Instalare
```bash
git clone https://github.com/Andreid2511/Proiect_RN.git
cd Proiect_RN
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```