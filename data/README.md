### 📂 README pentru `data/` (Dataset)
**Fișier:** `data/README.md`

# 💾 Setul de Date (Dataset)

Acest director conține datele utilizate pentru antrenarea și validarea sistemului SIA. Datele sunt 100% originale, generate prin simulare fizică.

## 📂 Structura Directorului

* **`train/`**: Conține `train.csv` (~70% din date). Folosit pentru ajustarea ponderilor (weights) rețelei.
* **`validation/`**: Conține `validation.csv` (~15% din date). Folosit pentru Early Stopping și reglarea hiperparametrilor.
* **`test/`**: Conține `test.csv` (~15% din date). Folosit EXCLUSIV pentru evaluarea finală a performanței (date nevăzute de model).

## 📝 Dicționar de Date

Fiecare fișier CSV conține următoarele coloane:

| Coloană | Tip | Unitate | Descriere |
| :--- | :--- | :--- | :--- |
| **rpm** | Float | rot/min | Turația motorului (800 - 7000) |
| **speed** | Float | km/h | Viteza vehiculului |
| **acceleration** | Float | m/s² | Derivata vitezei în timp |
| **throttle** | Float | % | Cât de apăsată e pedala de accelerație |
| **brake** | Float | % | Cât de apăsată e pedala de frână |
| **tilt** | Float | grade | Înclinația drumului (+ Urcare, - Coborâre) |
| **gear** | Int | - | Treapta curentă (1-8) |
| **style_label** | Int | - | **TARGET:** 0=Eco, 1=Normal, 2=Sport |

## ⚖️ Distribuția Claselor
Setul de date este **balansat**, conținând un număr egal de eșantioane pentru fiecare stil de condus (aprox. 60.000 eșantioane per clasă în total), pentru a evita bias-ul rețelei neuronale.