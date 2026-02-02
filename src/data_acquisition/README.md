### 📂 README pentru `src/data_acquisition/` (Generatorul)
**Fișier:** `src/data_acquisition/README.md`

# 📡 Modulul 1: Achiziție Date și Simulare Fizică

Acest director conține codul responsabil pentru generarea setului de date necesar antrenării rețelei neuronale.

## 🧪 Metodologie: Simulare vs. Date Reale
Deoarece seturile de date publice nu conțin informații detaliate despre înclinația drumului (`tilt`) corelate cu decizia șoferului, am optat pentru o **generare sintetică 100% originală** bazată pe ecuații fizice.

### Scriptul `generate_data.py`
Acest script simulează dinamica longitudinală a unui vehicul luând în calcul:
1.  **Forța de tracțiune:** Bazată pe curba de cuplu a unui motor pe benzină aspirat.
2.  **Rezistența la înaintare:** Frecarea la rulare.
3.  **Gravitația:** Componenta tangențială pe pante.

## 📊 Date Generate
Scriptul produce fișiere CSV în folderul `data/` cu următoarea structură:

* **Input Features (7):**
    * `rpm`: Turația motorului
    * `speed`: Viteza (km/h)
    * `acceleration`: Accelerația instantanee (m/s²)
    * `throttle`: Poziția pedalei (0-100)
    * `brake`: Poziția frânei (0-100)
    * `tilt`: Panta drumului (grade)
    * `gear`: Treapta de viteză
* **Target Label (1):**
    * `style_label`: 0 (Eco), 1 (Normal), 2 (Sport)

## ⚙️ Execuție
```bash
python src/data_acquisition/generate_data.py
```
## 📤 Output: 
Va genera automat folderele `data/train`, `data/validation`, `data/test` populate cu date(CSV).