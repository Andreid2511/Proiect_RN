### 📂 README pentru `src/app/` (Interfața Grafică)
**Fișier:** `src/app/README.md`

# 🖥️ Modulul 3: Interfața Grafică (Virtual Cockpit)

Acest modul implementează aplicația principală a sistemului SIA, simulând un bord digital de autovehicul în timp real.

## 📋 Descriere
Aplicația este construită folosind biblioteca **Tkinter** și servește drept punct de integrare pentru toate componentele proiectului:
1.  **Motorul Fizic:** Rulează în buclă la 30 FPS pentru a calcula viteza, RPM-ul și forțele.
2.  **Motorul AI:** Încarcă modelul antrenat (`.h5`) și efectuează inferențe la fiecare cadru.
3.  **Logica Hibridă:** Combină predicția AI cu reguli de siguranță (ex: Kickdown, Hill Descent).

## 🎮 Funcționalități UI
* **Turometru & Vitezometru:** Ceasuri analogice desenate dinamic.
* **Indicator Treaptă:** Afișează treapta curentă (P, R, N, D1-D8).
* **Panou Control:** Slidere pentru a simula:
    * Pedala de accelerație (0-100%)
    * Pedala de frână (0-100%)
    * Înclinația drumului (Pante +/- 15 grade)
* **Feedback Vizual:**
    * *AI Prediction + Confidence:* Ce stil a detectat rețeaua.
    * *Justification Text:* Ce inseamna stilul curent si de ce a aparut
    * *Informatii Sesiune:* Stilul dominant

## 🚀 Rulare
Din folderul rădăcină al proiectului:
```bash
python src/app/app_gui.py
```

## 🔧 Dependențe
- tkinter (inclus în Python standard)
- tensorflow (pentru încărcarea modelului)
- numpy
- joblib (pentru încărcarea scaler-ului)