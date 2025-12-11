import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np
import pandas as pd  # <--- AM ADAUGAT PANDAS
import os

# --- 1. SETUP CAI SI MODEL ---
current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.abspath(os.path.join(current_dir, "../config"))
model_path = os.path.join(config_dir, "driver_model.pkl")
scaler_path = os.path.join(config_dir, "scaler.pkl")

# Incarcam AI-ul Antrenat
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("AI Incarcat cu succes!")
except:
    print("EROARE: Nu gasesc modelul! Ruleaza intai train_model.py")
    exit()

class ModernCarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SIA Intelligent Dashboard - Driver Context Recognition")
        self.root.geometry("900x650")
        self.root.configure(bg="#1e272e") # Dark Blue-Grey Theme
        
        # --- VARIABILE FIZICE ---
        self.speed = 0.0
        self.rpm = 800.0
        self.gear = 1
        self.throttle = 0.0
        self.brake = 0.0
        self.tilt = 0.0
        
        # Istoric pentru netezirea predictiei AI
        self.pred_history = [] 
        self.aggression_timer = 0
        self.setup_ui()
        self.update_physics()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurare Stiluri Custom
        style.configure("TScale", background="#1e272e", troughcolor="#485460", sliderlength=20)
        style.configure("TProgressbar", thickness=30) 

        # --- HEADER ---
        header_frame = tk.Frame(self.root, bg="#0fb9b1", height=60)
        header_frame.pack(fill="x")
        lbl_title = tk.Label(header_frame, text="SIA - SMART ADAPTIVE GEARBOX", font=("Segoe UI", 20, "bold"), bg="#0fb9b1", fg="white")
        lbl_title.pack(pady=10)

        # --- ZONA PRINCIPALA (AI + DASHBOARD) ---
        main_frame = tk.Frame(self.root, bg="#1e272e")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Stanga: AI Prediction Box
        ai_frame = tk.Frame(main_frame, bg="black", bd=2, relief="sunken", width=300)
        ai_frame.pack(side="left", fill="y", padx=(0, 20))
        
        tk.Label(ai_frame, text="DRIVER STYLE (AI)", font=("Arial", 12), bg="black", fg="#bdc3c7").pack(pady=(20, 5))
        self.lbl_ai_text = tk.Label(ai_frame, text="ANALYZING...", font=("Arial", 22, "bold"), bg="black", fg="gray", wraplength=280)
        self.lbl_ai_text.pack(pady=10)
        
        self.lbl_ai_conf = tk.Label(ai_frame, text="Confidence: --%", font=("Consolas", 14), bg="black", fg="#2ecc71")
        self.lbl_ai_conf.pack(pady=5)
        
        tk.Label(ai_frame, text="GEARBOX STRATEGY", font=("Arial", 12), bg="black", fg="#bdc3c7").pack(pady=(40, 5))
        self.lbl_strategy = tk.Label(ai_frame, text="STANDARD", font=("Arial", 18, "bold"), bg="black", fg="white")
        self.lbl_strategy.pack(pady=5)
        
        # Dreapta: Ceasuri (Dashboard)
        dash_frame = tk.Frame(main_frame, bg="#1e272e")
        dash_frame.pack(side="left", fill="both", expand=True)

        # SPEED
        self.lbl_speed_val = tk.Label(dash_frame, text="0", font=("Impact", 70), bg="#1e272e", fg="#4bcffa")
        self.lbl_speed_val.pack()
        tk.Label(dash_frame, text="km/h", font=("Arial", 16), bg="#1e272e", fg="#d2dae2").pack()
        
        # RPM Bar
        tk.Label(dash_frame, text="RPM", font=("Arial", 10, "bold"), bg="#1e272e", fg="#ff5e57").pack(anchor="w", pady=(30, 0))
        self.prog_rpm = ttk.Progressbar(dash_frame, orient="horizontal", length=400, mode="determinate", maximum=7000)
        self.prog_rpm.pack(fill="x")
        self.lbl_rpm_text = tk.Label(dash_frame, text="800 RPM", font=("Consolas", 14), bg="#1e272e", fg="#ff5e57")
        self.lbl_rpm_text.pack(anchor="e")

        # GEAR Box
        self.lbl_gear_box = tk.Label(dash_frame, text="D1", font=("Arial", 50, "bold"), bg="#1e272e", fg="#f1c40f")
        self.lbl_gear_box.place(relx=0.80, rely=0.1) 

        # --- ZONA CONTROALE (JOS) ---
        ctrl_frame = tk.Frame(self.root, bg="#485460", bd=2, relief="raised")
        ctrl_frame.pack(fill="x", side="bottom")

        # Acceleratie
        tk.Label(ctrl_frame, text="ACCELERATIE", font=("Arial", 10, "bold"), bg="#485460", fg="#2ecc71").grid(row=0, column=0, padx=10, pady=10)
        self.scale_th = ttk.Scale(ctrl_frame, from_=0, to=100, orient='horizontal', command=self.update_inputs)
        self.scale_th.grid(row=0, column=1, sticky="ew", padx=10, ipadx=100)
        self.lbl_th_val = tk.Label(ctrl_frame, text="0%", font=("Consolas", 12, "bold"), bg="#485460", fg="#2ecc71", width=5)
        self.lbl_th_val.grid(row=0, column=2, padx=10)

        # Frana
        tk.Label(ctrl_frame, text="FRANA", font=("Arial", 10, "bold"), bg="#485460", fg="#ff5e57").grid(row=1, column=0, padx=10, pady=10)
        self.scale_br = ttk.Scale(ctrl_frame, from_=0, to=100, orient='horizontal', command=self.update_inputs)
        self.scale_br.grid(row=1, column=1, sticky="ew", padx=10, ipadx=100)
        self.lbl_br_val = tk.Label(ctrl_frame, text="0%", font=("Consolas", 12, "bold"), bg="#485460", fg="#ff5e57", width=5)
        self.lbl_br_val.grid(row=1, column=2, padx=10)

        # Buton Reset
        btn_reset = tk.Button(ctrl_frame, text="ELIBEREAZA PEDALELE\n(Idle)", bg="#ff3f34", fg="white", font=("Arial", 10, "bold"), command=self.reset_pedals)
        btn_reset.grid(row=0, column=3, rowspan=2, padx=20, sticky="ns")

        # Tilt
        self.scale_tilt = ttk.Scale(ctrl_frame, from_=-15, to=15, orient='horizontal', command=self.update_inputs)
        self.scale_tilt.grid(row=2, column=1, sticky="ew", padx=10, pady=5)
        self.lbl_tilt_val = tk.Label(ctrl_frame, text="Panta: 0°", bg="#485460", fg="white")
        self.lbl_tilt_val.grid(row=2, column=0, columnspan=3)

    def reset_pedals(self):
        """Resetare instantanee la relanti"""
        self.scale_th.set(0)
        self.scale_br.set(0)
        self.throttle = 0
        self.brake = 0
        self.lbl_th_val.config(text="0%")
        self.lbl_br_val.config(text="0%")

    def update_inputs(self, event=None):
        self.throttle = self.scale_th.get()
        self.brake = self.scale_br.get()
        self.tilt = self.scale_tilt.get()
        
        self.lbl_th_val.config(text=f"{int(self.throttle)}%")
        self.lbl_br_val.config(text=f"{int(self.brake)}%")
        self.lbl_tilt_val.config(text=f"Panta: {int(self.tilt)}°")

    def update_physics(self):
        # --- 1. CALCUL FIZIC (Delta Speed) ---
        factor_accel = 1.2
        factor_frana = 1.5
        gravity = 0.06
        frecare = 0.05
        
        eff_throttle = 0 if self.brake > 5 else self.throttle
        
        delta = (eff_throttle/100.0 * factor_accel) - (self.brake/100.0 * factor_frana)
        delta -= (self.tilt * gravity)
        delta -= frecare
        
        self.speed += delta
        if self.speed < 0: self.speed = 0
        if self.speed > 260: self.speed = 260

        # --- 2. AI PREDICTION ---
        ai_style = 1 
        try:
            input_data = np.array([[self.rpm, self.speed, self.throttle, self.brake, self.tilt, self.gear]])
            input_df = pd.DataFrame(
                input_data,
                columns=['rpm', 'speed', 'throttle', 'brake', 'tilt', 'gear']
            )
            input_scaled = scaler.transform(input_df)
            probs = model.predict_proba(input_scaled)[0]
            prediction = np.argmax(probs)
            confidence = probs[prediction]
            
            self.pred_history.append(prediction)
            if len(self.pred_history) > 10: self.pred_history.pop(0)
            ai_style = max(set(self.pred_history), key=self.pred_history.count)

            if ai_style == 2:
                self.aggression_timer = 50
            
            # Daca avem timer activ, fortam stilul sa ramana Agresiv
            if self.aggression_timer > 0:
                ai_style = 2
                self.aggression_timer -= 1
                
            labels = {0: "ECO / CALM", 1: "NORMAL", 2: "SPORT / AGRESIV"}
            colors = {0: "#05c46b", 1: "#ffa801", 2: "#ff3f34"}
            
            self.lbl_ai_text.config(text=labels[ai_style], fg=colors[ai_style])
            self.lbl_ai_conf.config(text=f"Confidence: {confidence*100:.0f}%")
        except:
            pass

        # --- 3. SMART GEARBOX LOGIC (FIXED) ---
        ratios = {1: 4.7, 2: 3.1, 3: 2.1, 4: 1.7, 5: 1.3, 6: 1.0, 7: 0.8, 8: 0.6}
        
        upshift_rpm = 3000 
        strategy_text = "STANDARD"
        
        if ai_style == 2: # Agresiv
            if self.speed < 60: 
                upshift_rpm = 2200 
                strategy_text = "FORCED ECO (City)"
                self.lbl_gear_box.config(fg="#05c46b")
            else:
                upshift_rpm = 5800
                strategy_text = "SPORT MODE"
                self.lbl_gear_box.config(fg="#ff3f34")
        elif ai_style == 0: # Eco
            upshift_rpm = 2000
            strategy_text = "MAX EFFICIENCY"
            self.lbl_gear_box.config(fg="#05c46b")
        else:
            self.lbl_gear_box.config(fg="#f1c40f")

        self.lbl_strategy.config(text=strategy_text)

        # Calcul RPM Curent
        ratio = ratios.get(self.gear, 1.0)
        self.rpm = self.speed * ratio * 30 + (self.throttle * 25)
        if self.rpm < 800: self.rpm = 800
        
        # --- LOGICA DE SCHIMBARE BLINDATĂ ---
        
        # 1. Calculam RPM-ul MECANIC (fara efectul de pedala) pentru treapta urmatoare
        # Asta ne spune daca motorul ar "muri" sau ar fi subturat daca schimbam
        next_ratio = ratios.get(self.gear + 1, 0.6)
        future_mechanical_rpm = self.speed * next_ratio * 30
        
        # CONDITIE UPSHIFT:
        # 1. RPM actual > Limita stabilita de AI
        # 2. Avem trepte disponibile (<8)
        # 3. (NOU) Daca schimbam, RPM-ul rezultat e sanatos (>1100)
        if self.rpm > upshift_rpm and self.gear < 8 and future_mechanical_rpm > 1100:
            self.gear += 1
            
        # CONDITIE DOWNSHIFT:
        # 1. RPM a scazut sub 1100 (moare motorul)
        # 2. SAU Kickdown: Pedala > 80% si avem loc de turatie (<3500)
        elif (self.rpm < 1100 or (self.throttle > 80 and self.rpm < 3500)) and self.gear > 1:
            self.gear -= 1

        # --- 4. UPDATE UI ---
        self.lbl_speed_val.config(text=f"{int(self.speed)}")
        self.prog_rpm['value'] = self.rpm
        self.lbl_rpm_text.config(text=f"{int(self.rpm)} RPM")
        
        if self.rpm > 5500: self.lbl_rpm_text.config(fg="red")
        else: self.lbl_rpm_text.config(fg="#ff5e57")
            
        self.lbl_gear_box.config(text=f"D{self.gear}")

        self.root.after(100, self.update_physics)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernCarApp(root)
    root.mainloop()