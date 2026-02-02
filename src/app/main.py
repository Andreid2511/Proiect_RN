import tkinter as tk
from tkinter import ttk
import numpy as np
import os
import math
import joblib
import warnings

# --- OPTIMIZARI CRITICE PENTRU LAG ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from keras.models import load_model

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.abspath(os.path.join(current_dir, "../../config"))
models_dir = os.path.abspath(os.path.join(current_dir, "../../models"))

model_path = os.path.join(models_dir, "optimized_model.h5") 
scaler_path = os.path.join(config_dir, "preprocessing_params.pkl")

MODEL_LOADED = False
print(f"Attempting to load model from: {model_path}")
print(f"Attempting to load scaler from: {scaler_path}")
try:
    scaler = joblib.load(scaler_path)
    print(f"✅ Scaler loaded successfully")
    model = load_model(model_path)
    print(f"✅ Model loaded successfully")
    MODEL_LOADED = True
    print("✅ Model Keras (CPU Optimized) Incarcat! Lag Eliminat.")
except Exception as e:
    print(f"EROARE MODEL: {e}")
    import traceback
    traceback.print_exc()

class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SIA Virtual Cockpit - Final")
        self.root.geometry("1024x600")
        self.root.configure(bg="#000000")
        
        # Variabile Fizice
        self.speed = 0.0
        self.prev_speed = 0.0 
        self.rpm = 800.0
        self.gear = 1
        self.throttle = 0.0
        self.brake = 0.0
        self.tilt = 0.0
        # Variabile Schimbare Treapta
        self.shift_timer = 0 
        self.pred_history = [] 
        self.ai_status_text = "INIT"
        self.ai_color = "#333333"
        self.strategy_text = "STANDARD"
        self.justification_text = "N/A"
        
        # Variabile pentru optimizare AI
        self.frame_count = 0
        self.last_ai_prediction = 1 # Default Normal
        self.last_confidence = 0.0
        # Statistici sesiune
        self.session_stats = {0: 0, 1: 0, 2: 0}
        self.total_ai_frames = 0
        self.dominant_style = "N/A"
        # Setup UI
        self.setup_dashboard()
        self.update_physics()
    
    def setup_dashboard(self):
        self.canvas = tk.Canvas(self.root, width=1024, height=600, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Layout Static (desenat o singura data la inceput, nu in loop)
        self.canvas.create_line(0, 50, 1024, 50, fill="#333333", width=2)
        self.canvas.create_text(512, 25, text="SIA INTELLIGENT TRANSMISSION", fill="#00a8ff", font=("Arial", 16, "bold"))

        # Fundal Turometru
        self.draw_arc(512, 350, 250, 225, -270, "#1a1a1a", 20) 
        self.canvas.create_text(300, 520, text="0", fill="gray", font=("Arial", 12))
        self.canvas.create_text(730, 520, text="7", fill="red", font=("Arial", 12, "bold"))

        # Texte Statice
        self.canvas.create_text(150, 100, text="INPUT TELEMETRY", fill="gray", font=("Arial", 10))
        self.canvas.create_text(885, 115, text="AI CONTEXT ANALYSIS", fill="gray", font=("Arial", 9))
        self.canvas.create_text(512, 410, text="km/h", fill="gray", font=("Arial", 12))

        # Chenare
        self.canvas.create_rectangle(100, 150, 130, 350, outline="#333333", width=2) # Throttle Box
        self.canvas.create_rectangle(170, 150, 200, 350, outline="#333333", width=2) # Brake Box
        self.canvas.create_rectangle(800, 100, 995, 250, outline="#333333", width=2) # AI Box
        self.canvas.create_rectangle(650, 320, 720, 390, outline="#333333", width=3) # Gear Box

        # Texte Pedale
        self.canvas.create_text(115, 370, text="ACCEL", fill="white", font=("Arial", 8))
        self.canvas.create_text(185, 370, text="BRAKE", fill="white", font=("Arial", 8))
        
        # Controale Jos
        self.controls_frame = tk.Frame(self.root, bg="#111111")
        self.canvas.create_window(512, 600, window=self.controls_frame, width=900, height=80)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Horizontal.TScale", background="#111111", troughcolor="#333333", sliderlength=20)
        # Slider Acceleratie, Frana, Panta
        tk.Label(self.controls_frame, text="ACCELERATIE", fg="#00ff00", bg="#111111").grid(row=0, column=0)
        self.scale_th = ttk.Scale(self.controls_frame, from_=0, to=100, command=self.update_inputs)
        self.scale_th.grid(row=0, column=1, sticky="ew", padx=10, ipadx=50)
        
        tk.Label(self.controls_frame, text="FRANA", fg="#ff0000", bg="#111111").grid(row=0, column=2)
        self.scale_br = ttk.Scale(self.controls_frame, from_=0, to=100, command=self.update_inputs)
        self.scale_br.grid(row=0, column=3, sticky="ew", padx=10, ipadx=50)

        tk.Label(self.controls_frame, text="PANTA", fg="#ffffff", bg="#111111").grid(row=0, column=4)
        self.scale_tilt = ttk.Scale(self.controls_frame, from_=-15, to=15, command=self.update_inputs)
        self.scale_tilt.grid(row=0, column=5, sticky="ew", padx=10, ipadx=50)
        # Buton Reset
        tk.Button(self.controls_frame, text="RESET", bg="red", fg="white", command=self.reset_pedals).grid(row=0, column=6, padx=10)

        self.canvas.create_rectangle(780, 280, 1020, 500, outline="#333333", width=2)
        
        self.canvas.create_text(880, 300, text="LOGICA DE FUNCTIONARE", fill="#00a8ff", font=("Arial", 10, "bold"))
        self.canvas.create_line(790, 315, 970, 315, fill="#333333", width=1)
        # Explicatii
        info_text = (
            "1. INPUT: Regleaza sliderele de jos\n"
            "   (Acceleratie, Frana, Panta).\n\n"
            "2. AI ANALYSIS: Reteaua Neuronala\n"
            "   detecteaza stilul (Eco/Sport)\n"
            "   bazat pe agresivitatea pedalei.\n\n"
            "3. TRANSMISIE: Cutia se adapteaza:\n"
            "   • ECO: Schimba la <2000 RPM\n"
            "   • FORCED_ECO: Tine turația >4000 RPM\n"
            "   • HILL: Retrogradeaza automat"
        )
        self.canvas.create_text(900, 405, text=info_text, fill="#cccccc", font=("Consolas", 8), justify="left")
    def draw_arc(self, x, y, r, start, extent, color, width, tags=None):
        self.canvas.create_arc(x-r, y-r, x+r, y+r, start=start, extent=extent, style=tk.ARC, outline=color, width=width, tags=tags)
        
    def draw_dynamic_ui(self):
        self.canvas.delete("dynamic") 

        # 1. Turometru Activ
        max_rpm = 7000
        angle_extent = (self.rpm / max_rpm) * 270
        rpm_color = "#00ff00" 
        if self.rpm > 4000: rpm_color = "#ffff00" 
        if self.rpm > 6000: rpm_color = "#ff0000" 
        
        self.draw_arc(512, 350, 250, 225, -angle_extent, rpm_color, 20, "dynamic")
        self.canvas.create_text(512, 450, text=f"{int(self.rpm)} RPM", fill=rpm_color, font=("Consolas", 14), tags="dynamic")

        # 2. Vitezometru
        self.canvas.create_text(512, 350, text=f"{int(self.speed)}", fill="white", font=("Impact", 90), tags="dynamic")

        # 3. Gearbox Info
        gear_color = "#00a8ff"
        if "SPORT" in self.strategy_text: gear_color = "#ff0000"
        if "ECO" in self.strategy_text: gear_color = "#00ff00"
        
        self.canvas.create_text(685, 355, text=f"D{self.gear}", fill=gear_color, font=("Arial", 30, "bold"), tags="dynamic")
        self.canvas.create_text(685, 290, text=self.strategy_text, fill=gear_color, font=("Arial", 10, "bold"), tags="dynamic")
        self.canvas.create_text(685, 270, text=self.justification_text, fill="white", font=("Arial", 9, "italic"), tags="dynamic")

        # 4. Pedale (Umplere)
        th_h = (self.throttle / 100) * 200 
        self.canvas.create_rectangle(100, 350 - th_h, 130, 350, fill="#00ff00", outline="", tags="dynamic")
        self.canvas.create_text(115, 140, text=f"{int(self.throttle)}%", fill="#00ff00", tags="dynamic")
        
        br_h = (self.brake / 100) * 200
        self.canvas.create_rectangle(170, 350 - br_h, 200, 350, fill="#ff0000", outline="", tags="dynamic")
        self.canvas.create_text(185, 140, text=f"{int(self.brake)}%", fill="#ff0000", tags="dynamic")

        # 5. Panta (Linia)
        cx, cy = 150, 450
        angle_rad = math.radians(self.tilt) 
        dx = 40 * math.cos(angle_rad)
        dy = 40 * math.sin(angle_rad)
        self.canvas.create_line(cx - 50, cy, cx + 50, cy, fill="#333333", width=2, tags="dynamic") 
        self.canvas.create_line(cx - dx, cy + dy, cx + dx, cy - dy, fill="white", width=4, tags="dynamic") 
        self.canvas.create_text(cx, cy - 50, text=f"Panta: {int(self.tilt)}°", fill="white", tags="dynamic")

        # 6. AI Status
        self.canvas.create_oval(872, 140, 922, 190, fill=self.ai_color, outline=self.ai_color, tags="dynamic")
        self.canvas.create_text(900, 210, text=self.ai_status_text, fill=self.ai_color, font=("Arial", 14, "bold"), tags="dynamic")
        
        conf_len = self.last_confidence * 200 
        self.canvas.create_rectangle(805, 240, 805 + conf_len - 15, 245, fill=self.ai_color, outline="", tags="dynamic")
        self.canvas.create_text(900, 255, text=f"Confidence: {int(self.last_confidence*100)}%", fill="white", font=("Arial", 10, "bold"), tags="dynamic")
        # 7. Dominant Style Sesiune
        stat_color = "#ffffff"
        if "ECO" in self.dominant_style: stat_color = "#00ff00"
        if "SPORT" in self.dominant_style: stat_color = "#ff0000"
        if "NORMAL" in self.dominant_style: stat_color = "#ffa500"
        
        self.canvas.create_text(865, 520, text="SESIUNE CURENTA:", fill="gray", font=("Arial", 9), tags="dynamic")
        self.canvas.create_text(865, 540, text=self.dominant_style, fill=stat_color, font=("Arial", 14, "bold"), tags="dynamic")
    # Reset Controale
    def reset_pedals(self):
        self.scale_th.set(0)
        self.scale_br.set(0)
        self.scale_tilt.set(0)
        self.throttle = 0
        self.brake = 0
    # Update Variabile la schimbare slider
    def update_inputs(self, event=None):
        self.throttle = self.scale_th.get()
        self.brake = self.scale_br.get()
        self.tilt = self.scale_tilt.get()
    # Calculul cuplului motor
    def get_engine_torque(self, rpm):
        if rpm < 1000: return 0.5 
        if rpm < 2000: return 0.5 + (rpm - 1000) * 0.001
        if rpm < 4500: return 1.5
        if rpm < 6000: return 1.2
        return 0.8
    # Logica Schimbare Viteze
    def update_gearbox(self, ratios, gravity, frecare, base_power):
        # Strategia default "NORMAL"
        upshift_rpm = 2500 # Turatie schimbare treapta NORMAL
        self.strategy_text = "STANDARD"
        kickdown_enabled = False # Variabila kickdown

        if self.shift_timer > 0: self.shift_timer -= 1 # Timer schimbare treapta

        if self.throttle > 90: # Activare kickdown
            kickdown_enabled = True
        else:
            kickdown_enabled = False

        downshift_rpm = 1100 # Turatie retrogradare NORMAL 

        if self.tilt > 2: # Ajustare praguri pentru urcare
            upshift_rpm += (self.tilt * 120) 
            downshift_rpm = 1800 
        
        if self.tilt < -8 and self.throttle < 5: # Coborare panta
            self.strategy_text = "HILL DESCENT"
            downshift_rpm = 3000 # Tine treapta mica la coborare
            upshift_rpm = 6000   
        elif self.last_ai_prediction == 2: # Sportiv
            if self.speed < 65 and self.tilt < 5 and kickdown_enabled == False:
                upshift_rpm = 2800; # Imbunatatire consum la viteze mici
                self.strategy_text = "FORCED ECO"
            else:
                upshift_rpm = 5800 # Turatie schimbare treapta SPORT
                self.strategy_text = "SPORT MODE"
        elif self.last_ai_prediction == 0: # ECO
            if self.tilt > 5: # Urcare minim 5 grade
                upshift_rpm = max(upshift_rpm, 3500) 
                self.strategy_text = "HILL CLIMB"
            else:
                upshift_rpm = 2000 # Turatie schimbare treapta ECO
                self.strategy_text = "MAX EFFICIENCY"

        gear_ratio = ratios.get(self.gear, 1.0)
        self.rpm = self.speed * gear_ratio * 30 + (self.throttle * 10) # Simulare usoara incarcare motor
        if self.rpm < 800: self.rpm = 800
        
            # Logica schimbare treapta urcare
        emergency_downshift = (self.rpm < 1100 and self.gear > 1)
        if self.shift_timer == 0 or emergency_downshift:
            # Verificare putere viitoare pentru a decide schimbarea treptei 
            next_ratio = ratios.get(self.gear + 1, 0.6)
            future_rpm = self.speed * next_ratio * 30
            
            # Calcul rezistenta la urcare 
            force_resist = (self.tilt * gravity) + frecare
            future_torque = self.get_engine_torque(future_rpm)
            future_traction = (self.throttle/100.0) * future_torque * next_ratio * base_power
            can_climb = future_traction > (force_resist * 1.1)
            # Prag minim viitor RPM pentru urcare 
            min_future = 1700 
            if ("SPORT" in self.strategy_text):
                min_future = 2500

            if self.rpm > upshift_rpm and self.gear < 8 and future_rpm > min_future and can_climb:
                self.gear += 1
                self.shift_timer = 6
            elif (self.rpm < downshift_rpm or (kickdown_enabled and self.rpm < 3500)) and self.gear > 1:
                self.gear -= 1
                self.shift_timer = 6 
            if self.speed < 2: self.gear = 1   
    
    def update_physics(self):
        # --- CALCUL FIZIC (Ruleaza la fiecare frame) ---
        ratios = {1: 4.7, 2: 3.1, 3: 2.1, 4: 1.7, 5: 1.3, 6: 1.0, 7: 0.8, 8: 0.6}
        gear_ratio = ratios.get(self.gear, 1.0)
        engine_torque = self.get_engine_torque(self.rpm)
        base_power = 0.14
        factor_frana = 2.5
        gravity = 0.018
        frecare = 0.01
        
        eff_throttle = 0 if self.brake > 5 else self.throttle
        push_force = (eff_throttle/100.0) * engine_torque * gear_ratio * base_power
        
        engine_braking = 0
        if self.throttle < 5:
            if self.speed > 15: 
                engine_braking = (self.rpm / 7000) * gear_ratio * 0.10 
        # Rezistenta totala
        delta = push_force - (self.brake/100.0 * factor_frana)
        delta -= engine_braking 
        delta -= (self.tilt * gravity)
        # Rezistenta la rulare 
        delta -= frecare
        
        self.speed += delta
        if self.speed < 0: self.speed = 0
        if self.speed > 260: self.speed = 260

        acceleration = self.speed - self.prev_speed
        self.prev_speed = self.speed 

        # --- AI INFERENCE (Ruleaza la fiecare 5 frame-uri pentru a reduce lag-ul) ---
        self.frame_count += 1
        
        if MODEL_LOADED and self.frame_count % 3 == 0: # Optimizare frecventa (33ms physics, 100ms AI) 
            try:
                # Folosim NumPy direct (fara Pandas DataFrame care e lent) 
                input_arr = np.array([[self.rpm, self.speed, acceleration, self.throttle, self.brake, self.tilt, self.gear]])
                
                # Scalare
                input_scaled = scaler.transform(input_arr)
                
                # Direct Call (__call__) in loc de .predict()
                # .predict e facut pentru batch-uri mari si are overhead imens
                # model(x, training=False) este instant pentru 1 sample
                probs_tensor = model(input_scaled, training=False)
                probs = probs_tensor.numpy()[0]
                # Predictie si logica post-processare
                raw_prediction = np.argmax(probs)
                if self.tilt < -2 and self.throttle < 5:
                    prediction = 0
                else:
                    prediction = raw_prediction

                self.last_confidence = probs[raw_prediction]
                
                self.session_stats[prediction] += 1
                self.total_ai_frames += 1
                # Determinare stil dominant sesiune
                best_style = max(self.session_stats, key=self.session_stats.get)
                style_names = {0: "ECO", 1: "NORMAL", 2: "SPORT"}
                # Calcul procent dominant
                pct = (self.session_stats[best_style] / self.total_ai_frames) * 100
                self.dominant_style = f"{style_names[best_style]} ({int(pct)}%)"
                # Salvare istoric predictii pentru stabilitate
                self.pred_history.append(prediction)
                if len(self.pred_history) > 5: self.pred_history.pop(0) # Buffer mai mic 
                self.last_ai_prediction = max(set(self.pred_history), key=self.pred_history.count)

            except Exception:
                pass
            
            labels = {0: "ECO MODE", 1: "NORMAL", 2: "SPORT / AGRESIV"}
            colors = {0: "#00ff00", 1: "#ffa500", 2: "#ff0000"}

            if self.throttle < 2 and self.brake < 2 and self.speed > 5:
                self.justification_text = "Rulare libera (Inertie)"
            elif self.throttle < 2 and self.brake < 2:
                # Stationare sau mers incet fara pedale 
                self.ai_status_text = labels.get(0, "Unknown")
                self.ai_color = colors.get(0, "white")
                self.justification_text = "Stationare / Relanti"
            else:
                self.ai_status_text = labels.get(self.last_ai_prediction, "Unknown")
                self.ai_color = colors.get(self.last_ai_prediction, "white")
                
                # Justificari detaliate
                if self.last_ai_prediction == 2 and self.throttle > 90: # Kickdown
                    self.justification_text = "Kickdown Activat"
                elif self.last_ai_prediction == 2 and self.speed < 61: # Sport
                    self.justification_text = "Imbunatatire consum (agresiv la viteze mici)"
                    if self.tilt > 3: self.justification_text += " + Panta"
                elif self.last_ai_prediction == 2: # Sport
                    self.justification_text = "Conditii sportive (Viteza mare)"
                    if self.tilt > 3: self.justification_text += " + Panta"
                elif self.last_ai_prediction == 0: # Eco
                    self.justification_text = "Pedala Constanta (Trafic Urban)"
                    if self.tilt > 3: self.justification_text += " + Panta"
                else: # Normal
                    self.justification_text = "Regim de condus mixt"
                    if self.tilt > 3: self.justification_text += " + Panta"
                    
        self.update_gearbox(ratios, gravity, frecare, base_power)
        self.draw_dynamic_ui()
        # 30 FPS constant (33ms) 
        self.root.after(33, self.update_physics)
# Rulare Aplicatie
if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()