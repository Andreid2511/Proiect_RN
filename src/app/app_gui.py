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
config_dir = os.path.abspath(os.path.join(current_dir, "../config"))
models_dir = os.path.abspath(os.path.join(current_dir, "../models"))

model_path = os.path.join(models_dir, "trained_model.keras") 
scaler_path = os.path.join(config_dir, "scaler.pkl")

MODEL_LOADED = False
try:
    scaler = joblib.load(scaler_path)
    model = load_model(model_path)
    MODEL_LOADED = True
    print("✅ Model Keras (CPU Optimized) Incarcat! Lag Eliminat.")
except Exception as e:
    print(f"EROARE MODEL: {e}")

class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SIA Virtual Cockpit - Ultra Fast")
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
        
        self.shift_timer = 0 
        self.pred_history = [] 
        self.ai_status_text = "INIT"
        self.ai_color = "#333333"
        self.strategy_text = "STANDARD"
        
        # Variabile pentru optimizare AI
        self.frame_count = 0
        self.last_ai_prediction = 1 # Default Normal
        self.last_confidence = 0.0

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
        self.canvas.create_text(280, 480, text="0", fill="gray", font=("Arial", 12))
        self.canvas.create_text(744, 480, text="7", fill="red", font=("Arial", 12, "bold"))

        # Texte Statice
        self.canvas.create_text(150, 100, text="INPUT TELEMETRY", fill="gray", font=("Arial", 10))
        self.canvas.create_text(865, 115, text="AI CONTEXT ANALYSIS", fill="gray", font=("Arial", 9))
        self.canvas.create_text(512, 400, text="km/h", fill="gray", font=("Arial", 12))

        # Chenare
        self.canvas.create_rectangle(100, 150, 130, 350, outline="#333333", width=2) # Throttle Box
        self.canvas.create_rectangle(170, 150, 200, 350, outline="#333333", width=2) # Brake Box
        self.canvas.create_rectangle(750, 100, 980, 250, outline="#333333", width=2) # AI Box
        self.canvas.create_rectangle(650, 320, 720, 390, outline="#333333", width=3) # Gear Box

        # Texte Pedale
        self.canvas.create_text(115, 370, text="ACCEL", fill="white", font=("Arial", 8))
        self.canvas.create_text(185, 370, text="BRAKE", fill="white", font=("Arial", 8))
        
        # Controale Jos
        self.controls_frame = tk.Frame(self.root, bg="#111111")
        self.canvas.create_window(512, 550, window=self.controls_frame, width=900, height=80)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Horizontal.TScale", background="#111111", troughcolor="#333333", sliderlength=20)

        tk.Label(self.controls_frame, text="ACCELERATIE", fg="#00ff00", bg="#111111").grid(row=0, column=0)
        self.scale_th = ttk.Scale(self.controls_frame, from_=0, to=100, command=self.update_inputs)
        self.scale_th.grid(row=0, column=1, sticky="ew", padx=10, ipadx=50)
        
        tk.Label(self.controls_frame, text="FRANA", fg="#ff0000", bg="#111111").grid(row=0, column=2)
        self.scale_br = ttk.Scale(self.controls_frame, from_=0, to=100, command=self.update_inputs)
        self.scale_br.grid(row=0, column=3, sticky="ew", padx=10, ipadx=50)

        tk.Label(self.controls_frame, text="PANTA", fg="#ffffff", bg="#111111").grid(row=0, column=4)
        self.scale_tilt = ttk.Scale(self.controls_frame, from_=-15, to=15, command=self.update_inputs)
        self.scale_tilt.grid(row=0, column=5, sticky="ew", padx=10, ipadx=50)

        tk.Button(self.controls_frame, text="RESET", bg="red", fg="white", command=self.reset_pedals).grid(row=0, column=6, padx=10)

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
        self.canvas.create_oval(840, 140, 890, 190, fill=self.ai_color, outline=self.ai_color, tags="dynamic")
        self.canvas.create_text(865, 210, text=self.ai_status_text, fill=self.ai_color, font=("Arial", 14, "bold"), tags="dynamic")
        
        conf_len = self.last_confidence * 200 
        self.canvas.create_rectangle(750, 240, 750 + conf_len, 245, fill=self.ai_color, outline="", tags="dynamic")

    def reset_pedals(self):
        self.scale_th.set(0)
        self.scale_br.set(0)
        self.throttle = 0
        self.brake = 0

    def update_inputs(self, event=None):
        self.throttle = self.scale_th.get()
        self.brake = self.scale_br.get()
        self.tilt = self.scale_tilt.get()

    def get_engine_torque(self, rpm):
        if rpm < 1000: return 0.5 
        if rpm < 2000: return 0.5 + (rpm - 1000) * 0.001
        if rpm < 4500: return 1.5
        if rpm < 6000: return 1.2
        return 0.8

    def update_physics(self):
        # --- CALCUL FIZIC (Ruleaza la fiecare frame) ---
        ratios = {1: 4.7, 2: 3.1, 3: 2.1, 4: 1.7, 5: 1.3, 6: 1.0, 7: 0.8, 8: 0.6}
        gear_ratio = ratios.get(self.gear, 1.0)
        engine_torque = self.get_engine_torque(self.rpm)
        base_power = 0.3
        factor_frana = 2.5
        gravity = 0.09
        frecare = 0.06
        
        eff_throttle = 0 if self.brake > 5 else self.throttle
        push_force = (eff_throttle/100.0) * engine_torque * gear_ratio * base_power
        
        engine_braking = 0
        if self.throttle < 5: 
            engine_braking = (self.rpm / 1000) * gear_ratio * 0.15 

        delta = push_force - (self.brake/100.0 * factor_frana)
        delta -= engine_braking 
        delta -= (self.tilt * gravity)
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
                
                prediction = np.argmax(probs)
                self.last_confidence = probs[prediction]
                
                self.pred_history.append(prediction)
                if len(self.pred_history) > 5: self.pred_history.pop(0) # Buffer mai mic
                self.last_ai_prediction = max(set(self.pred_history), key=self.pred_history.count)

            except Exception:
                pass

        # Update UI Labels based on cached prediction
        labels = {0: "ECO MODE", 1: "NORMAL", 2: "SPORT / AGRESIV"}
        colors = {0: "#00ff00", 1: "#ffa500", 2: "#ff0000"}
        
        # Coasting Override
        if self.throttle < 5 and self.brake < 5:
            self.last_ai_prediction = 0
            self.last_confidence = 1.0
            
        self.ai_status_text = labels.get(self.last_ai_prediction, "Unknown")
        self.ai_color = colors.get(self.last_ai_prediction, "white")
        
        if self.throttle < 5 and self.brake < 5 and self.speed > 10:
            self.ai_status_text = "COASTING"
            self.ai_color = "#00a8ff"

        # --- Gearbox Logic ---
        upshift_rpm = 3000 
        self.strategy_text = "STANDARD"
        
        if self.shift_timer > 0: self.shift_timer -= 1
        
        downshift_rpm = 1100 
        if self.tilt > 2:
            upshift_rpm += (self.tilt * 120) 
            downshift_rpm = 1800 
        
        if self.tilt < -2 and self.throttle < 5:
            self.strategy_text = "HILL DESCENT"
            downshift_rpm = 3000 
            upshift_rpm = 6000   
        elif self.last_ai_prediction == 2: 
            if self.speed < 60: 
                upshift_rpm = 2500; 
                self.strategy_text = "FORCED ECO"
            else:
                upshift_rpm = 5800
                self.strategy_text = "SPORT MODE"
        elif self.last_ai_prediction == 0: 
            if self.tilt > 5:
                upshift_rpm = max(upshift_rpm, 3500)
                self.strategy_text = "HILL CLIMB"
            else:
                upshift_rpm = 2000
                self.strategy_text = "MAX EFFICIENCY"

        self.rpm = self.speed * gear_ratio * 30 + (self.throttle * 10)
        if self.rpm < 800: self.rpm = 800
        
        emergency_downshift = (self.rpm < 1100 and self.gear > 1)
        
        if self.shift_timer == 0 or emergency_downshift:
            next_ratio = ratios.get(self.gear + 1, 0.6)
            future_rpm = self.speed * next_ratio * 30
            
            force_resist = (self.tilt * gravity) + frecare
            future_torque = self.get_engine_torque(future_rpm)
            future_traction = (self.throttle/100.0) * future_torque * next_ratio * base_power
            can_climb = future_traction > (force_resist * 1.1)

            min_future = 1300 if ("FORCED ECO" in self.strategy_text) else 1300

            if self.rpm > upshift_rpm and self.gear < 8 and future_rpm > min_future and can_climb:
                self.gear += 1
                self.shift_timer = 10 
            elif (self.rpm < downshift_rpm or (self.throttle > 85 and self.rpm < 3500)) and self.gear > 1:
                self.gear -= 1
                self.shift_timer = 10 

        if self.speed < 2: self.gear = 1

        self.draw_dynamic_ui()
        # 30 FPS constant (33ms)
        self.root.after(33, self.update_physics)

if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()