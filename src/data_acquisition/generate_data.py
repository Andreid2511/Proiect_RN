import pandas as pd
import numpy as np
import random
import os

def simulate_behavior(num_samples, style):
    data = []
    
    # Init
    speed = 0.0
    rpm = 800.0
    gear = 1
    throttle = 0.0
    brake = 0.0
    
    # --- CONSTANTE FIZICE (Configuratie VW Polo) ---
    ratios = {1: 4.7, 2: 3.1, 3: 2.1, 4: 1.7, 5: 1.3, 6: 1.0, 7: 0.8, 8: 0.6}
    factor_accel = 1.2   
    factor_frana = 2.0
    gravity = 0.06       
    frecare = 0.04
    
    current_speed_limit = 50 
    limit_timer = 0
    
    for i in range(num_samples):
        time = i * 0.1
        
        limit_timer += 1
        if limit_timer > random.randint(300, 600):
            current_speed_limit = random.choice([30, 50, 60, 90, 100])
            limit_timer = 0

        # --- GENERARE PANTA REALISTĂ (-15 ... +15 grade) ---
        # 15 grade inseamna o panta de 27% (foarte abrupta, dar posibila)
        tilt = np.sin(i/600) * 15
        
        # --- 1. COMPORTAMENTUL ȘOFERULUI ---
        target_throttle = 0
        target_brake = 0
        
        # Compensare pentru deal: Cu cat e panta mai mare, cu atat "ai voie" sa apesi mai tare
        # fara sa fii considerat agresiv.
        hill_compensation = max(0, tilt * 3.0) 
        
        if style == 0: # ECO / CALM
            base_throttle = random.uniform(10, 40)
            allowed_throttle = base_throttle + hill_compensation
            
            if speed < current_speed_limit:
                if random.random() < 0.05: target_throttle = allowed_throttle
                else: target_throttle = throttle
            else:
                target_throttle = 0
                if speed > current_speed_limit + 5: target_brake = 10
            
            smooth_factor = 1.5

        elif style == 2: # SPORT / AGRESIV
            base_throttle = random.uniform(60, 100)
            
            if speed < current_speed_limit + 15:
                if random.random() < 0.1: target_throttle = base_throttle
                else: target_throttle = throttle
            else:
                target_throttle = 0
                target_brake = 50
            
            smooth_factor = 6.0

        else: # NORMAL
            base_throttle = random.uniform(20, 60)
            allowed_throttle = base_throttle + hill_compensation
            
            if speed < current_speed_limit + 5:
                if random.random() < 0.05: target_throttle = allowed_throttle
                else: target_throttle = throttle
            else:
                target_throttle = 0
                target_brake = 20
            
            smooth_factor = 3.0

        # Limite Pedale
        if target_throttle > throttle: throttle += smooth_factor
        elif target_throttle < throttle: throttle -= smooth_factor
        
        brake = target_brake
        
        if throttle < 0: throttle = 0
        if throttle > 100: throttle = 100

        # --- 2. FIZICA ---
        air_drag = (speed * speed) * 0.00005
        
        delta = (throttle/100.0 * factor_accel) - (brake/100.0 * factor_frana)
        delta -= (tilt * gravity)
        delta -= frecare
        delta -= air_drag
        
        speed += delta
        if speed < 0: speed = 0
        
        # --- 3. CUTIE ---
        ratio = ratios.get(gear, 1.0)
        rpm = speed * ratio * 30 + (throttle * 25)
        if rpm < 800: rpm = 800
        
        upshift_point = 2800 + (tilt * 50) # Schimba tarziu la deal
        if throttle > 80: upshift_point = 5500
        
        if rpm > upshift_point and gear < 8: gear += 1
        elif (rpm < 1100 or (throttle > 80 and rpm < 3000)) and gear > 1: gear -= 1
            
        data.append([rpm, speed, throttle, brake, tilt, time, gear, style])

    return data

# --- MAIN ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, "../../data"))

print("Generez date de antrenare pentru stiluri de condus...")
os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)

cols = ['rpm', 'speed', 'throttle', 'brake', 'tilt', 'time', 'gear', 'style_label']

all_data = []
all_data.extend(simulate_behavior(20000, 0)) 
all_data.extend(simulate_behavior(20000, 1)) 
all_data.extend(simulate_behavior(20000, 2)) 

df = pd.DataFrame(all_data, columns=cols)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df)
train_df = df.iloc[:int(n*0.7)]
val_df = df.iloc[int(n*0.7):int(n*0.85)]
test_df = df.iloc[int(n*0.85):]

train_df.to_csv(os.path.join(data_dir, "train", "train.csv"), index=False)
val_df.to_csv(os.path.join(data_dir, "validation", "validation.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "test", "test.csv"), index=False)

print("SUCCESS! Datele sunt gata.")