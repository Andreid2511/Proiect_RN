import pandas as pd
import numpy as np
import random
import os

# --- CONSTANTE FIZICE (IDENTICE CU APP_GUI) ---
# Time step 0.033s = 30 FPS (Exact ca in aplicatie)
DT = 0.033 

def get_engine_torque(rpm):
    if rpm < 1000: return 0.5 
    if rpm < 2000: return 0.5 + (rpm - 1000) * 0.001 
    if rpm < 4500: return 1.5 
    if rpm < 6000: return 1.2 
    return 0.8 

def simulate_behavior(num_samples, style):
    data = []
    
    speed = 0.0
    prev_speed = 0.0
    rpm = 800.0
    gear = 1
    throttle = 0.0
    brake = 0.0
    
    ratios = {1: 4.7, 2: 3.1, 3: 2.1, 4: 1.7, 5: 1.3, 6: 1.0, 7: 0.8, 8: 0.6}
    
    factor_frana = 2.5
    gravity = 0.018
    frecare = 0.01
    base_power = 0.14 
    
    current_speed_limit = 50 
    limit_timer = 0
    
    for i in range(num_samples):
        time = i * DT
        
        limit_timer += 1
        if limit_timer > random.randint(300, 600):
            current_speed_limit = random.choice([30, 50, 60, 90, 100])
            limit_timer = 0

        # Panta variabila (-15 la 15)
        tilt = np.sin(i/1000) * 15 # Am incetinit frecventa pantei pt 30fps
        
        target_throttle = 0
        target_brake = 0
        
        # --- LOGICA COMPORTAMENT ---
        is_climbing_hard = (tilt > 8)

        if style == 0: # ECO
            if is_climbing_hard:
                base_th = random.uniform(60, 90)
            else:
                base_th = random.uniform(10, 40) # Panta usoara => pedala mica
            
            if speed < current_speed_limit:
                if random.random() < 0.02: target_throttle = base_th # Reactie mai lenta
                else: target_throttle = throttle
            else:
                target_throttle = 0
                if speed > current_speed_limit + 5: target_brake = 10
            
            smooth = 0.5 # Pedala foarte fina (Eco)

        elif style == 2: # AGRESIV
            base_th = random.uniform(85, 100) 
            if speed < current_speed_limit + 20:
                if random.random() < 0.1: target_throttle = base_th
                else: target_throttle = throttle
            else:
                target_throttle = 0
                target_brake = 60 
            smooth = 5.0 # Pedala brusca (Sport)

        else: # NORMAL
            if is_climbing_hard:
                base_th = random.uniform(70, 95)
            else:
                base_th = random.uniform(30, 65)

            if speed < current_speed_limit + 5:
                if random.random() < 0.05: target_throttle = base_th
                else: target_throttle = throttle
            else:
                target_throttle = 0
                target_brake = 25
            smooth = 1.5

        # Smoothing Pedala
        if target_throttle > throttle: throttle += smooth
        elif target_throttle < throttle: throttle -= smooth
        brake = target_brake
        
        if throttle < 0: throttle = 0
        if throttle > 100: throttle = 100

        # --- FIZICA (Aceeasi formula ca in GUI) ---
        engine_torque = get_engine_torque(rpm)
        gear_ratio = ratios.get(gear, 1.0)
        
        
        eff_throttle = throttle
        if brake > 5: eff_throttle = 0

        push_force = (eff_throttle/100.0) * engine_torque * gear_ratio * base_power
        
        engine_braking = 0
        if throttle < 5: 
            engine_braking = (rpm / 1000) * gear_ratio * 0.15 
        
        delta = push_force - (brake/100.0 * factor_frana)
        delta -= engine_braking
        delta -= (tilt * gravity)
        delta -= frecare
        
        # Aerodinamica simpla
        delta -= (speed * speed) * 0.00005

        speed += delta
        if speed < 0: speed = 0
        
        acceleration = speed - prev_speed
        prev_speed = speed 

        # Cutie
        rpm = speed * gear_ratio * 30 + (throttle * 10) 
        if rpm < 800: rpm = 800
        
        upshift_point = 3000
        if tilt > 5: upshift_point = 4500 
        if style == 2: upshift_point = 5500 # Sport schimba sus
        if style == 0: upshift_point = 2200 # Eco schimba jos

        if rpm > upshift_point and gear < 8: gear += 1
        elif (rpm < 1100) and gear > 1: gear -= 1
            
        # SALVARE DATE
        data.append([
            rpm, 
            speed, 
            acceleration, 
            throttle, 
            brake, 
            tilt, 
            gear, 
            style
        ])

    return data

# --- MAIN ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, "../../data"))

print("Generez date (30 FPS Raw Data)...")
os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)

cols_simple = ['rpm', 'speed', 'acceleration', 'throttle', 'brake', 'tilt', 'gear', 'style_label']

all_data = []
# Generam mai multe sample-uri pentru ca pasul de timp este mic (0.033)
all_data.extend(simulate_behavior(60000, 0)) 
all_data.extend(simulate_behavior(60000, 1)) 
all_data.extend(simulate_behavior(60000, 2)) 

df = pd.DataFrame(all_data, columns=cols_simple)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df)
train_df = df.iloc[:int(n*0.7)]
val_df = df.iloc[int(n*0.7):int(n*0.85)]
test_df = df.iloc[int(n*0.85):]

train_df.to_csv(os.path.join(data_dir, "train", "train.csv"), index=False)
val_df.to_csv(os.path.join(data_dir, "validation", "validation.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "test", "test.csv"), index=False)

print("✅ Dataset generat! Acum ruleaza train_model.py")