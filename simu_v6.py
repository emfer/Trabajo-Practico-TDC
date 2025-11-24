import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# --- CONFIGURACIÓN ---
DT = 2  
HISTORIAL = 200
MAX_BRAKE_FORCE = 400.0
SLIP_FALLA = 0.80  # Umbral de falla crítica (80%)

# CONTROLADOR PID
KP = 5.0  
KI = 5.0   
KD = 20.0   

class SistemaInteligente:
    def __init__(self):
        self.vel_auto = 80.0
        self.vel_rueda = 80.0
        self.target_slip = 0.20 # Referencia dinámica
        
        self.tendencia_base = 0.3
        self.factor_potencia = 1.0 
        self.freno_aplicado = 0.0
        
        self.error_previo = 0.0
        self.integral = 0.0
        
        # Bandera de Estado de Falla
        self.en_falla = False

    def actualizar(self):
        # 1. CALCULAR SLIP
        if self.vel_auto > 0.1:
            slip = (self.vel_rueda - self.vel_auto) / self.vel_auto
        else:
            slip = 0.0
            
        # --- LÓGICA DE FALLA ---
        if slip >= SLIP_FALLA:
            self.en_falla = True
            
        if self.en_falla:
            # En falla, se apaga todo por seguridad
            self.freno_aplicado = 0.0
            self.factor_potencia = 0.0 # Cortamos motor
            
            # Física de inercia (sin control)
            empuje_motor = self.tendencia_base * self.factor_potencia
            cambio = (empuje_motor) * DT # Solo acelera si hay tendencia natural
            self.vel_rueda += cambio
            return slip, 0.0, "FALLA"

        # 2. CONTROLADOR PID (Si no hay falla)
        error = slip - self.target_slip
        
        self.integral += error * DT
        self.integral = np.clip(self.integral, -50, 50)
        
        derivada = (error - self.error_previo) / DT
        self.error_previo = error
        
        salida_pid = (KP * error) + (KI * self.integral) + (KD * derivada)
        
        if salida_pid > 0:
            self.freno_aplicado = salida_pid
        else:
            self.freno_aplicado = 0.0
            self.integral *= 0.9
            
        self.freno_aplicado = np.clip(self.freno_aplicado, 0, MAX_BRAKE_FORCE)
        
        # 3. LÓGICA MOTOR
        if self.freno_aplicado > 5:
            self.factor_potencia -= 0.1
        else:
            self.factor_potencia += 0.005
            
        self.factor_potencia = np.clip(self.factor_potencia, 0.2, 1.0)
        
        # 4. FÍSICA
        empuje_motor = self.tendencia_base * self.factor_potencia
        cambio_velocidad = (empuje_motor - self.freno_aplicado) * DT
        
        self.vel_rueda += cambio_velocidad
        
        if self.vel_rueda < self.vel_auto:
            self.vel_rueda = self.vel_auto
            self.integral = 0
            
        # Determinar estado para mostrar
        estado_str = "ESTABLE" if abs(error) < 0.02 else "TRANSITORIO"
            
        return slip, self.freno_aplicado, estado_str

# Arrancar programa
sistema = SistemaInteligente()

slip_data = [0.0] * HISTORIAL
freno_data_pct = [0.0] * HISTORIAL
ref_data = [0.20] * HISTORIAL

# Graficos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
plt.subplots_adjust(bottom=0.30) # Más espacio para slider y botones

# Gráfico 1: Deslizamiento
ax1.set_title("MONITOR DE DESLIZAMIENTO")
ax1.set_ylabel("Slip")
ax1.set_ylim(-0.05, 1.0) 
ax1.axhline(y=SLIP_FALLA, color='red', linestyle=':', label='Falla (0.8)') # Línea roja de falla
ax1.grid(True, linestyle='--')

line_slip, = ax1.plot(slip_data, 'r-', lw=2, label='Slip Real')
line_ref, = ax1.plot(ref_data, 'g--', lw=2, label='Set Point')
ax1.legend(loc='upper right')

# Cuadro de Estado
text_estado = ax1.text(0.02, 0.9, "ESTADO: ESTABLE", transform=ax1.transAxes, 
                       bbox=dict(facecolor='#ccffcc', alpha=0.8, edgecolor='black'), fontweight='bold')

# Gráfico 2: Frenos
ax2.set_title("ACCIÓN DE CONTROL (Freno)")
ax2.set_ylabel("Intensidad (%)")
ax2.set_ylim(0, 105)
ax2.grid(True, linestyle='--')
line_freno, = ax2.plot(freno_data_pct, 'b-', lw=2, label='Freno TCS')
fill_obj = None 

# --- CONTROLES DE INTERFAZ ---

# 1. Set Point
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
s_ref = Slider(ax_slider, 'Set Point Slip', 0.05, 0.30, valinit=0.20, color='green')

def update_ref(val):
    sistema.target_slip = val
s_ref.on_changed(update_ref)

# 2. Botones de Escenario
ax_asfalto = plt.axes([0.1, 0.03, 0.15, 0.05])
ax_lluvia = plt.axes([0.26, 0.03, 0.15, 0.05])
ax_lluvia_f = plt.axes([0.42, 0.03, 0.15, 0.05])
ax_nieve = plt.axes([0.58, 0.03, 0.15, 0.05])
ax_hielo = plt.axes([0.74, 0.03, 0.15, 0.05])
ax_reset = plt.axes([0.80, 0.15, 0.10, 0.05]) # Reset arriba a la derecha

b_asfalto = Button(ax_asfalto, 'ASFALTO', color='#ccffcc')
b_lluvia = Button(ax_lluvia, 'LLUVIA', color='#ccebff')
b_lluvia_f = Button(ax_lluvia_f, 'LLUVIA FUERTE', color='#99ccff')
b_nieve = Button(ax_nieve, 'NIEVE', color='#f2f2f2')
b_hielo = Button(ax_hielo, 'HIELO (Falla)', color='#ffcccc') # Hielo extremo para probar falla
b_reset = Button(ax_reset, 'RESET SYSTEM', color='#ff6666', hovercolor='#ff3333')

def set_escenario(tendencia):
    sistema.tendencia_base = tendencia

# Ajusté los valores para que HIELO (15.0) sea tan agresivo que cause la falla
b_asfalto.on_clicked(lambda x: set_escenario(0.0))
b_lluvia.on_clicked(lambda x: set_escenario(2.0))
b_lluvia_f.on_clicked(lambda x: set_escenario(4.0))
b_nieve.on_clicked(lambda x: set_escenario(6.0))
b_hielo.on_clicked(lambda x: set_escenario(35.0)) # Valor muy alto para forzar falla

def reset_sim(event):
    sistema.vel_rueda = 80.0
    sistema.freno_aplicado = 0.0
    sistema.factor_potencia = 1.0
    sistema.integral = 0.0
    sistema.error_previo = 0.0
    sistema.en_falla = False
    sistema.tendencia_base = 0.0 # Vuelve a asfalto
    
    slip_data[:] = [0.0] * HISTORIAL
    freno_data_pct[:] = [0.0] * HISTORIAL

b_reset.on_clicked(reset_sim)

# --- BUCLE ---
def update(frame):
    global fill_obj
    
    s, f_raw, estado_actual = sistema.actualizar()
    
    # Convertir a porcentaje
    f_pct = (f_raw / MAX_BRAKE_FORCE) * 100
    f_pct = min(f_pct, 100.0)
    
    # Actualizar datos
    slip_data.pop(0); slip_data.append(s)
    freno_data_pct.pop(0); freno_data_pct.append(f_pct)
    ref_data.pop(0); ref_data.append(sistema.target_slip)
    
    # Líneas
    line_slip.set_ydata(slip_data)
    line_ref.set_ydata(ref_data)
    line_freno.set_ydata(freno_data_pct)
    
    # Relleno
    if fill_obj: fill_obj.remove()
    fill_obj = ax2.fill_between(range(HISTORIAL), freno_data_pct, color='blue', alpha=0.2)
    
    # Actualizar Texto de Estado y Color
    text_estado.set_text(f"ESTADO: {estado_actual}")
    
    if estado_actual == "ESTABLE":
        text_estado.set_backgroundcolor('#ccffcc') # Verde
    elif estado_actual == "TRANSITORIO":
        text_estado.set_backgroundcolor('#ffcc00') # Naranja
    elif estado_actual == "FALLA":
        text_estado.set_backgroundcolor('#ff3333') # Rojo Alarma
        
    return line_slip, line_freno, line_ref, text_estado

ani = FuncAnimation(fig, update, interval=30, blit=False)
plt.show()