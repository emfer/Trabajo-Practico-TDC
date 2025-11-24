import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# --- CONFIGURACIÓN ---
DT = 1 # 50ms (Velocidad de simulación adecuada)
HISTORIAL = 200
TARGET_SLIP = 0.20
MAX_BRAKE_FORCE = 400.0 # Esto será el 100% de la barra

# CONTROLADOR PID
# Ajustado para que el freno sea firme y baje el slip por debajo de la linea
KP =8.0  
KI = 32.0   
KD = 35.0   

class SistemaInteligente:
    def __init__(self):
        self.vel_auto = 80.0
        self.vel_rueda = 80.0
        
        # Tendencia a patinar (Fuerza del motor que intenta acelerar la rueda)
        # 0.5 = Asfalto (Agarre normal)
        # 8.0 = Hielo (Se dispara)
        self.tendencia_base = 0.0 
        
        self.factor_potencia = 1.0 
        self.freno_aplicado = 0.0
        
        # Variables PID
        self.error_previo = 0.0
        self.integral = 0.0

    def actualizar(self):
        # 1. CALCULAR SLIP
        # Evitamos división por cero
        if self.vel_auto > 0.1:
            slip = (self.vel_rueda - self.vel_auto) / self.vel_auto
        else:
            slip = 0.0
        
        # 2. CONTROLADOR PID
        # Error positivo = Slip alto (hay que frenar)
        error = slip - TARGET_SLIP 
        
        # Integral (La memoria que fuerza a bajar por debajo del target)
        self.integral += error * DT
        # Anti-windup (Límites para que no acumule error infinito)
        self.integral = np.clip(self.integral, -100, 100)
        
        # Derivativo (Reacción rápida)
        derivada = (error - self.error_previo) / DT
        self.error_previo = error
        
        # Salida PID
        salida_pid = (KP * error) + (KI * self.integral) + (KD * derivada)
        
        if salida_pid > 0:
            self.freno_aplicado = salida_pid
        else:
            self.freno_aplicado = 0.0
            # Si no frenamos, relajamos la integral suavemente para que no quede "pegada"
            self.integral *= 0.95
            
        # Limitamos físico (0 a MAX)
        self.freno_aplicado = np.clip(self.freno_aplicado, 0, MAX_BRAKE_FORCE)
        
        # 3. LÓGICA DE CORTE DE MOTOR (Ayuda al freno)
        if self.freno_aplicado > 10:
            self.factor_potencia -= 0.05 # Baja potencia rápido
        else:
            self.factor_potencia += 0.005 # Recupera potencia muy lento
            
        self.factor_potencia = np.clip(self.factor_potencia, 0.2, 1.0)
        
        # 4. FÍSICA SIMPLIFICADA
        # Velocidad = Anterior + (Motor - Freno) * tiempo
        empuje_motor = self.tendencia_base * self.factor_potencia
        
        # El freno resta velocidad
        cambio_velocidad = (empuje_motor - self.freno_aplicado) * DT
        
        self.vel_rueda += cambio_velocidad
        
        # La rueda no puede girar más lento que el auto (tracción total)
        if self.vel_rueda < self.vel_auto:
            self.vel_rueda = self.vel_auto
            self.integral = 0 # Reset integral si ya no patina
            
        return slip, self.freno_aplicado

# --- INICIALIZACIÓN ---
sistema = SistemaInteligente()

# Buffers de datos
slip_data = [0.0] * HISTORIAL
freno_data_pct = [0.0] * HISTORIAL
ref_data = [TARGET_SLIP] * HISTORIAL

# --- GRÁFICOS ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(bottom=0.25)

# Gráfico 1: Deslizamiento
ax1.set_title("MONITOR DE DESLIZAMIENTO (PID)")
ax1.set_ylabel("Slip")
ax1.set_ylim(-0.05, 0.8)
ax1.grid(True, linestyle='--', alpha=0.6)
line_slip, = ax1.plot(slip_data, 'r-', lw=2, label='Slip Real')
line_ref, = ax1.plot(ref_data, 'g--', lw=2, label='Límite (0.20)')
ax1.legend(loc='upper right')
text_estado = ax1.text(0.02, 0.9, "SUELO: ASFALTO", transform=ax1.transAxes, 
                       bbox=dict(facecolor='#ccffcc', alpha=0.8, edgecolor='none'))

# Gráfico 2: Frenos (ESCALA 0-100%)
ax2.set_title("ACCIÓN DE CONTROL (Intensidad de Frenado)")
ax2.set_ylabel("Freno Aplicado N/M")
ax2.set_ylim(0, 10) # Fijo de 0 a 100
ax2.grid(True, linestyle='--', alpha=0.6)

# Linea azul del freno
line_freno, = ax2.plot(freno_data_pct, 'b-', lw=2, label='Freno TCS')
# Referencia para el relleno (se actualiza en el loop)
fill_obj = None 

ax2.legend(loc='upper right')

# --- BOTONES ---
ax_asfalto = plt.axes([0.1, 0.12, 0.15, 0.07])
ax_lluvia = plt.axes([0.26, 0.12, 0.15, 0.07])
ax_lluvia_f = plt.axes([0.42, 0.12, 0.15, 0.07])
ax_nieve = plt.axes([0.58, 0.12, 0.15, 0.07])
ax_hielo = plt.axes([0.74, 0.12, 0.15, 0.07])
ax_reset = plt.axes([0.42, 0.03, 0.15, 0.07])

b_asfalto = Button(ax_asfalto, 'ASFALTO', color='#ccffcc')
b_lluvia = Button(ax_lluvia, 'LLUVIA', color='#ccebff')
b_lluvia_f = Button(ax_lluvia_f, 'LLUVIA FUERTE', color='#99ccff')
b_nieve = Button(ax_nieve, 'NIEVE', color='#f2f2f2')
b_hielo = Button(ax_hielo, 'HIELO', color='#ffcccc')
b_reset = Button(ax_reset, 'RESET', color='#ff6666', hovercolor='#ff3333')

def set_escenario(tendencia, nombre, color):
    sistema.tendencia_base = tendencia
    text_estado.set_text(f"SUELO: {nombre}")
    text_estado.set_backgroundcolor(color)

b_asfalto.on_clicked(lambda x: set_escenario(0.0, "ASFALTO", "#ccffcc"))
b_lluvia.on_clicked(lambda x: set_escenario(2.0, "LLUVIA", "#ccebff"))
b_lluvia_f.on_clicked(lambda x: set_escenario(4.0, "LLUVIA FUERTE", "#99ccff"))
b_nieve.on_clicked(lambda x: set_escenario(6.0, "NIEVE", "#f2f2f2"))
b_hielo.on_clicked(lambda x: set_escenario(8.0, "HIELO", "#ffcccc"))

def reset_sim(event):
    sistema.vel_rueda = 80.0
    sistema.freno_aplicado = 0.0
    sistema.factor_potencia = 1.0
    sistema.integral = 0.0
    sistema.error_previo = 0.0
    set_escenario(0.0, "ASFALTO", "#ccffcc")
    
    global slip_data, freno_data_pct
    slip_data[:] = [0.0] * HISTORIAL
    freno_data_pct[:] = [0.0] * HISTORIAL

b_reset.on_clicked(reset_sim)

# --- BUCLE ---
def update(frame):
    global fill_obj
    
    s, f_raw = sistema.actualizar()
    
    # Convertir a porcentaje (0 a 100%)
    f_pct = (f_raw / MAX_BRAKE_FORCE) * 100
    f_pct = min(f_pct, 100.0) # Clip visual
    
    # Actualizar datos
    slip_data.pop(0)
    slip_data.append(s)
    
    freno_data_pct.pop(0)
    freno_data_pct.append(f_pct)
    
    # Actualizar lineas
    line_slip.set_ydata(slip_data)
    line_freno.set_ydata(freno_data_pct)
    
    # Actualizar relleno de forma segura
    if fill_obj:
        fill_obj.remove()
    fill_obj = ax2.fill_between(range(HISTORIAL), freno_data_pct, color='blue', alpha=0.2)
    
    return line_slip, line_freno

ani = FuncAnimation(fig, update, interval=30, blit=False)
plt.show()