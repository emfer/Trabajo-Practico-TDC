import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
DT = 0.05
HISTORIAL = 150 

# Estado inicial
estado = {
    't': 0,
    'vel_vehiculo': 8.0,  # 8 m/s ~ 28 km/h
    'vel_rueda': 8.0,
    'integral_error': 0.0,
    'error_previo': 0.0,
    'acelerador_driver': 0.95 # Conductor exigente
}

PARAMS = {
    'masa': 350,       
    'radio': 0.3,      
    'inercia': 1.0,    
    'torque_max': 600, 
    'kp': 4.0,         
    'ki': 0.1,         
    'kd': 0.05 
}

# Buffers de datos
slip_data = np.zeros(HISTORIAL)
ref_data = np.zeros(HISTORIAL) + 0.15
torque_real_data = np.zeros(HISTORIAL)
torque_driver_data = np.zeros(HISTORIAL)

# ==========================================
# 2. FÍSICA Y CONTROLADOR
# ==========================================
def actualizar_fisica(val_friccion, val_referencia):
    global estado
    
    # 1. Calculo de Slip
    denominador = max(estado['vel_rueda'], 0.1)
    slip = (estado['vel_rueda'] - estado['vel_vehiculo']) / denominador
    slip = np.clip(slip, 0, 1.0)
    
    # 2. CONTROLADOR TCS (PID)
    error = val_referencia - slip
    estado['integral_error'] += error * DT
    derivada = (error - estado['error_previo']) / DT
    
    # Si el error es grande (patina), esto da negativo para bajar el acelerador
    ajuste = (PARAMS['kp'] * error) + (PARAMS['ki'] * estado['integral_error']) + (PARAMS['kd'] * derivada)
    
    # El acelerador final nunca supera al del conductor, ni baja de 0
    throttle_final = np.clip(estado['acelerador_driver'] + ajuste, 0.0, estado['acelerador_driver'])
    
    # 3. DINÁMICA FÍSICA (CORREGIDA)
    torque_motor = throttle_final * PARAMS['torque_max']
    
    # Fuerza máxima que el suelo aguanta antes de patinar (Límite de Fricción)
    fuerza_max_suelo = val_friccion * PARAMS['masa'] * 9.81
    torque_max_suelo = fuerza_max_suelo * PARAMS['radio']
    
    # --- CORRECCIÓN CLAVE AQUÍ ---
    # Si la rueda patina (gira más rápido que el auto), el suelo ejerce TODA su fricción para frenarla.
    # Si no patina, el suelo solo resiste lo que el motor empuja (Newton).
    if estado['vel_rueda'] > estado['vel_vehiculo'] + 0.1: 
        # La rueda está patinando: El suelo frena la rueda con fuerza máxima
        torque_resistente = torque_max_suelo
    else:
        # Hay agarre: El suelo resiste lo mismo que empujas (equilibrio)
        torque_resistente = min(torque_motor, torque_max_suelo)
    
    # La fuerza que realmente mueve al auto es la resistencia que opuso el suelo
    fuerza_traccion = torque_resistente / PARAMS['radio']
    
    # Ecuaciones de movimiento
    accel_vehiculo = fuerza_traccion / PARAMS['masa']
    
    # Suma de momentos: Motor acelera, Suelo frena
    accel_rueda = (torque_motor - torque_resistente) / PARAMS['inercia']
    
    # Integración
    estado['vel_vehiculo'] += accel_vehiculo * DT
    estado['vel_rueda'] += (accel_rueda * PARAMS['radio']) * DT
    
    # Evitar que la rueda gire hacia atrás por error matemático
    if estado['vel_rueda'] < estado['vel_vehiculo']: estado['vel_rueda'] = estado['vel_vehiculo']
    
    estado['error_previo'] = error
    estado['t'] += DT
    
    return slip, torque_motor, (estado['acelerador_driver'] * PARAMS['torque_max'])

# ==========================================
# 3. GRÁFICOS E INTERFAZ
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(bottom=0.25, hspace=0.3)

# Estilos
ax1.set_title("MONITOR DE DESLIZAMIENTO (TCS)")
ax1.set_ylabel("Deslizamiento (%)")
ax1.set_ylim(-0.05, 0.8) 
ax1.grid(True)

line_slip, = ax1.plot([], [], 'r-', label='Real (Sensor)', lw=2.5)
line_ref, = ax1.plot([], [], 'g--', label='Referencia (Objetivo)', lw=2)
text_status = ax1.text(0.02, 0.85, "", transform=ax1.transAxes, fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')

ax2.set_title("ACCIÓN DE CONTROL")
ax2.set_ylabel("Torque (Nm)")
ax2.set_ylim(0, 650)
ax2.grid(True)

line_driver, = ax2.plot([], [], 'k:', label='Pedido Conductor', alpha=0.5)
line_tcs, = ax2.plot([], [], 'b-', label='Salida TCS', lw=2)
ax2.legend(loc='upper right')

# Controles
ax_friccion = plt.axes([0.15, 0.1, 0.6, 0.03], facecolor='lightyellow')
ax_ref = plt.axes([0.15, 0.05, 0.6, 0.03], facecolor='lightyellow')
ax_reset = plt.axes([0.8, 0.05, 0.1, 0.08]) # Botón a la derecha

s_friccion = Slider(ax_friccion, 'PERTURBACIÓN\n(Fricción)', 0.1, 1.0, valinit=0.9)
s_ref = Slider(ax_ref, 'REFERENCIA\n(Set Point)', 0.05, 0.30, valinit=0.15)
b_reset = Button(ax_reset, 'RESET', color='orange', hovercolor='red')

def reset_sim(event):
    global estado, slip_data, torque_real_data
    estado['vel_vehiculo'] = 8.0
    estado['vel_rueda'] = 8.0
    estado['integral_error'] = 0.0
    estado['acelerador_driver'] = 0.95
    slip_data[:] = 0
    torque_real_data[:] = 0
    print("Simulación reiniciada.")

b_reset.on_clicked(reset_sim)

def update(frame):
    # Aceleramos x3 la física para fluidez
    for _ in range(3):
        val_mu = s_friccion.val
        val_target = s_ref.val
        s, t_real, t_driver = actualizar_fisica(val_mu, val_target)
        
        global slip_data, ref_data, torque_real_data, torque_driver_data
        slip_data = np.roll(slip_data, -1); slip_data[-1] = s
        ref_data = np.roll(ref_data, -1); ref_data[-1] = val_target
        torque_real_data = np.roll(torque_real_data, -1); torque_real_data[-1] = t_real
        torque_driver_data = np.roll(torque_driver_data, -1); torque_driver_data[-1] = t_driver
    
    line_slip.set_data(np.arange(HISTORIAL), slip_data)
    line_ref.set_data(np.arange(HISTORIAL), ref_data)
    line_tcs.set_data(np.arange(HISTORIAL), torque_real_data)
    line_driver.set_data(np.arange(HISTORIAL), torque_driver_data)
    
    ax1.set_xlim(0, HISTORIAL); ax2.set_xlim(0, HISTORIAL)
    
    # Lógica de colores
    if s > val_target + 0.05:
        text_status.set_text("⚠️ ESTADO: PATINANDO")
        text_status.set_color("red")
        ax1.set_facecolor('#fff0f0')
    else:
        text_status.set_text("✅ ESTADO: ESTABLE")
        text_status.set_color("green")
        ax1.set_facecolor('white')
        
    return line_slip, 

ani = FuncAnimation(fig, update, interval=20, blit=False)
plt.show()