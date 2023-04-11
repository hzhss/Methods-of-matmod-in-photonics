import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import pandas

#%% Параметры (волокно двойка)
d_core = 10e-6 # Диаметр жилы, [м]
d_clad = 125e-6 # Диаметр оболочки, [м]
N = 4000 * 6.62e22# концентрация ионов итербия (6.62e22 - Перевод из ppm
                                # в обычную концентрацию в обратных кубометрах)
L = 10 # Длина волокна, м
tau = 1.54e-3 # Время жизни в возбужденном состоянии, [сек]

h = 6.626e-34
c = 3e8 / 1.44 # Скорость света в волокне, [м/с]

Aeff_s = pi * d_core**2 / 4 # Площадь моды сигнала
Aeff_p = pi * d_clad**2 / 4 * 2 # Площадь моды накачки (*2 т.к. две жилы)
G = Aeff_s / Aeff_p #Интеграл перекрытия

lmbda_s = 1064 # Длина волны сигнала,[nm]
lmbda_p = 962   # Длина волны накачки, [nm]

P_s = 0.05  # Средняяя мощность импульса, [Вт]

t_s = 1e-6 # Длительность импульса, [с]
f = 1e5 # Частота повторения импульсов сигнала, [Гц]
t_repeat = 1e-5 # Время между импульсами, [с]
N_grid = 100

#Сечения взаимодействия из файла:
sigma = pandas.read_csv('Yb_cross_sections.csv', sep=';').to_numpy()

# Коэффициенты поглощения и люминисценции для сигнала и накачки:
s_12_s = np.interp(lmbda_s, sigma[:,0], sigma[:, 2])
s_21_s = np.interp(lmbda_s, sigma[:,0], sigma[:, 1])
s_12_p = np.interp(lmbda_p, sigma[:,0], sigma[:, 2])
s_21_p = np.interp(lmbda_p, sigma[:,0], sigma[:, 1])

# %% Итербиевый импульсный усилитель

# Разобьем импульс на слои малой длительности t = j dt
t = np.linspace(0, t_repeat, N_grid)

#Сетка по х:
x = np.linspace(0, L, N_grid)

dx = x[1]-x[0]
dt = t[1]-t[0]

q_s = np.zeros(N_grid)
q_p = np.zeros(N_grid)
N2 = np.zeros(N_grid)
N2_0 = np.zeros(N_grid)

Imp = np.where(t <= t_s, P_s, 0)

#массивы для всех импульсов и инверсий
N_inv = np.zeros((N_grid, N_grid)) 
Q = np.zeros((N_grid, N_grid))

#количество импульсов
N_pulse = int(input("Введите номер импульса N_pulse\n"))

#мощность накачки
P_p = int(input("Введите мощность накачки P_p\n"))

form = int(input("Какая форма импульса form? 0 - меандр, 1 - гаусс\n"))
if form:
    gauss =  ((2 * P_s * t_repeat) / (t_s * np.sqrt(np.pi)) *
                             np.exp(- (t - 2* t_s)**2 / ((t_s / 2)**2)))

def Yb_Impulse_Amp(P_p, N_pulse, form):
    
    #Цикл по числу импульсов
    for k in range(N_pulse):
        
        #Цикл по времени
        for j in range(N_grid):
            if form:
                q = gauss[j] / c / Aeff_s / (h * c / (lmbda_s*1e-9))
                for i in range(0, N_grid):
            
                    q = q * np.exp((s_21_s * N2[i] - s_12_s * (N-N2[i])) * dx)
                    N2[i] = (N2[i] + c * dt * q *((s_12_s * (N - N2[i]) - 
                                                s_21_s * N2[i]))) * np.exp(-dt/tau)
                q_s[j] = q
            #Если время до 1мкс, то запущен импульс, если больше, то накачка.
            elif j < int(N_grid * t_s * f):
                
                q = P_s / c / Aeff_s / (h * c / lmbda_s * 1e9)
                
                for i in range(0, N_grid):
            
                    q = q * np.exp((s_21_s * N2[i] - s_12_s * (N-N2[i])) * dx)
                    N2[i] = (N2[i] + c * dt * q *((s_12_s * (N - N2[i]) - 
                                                s_21_s * N2[i]))) * np.exp(-dt/tau)
                q_s[j] = q
                
            if (j >= int(N_grid * t_s * f)):
                q = P_p / c / Aeff_p / (h * c / lmbda_p * 1e9)
                for i in range(0, N_grid):        
                
                    q = q * np.exp(G * (s_21_p * N2[i] - s_12_p * (N-N2[i]))*dx)
                    N2[i] = (N2[i] + c * dt * q *((s_12_p * (N - N2[i]) - 
                                            s_21_p * N2[i]))) * np.exp(-dt/tau)
                q_p[j] = q
                
        N_inv[k] = N2
        Q[k] = q_s * c * Aeff_s * (h * c / lmbda_s * 1e9)
    return Q, N_inv

# Inversion = np.zeros((N_grid, N_grid)) 
# Impulse = np.zeros((N_grid, N_grid))

Impulse, Inversion  = Yb_Impulse_Amp(P_p, N_pulse-1, form)
#Цикл по времени

for j in range(N_grid):
    if form:
        q = gauss[j] / c / Aeff_s / (h * c / (lmbda_s*1e-9))
        for i in range(0, N_grid):
    
            q = q * np.exp((s_21_s * N2[i] - s_12_s * (N-N2[i])) * dx)
            N2[i] = (N2[i] + c * dt * q *((s_12_s * (N - N2[i]) - 
                                        s_21_s * N2[i]))) * np.exp(-dt/tau)
        q_s[j] = q
    #Если время до 1мкс, то запущен импульс, если больше, то накачка.
    elif j < int(N_grid * t_s * f):
        
        q = P_s / c / Aeff_s / (h * c / lmbda_s * 1e9)
        
        for i in range(0, N_grid):

            q = q * np.exp((s_21_s * N2[i] - s_12_s * (N-N2[i])) * dx)
            N2[i] = (N2[i] + c * dt * q *((s_12_s * (N - N2[i]) - 
                                    s_21_s * N2[i]))) * np.exp(-dt/tau)
        q_s[j] = q
        
Inversion[N_pulse-1] = N2
Impulse[N_pulse-1] = q_s * c * Aeff_s * (h * c / lmbda_s * 1e9)

# %% Графики 
if form:
    plt.figure('Импульс до усиления', clear=True)
    plt.plot(t, gauss)   
    plt.xlabel('t, с')
    plt.ylabel('$P_s$, Вт')      
    # plt.xlim([0, 0.4 * t_repeat])
    plt.show()
else:
    plt.figure('Импульс до усиления', clear=True)
    plt.plot(t, Imp)    
    plt.xlabel('t, с')
    plt.ylabel('$P_s$, Вт')     
    # plt.xlim([0, 0.4 * t_repeat])
    plt.show()
                      
plt.figure('N-ый импульc после усиления ', clear=True)
plt.plot(t, Impulse[N_pulse - 1])
plt.xlabel('t, с')
plt.ylabel('$P_s$, Вт')
plt.legend('pulse')
# plt.xlim([0, 0.4 * t_repeat])
plt.show()

plt.figure('Инверсия до и после прохождения импульса', clear=True)
plt.plot(x, Inversion[N_pulse-2]/N)  
plt.plot(x, Inversion[N_pulse-1]/N)  
plt.legend(('До', 'После'))
plt.show()



















