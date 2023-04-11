import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import pandas
from scipy.integrate import solve_bvp #ф-я решения инт уравнения

#%% Параметры (волокно двойка)
d_core = 10e-6 # Диаметр жилы, [м]
d_clad = 125e-6 # Диаметр оболочки, [м]
N = 4000 * 6.62e22 # концентрация ионов итербия (6.62e22 - Перевод из ppm
                                # в обычную концентрацию в обратных кубометрах)
L = 3 # Длина волокна, м
tau = 1.54e-3 # Время жизни в возбужденном состоянии, [сек]

c = 3e8 / 1.44 # Скорость света, [м/с]
Aeff_s = pi * d_core**2 / 4 # Площадь моды сигнала
Aeff_p = pi * d_clad**2 / 4 * 2 # Площадь моды накачки (*2 т.к. две жилы)

R_1 = 0.1 # Коэффициент отражения глухой решетки
R2 = 0.95 # Коэффициент отражения глухой решетки

P_p = 0.5 # Мощность накачки с каждой стороны, [Вт]

lmbda_s = 1064 # Длина волны сигнала,[nm]
lmbda_p = 975   # Длина волны накачки, [nm]

#Сечения взаимодействия из файла:
sigma = pandas.read_csv('Yb_cross_sections.csv', sep=';').to_numpy()

# Коэффициенты поглощения и люминисценции для сигнала и накачки:
s_12_s = np.interp(lmbda_s, sigma[:,0], sigma[:, 2])
s_21_s = np.interp(lmbda_s, sigma[:,0], sigma[:, 1])
s_12_p = np.interp(lmbda_p, sigma[:,0], sigma[:, 2])
s_21_p = np.interp(lmbda_p, sigma[:,0], sigma[:, 1])

#%% Распределение мощностей сигнала:

eff = np.zeros(95) # лист для кпд
R1 = np.linspace(0.01, 0.95, 95) #лист для разных показателей отражения зеркала

for i in range(95):
    
    # Инверсия
    def Inv(P):
        
        #концентрация фотонов накачки и сигнала:                                                     
        n_pp = P[0] / c / Aeff_p / (1240 / lmbda_p * 1.6e-19)
        n_pm = P[1] / c / Aeff_p / (1240 / lmbda_p * 1.6e-19) 
        n_sp = P[2] / c / Aeff_s / (1240 / lmbda_s * 1.6e-19)
        n_sm = P[3] / c / Aeff_s / (1240 / lmbda_s * 1.6e-19)
        
        N2 = N * ((((n_pp + n_pm) * s_12_p) + ((n_sp + n_sm) * s_12_s)) /
                  ((n_pp + n_pm) * (s_12_p + s_21_p) + (n_sp + n_sm) * 
                            (s_12_s + s_21_s) + 1 / (c * tau)))
        return N2
    
    # Уравнение
    def ODE(z, P):
        N2 = Inv(P)
        N1 = N - N2
        dP = np.zeros((4, len(P[0, :])))
        dP[0] = Aeff_s / Aeff_p * P[0] * (s_21_p * N2 - s_12_p * N1)
        dP[1] = -Aeff_s / Aeff_p * P[1] * (s_21_p * N2 - s_12_p * N1)
        dP[2] = (P[2] + 1e-10) * (s_21_s * N2 - s_12_s * N1)
        dP[3] = -(P[3]) * (s_21_s * N2 - s_12_s * N1)
        return dP
    
    # Граничные условия
    def bc(P_a, P_b):
        return np.array([P_a[0] - P_p, P_b[1] - P_p, P_a[2] - P_a[3] * R2,
                          P_b[3] - P_b[2] * R1[i]])
       
    x = np.linspace(0, L, 1000)
    y = np.zeros((4, 1000)) 
    
    y[0] = P_p * np.exp(-x)
    y[1] = P_p * np.exp(x - L)
    y[2] = P_p #сигнал в положительном направлении  np.linspace(0.1, 2.5, 1000) *
    y[3] =  np.linspace(2.5, 5.0, 1000) * P_p #сигнал в отр. напр.

    sol = solve_bvp(ODE, bc, x, y)
    P_sol = sol.sol(x)
    
    if i == 9: #если R == 0.1
        
        #График инверсии
        plt.figure('Инверсия')
        plt.plot(x, Inv(P_sol))
        plt.ylabel('N2')
        plt.xlabel('Длина волокна, м')
        plt.grid()   
        
        #График мощности накачки и сигнала   
        plt.figure('Мощность')
        plt.plot(x, P_sol[0, :], x, P_sol[1, :],
                             x, P_sol[2, :], x, P_sol[3, :])
        plt.legend(('Накачка в прямом направлении','Накачка в обратном',
                                    'Сигнал в прямом','Сигнал в обратном'))
        plt.ylabel('Мощность, Вт')
        plt.xlabel('Длина волокна, м')
        plt.grid()  
        
    # КПД
    eff[i] =  (P_sol[2, -1] - P_sol[3,-1]) / (2 * P_p)
    
    
#КПД в зависимости от R1:
plt.figure('КПД')
plt.plot(R1, eff * 100)
plt.ylabel('КПД')
plt.xlabel('Коэффициент отражения полупрозрачного зеркала')
plt.grid()
        
    
    
    
    
    