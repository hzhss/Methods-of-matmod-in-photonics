import numpy as np
import matplotlib.pyplot as plt

# %% Параметры
alpha = 0 # Коэффициент поглощения, 1/м
beta1 = 0 # Фазовая скорость , м/с
beta2 = -22e-27  # Коэффициент ДСГ, с**2 / м
gamma = 5e-3  # Коэффициент нелинейности, (Вт * м)**-1

T = 2000e-15  # Временной промежуток, с
nt = 2**9  # Число точек по времени
t = np.linspace(-T/2, T/2, nt)  # Массив точек , с
dt = T / nt  # Шаг по времени, с
w = np.fft.fftfreq(nt, dt) * 2 * np.pi

L = 1  # Длина оптического волокна, м
nz = 1001  # Число точек по координате
n_plot = 5  # Количество промежуточных графиков
dz = L / nz  # Шаг по координате, s

t0 = 200e-15  # Временной параметр, s

# %% Основная часть

# Фундаментальное решение
u = np.sqrt(np.absolute(beta2) / (gamma * t0**2)) / np.cosh(t / t0)

# вошли в ф-п-во
ufft = np.fft.fft(u)
# нелинейные эффекты учитываются в середине шага в объеме за весь шаг
ufft = (ufft * np.exp((1j * beta2 / 2 * w**2 +
                       1j * beta1 * w -
                       alpha/2) * dz/2))
A1_plot = []

for ii in range(nz):
    u = np.fft.ifft(ufft)
    u = u * np.exp(1j * gamma * np.absolute(u)**2 * dz)
    if ii % (nz // (n_plot - 1)) == 0:
        A1_plot.append(u)
    ufft = np.fft.fft(u)
    ufft = (ufft * np.exp(1j * beta2 / 2 * w**2 * dz))


# Фундаментальное решение с небольшим коэффициентом поглощения alpha
u = np.sqrt(np.absolute(beta2) / (gamma * t0**2)) / np.cosh(t / t0)

# вошли в ф-п-во
ufft = np.fft.fft(u)
# нелинейные эффекты учитываются в середине шага в объеме за весь шаг
ufft = (ufft * np.exp((1j * beta2 / 2 * w**2 +
                       1j * beta1 * w -
                       1000/2) * dz/2))
A1_alpha_plot = []

for ii in range(nz):
    u = np.fft.ifft(ufft)
    u = u * np.exp(1j * gamma * np.absolute(u)**2 * dz)
    if ii % (nz // (n_plot - 1)) == 0:
        A1_alpha_plot.append(u)
    ufft = np.fft.fft(u)
    ufft = (ufft * np.exp(1j * beta2 / 2 * w**2 * dz))



# Решение второго порядка
u = 2 * np.sqrt(np.absolute(beta2) / (gamma * t0**2)) / np.cosh(t / t0)

ufft = np.fft.fft(u)
ufft = (ufft * np.exp((1j * beta2 / 2 * w**2 +
                       1j * beta1 * w -
                       alpha/2) * dz/2))
A2_plot = []

for ii in range(nz):
    u = np.fft.ifft(ufft)
    u = u * np.exp(1j * gamma * np.absolute(u)**2 * dz)
    if ii % (nz // (n_plot - 1)) == 0:
        A2_plot.append(u)
    ufft = np.fft.fft(u)
    ufft = (ufft * np.exp(1j * beta2 / 2 * w**2 * dz))
    
# Решение третьего порядка
ufft = (ufft * np.exp((1j * beta2 / 2 * w**2 +
                       1j * beta1 * w -
                       alpha/2) * dz/2))

ufft = np.fft.fft(u)
ufft = ufft = (ufft * np.exp((1j * beta2 / 2 * w**2 +
                       1j * beta1 * w -
                       alpha/2) * dz/2))
A3_plot = []

for ii in range(nz):
    u = np.fft.ifft(ufft)
    u = u * np.exp(1j * gamma * np.absolute(u)**2 * dz)
    if ii % (nz // (n_plot - 1)) == 0:
        A3_plot.append(u)
    ufft = np.fft.fft(u)
    ufft = (ufft * np.exp(1j * beta2 / 2 * w**2 * dz))


# %% Графики

# Фундаментальное решение
fig1, ax1 = plt.subplots()
ax1.set_xlim(-T/2 * 1e15, T/2 * 1e15)
ax1.grid()
ax1.set_title('Фундаментальное решение')
ax1.set_xlabel('Время [fs]')
ax1.set_ylabel('Мощность [c.u.]')

for i in range(n_plot):
    ax1.plot(t * 1e15, np.absolute(A1_plot[i])**2, 
             label=(str(round(L / (n_plot - 1) * i, 2)) + 'м'))
ax1.legend()
    
# Решение второго порядка
fig2, ax2 = plt.subplots()
ax2.set_xlim(-T/2 * 1e15, T/2 * 1e15)
ax2.grid()
ax2.set_title('Решение второго порядка')
ax2.set_xlabel('Время [fs]')
ax2.set_ylabel('Мощность [c.u.]')

for i in range(n_plot):
    ax2.plot(t * 1e15, np.absolute(A2_plot[i])**2, 
             label=(str(round(L / (n_plot - 1) * i, 2)) + 'm'))
ax2.legend()

# Решение третьего порядка
fig3, ax3 = plt.subplots()
ax3.set_xlim(-T/2 * 1e15, T/2 * 1e15)
ax3.grid()
ax3.set_title('Решение третьего порядка')
ax3.set_xlabel('Время [fs]')
ax3.set_ylabel('Мощность [c.u.]')

for i in range(n_plot):
    ax3.plot(t * 1e15, np.absolute(A3_plot[i])**2, 
             label=(str(round(L / (n_plot - 1) * i, 2)) + 'm'))
ax3.legend()

fig4, ax4 = plt.subplots()
ax4.set_xlim(-T/2 * 1e15, T/2 * 1e15)
ax4.grid()
ax4.set_title('Фундаментальное решение с поглощением')
ax4.set_xlabel('Время [fs]')
ax4.set_ylabel('Мощность [c.u.]')

for i in range(n_plot):
    ax4.plot(t * 1e15, np.absolute(A1_alpha_plot[i])**2, 
             label=(str(round(L / (n_plot - 1) * i, 2)) + 'm'))
ax4.legend()
  