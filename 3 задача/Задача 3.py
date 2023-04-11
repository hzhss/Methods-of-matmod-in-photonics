import numpy as np
import matplotlib.pyplot as plt
import pandas

from scipy.integrate import solve_ivp
from numpy import pi

lmbda_s = 1064
lmbda_p = 962

dp = 125e-6
d = 10e-6
N = 4000 * 6.22e22      #concentration  m**-3
Ngrid = 100
L = 10
h =  6.62607015e-34
tau =  1.54e-3
c = 3e8 / 1.44
Aeff_s = pi * d**2 / 4
Aeff_p = pi * dp**2 / 4 * 2
Ppulse = 50e-3
P_p = 20
tpulse = 1e-6
freq = 1e5
I = Aeff_s / Aeff_p

sigma = pandas.read_csv('Yb_cross_sections.csv', sep=';').to_numpy()

s_12_s = np.interp(lmbda_s, sigma[:, 0], sigma[:,2])
s_21_s = np.interp(lmbda_s, sigma[:, 0], sigma[:,1])
s_12_p = np.interp(lmbda_p, sigma[:, 0], sigma[:,2])
s_21_p = np.interp(lmbda_p, sigma[:, 0], sigma[:,1])
#%%
t = np.linspace(0, 1/freq, Ngrid)

pulse = np.where(t <= tpulse, Ppulse, 0)
Pump = np.where(t >= tpulse, P_p, 0)

plt.figure('pulse', clear=True)
plt.title('Pulse')
plt.plot(t, pulse, t, Pump/200)

z = np.linspace(0, L, Ngrid)
dz = z[1]-z[0]
dt = t[1]-t[0]
    
qo = pulse / c / Aeff_s / (h * c / lmbda_s *1e9)
qf = pulse / c / Aeff_s / (h * c / lmbda_s *1e9)

qp = np.zeros(Ngrid)
qp_0 = qp
N2 = np.zeros(Ngrid)
N2o = np.zeros(Ngrid)

N_inv = np.zeros((Ngrid, Ngrid)) 
Q = np.zeros((Ngrid, Ngrid))

for k in range(10):
    
    for j in range(Ngrid):
        
        if j < int(Ngrid * tpulse * freq):
            q = Ppulse / c / Aeff_s / (h * c / lmbda_s *1e9)
            # q = qo[j]
            for i in range(0, Ngrid):
                N2prev = N2[i]
                q = q * np.exp((s_21_s * N2prev - s_12_s * (N-N2prev)) * dz)
                
                N2[i] = (N2prev + c * dt * q *((s_12_s * (N - N2prev) - 
                                            s_21_s * N2prev))) * np.exp(-dt/tau)
            qf[j] = q
        if j >= (Ngrid * tpulse * freq):
            q = P_p / c / Aeff_p / (h * c / lmbda_p *1e9)
            for i in range(0, Ngrid):        
                N2prev = N2[i]
                q = q * np.exp(I * (s_21_p * N2prev - s_12_p * (N-N2prev))*dz)
                
                N2[i] = (N2prev + c * dt * q *((s_12_p * (N - N2prev) - 
                                                    s_21_p * N2prev))) * np.exp(-dt/tau)
            qp[j] = q
    N_inv[k] = N2
    Q[k] = qf * c * Aeff_s * (h * c / lmbda_s *1e9)

plt.figure('Импульс')
# plt.plot(t, qf * c * Aeff_s * (h * c / lmbda_s *1e9))
plt.plot(t, Q[0])
plt.plot(t, Q[2])
plt.plot(t, Q[4])
plt.plot(t, Q[7])
plt.plot(t, Q[9])
# plt.plot(t, Q[19])
# plt.plot(t, Q[29])
plt.plot(t, pulse)
plt.legend(('new pulse 0', 'new inversion 2 ',  'new inversion4', 'new pulse 7','new pulse 9' , 'old inversion'))
plt.show()

plt.figure('Инверсия')
plt.plot(z, N_inv[0]/N)  
plt.plot(z, N_inv[7]/N)  
plt.plot(z, N_inv[15]/N)  
plt.plot(z, N_inv[20]/N)  
plt.plot(z, N_inv[29]/N)  
plt.plot(z, N2o/N)  
plt.legend(('new inversion 0', 'new inversion20', 'new inversion40', 'new inversion60', 'new inversion80', 'old inversion'))
plt.show()

# P1 = answer.sol(z)
# plt.figure('Amp_beautiful', clear=True)
# plt.plot(z, P1[0, :], z, P1[1, :])
        

