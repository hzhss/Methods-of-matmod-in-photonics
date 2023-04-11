import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy import pi

#%%Параметры (укажем все параметры в СГС)

#показатели преломления обыкновенной и необыкновенной волн:
def n_o(lmbda):
    lmbda *= 1e4
    return (4.9130 + (0.1173 + 1.65e-8 * T**2) /
            (lmbda**2 - (0.212 + 2.7e-8 * T**2)**2) - 0.0278*lmbda**2)**0.5

def n_e(lmbda):
    lmbda *= 1e4
    return (4.5567 + 2.605e-7 * T**2 + (0.097 + 2.7e-8 * T**2) /
            (lmbda**2 - (0.201 + 5.4e-8 * T**2)**2) - 0.0224 * lmbda**2)**0.5

# функция определения диаметра перетяжки:
def w_0(w_in, z0):
    
    #коэфы для определения диаметра перетяжки:
    coef = [1,0, -w_in**2,0, (lmbda * z0 / pi / n_o(lmbda) )**2] 
    w0 = np.abs(np.roots(coef))
    if min(w0) != 0:
        return min(w0)# диаметр перетяжки, [см]


T = 300 #температура, [K]
c = 3e10 #скорость света, [см/c]

lmbda =1.55e-4 #длина волны, [см]
w = 2*pi*c/lmbda #угловая частота, [1/с]

k = w/c *n_o(lmbda) #волновой вектор для частоты w, [1/см]
K = 2*w/c *n_e(lmbda/2) #волновой вектор для частоты 2w, [1/см]

dk = 0 #волновая расстройка, [см^-1]

L = 1 # Длина кристалла ниобата лития, [см]

d = 4e-12 #усредненный коэффициент нелинейности кристалла, [м / В]
d *= 3e4 # перевод из СИ в СГС, [см / статВ]

#коэффициенты нелинейной связи в СГС, [1/статВ]:
sigma1 = 2 * pi * (k) * d / (n_o(lmbda))**2 
sigma2 = sigma1

w_in= float(input('введите диаметр пучка d на входе в кристалл, [мкм]\n'))
#перевод в см:
w_in *= 1e-4

P_in_ci = 10 # начальная мощность первой гармоники, [Вт]
P_in = P_in_ci * 1e7 # мощность первой гармоники, [эрг/с]

I_in = P_in / pi / (w_in / 2) ** 2 # интенсивность первой гармоники, [эрг / см^2]

A1 = complex(np.sqrt(8 * pi * I_in / c / n_o(lmbda))) # комплексная
                                            # амплитуда первой гармоники в СГС

A2 = complex(0) #комплексная амплитуда второй гармоники


#%% Вычисление

def SHG_calc(A1, A2, sigma1, sigma2, dk, L, z0, w0):
    def shortened_eq(z, A):
        dA = np.zeros(2, dtype=complex)
        dA[0] = (-1j * sigma1 * np.conjugate(A[0]) * A[1] * np.exp(-1j * dk * z) -
                          A[0]*((lmbda / (n_o(lmbda)*pi*w0**2))**2 * (z-z0) /
                      (1 + (lmbda*(z-z0) / (n_o(lmbda)*pi*w0**2))**2)))
        dA[1] = (-1j * sigma2  * A[0]**2 * np.exp(1j * dk * z) -
                      A[1]*(((lmbda/2) / (n_e(lmbda/2)*pi*w0**2))**2 * (z-z0) /
                        (1 + ((lmbda/2)*(z-z0) / (n_e(lmbda/2)*pi*w0**2))**2)))
        return dA
    answer = solve_ivp(shortened_eq, [0, L], [A1, A2], dense_output=True,
                        rtol=1e-7, atol=1e-6)
                       
    return answer    

P2 = []

P2_max = 0
z0_opt = 0

if w_in * 1e4 < 100:
    z = np.linspace(0.001, w_in**2 * 10000, 100)
else:
    z = np.linspace(0.001, L, 100)
for z0 in z:
    w0 = w_0(w_in, z0)
    answer = SHG_calc(A1, A2, sigma1, sigma2, dk, L, z0, w0)

    I2 = (c * n_e(lmbda/2) * np.absolute(answer.y[1,-1])**2 / 8 / pi) * 1e-7
    I2 *= pi * (w0 * np.sqrt(1 + (lmbda/2 * (L - z0) /
                                 (n_e(lmbda/2) * pi * w0 ** 2)) ** 2) / 2) ** 2
    if z0 == z[5] or z0 == z[49] or z0 == z[99]:
        Answ = answer.sol(z)
    
        w1 = (w0 * np.sqrt(1 + (lmbda * (z - z0) /
                                        (n_o(lmbda) * pi * w0 ** 2)) ** 2))
        w2 = (w0 * np.sqrt(1 + (lmbda * (z - z0) /
                                      (n_e(lmbda/2) * pi * w0 ** 2)) ** 2))
    
        
        P2w = ((c * n_e(lmbda/2) * np.abs(Answ[1])**2 / 8 / pi) * 1e-7 
                  * pi * (w2 / 2) ** 2)
        Pw = ((c * (n_o(lmbda) * np.abs(Answ[0])**2 / 8 / pi)
                                                  * 1e-7 * pi * (w1 / 2) ** 2))
        
        fig,ax = plt.subplots()
        ax.set_xlabel('$z$ [cm]')
        ax.set_ylabel('$P1$, Вт')
        ax.grid()
        if z0 == z[5]: title = 'начале'
        if z0 == z[49]: title = 'середине'
        if z0 == z[99]: title = 'конце'
        ax.set_title('График P(z), перетяжка в ' + title)
        ax.plot(z, Pw, 'r')
        ax2 = ax.twinx()
        ax2.plot(z, P2w)
        ax2.set_xlabel('$z$, cm')
        ax2.set_ylabel('$P2$, Вт')
        fig.legend(('Первая гармоника', 'Вторая гармоника'), loc='lower left')
        plt.show()    
   
    P2.append(I2)
    if P2_max < I2:
        P2_max = I2
        z0_opt = z0

print('z0_opt = ',  round(z0_opt,5) , 'cm', ' P2_maxx = ', round(P2_max,5), 'Вт')



#%% Графики

plt.figure('2', clear=True) 
plt.title("График зависимости P(z0)")
plt.xlabel("Координата перетяжки z0, см")
plt.ylabel("Мощность второй гармоники P, Вт")
plt.plot(z, P2, 'r')
plt.grid()

# plt.figure('2', clear=True) 
# plt.title("График зависимости второй гармоники от координаты перетяжки P(z0)")
# plt.xlabel("Координата перетяжки z0, см")
# plt.ylabel("Мощность второй гармоники P, Вт")
# plt.plot((z[0], z[50], z[99]), (P2[0], P2[50], P2[99]), 'r')
# plt.grid()

# plt.figure('3', clear=True) 
# plt.title("График зависимости первой гармоники от координаты перетяжки P(z0)")
# plt.xlabel("Координата перетяжки z0, см")
# plt.ylabel("Мощность второй гармоники P, Вт")
# plt.plot((z[0], z[50], z[99]), (P1[0], P1[50], P1[99]), 'r')
# plt.grid()