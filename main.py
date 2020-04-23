import copy
import random
from env import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
F.X = odeint(state,F.X[-1,:],time,args=(u,0))
from fnn import fuzzy_nerual_networks


def main(F, case, example, t1, save=False):
    if example == 1:
        x1, x2 = 3, 3
        x1_,x2_=-1, -1
        p = 50
    else:
        x1, x2 = -(np.pi/60), 0
        x1_,x2_= -(1/2), 0
        p = 100
    xhat = np.array([x1_,x2_])

    F.X[:,:] = [x1,x2]
    F.E[:,:] = list(ym(case) - xhat)

    X = [x1]
    Xhat = [x1_]
    T = [0]
    U = []
    UF = []
    US = []
    UD = []
    E = []

    weight = F.W()

    time_step = 0.001
    size = int(t1 / time_step)
    times = np.linspace(0, t1, size)
    t0 = 0

    for count, t in enumerate(times[1:]):
        time = [t0, t]

        # error generation
        e_wave_array = ym(case,t) - F.X[-1,:]
        e_wave_array = e_wave_array - F.E[-1,:]
        e_wave = e_wave_array[0]

        # error diffential    
        F.E = odeint(error,F.E[-1,:],time,args=(e_wave,0))
        F.set_e(F.E)

        # u generation
        V = 1/2 * F.dot(e_wave_array.reshape((1,2)), F.P(), e_wave_array)
        uf, weight = F.RGA(e_wave, weight, t, t1)
        ud = F.Ud(e_wave, p=p)
        us = F.Us(e_wave, uf, V)
        u = float(uf + us + ud)
        
        # plant
        if example == 1:
            F.X = odeint(state,F.X[-1,:],time,args=(u,0))
        else:
            F.X = odeint(state1,F.X[-1,:],time,args=(u,0))

        # data collection
        Xhat.append(ym(case,t)-F.E[-1,0])
        for x in F.X[:,0]:
            X.append(x)
        for t in time:
            T.append(t)

        E.append(e_wave)
        U.append(u)
        UF.append(uf)
        US.append(us)
        UD.append(ud)

        t0 = t
        print(round(count/size*100,2),'\r',end='')

    name = f"example{example}_case{case}_time{t1}"

    plt.plot(T, X, c='g')
    plt.plot(times, Xhat, c='r')
    if save:
        plt.savefig(f'{name}_output.pdf')
        plt.close()
    else:
        plt.show()

    plt.plot(times[1:], E)
    if save:
        plt.savefig(f'{name}_e_hat.pdf')
        plt.close()
    else:
        plt.show()

    # plt.plot(times[1:], UF, c='r')
    # plt.plot(times[1:], US, c='g')
    # plt.plot(times[1:], UD, c='b')
    plt.plot(times[1:], U, c='k')
    if save:
        plt.savefig(f'{name}_control_input_u.pdf')
        plt.close()
    else:
        plt.show()

#[1, 2]
examples = [1]
cases = [1]
ts = [1]
F = fuzzy_nerual_networks()
for t1 in ts:
    for example in examples:
        for case in cases:
            print(f't:{t1}, case:{case}, example:{example}')
            main(F, case, example, t1, save=False)
      
