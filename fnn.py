from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

class fuzzy_nerual_networks():
    def __init__(self,
                 size=100
                ):
        self.X = np.zeros((size,2))
        self.E = np.zeros((size,2))

        self.D = [-10, 10]
        self.Q = np.eye(2) * 500
        self.Kc = np.array([[144, 24]])
        
    def dot(self, array, *args):
        for arg in args:
            array = np.dot(array, arg)
        return array
    
    def W(self):
        weight = np.zeros((4,49))
        for i in range(4):
            for j in range(49):
                weight[i,j] = np.random.uniform(self.D[0], self.D[1])
        return weight
    
    def RGA(self, e_wave, weight, t, T):
        weight, j = self.SSCP(e_wave, weight)
        weight = self.crossover(weight, j)
        weight = self.sort(e_wave, weight)
        weight = self.mutation(weight, j, t, T)
        uf = self.Uf(weight[0])
        return uf, weight
    
    def SSCP(self, e_wave, weight):
        old = weight.copy()
        for i in range(49):
            weight = self.crossover(weight, i)
            if self.fitness(e_wave, weight[0,:]) < 0:
                return weight, i
        return old, i
    
    def crossover(self, weight, i):
        for j in range(4):
            α = random.random()
            if j < 2:
                weight[j,i] = weight[j,i]*(1-α) + weight[j+2,i]*α
            else:
                weight[j,i] = weight[j,i]*(1-α) + weight[j-2,i]*α
        return weight
    
    def sort(self, e_wave, weight):
        fits = self.fitness(e_wave, weight)
        return weight[np.argsort(fits[:,0]),:]
        
    def mutation(self, weight, j, t, T, γ=0.05):
        delta = (1 - t/T)*γ + γ
        if random.random() > 0.5:
            weight[2,j] = weight[0,j] + delta
        else:
            weight[2,j] = weight[0,j] - delta
        return weight

    def fitness(self, e_wave, weight):
        u_star = self.U_star()
        uf = self.Uf(weight)
        fit = -1/2*self.Q[0,0]*(e_wave**2) + abs(e_wave)*(abs(uf) + abs(u_star))
        return fit
    
    def Uf(self, weight):
        return np.dot(weight, self.ϕA())
    
    def ϕA(self):
        ϕ = np.zeros((49,1))
        for i in range(7):
            for j in range(7):
                ϕ[7 * i + j] = self.μA(i, self.E[-1,0]) * self.μA(j, self.E[-1,1])
        return ϕ / ϕ.sum()
    
    def Ud(self, e_wave, p=50 ,α=0.01):
        if e_wave >= 0 and abs(e_wave) > α:
            return p
        elif e_wave < 0 and abs(e_wave) > α:
            return -p
        elif abs(e_wave) < α:
            return p*e_wave/α
    
    def set_e(self, E):
        self.E = E

    def get_e(self):
        return self.E[-1,:]

    def Us(self, e_wave, uf, V, V_=0.005):
        u_star = self.U_star()
        return np.sin(e_wave)*(abs(uf) + abs(u_star)) if V > V_ and V > 0 else 0
        
    def U_star(self):
        e = self.get_e().reshape((2,1))
        return 15.3 + 0 + abs(np.dot(self.Kc, e))
        
    def P(self):
        a = (-500 + 900 * -500) / (-120)
        b = -250
        c = -500 - b
        d = (-a + 60 * b) / (-900)
        return np.array([[a, b],[c, d]])
    
    def μA(self, i, e):
        i += 1
        bound = 136
        if e < -bound:
            e = -bound
        elif e > bound:
            e = bound
        if i == 1:
            return 1 / (1 + np.exp(5 * (e + 3)))
        elif i == 2:
            return np.exp(-((e + 2)**2))
        elif i == 3:
            return np.exp(-((e + 1)**2))
        elif i == 4:
            return np.exp(-((e)**2))
        elif i == 5:
            return np.exp(-((e - 1)**2))
        elif i == 6:
            return np.exp(-((e - 2)**2))
        elif i == 7:
            return 1 / (1 + np.exp(-5 * (e - 3)))