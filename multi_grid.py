# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 07:37:48 2021

@author: Yanis Zatout
"""

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import time as t

def gauss_seidel(L,f,u,n):
    for iter in range(n):
        for k in range(np.size(L,axis=0)):
            u[k] = (f[k] - L[k,0:k]@u[0:k] - L[k,k+1:]@u[k+1:])/L[k,k]
    return u

def vcycle(L,f):
    """
    """
    sizeF=np.size(L,axis=0)
    
    if sizeF<2:
        u=npl.solve(L,f)
        #résolution de la librairie np.linalg
        return u
    
    N1=5
    u=np.zeros(sizeF)
    u = gauss_seidel(L,f,u,N1)

    sizeC=int((sizeF-1)/2+1)
    
    P=np.zeros((sizeF,sizeC))
    d=np.arange(sizeC)
    d_=d[:-1]
    P[2*d,d]=1
    P[2*d_+1,d_]=.5
    P[2*d_+1,d_+1]=.5
    
    residu = f - L @ u
    
    
    residuC = P.T @ residu
    
    LC = P.T @ L @ P
    
    uC = vcycle(LC,residuC)
    
    u = P @ uC
    
    N2 = 5
    u = gauss_seidel(L,f,u,N2)     
    return u


N=2**9+1
x=np.linspace(0,1,N)

h=x[1]-x[0]

L = np.diag(2.*np.ones(N)) - np.diag(np.ones(N-1),  1) - np.diag(np.ones(N-1),  -1)

f=np.ones(N, dtype=float) 
t2=t.time()
udirect=npl.solve(L,f)
temps_direct=t.time()-t2



u=np.zeros(N)
e=[]
temps_vcycle=0
for iter in range(100):
    r = f - L @ u 
    
    if npl.norm(r)/npl.norm(f) < 1.e-10:
        break
    
    
    t3=t.time()
    du=vcycle(L,r)
    temps_vcycle+=(t.time()-t3)
    u+=du
    erreur=npl.norm(u-udirect)/npl.norm(udirect)
    e.append(erreur)
    print("step {step}, rel error={err}".format(step=iter+1,err= erreur))

print("Temps d'execution méthode directe: " +str(temps_direct) + " s")
print("Temps d'execution méthode multi-grille: " +str(temps_vcycle) + " s")

fig, ax = plt.subplots(); 

ax.plot(e)
plt.title("Erreurs du V-Cycle")
plt.show()


fig, (ax1,ax2) = plt.subplots(1,2); 
fig.suptitle("Comparaison des solutions")


ax1.plot(udirect)
ax1.set_title("Solution directe")

ax2.plot(u)
ax2.set_title("Solution du v-cycle")

plt.show()
