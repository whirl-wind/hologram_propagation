# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:13:24 2020

@author: ZhanZhang

Email: whirlwind@mail.ustc.edu.cn

"""

import torch

###############################################################################
#FFTShift
###############################################################################

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(0,len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, -1, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

###############################################################################
#Sampling
###############################################################################

def up_sampling(I,Tx,Ty):
    target = torch.zeros(Ty,Tx)
    Ix = I.size(1)
    Iy = I.size(0)
    
    def sample(idx):
        i = idx//Ty
        j = idx%Tx
        return I[i*(Iy-1)//(Ty-1),j*(Ix-1)//(Tx-1)]
    
    idx = torch.arange(Tx*Ty)
    b = list(map(sample,idx))
    target = torch.tensor(b).reshape(Ty,Tx)
        
    return target

def roll_extend(I,Tx,Ty):
    target = torch.zeros(Ty,Tx)
    Ix = I.size(1)
    Iy = I.size(0)
    if(Tx<=Ix or Ty<=Iy):
        print("Size is too small!")
        return I
    
    def sample(idx):
        i = idx//Ty
        j = idx%Tx
        return I[(i-(Ty-Iy)//2)%(Iy-1),(j-(Tx-Ix)//2)%(Ix-1)]
    
    idx = torch.arange(Tx*Ty)
    b = list(map(sample,idx))
    target = torch.tensor(b).reshape(Ty,Tx)
    
    return target

def i_roll_extend(T,Ix,Iy):
    target = torch.zeros(Iy,Ix)
    Tx = T.size(1)
    Ty = T.size(0)
    target = T[int((Ty-Iy)/2):int((Ty-Iy)/2)+Iy,int((Tx-Ix)/2):int((Tx-Ix)/2)+Ix]
    return target