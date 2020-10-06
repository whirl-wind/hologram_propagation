# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:38:23 2020

@author: ZhanZhang

Email: whirlwind@mail.ustc.edu.cn

"""

import math
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import Propagation as Prop
import math_utils

def Opt_GS(phase,target,rs_p,iter_num,propgation_method,zero_padding):
    with torch.no_grad():
        for it in range(iter_num):
            rs_p.prop(phase,zero_padding,propgation_method)
            norm = torch.sqrt(torch.sum(rs_p.target.result*rs_p.target.result,1))
            norm = torch.stack((norm,norm),1)
            real,imag =  torch.unbind((rs_p.target.result/norm).reshape(rs_p.target.Dy,rs_p.target.Dx,2),-1)
            zero = torch.zeros_like(imag)
            one = torch.ones_like(imag)
            phase = torch.acos(real)+torch.where(imag<0,one,zero)*math.pi
            rs_p.target.amplitude = torch.sqrt(target)
            phase = rs_p.iprop(phase,zero_padding,propgation_method)   
        return phase

def Opt_autodiff(phase,target,rs_p,iter_num,propgation_method,optimizer_method,zero_padding):
    SLM_Dx = rs_p.aperture.Ax
    SLM_Dy = rs_p.aperture.Ay
    phase.requires_grad_(True)
    if optimizer_method == 'SGD':
        optimizer = torch.optim.SGD(params=[phase], lr=0.1,momentum=0.9)
    elif optimizer_method == 'Adagard':
        optimizer = torch.optim.Adagrad(params=[phase], lr=0.01)
    elif optimizer_method == 'Adam':
        optimizer = torch.optim.Adam(params=[phase], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    for t in range(iter_num):
        optimizer.zero_grad() 
        rs_p.Angular_spectrum(phase,zero_padding)
        im = torch.sum(rs_p.target.result*rs_p.target.result,1).reshape(rs_p.target.Dy,rs_p.target.Dx)
        recon = math_utils.i_roll_extend(im,SLM_Dx,SLM_Dy)
        #recon = (im-torch.min(im))/(torch.max(im)-torch.min(im))
        
        loss = (recon - target).pow(2).sum()/(SLM_Dx*SLM_Dy)
        if t % 100 == 99:
            print(t, loss.item())
    
        loss.backward()
        with torch.no_grad():
            #print(phase.grad)
            optimizer.step()
            scheduler.step(loss)
            #phase -= learning_rate * phase.grad
            #phase.grad.zero_()
    return phase
    
def demo():
    
    dtype = torch.float
    device = torch.device("cpu")
    
    SLM_Dx = 256
    SLM_Dy = 256
    T_Dx = 512
    T_Dy = 512
    T_Dz = 10000
    iter_num = 1000
    
    zero_m = torch.nn.ZeroPad2d(((T_Dx-SLM_Dx)//2,(T_Dx-SLM_Dx)//2,(T_Dy-SLM_Dy)//2,(T_Dy-SLM_Dy)//2))
    
    rs_p = Prop.Propagation(Prop.Aperture(T_Dx,T_Dy,SLM_Dx,SLM_Dy,'rectangle'), Prop.Target(T_Dx,T_Dy,T_Dz,'parallel_plane'), Prop.Light())
    rs_p.aperture.update_sampleratio(math.sqrt(2*T_Dz*rs_p.target.sample_ratio/T_Dy)) #set aperture to make Fraunhofer the same as ASM
    rs_p.aperture.roll_phase()
    phase = torch.rand((SLM_Dy,SLM_Dx), device=device, dtype=dtype)*math.pi*2
        
    I = mpimg.imread('./data/test_data/texture_test.bmp')
    Target = torch.sum(torch.tensor(I),dim=2).float()/3/255
    Target_0 = math_utils.up_sampling(Target, SLM_Dx, SLM_Dy)
    Target = math_utils.roll_extend(Target_0,T_Dx,T_Dy)
    
    init_phase = Opt_GS(phase,Target,rs_p,30,"ASM",zero_m)
    
    begin_time = time.time()
    phase = Opt_autodiff(init_phase,Target_0,rs_p,iter_num,propgation_method="ASM",optimizer_method="Adam",zero_padding=zero_m)
    end_time = time.time()
    
    print("Time：%f s"% (end_time - begin_time))
    
    with torch.no_grad():
        fig = plt.figure()
            
        ax1 = fig.add_subplot(221)
        ax1.set_title("Target")
        #ax1.imshow(Target,cmap="gray")
        ax1.imshow(math_utils.i_roll_extend(Target,SLM_Dx,SLM_Dy),cmap="gray")
        
        ax2 = fig.add_subplot(222)
        ax2.set_title("Recon")
        rs_p.Angular_spectrum(phase,zero_m)
        #ax2.imshow(rs_p.target.image(),cmap="gray")
        ax2.imshow(math_utils.i_roll_extend(rs_p.target.image(),SLM_Dx,SLM_Dy),cmap="gray")
        
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    demo()