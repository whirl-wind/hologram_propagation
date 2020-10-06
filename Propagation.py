# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:11:37 2020

@author: ZhanZhang

Email: whirlwind@mail.ustc.edu.cn

"""

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import math_utils

dtype = torch.float
device = torch.device("cpu")

class Light:
    def __init__(self):
        self._lambda = 1
        self.k = 2*math.pi/self._lambda

class Aperture:
    def __init__(self, Dx, Dy, Ax, Ay, pattern):
        self.Dx = Dx
        self.Dy = Dy
        self.Ax = Ax
        self.Ay = Ay
        if pattern == 'rectangle':
            if(Ax==Dx and Ay==Dy):
                self.amplitude = torch.ones((Dy,Dx), device=device, dtype=dtype)
            else:
                self.amplitude = torch.zeros((Dy,Dx), device=device, dtype=dtype)
                for i in range(Ay):
                    for j in range(Ax):
                        self.amplitude[i+int((Dy-Ay)/2),j+int((Dx-Ax)/2)] = 1
        elif pattern == 'circle':
            self.amplitude = torch.zeros((Dy,Dx), device=device, dtype=dtype)
            for i in range(Ay):
                for j in range(Ax):
                    x = j-(Ax-1)/2
                    y = i-(Ay-1)/2
                    if (x*x)/((Ax-1)*(Ax-1))+(y*y)/((Ay-1)*(Ay-1))<=0.25:
                        self.amplitude[i+int((Dy-Ay)/2),j+int((Dx-Ax)/2)] = 1
        self.phase = torch.ones((Dy,Dx), device=device, dtype=dtype)*math.pi
        self.sample_ratio = 20 # lambda/2
        self.n = torch.cat([torch.zeros((Dy,Dx,1)),torch.zeros((Dy,Dx,1)),torch.ones((Dy,Dx,1))],2)
        idx = torch.stack([torch.arange(Dy).reshape((Dy,1)).repeat(1,Dx),torch.arange(Dx).reshape((1,Dx)).repeat(Dy,1)],2)
        idx = torch.cat([idx.long(),torch.zeros((Dy,Dx,1)).long()],2)
        self.position = idx.float()-torch.tensor([[[(Dy-1)/2,(Dx-1)/2,0]]])
    
    def update_sampleratio(self,sample_ratio):
        self.sample_ratio = sample_ratio
    def set_position_with_light(self,light):
        idx = torch.stack([torch.arange(self.Dy).reshape((self.Dy,1)).repeat(1,self.Dx),torch.arange(self.Dx).reshape((1,self.Dx)).repeat(self.Dy,1)],2)
        idx = torch.cat([idx.long(),torch.zeros((self.Dy,self.Dx,1)).long()],2)
        self.position = idx.float()-torch.tensor([[[(self.Dy-1)/2,(self.Dx-1)/2,0]]])
        self.position = self.position*self.sample_ratio*light._lambda/2
    def roll_phase(self):
        self.phase = torch.rand((self.Ay,self.Ax), device=device, dtype=dtype)*math.pi*2
    def show(self):
        plt.imshow(self.amplitude*self.amplitude)
        
class Target:
    def __init__(self,Dx,Dy,Dz,pattern):
        self.Dx = Dx
        self.Dy = Dy
        self.Dz = Dz
        self.sample_ratio = 2 # lambda/2
        self.pattern = pattern
        if pattern == 'parallel_plane': 
            self.size = Dx*Dy
            idx = torch.stack([torch.arange(Dy).reshape((Dy,1)).repeat(1,Dx),torch.arange(Dx).reshape((1,Dx)).repeat(Dy,1)],2)
            idx = torch.cat([idx.long(),torch.zeros((Dy,Dx,1)).long()],2)
            self.position = idx.float()-torch.tensor([[[(Dy-1)/2,(Dx-1)/2,-Dz]]])
        elif pattern == 'vertical_plane': 
            self.size = Dy*Dz
            idx = torch.stack([torch.arange(Dy).reshape((Dy,1)).repeat(1,Dz),torch.arange(Dz).reshape((1,Dz)).repeat(Dy,1)],2)
            idx = torch.cat([torch.zeros((Dy,Dz,1)).long(),idx.long()],2)
            self.position = idx.float()-torch.tensor([[[0,(Dy-1)/2,0]]])
        self.position = self.position.reshape(self.size,3)
    def set_position_with_light(self,light):
        if self.pattern == 'parallel_plane': 
            idx = torch.stack([torch.arange(self.Dy).reshape((self.Dy,1)).repeat(1,self.Dx),torch.arange(self.Dx).reshape((1,self.Dx)).repeat(self.Dy,1)],2)
            idx = torch.cat([idx.long(),torch.zeros((self.Dy,self.Dx,1)).long()],2)
            self.position = idx.float()-torch.tensor([[[(self.Dy-1)/2,(self.Dx-1)/2,-self.Dz]]])
            self.position = self.position.reshape(self.size,3)
            self.position = self.position*self.sample_ratio*light._lambda/2
        elif self.pattern == 'vertical_plane': 
            idx = torch.stack([torch.arange(self.Dy).reshape((self.Dy,1)).repeat(1,self.Dz),torch.arange(self.Dz).reshape((1,self.Dz)).repeat(self.Dy,1)],2)
            idx = torch.cat([torch.zeros((self.Dy,self.Dz,1)).long(),idx.long()],2)
            self.position = idx.float()-torch.tensor([[[0,(self.Dy-1)/2,0]]])
            self.position = self.position.reshape(self.size,3)
            self.position = self.position*self.sample_ratio*light._lambda/2
    def update_sampleratio(self,sample_ratio):
        self.Dz = self.sample_ratio*self.Dz/sample_ratio
        self.sample_ratio = sample_ratio
    def image(self):
        I = torch.sum(self.result*self.result,1)
        if self.pattern == 'parallel_plane':
            I = I.reshape(self.Dy,self.Dx)
        elif self.pattern == 'vertical_plane':
            I = I.reshape(self.Dy,self.Dz)
        return I
    def show(self):
        plt.imshow(self.image())

class Propagation:
    def __init__(self, aperture, target, light):
        self.aperture = aperture
        self.target = target
        self.light = light
        self.target.set_position_with_light(light)
        self.aperture.set_position_with_light(light)
        
    def diffraction_RS(self,phase,zero_padding):
        _2pi = 1/(2*math.pi)
        if zero_padding:
            self.aperture.phase = phase
            self.aperture.phase = zero_padding(phase)
            self.aperture.real = self.aperture.amplitude * torch.cos(zero_padding(phase))
            self.aperture.imag = self.aperture.amplitude * torch.sin(zero_padding(phase))
        else:
            self.aperture.phase = phase
            self.aperture.real = self.aperture.amplitude * torch.cos(phase)
            self.aperture.imag = self.aperture.amplitude * torch.sin(phase)
        idx = torch.arange(self.target.size)
        def RS_integration(idx):
            position = self.target.position[idx]
            r = F.torch.norm(position-self.aperture.position,p=2,dim=2)
            cos_rn = torch.sum(self.aperture.n*F.normalize(position-self.aperture.position, p=2, dim=2),dim=2)
            real = cos_rn*(torch.cos(self.light.k*r)/(r*r)+self.light.k*torch.sin(self.light.k*r)/r)
            imag = cos_rn*(torch.sin(self.light.k*r)/(r*r)-self.light.k*torch.cos(self.light.k*r)/r)
            target_real = _2pi*torch.sum(self.aperture.real*real-self.aperture.imag*imag)
            target_imag = _2pi*torch.sum(self.aperture.real*imag+self.aperture.imag*real)
            return [target_real.item(),target_imag.item()]
        b = list(map(RS_integration,idx))
        self.target.result = torch.tensor(b)
        return self.target.result
    
    def idiffraction_RS(self,phase,zero_padding):
        _2pi = 1/(2*math.pi)
        if zero_padding:
            self.aperture.phase = phase
            self.aperture.phase = zero_padding(phase)
            self.aperture.real = self.aperture.amplitude * torch.cos(zero_padding(phase))
            self.aperture.imag = self.aperture.amplitude * torch.sin(zero_padding(phase))
        else:
            self.aperture.phase = phase
            self.aperture.real = self.aperture.amplitude * torch.cos(phase)
            self.aperture.imag = self.aperture.amplitude * torch.sin(phase)
        idx = torch.arange(self.target.size)
        def iRS_integration(idx):
            position = self.aperture.position[idx]
            r = F.torch.norm(position-self.aperture.position,p=2,dim=2)
            cos_rn = torch.sum(self.aperture.n*F.normalize(position-self.aperture.position, p=2, dim=2),dim=2)
            real = cos_rn*(torch.cos(-self.light.k*r)/(r*r)+self.light.k*torch.sin(-self.light.k*r)/r)
            imag = cos_rn*(torch.sin(-self.light.k*r)/(r*r)-self.light.k*torch.cos(-self.light.k*r)/r)
            target_real = _2pi*torch.sum(self.target.real*real-self.target.imag*imag)
            target_imag = _2pi*torch.sum(self.target.real*imag+self.target.imag*real)
            return [target_real.item(),target_imag.item()]
        b = list(map(iRS_integration,idx))
        norm = torch.sqrt(torch.sum(b*b,1))
        norm = torch.stack((norm,norm),1)
        real,imag = torch.unbind(torch.tensor(b/norm),-1)
        zero = torch.zeros_like(imag)
        one = torch.ones_like(imag)
        self.aperture.phase = torch.acos(real)+torch.where(imag<0,one,zero)*math.pi
        return self.aperture.phase
    
    def FraunhoferFFT_sampleratio_adjust(self):
        self.target.update_sampleratio(2*self.target.Dz*self.target.sample_ratio/self.target.Dy/self.aperture.sample_ratio)
        self.target.set_position_with_light(self.light)
        
    def Fraunhofer_approximation(self,phase,fft_mod,zero_padding):
        idx = torch.arange(self.target.size)
        if zero_padding:
            self.aperture.phase = phase
            self.aperture.phase = zero_padding(phase)
            self.aperture.real = self.aperture.amplitude * torch.cos(zero_padding(phase))
            self.aperture.imag = self.aperture.amplitude * torch.sin(zero_padding(phase))
        else:
            self.aperture.phase = phase
            self.aperture.real = self.aperture.amplitude * torch.cos(phase)
            self.aperture.imag = self.aperture.amplitude * torch.sin(phase)
        lambda_1 = 1/self.light._lambda
        z = self.target.Dz*self.target.sample_ratio*self.light._lambda/2
        def Fraunhofer_integration(idx):
            position = self.target.position[idx]
            temp = -torch.sum(self.light.k*position*self.aperture.position/z,dim=2)
            
            real_1 = torch.sin(self.light.k*((position[0]*position[0]+position[1]*position[1])*0.5/z+z))*lambda_1/z
            imag_1 = -torch.cos(self.light.k*((position[0]*position[0]+position[1]*position[1])*0.5/z+z))*lambda_1/z
            real_2 = torch.cos(temp)
            imag_2 = torch.sin(temp)
            
            real = real_1*real_2-imag_1*imag_2
            imag = real_1*imag_2+real_2*imag_1
            
            target_real = torch.sum(self.aperture.real*real-self.aperture.imag*imag)
            target_imag = torch.sum(self.aperture.real*imag+self.aperture.imag*real)
            return [target_real.item(),target_imag.item()]
        
        if self.target.pattern == 'parallel_plane':
            if fft_mod:
                self.FraunhoferFFT_sampleratio_adjust()
                
                aperture = torch.stack((self.aperture.real,self.aperture.imag),2)
                position = self.target.position - torch.tensor([[[0,0,z]]])
                real_1 = torch.sin(self.light.k*(torch.sum(position*position,dim=2)*0.5/z+z))*self.light._lambda*z
                imag_1 = -torch.cos(self.light.k*(torch.sum(position*position,dim=2)*0.5/z+z))*self.light._lambda*z
                real_1 = real_1.reshape(self.target.Dy,self.target.Dx)
                imag_1 = imag_1.reshape(self.target.Dy,self.target.Dx)
                real_2,imag_2 = torch.unbind(math_utils.batch_ifftshift2d(torch.fft(math_utils.batch_fftshift2d(aperture),2)),-1)
                
                target_real = real_2*real_1-imag_2*imag_1
                target_imag = real_2*imag_1+imag_2*real_1
                
                b = torch.stack((target_real,target_imag),2)
                self.target.result = b.reshape(self.target.size,2)
                return self.target.result
            else:
                b = list(map(Fraunhofer_integration,idx))
                self.target.result = torch.tensor(b)
                return self.target.result
        else:
            print('Using RS')
            return self. diffraction_RS()

    def FresnelFFT_sampleratio_adjust(self):
        self.target.update_sampleratio(2*self.target.Dz*self.target.sample_ratio/self.target.Dy/self.aperture.sample_ratio)
        self.target.set_position_with_light(self.light)

    def Fresnel_approximation(self,phase,fft_mod,zero_padding):
        idx = torch.arange(self.target.size)
        if zero_padding:
            self.aperture.phase = phase
            self.aperture.phase = zero_padding(phase)
            self.aperture.real = self.aperture.amplitude * torch.cos(zero_padding(phase))
            self.aperture.imag = self.aperture.amplitude * torch.sin(zero_padding(phase))
        else:
            self.aperture.phase = phase
            self.aperture.real = self.aperture.amplitude * torch.cos(phase)
            self.aperture.imag = self.aperture.amplitude * torch.sin(phase)
        lambda_1 = 1/self.light._lambda
        z = self.target.Dz*self.target.sample_ratio*self.light._lambda/2
        def Fresnel_integration(idx):
            position = self.target.position[idx]
            temp = torch.sum(0.5*self.light.k*(position-self.aperture.position)*(position-self.aperture.position)/z,dim=2)
            
            real_1 = torch.sin(torch.tensor(self.light.k*z))*lambda_1/z
            imag_1 = -torch.cos(torch.tensor(self.light.k*z))*lambda_1/z
            real_2 = torch.cos(temp)
            imag_2 = torch.sin(temp)
            
            real = real_1*real_2-imag_1*imag_2
            imag = real_1*imag_2+real_2*imag_1
            
            target_real = torch.sum(self.aperture.real*real-self.aperture.imag*imag)
            target_imag = torch.sum(self.aperture.real*imag+self.aperture.imag*real)
            return [target_real.item(),target_imag.item()]

        if self.target.pattern == 'parallel_plane':
            if fft_mod:
                self.FresnelFFT_sampleratio_adjust()
                
                position = self.target.position - torch.tensor([[[0,0,z]]])
                temp = -torch.sum(0.5*self.light.k*self.aperture.position*self.aperture.position/z,dim=2)
                
                real_1 = torch.sin(self.light.k*(torch.sum(position*position,dim=2)*0.5/z+z))*self.light._lambda*z
                imag_1 = -torch.cos(self.light.k*(torch.sum(position*position,dim=2)*0.5/z+z))*self.light._lambda*z
                real_1 = real_1.reshape(self.target.Dy,self.target.Dx)
                imag_1 = imag_1.reshape(self.target.Dy,self.target.Dx)
                real_2 = torch.cos(temp)
                imag_2 = torch.sin(temp)
                
                real = self.aperture.real*real_2-self.aperture.imag*imag_2
                imag = self.aperture.real*imag_2+real_2*self.aperture.imag
                temp = torch.stack((real,imag),2)
                
                real_2,imag_2 = torch.unbind(math_utils.batch_ifftshift2d(torch.fft(math_utils.batch_fftshift2d(temp),2)),-1)
                
                target_real = real_2*real_1-imag_2*imag_1
                target_imag = real_2*imag_1+imag_2*real_1
                
                b = torch.stack((target_real,target_imag),2)
                self.target.result = b.reshape(self.target.size,2)
                return self.target.result
            else:
                b = list(map(Fresnel_integration,idx))
                self.target.result = torch.tensor(b)
                return self.target.result
        else:
            print('Using RS')
            return self. diffraction_RS()
    
    def ASM_sampleratio_adjust(self):
        self.target.update_sampleratio(self.aperture.sample_ratio)
        self.target.set_position_with_light(self.light)
    
    def Angular_spectrum(self, phase, zero_padding):
        self.ASM_sampleratio_adjust()
        if zero_padding:
            self.aperture.phase = phase
            self.aperture.phase = zero_padding(phase)
            self.aperture.real = self.aperture.amplitude * torch.cos(zero_padding(phase))
            self.aperture.imag = self.aperture.amplitude * torch.sin(zero_padding(phase))
        else:
            self.aperture.phase = phase
            self.aperture.real = self.aperture.amplitude * torch.cos(phase)
            self.aperture.imag = self.aperture.amplitude * torch.sin(phase)
    
        idx = torch.stack([torch.arange(self.aperture.Dy).reshape((self.aperture.Dy,1)).repeat(1,self.aperture.Dx),torch.arange(self.aperture.Dx).reshape((1,self.aperture.Dx)).repeat(self.aperture.Dy,1)],2)
        idx = torch.cat([idx.long(),torch.zeros((self.aperture.Dy,self.aperture.Dx,1)).long()],2)
        
        if self.target.pattern == 'parallel_plane':
            aperture = torch.stack((self.aperture.real,self.aperture.imag),2)
            real_1,imag_1 = torch.unbind( math_utils.batch_ifftshift2d(torch.fft(math_utils.batch_fftshift2d(aperture),2)),-1)
            
            #alpha,beta,gamma = torch.unbind(F.normalize(self.target.position.reshape(self.target.Dy,self.target.Dy,3), p=2, dim=2),-1)
            alpha = 2/self.aperture.Dy/self.target.sample_ratio
            beta = 2/self.aperture.Dx/self.target.sample_ratio
            n_p = (idx.float()-torch.tensor([[[(self.aperture.Dy-1)/2,(self.aperture.Dx-1)/2,0]]]))*torch.tensor([[[alpha,beta,0]]])
            temp = 1-torch.sum(n_p*n_p,dim=2)
            zero = torch.zeros_like(temp)
            gamma = torch.sqrt(torch.where(temp < 0, zero, temp))
            
            real_2 = torch.cos(math.pi*self.target.sample_ratio*self.target.Dz*gamma)
            imag_2 = torch.sin(math.pi*self.target.sample_ratio*self.target.Dz*gamma)
            
            target_real = real_2*real_1-imag_2*imag_1
            target_imag = real_2*imag_1+imag_2*real_1
            
            target = torch.stack((target_real,target_imag),2)
            b = math_utils.batch_ifftshift2d(torch.ifft( math_utils.batch_fftshift2d(target),2))
            self.target.result = b.reshape(self.target.size,2)
            return self.target.result
        else:
            print('Using RS')
            return self. diffraction_RS()
    
    def iAngular_spectrum(self, phase,zero_padding):
        self.ASM_sampleratio_adjust()
        self.target.phase = phase
        self.target.real = self.target.amplitude * torch.cos(phase)
        self.target.imag = self.target.amplitude * torch.sin(phase)
    
        idx = torch.stack([torch.arange(self.aperture.Dy).reshape((self.aperture.Dy,1)).repeat(1,self.aperture.Dx),torch.arange(self.aperture.Dx).reshape((1,self.aperture.Dx)).repeat(self.aperture.Dy,1)],2)
        idx = torch.cat([idx.long(),torch.zeros((self.aperture.Dy,self.aperture.Dx,1)).long()],2)
        
        if self.target.pattern == 'parallel_plane':
            b = torch.stack((self.target.real,self.target.imag),2)
            target = math_utils.batch_fftshift2d(torch.fft( math_utils.batch_ifftshift2d(b),2))
            target_real, target_imag = torch.unbind(target,-1)
                   
            alpha = 2/self.aperture.Dx/self.target.sample_ratio
            beta = 2/self.aperture.Dy/self.target.sample_ratio
            n_p = (idx.float()-torch.tensor([[[(self.aperture.Dy-1)/2,(self.aperture.Dx-1)/2,0]]]))*torch.tensor([[[alpha,beta,0]]])
            temp = 1-torch.sum(n_p*n_p,dim=2)
            zero = torch.zeros_like(temp)
            gamma = torch.sqrt(torch.where(temp < 0, zero, temp))
            real_2 = torch.cos(-math.pi*self.target.sample_ratio*self.target.Dz*gamma)
            imag_2 = torch.sin(-math.pi*self.target.sample_ratio*self.target.Dz*gamma)
            
            real_1 = real_2*target_real-imag_2*target_imag
            imag_1 = real_2*target_imag+imag_2*target_real
            
            z_1 = torch.stack((real_1,imag_1),2)
            aperture = math_utils.batch_fftshift2d(torch.ifft(math_utils.batch_ifftshift2d(z_1),2))
            
            norm = torch.sqrt(torch.sum(aperture*aperture,2))
            norm = torch.stack((norm,norm),2)
            real,imag = torch.unbind(aperture/norm,-1)
            zero = torch.zeros_like(imag)
            one = torch.ones_like(imag)
            all_phase = torch.acos(real)+torch.where(imag<0,one,zero)*math.pi
            if zero_padding:
                self.aperture.phase = all_phase[int((self.aperture.Dy-self.aperture.Ay)/2):int((self.aperture.Dy+self.aperture.Ay)/2),int((self.aperture.Dx-self.aperture.Ax)/2):int((self.aperture.Dx+self.aperture.Ax)/2)]
            return self.aperture.phase
        else:
            print('Using RS')
            return self.idiffraction_RS(phase)
    
    def prop(self,phase,zero_padding,propgation_method):
        if propgation_method == 'RS':
            return self.diffraction_RS(phase, zero_padding)
        elif propgation_method == 'ASM':
            return self.Angular_spectrum(phase,zero_padding)
        elif propgation_method == 'Fraunhofer_FFT':
            return self.Fraunhofer_approximation(phase,True,zero_padding)
        elif propgation_method == 'Fresnel_FFT':
            return self.Fresnel_approximation(phase,True,zero_padding)
        elif propgation_method == 'Fraunhofer':
            return self.Fraunhofer_approximation(phase,False,zero_padding)
        elif propgation_method == 'Fresnel':
            return self.Fresnel_approximation(phase,False,zero_padding)
        
    def iprop(self,phase,zero_padding,propgation_method):
        if propgation_method == 'RS':
            return self.idiffraction_RS(phase, zero_padding)
        elif propgation_method == 'ASM':
            return self.iAngular_spectrum(phase,zero_padding)
#        elif propgation_method == 'Fraunhofer_FFT':
#            return self.iFraunhofer_approximation(phase,True,zero_padding)
#        elif propgation_method == 'Fresnel_FFT':
#            return self.iFresnel_approximation(phase,True,zero_padding)
#        elif propgation_method == 'Fraunhofer':
#            return self.iFraunhofer_approximation(phase,False,zero_padding)
#        elif propgation_method == 'Fresnel':
#            return self.iFresnel_approximation(phase,False,zero_padding)
        
def demo():
    test = False
    if test:
        light = Light()
        s_x = 1920
        s_y = 1080
        a_x = 4
        a_y = 4
        aperture = Aperture(s_x,s_y,a_x,a_y,'circle')
        aperture.roll_phase()
        #plt.imshow(aperture.amplitude)
        t_x = 1920
        t_y = 1080
        t_z = 100000
        target = Target(t_x,t_y,t_z,'parallel_plane')
        aperture.update_sampleratio(math.sqrt(2*t_z*target.sample_ratio/t_y)) #set aperture to make Fraunhofer the same as ASM
        
        zero_m = torch.nn.ZeroPad2d(((s_x-a_x)//2,(s_x-a_x)//2,(s_y-a_y)//2,(s_y-a_y)//2))
        phase = torch.ones((a_y,a_x), device=device, dtype=dtype)*math.pi*2
        
        rs_p = Propagation(aperture, target, light)
        
        fig = plt.figure()
        
        ax1 = fig.add_subplot(221)
        ax1.set_title("Aperture")
        ax1.imshow(zero_m(aperture.amplitude*aperture.amplitude))
        
        ax2 = fig.add_subplot(222)
        ax2.set_title("ASM")
        rs_p.Angular_spectrum(phase,zero_padding=zero_m)
        with torch.no_grad():
            ax2.imshow(target.image())
        
        ax3 = fig.add_subplot(223)
        ax3.set_title("Fraunhofer_FFT")
        rs_p.Fraunhofer_approximation(phase,fft_mod = True,zero_padding=zero_m)
        with torch.no_grad():
            ax3.imshow(target.image())
        i3 = target.image()
        
        ax4 = fig.add_subplot(224)
        ax4.set_title("Fresnel_FFT")
        rs_p.Fresnel_approximation(phase,fft_mod = True, zero_padding=zero_m)
        with torch.no_grad():
            ax4.imshow(target.image())
        i4 = target.image()
        
        fig.tight_layout()
        plt.show()
        print("Requires_Grad: ", target.image().requires_grad)
        print("Fraunhofer-Fresnel: ", torch.sum(i3/torch.max(i3)-i4/torch.max(i4)).item())
        
    else:
        light = Light()
        s_x = 65
        s_y = 65
        a_x = 35
        a_y = 35
        aperture = Aperture(s_x,s_y,a_x,a_y,'circle')
        #aperture.roll_phase()
        #plt.imshow(aperture.amplitude)
        t_x = 65
        t_y = 65
        t_z = 3000
        target = Target(t_x,t_y,t_z,'parallel_plane')
        aperture.update_sampleratio(math.sqrt(2*t_z*target.sample_ratio/t_y)) #set aperture to make Fraunhofer the same as ASM
        
        zero_m = torch.nn.ZeroPad2d(((s_x-a_x)//2,(s_x-a_x)//2,(s_y-a_y)//2,(s_y-a_y)//2))
        phase = torch.rand((a_y,a_x), device=device, dtype=dtype)*math.pi*2
        #phase = torch.ones((a_y,a_x), device=device, dtype=dtype)*math.pi*2
        rs_p = Propagation(aperture, target, light)
        with torch.no_grad():
            fig = plt.figure()
            
            ax2 = fig.add_subplot(222)
            ax2.set_title("ASM")
            rs_p.prop(phase, zero_padding=zero_m,propgation_method="ASM")
            ax2.imshow(target.image())
            
            ax1 = fig.add_subplot(221)
            ax1.set_title("RS")
            rs_p.prop(phase, zero_padding=zero_m,propgation_method="RS")
            ax1.imshow(target.image())
            
            ax3 = fig.add_subplot(223)
            ax3.set_title("Fraunhofer_FFT")
            rs_p.prop(phase, zero_padding=zero_m,propgation_method="Fraunhofer_FFT")
            ax3.imshow(target.image())
            
            ax4 = fig.add_subplot(224)
            ax4.set_title("Fresnel_FFT")
            rs_p.prop(phase, zero_padding=zero_m,propgation_method="Fresnel_FFT")
            ax4.imshow(target.image())
            
            fig.tight_layout()
            plt.show()

if __name__ == "__main__":
    demo()