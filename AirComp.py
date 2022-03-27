# -*- coding: utf-8 -*-

import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
#import copy
# import argparse


def transmission(libopt,d,signal,x,f,h):
    index=(x==1)#被选中的设备序号
    #I=sum(x)
    N=libopt.N
    #M=libopt.M
    K=libopt.K[index]
    K2=K**2
    #print(x)

    
    
    inner=f.conj()@h[:,index]#这个inner不知道是干啥的。如果len(index)=35, h[:,index].shape=[5,35],f.shape=[5,1]
    inner2=np.abs(inner)**2
    #print(inner[index])
    g=signal#g.shape=[35,21921]
    #mean and variance
    mean=np.mean(g,axis=1)#对于每个设备而言，求所有梯度的平均值，因此axis=1
    g_bar=K@mean#式(11)的第二项
    
    
    
    var=np.var(g,axis=1)
    
    var[var<1e-3]=1e-3
#    if min(var)<1e-5:
#         var=1
         
    var_sqrt=var**0.5

    eta=np.min(libopt.transmitpower*inner2/K2/var)#η normalization scalar
    eta_sqrt=eta**0.5
    b=K*eta_sqrt*var_sqrt*inner.conj()/inner2
    
    
    noise_power=libopt.sigma*libopt.transmitpower
    
    
    n=(np.random.randn(N,d)+1j*np.random.randn(N,d))/(2)**0.5*noise_power**0.5
#    n=0
    x_signal=np.tile(b/var_sqrt,(d,1)).T*(g-np.tile(mean,(d,1)).T)#[35,21921]
    y=h[:,index]@x_signal+n#式(6) h[:,index].shape=5,35
    w=np.real((f.conj()@y/eta_sqrt+g_bar))/sum(K)#除以sum(K)的根据是式(4)  #这个式子指的是(11)
    

    #print(abs(inner*b/eta_sqrt/var_sqrt-K))
    
        
#    true_w=K@g/sum(K)
#    avg_mse=np.linalg.norm((true_w-w))**2/d
    
    #print(np.linalg.norm((true_w-w))**2/d)
    #print(libopt.sigma/eta/sum(K)**2)
    #print(libopt.sigma/sum(K)**2*np.max(K2*np.linalg.norm(g,axis=1)**2/inner2)/d)
    return w


if __name__ == '__main__':
   pass
    
    