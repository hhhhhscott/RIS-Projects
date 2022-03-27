# -*- coding: utf-8 -*-
import copy
import numpy as np
np.set_printoptions(precision=6,threshold=1e3)
from scipy.optimize import minimize



def sca_fmincon(libopt,h_d,G,f,theta,x,K2,RISON):
    N=libopt.N#参数服务器天线数
    L=libopt.L#反射元件数
    I=sum(x)#被选中设备的数量
    tau=libopt.tau
    if theta is None:
        theta=np.ones([L],dtype=complex)#赋初值或者从外面传进来
    if not RISON:#讨论没有RIS的情况
        theta=np.zeros([L],dtype=complex)
    result=np.zeros(libopt.nit)#NIT是SCA循环的次数，nit n iteration
    h=np.zeros([N,I],dtype=complex)
    for i in range(I):
        h[:,i]=h_d[:,i]+G[:,:,i]@theta#式（5）,维度N*被选中的设备数（对应main中的N*全部设备数），即总信道系数（直连+反射信道）
        
    if f is None:#赋初值或者传进来
        f=h[:,0]/np.linalg.norm(h[:,0])#shape N*1,保证模长为1，但为什么这么做。。而不是初始化为相等的，还是只是随便初始化一个
   
    obj=min(np.abs(np.conjugate(f)@h)**2/K2)#式（24）算法1 第3行
    threshold=libopt.threshold
    
    for it in range(libopt.nit):
        obj_pre=copy.deepcopy(obj)
        a=np.zeros([N,I],dtype=complex)
        b=np.zeros([L,I],dtype=complex)
        c=np.zeros([1,I],dtype=complex)
        F_cro=np.outer(f,np.conjugate(f));
        for i in range(I):
            a[:,i]=tau*K2[i]*f+np.outer(h[:,i],np.conjugate(h[:,i]))@f
                       #↑这个K2是为什么（似乎是根据数据数量自适应调节的tau参数
            if RISON:

                b[:,i]=tau*K2[i]*theta+G[:,:,i].conj().T@F_cro@h[:,i]
                          #↑这个K2是为什么
                c[:,i]=np.abs(np.conjugate(f)@h[:,i])**2+2*tau*K2[i]*(L+1)+2*np.real((theta.conj().T)@(G[:,:,i].conj().T)@F_cro@h[:,i])#这个地方经过证明是和论文里的是等价的
            else:
                c[:,i]=np.abs(np.conjugate(f)@h[:,i])**2+2*tau*K2[i]
        
        
        #print(c.shape)
        
        fun=lambda mu: np.real(2*np.linalg.norm(a@mu)+2*np.linalg.norm(b@mu,ord=1)-c@mu)#式(29a)
        
        cons = ({'type': 'eq', 'fun': lambda mu:  K2@mu-1})#式(29b)
        bnds=((0,None) for i in range(I))#要求zeta_m＞0
        res = minimize(fun, 1/K2,   bounds=tuple(bnds), constraints=cons)
        if ~res.success:
            pass
            #print('Iteration: {}, solution:{} obj:{:.6f}'.format(it,res.x,res.fun[0]))
            #print(res.message)
            #return
        fn=a@res.x
        thetan=b@res.x
        fn=fn/np.linalg.norm(fn)#左值为f_n+1 这里是对总体求二范数，fn的模长为1
#        thetan=thetan/np.abs(thetan)
        if RISON:
            thetan=thetan/np.abs(thetan)#左值为theta_n+1 这里是对每一个分量求绝对值，对应位置相除，theta每个分量模长为1，总体模长为L
            theta=thetan
        f=fn
        for i in range(I):
            h[:,i]=h_d[:,i]+G[:,:,i]@theta
        obj=min(np.abs(np.conjugate(f)@h)**2/K2)
        result[it]=copy.deepcopy(obj)#第it次迭代的结果
        if libopt.verbose>2:
            print('  Iteration {} Obj {:.6f} Opt Obj {:.6f}'.format(it,result[it],res.fun[0]))
        if np.abs(obj-obj_pre)/min(1,abs(obj))<=threshold:
            break
        
        #print(res)
    if libopt.verbose>1:
        print(' SCA Take {} iterations with final obj {:.6f}'.format(it+1,result[it]))
    result=result[0:it]#由于有早停规则，所以有可能还没到预设的试验次数就终止迭代，所以这里面用it而不用nit
    return f,theta,result




def find_obj_inner(libopt,x,K,K2,Ksum2,h_d,G,f0,theta0,RISON):
    N=libopt.N
    L=libopt.L
    M=libopt.M
    if sum(x)==0:#避免所有设备都不被选择
        obj=np.inf
        
        theta=np.ones([L],dtype=complex)
        f=h_d[:,0]/np.linalg.norm(h_d[:,0])#保证f模长为1
        if not RISON:
            theta=np.zeros([L])
    else:
         index=(x==1)#存储了一个被选择设备位置的真值表
         #print(index)

         f,theta,_=sca_fmincon(libopt,h_d[:,index],G[:,:,index],f0,theta0,x,K2[index],RISON)
         #这里的index是个boolean元组，会把所有index为True对应的位置传进去（可以看到这里不是循环）

         h=np.zeros([N,M],dtype=complex)
         for i in range(M):
             h[:,i]=h_d[:,i]+G[:,:,i]@theta
         gain=K2/(np.abs(np.conjugate(f)@h)**2)*libopt.sigma#这里和式(23)有关
         #print(gain)
         #print(gain)
         #print(2/Ksum2*(sum(K[~index]))**2)
         #print(np.max(gain[index])/(sum(K[index]))**2)
         obj=np.max(gain[index])/(sum(K[index]))**2+4/Ksum2*(sum(K[~index]))**2#这里和式(23)有关,P0被设置成1了？还是说考虑了ref
    return obj,x,f,theta
def Gibbs(libopt,h_d,G,x0,RISON,Joint):#RISON=RIS ON?
    #initial
    
    N=libopt.N#服务器天线
    L=libopt.L#反射元件
    M=libopt.M#数量
    Jmax=libopt.Jmax#Gibbs loop
    K=libopt.K/np.mean(libopt.K) #normalize K to speed up floating computation
    K2=K**2
    Ksum2=sum(K)**2#这里总会是M的平方//a b c d的分别处以他们四个平均数，再求和，总会是数据个数
    x=x0
    # inital the return values
    obj_new=np.zeros(Jmax+1)
    f_store=np.zeros([N,Jmax+1],dtype = complex)
    theta_store=np.zeros([L,Jmax+1],dtype = complex)
    x_store=np.zeros([Jmax+1,M],dtype=int)
    
    #the first loop，第一次试验的结果是在所有设备都被选中情况下进行的
    ind=0
    [obj_new[ind],x_store[ind,:],f,theta]=find_obj_inner(libopt,x,K,K2,Ksum2,h_d,G,None,None,RISON)
    
    theta_store[:,ind]=copy.deepcopy(theta)
    f_store[:,ind]=copy.deepcopy(f)
#    beta=min(max(obj_new[ind],1)
    beta=min(1,obj_new[ind])#beta值其实并不是用obj来算的，这里只是用obj来表=赋个初值
    # print(beta)
    alpha=0.9##############
    if libopt.verbose>1:
        print('The inital guess: {}, obj={:.6f}'.format(x,obj_new[ind]))
    elif libopt.verbose==1:
        print('The inital guess obj={:.6f}'.format(obj_new[ind]))
    f_loop=np.tile(f,(M+1,1))#f_loop的每一行是第j次试验中和给定的x在第m个位置不同的策略的SCA搜索结果，这个数组用于在算法2的每次迭代中传递上次试验的结果
    #就如arxiv第20页的描述，这样做可以减少搜索时间

    theta_loop=np.tile(theta,(M+1,1))
    #print(theta_loop.shape)
    #print(theta_loop[0].shape)
    for j in range(Jmax):
        if libopt.verbose>1:
            print('This is the {}-th Gibbs sampling iteration, beta= {:.6f}'.format(j+1,beta));
        
        #store the possible transition solution and their objectives
        X_sample=np.zeros([M+1,M],dtype=int)
        Temp=np.zeros(M+1)#这里的Temp实际上是论文里的J(x_j)
        #the first transition => no change
        X_sample[0,:]=copy.deepcopy(x)
        Temp[0]=copy.deepcopy(obj_new[ind])############
        f_loop[0]=copy.deepcopy(f)
        theta_loop[0]=copy.deepcopy(theta)
        #2--M+1-th trnasition, change only 1 position
        for m in range(M):#这个循环里面对每一个m位置改变后的选择策略进行SCA求解，然后将得到的式23的值以及对应的f,theta存入Temp,f_loop,theta_loop中
            if libopt.verbose>1:
                print('the {}-th:'.format(m+1))
            #filp the m-th position
            x_sam=copy.deepcopy(x)
            x_sam[m]=copy.deepcopy((x_sam[m]+1)%2)
            X_sample[m+1,:]=copy.deepcopy(x_sam);#将第m个位置不同于x的选择策略存进X_sample中
            Temp[m+1],_,f_loop[m+1],theta_loop[m+1]=find_obj_inner(libopt,
                x_sam,K,K2,Ksum2,h_d,G,f_loop[m+1],theta_loop[m+1],RISON)#在给定f_loop[m+1]和theta_loop[m+1]的条件下进行搜索，减少计算时间
            if libopt.verbose>1:
                print('          sol:{} with obj={:.6f}'.format(x_sam,Temp[m+1]))
        temp2=Temp;
        
        Lambda=np.exp(-1*temp2/beta);
        Lambda=Lambda/sum(Lambda);
        while np.isnan(Lambda).any():
            if libopt.verbose>1:
                print('There is NaN, increase beta')
            beta=beta/alpha;
            Lambda=np.exp(-1.*temp2/beta);
            Lambda=Lambda/sum(Lambda);
        
        if libopt.verbose>1:
            print('The obj distribution: {}'.format(temp2))
            print('The Lambda distribution: {}'.format(Lambda))
        kk_prime=np.random.choice(M+1,p=Lambda)
        x=copy.deepcopy(X_sample[kk_prime,:])
        f=copy.deepcopy(f_loop[kk_prime])
        theta=copy.deepcopy(theta_loop[kk_prime])
        ind=ind+1
        obj_new[ind]=copy.deepcopy(Temp[kk_prime])
        x_store[ind,:]=copy.deepcopy(x)
        theta_store[:,ind]=copy.deepcopy(theta)
        f_store[:,ind]=copy.deepcopy(f)
        
        if libopt.verbose>1:
            print('Choose the solution {}, with objective {:.6f}'.format(x,obj_new[ind]))
            
        if libopt.verbose:
            print('This is the {}-th Gibbs sampling iteration, beta= {:.6f},obj={:.6f}'.format(j+1,beta,obj_new[ind]));
        beta=max(alpha*beta,1e-4);
        
    return x_store,obj_new,f_store,theta_store