
import numpy as np
import torch
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from scipy.special import gammaln #equivalent to lgamma r funct
import math 
import pandas as pd
from scipy.special import logsumexp
from scipy.special import factorial
import random
import sys

'''
studentpdf, studentpdf1, const_hazard are three function used in the online change point detection function.
'''

def studentpdf(x, mu, var, nu):
    c= np.exp(gammaln(nu/2+0.5)- gammaln(nu/2)).reshape(-1,1)/np.sqrt(nu.reshape(-1,1)*math.pi*var)
   
         
    return c*(1+(1/(nu.reshape(-1,1)*var))*((x-mu))**2)**(-(nu.reshape(1,-1)+1)/2).T
def const_hazard(r, p):
    """
    Constant prior for every datapoint
    Arguments:
        p - probability of event
    """
    
    return 1/p*np.ones(r)
def studentpdf1(x, mu, var, nu):
    c= np.asarray(np.exp(gammaln(nu/2+0.5)- gammaln(nu/2))/np.sqrt(nu*np.pi*var))
   
         
    return c*(np.asarray(1+(1/(nu*var)))*np.asarray(np.square(x - mu)))**(-(nu+1)/2)
'''
Online change point detection function. If u=0 the offline function is activated for initialization
'''
def onlineCPD(datapt, hazard_func = const_hazard, oCPD = None, timept = None):
    if oCPD is None:
        return offlineCPD(np.transpose(datapt), timept)
    T = oCPD['R'].shape[0]
    Q = len(datapt)
    R2 = np.zeros((T + 1, T + 1))
    R2[:T, :T] = oCPD['R']
    lambda_ = 2000
    predProbs = studentpdf(datapt, np.asarray(oCPD['mu']), np.asarray(oCPD['beta']) * ((oCPD['kappa'] + 1)/(oCPD['alpha'] * oCPD['kappa'])).reshape(-1,1), 2 * oCPD['alpha'])
    H = hazard_func(T, lambda_)
    R2[1:(T+1), T] = R2[:T, (T-1)] * np.apply_along_axis(np.prod, 1, predProbs) * (1 - H)
    R2[0, T] = np.sum(R2[:T, (T-1)] * np.apply_along_axis(np.prod, 1, predProbs) * H)
    #print("sum", np.sum(R2[:, T]))
    R2[:, T] = R2[:, T]/np.sum(R2[:, T])
    tempmu = np.r_[np.zeros((Q)).reshape(1,-1) , (np.asarray(oCPD['mu'])*oCPD['kappa'].reshape(-1,1) + datapt)/(oCPD['kappa'].reshape(-1,1)  + 1)]
    tempkappa = np.append(0.01, (oCPD['kappa'] + 1))
    tempalpha = np.append(0.01, (oCPD['alpha'] + 0.5))
    tempbeta = np.r_[np.array((np.repeat(1e-4, Q))).reshape(1,-1), (oCPD['beta'] + (oCPD['kappa'].reshape(-1,1) * np.asarray(np.square(datapt - oCPD['mu'])))/(2 * (oCPD['kappa'].reshape(-1,1) + 1)))]
    maxes = np.append(oCPD['max'], np.argmax(R2[:, T]))
    cps = np.append(oCPD['changes'], T -1 - maxes[T])
    cps = np.sort(np.unique(cps))
    result = {'R': R2, 'predProbs':predProbs,'alpha': tempalpha, 'beta': tempbeta, 
              'kappa': tempkappa, 'mu': tempmu, 'max': maxes, 'changes': cps}
    return result


'''
Offline change point detection function, used only at the first time instant t= 0
'''
def offlineCPD(data, time=None, hazard_func=const_hazard, m=0, k=0.01, a=0.01, b=1e-04):
    data = np.matrix(data) if type(data) is np.ndarray else np.matrix(data)
    lambda_ = 2000
    T = data.shape[0]
    dim = data.shape[1]
    muT = mu0 = np.matrix(np.repeat(m, dim))
    kappaT = kappa0 = k
    betaT = beta0 = np.matrix(np.repeat(b, dim))
    alphaT = alpha0 = a
    R = np.matrix(np.zeros((T+1, T+1)))
    R[0,0] = 1
    maxes = np.zeros(T+1, dtype=int)
    cps = np.zeros(T+1, dtype=int)
    for t in range(1,T+1):
        #print('t',t)
        predProbs = studentpdf1(data, muT, betaT * (kappaT + 1)/(alphaT * kappaT), 2 * alphaT)
        #ht = t+1
        H = hazard_func(t, lambda_)
        R[1:t+1, t] = R[0:t, (t-1)] * np.prod(predProbs, axis=1) * (1 - H)
        R[0, t] = np.sum(R[0:t, (t-1)] * np.prod(predProbs, axis=1) * H)
        R[:, t] = R[:, t]/np.sum(R[:, t])
        tempmu = np.vstack((mu0, (kappaT * muT + data)/(kappaT + 1)))
        tempkappa = np.append(kappa0, kappaT + 1)
        tempalpha = np.append(alpha0, alphaT + 0.5)
        tempbeta = np.vstack((beta0, betaT + (kappaT * np.asarray((data - muT))**2)/(2 * (kappaT + 1))))
        muT = tempmu
        kappaT = tempkappa
        alphaT = tempalpha
        betaT = tempbeta
        
        maxes[t] = np.argmax(R[:, t])
        cps[t] = t - maxes[t]
    cps = np.sort(np.unique(cps))
    result = {'R': R, 'alpha': alphaT, 'beta': betaT, 'kappa': kappaT, 'mu': muT, 'max': maxes, 'changes': cps}
    return result

def SimulateData_Py(M = 200, P=250, U =150):
    Q = 3 
    L = 2
    alpha = np.zeros(shape=(U,Q))
    beta = np.zeros(shape=(U,L))
    pi = np.zeros(shape=(U,2))
    
    
    
    cp_alpha = np.array([U//9,U//3,U//4])
    alpha[:,0] =  np.concatenate((np.random.normal(0, 0.1, cp_alpha[0]+ cp_alpha[1]), 
                                 np.random.normal(0.3, 0.1, cp_alpha[2]), 
                                 np.random.normal(0.8, 0.1, U-np.cumsum(cp_alpha)[2])))
    alpha[:,1] =  np.concatenate((np.random.normal(0.7, 0.1, cp_alpha[0]), 
                                 np.random.normal(0.2, 0.1, cp_alpha[1]), 
                                 np.random.normal(0.4, 0.1, cp_alpha[2]), 
                                 np.random.normal(0.6, 0.1, U-np.cumsum(cp_alpha)[2])))
    alpha[:,2] = np.ones((U))-(alpha[:,0]+alpha[:,1])
    b = np.exp(alpha)
    alpha =b/b.sum(axis=1)[:,None]
    
    cp_beta = np.array([U//6,U//4,U//3])
    beta[:,0] =  np.concatenate((np.random.normal(0.1, 0.12, cp_beta[0]), 
                                 np.random.normal(0.7, 0.12, cp_beta[1]),
                                 np.random.normal(0.3, 0.12, cp_beta[2]),
                                 np.random.normal(0.6, 0.12, U-np.cumsum(cp_beta)[2])))
    beta[:,1] = np.ones((U))-(beta[:,0])

    b = np.exp(beta)
    beta =b/b.sum(axis=1)[:,None]
    
    cp_pi = np.array([U//9,U//3,U//4])
    pi[:,0] =  np.concatenate((np.random.normal(0.35, 0.1, cp_pi[0]), 
                                 np.random.normal(0.7, 0.1, cp_pi[1]),
                                 np.random.normal(0.25, 0.1, cp_pi[2]), 
                                 np.random.normal(0.8, 0.1, U-np.cumsum(cp_pi)[2])))
    
    pi[:,1] = np.ones((U))-(pi[:,0])
    b = np.exp(pi)
    pi =b/b.sum(axis=1)[:,None]
    fig, ax = plt.subplots(3, figsize=[11, 11], sharex=True)
    ax = fig.add_subplot(3, 1, 1)
    ax.plot(alpha)
    
    plt.title('Estimated alpha')
    plt.ylim(0, 1)
    
    ax = fig.add_subplot(3, 1, 2, sharex=ax)
    ax.plot(beta)
    
    plt.title('Estimated beta')
    plt.ylim(0, 1)
    
    ax = fig.add_subplot(3, 1, 3, sharex=ax)
    ax.plot(pi[:,0])
    
    plt.title('Estimated pi')
    plt.ylim(0, 1)
    plt.show()
    
    Lambda = np.array([[6, 4,],[1, 2],[7, 3]])
    
    '''Variable Simulations'''
    Zinit = np.zeros((M, Q, U))
    Winit = np.zeros((P, L, U))
    A = np.zeros((M, P, U))
    X = np.zeros((M, P, U))
    
    
    for u in range(U):
        Zinit[:,:,u] = np.random.multinomial(1,np.asarray(alpha[u,:]).astype('float64'), M)
        Winit[:,:,u] = np.random.multinomial(1,np.asarray(beta[u,:]).astype('float64'), P)
        A[:,:,u] = np.random.binomial(1, np.asarray(pi[u,0]).astype('float64'),M*P).reshape(M, P)
        
        Z = Zinit.argmax(axis=1)
        W = Winit.argmax(axis=1)
        
        for q in range(0,Q):
            for l in range (0,L):
                sel_z = np.argwhere(Z[:,u]==q)[:,0]
                sel_w = np.argwhere(W[:,u]==l)[:,0]
                #print('u = %s / q= %s / l=%s / sel_z = %s / sel_w = %s'%(u, q, l, sel_z.shape, sel_w.shape))
                #print(pi[u,0]/(pi[u,0]+((1-pi[u,0])*np.exp(-Lambda[q, l]))))
                X[sel_z[:,None],sel_w,u] = np.random.poisson(Lambda[q, l],len(sel_z)*len(sel_w)). reshape(len(sel_z), len(sel_w))
    X[(A==1)]=0
    result = {'X': X, 'row_clusr':Z,'col_clust': W, 'A': A,'Zinit': Zinit, 'Winit': Winit, 
              'Lambda': Lambda, 'pi': pi, 'alpha': alpha, 'beta': beta, 
              'cp_alpha':cp_alpha, 'cp_beta':cp_beta, 'cp_pi':cp_pi}
    
    
    return result




'''Defining the Neural Net'''
'''
Fully connected neural network. Used for the prediction of alpha and beta only when u<mov_U. To inizialize the input of the LSTM.
Two layer, with relu activation functions
'''
class Two_layer_nn(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu2= nn.ReLU()
        self.output = nn.Linear(hidden_size_2, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x



'''
This function is used to inizialize the model parameters every time a new time instant is added to the algorithm
Same as the one that for Zip-dLBM was outside the algorithm 
'''

def LBM_ZIP(X, Q, L,max_iter, alpha_init, beta_init, Lambda_init, pi_init):
    alpha = np.copy(alpha_init)
    beta = np.copy(beta_init)
    Lambda = np.copy(Lambda_init)
    pi = np.copy(pi_init)
    M = X.shape[0]
    P = X.shape[1]
    '''Data initialization '''
    delta = np.ones((M,P))
    delta[X>0]= 0 
    alpha = alpha/np.sum(alpha)
    beta = beta/np.sum(beta)

    tau = np.random.multinomial(1, np.asarray(alpha).astype('float64'), M)
    eta = np.random.multinomial(1, np.asarray(beta).astype('float64'), P)
    # 
    low_bound = np.zeros((max_iter))
    e_it = 3

    pi = np.mean(delta)
    tau[tau<1e-16] = 1e-16
    eta[eta<1e-16] = 1e-16
    alpha[alpha<1e-16] = 1e-16
    beta[beta<1e-16] = 1e-16
    Lambda[Lambda<1e-16] = 1e-16
    for i in range(0,max_iter):
      for e in range(0,e_it):
        #print(np.mean(delta))
        ''' E - Step '''
          
        '''Delta Estimation'''
        q_ij = pi*(X==0)*np.exp(-np.matmul(np.matmul(tau, np.log(Lambda)), np.transpose(eta))*X +np.matmul(np.matmul(tau, Lambda), np.transpose(eta)) + np.log(factorial(X)) -np.log(1-pi))
        delta = q_ij/(1+q_ij)
        delta[(X>0)]=0 ##Corect! Checked!
        delta[delta<1e-16] = 1e-16
        ''' Tau Estimation '''
          
        tau[tau<1e-16] = 1e-16
        eta[eta<1e-16] = 1e-16
        
        mat_R = np.matmul(np.matmul((1-delta)*X, eta), np.transpose(np.log(Lambda))) - np.matmul(np.matmul((1-delta),eta), np.transpose(Lambda)) + np.log(alpha)
        z_q = logsumexp(mat_R, axis = 1).reshape(-1,1)
        log_r_iq = mat_R - z_q
        tau = np.exp(log_r_iq)
        
        ''' Eta Estimation '''
        mat_S = np.matmul(np.matmul(np.transpose((1-delta)*X),tau),np.log(Lambda)) -np.matmul(np.matmul(np.transpose((1-delta)),tau),Lambda)+np.log(beta) 
        w_l = logsumexp(mat_S, axis =1).reshape(-1,1)
        log_s_jl =  mat_S - w_l
        eta = np.exp(log_s_jl)
      
      tau[tau<1e-16] = 1e-16
      eta[eta<1e-16] = 1e-16
      
      ''' M - Step : Lambda '''
      X_delta = X*delta
      X_ql = np.matmul(np.transpose(tau),np.matmul(X,eta))
      dell = np.matmul(np.transpose(tau),np.matmul(X_delta,eta))
      den = np.matmul(np.transpose(tau),np.matmul((1-delta), eta))
      Lambda = (X_ql - dell)/den
      '''Alpha, Beta:'''
      alpha = np.mean(tau, axis =0)  
      beta = np.mean(eta, axis =0)  
      pi = np.mean(delta)
      
      alpha[alpha<1e-16] = 1e-16
      beta[beta<1e-16] = 1e-16
      Lambda[Lambda<1e-16] = 1e-16
   
      ''' Lower Bound Computation '''
      
      p_x1 = np.sum(delta[X==0]*np.log(pi))
      p_x2 = np.sum(X*(1-delta)*np.matmul(np.matmul(tau, np.log(Lambda)),np.transpose(eta))) + np.sum((1-delta)*np.log(1-pi))
      p_x2b = np.sum(np.matmul(np.matmul(tau,Lambda),np.transpose(eta))*(1-delta))
      p_tau = np.sum(np.matmul(tau, np.log(alpha)))
      p_eta = np.sum(np.matmul(eta, np.log(beta)))                  
      ent_tau = np.sum(tau*np.log(tau))
      ent_eta = np.sum(eta*np.log(eta))
      ent_delta = np.sum(delta*np.log(delta)+(1-delta)*np.log(1-delta))

      low_bound[i]  =  p_x1 + p_x2  - p_x2b  + p_tau + p_eta - ent_eta - ent_tau - ent_delta
      
    crit = low_bound[i] - (Q-1)/2*np.log(M) - (L-1)/2*np.log(P) - (Q*L)/2*np.log(M*P) - 1/2*np.log(M*P) 
    return tau, eta, delta, alpha, beta, pi, low_bound, Lambda, crit

 
'''
LSTM function, to estimate the parameter alpha and beta when u>mov_U

'''

class ShallowRegressionLSTM(nn.Module):
    def __init__(self, out_features, num_sensors, hidden_size, proj_size, u, mov_U):
        super().__init__()
        self.out_features = out_features
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_size = hidden_size #hidden state
        self.num_layers = 1
        self.proj_size = proj_size
        #self.dropout = 0.3
        self.u = u
        self.mov_U = mov_U
        batch_size = mov_U
        self.hn = nn.ParameterList((nn.Parameter(torch.zeros(self.num_layers, batch_size, self.proj_size)),
                                    nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_size))))
        
        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_size,
            num_layers=self.num_layers, 
            proj_size= proj_size, 
            #dropout = self.dropout
        )
        
        #self.linear = nn.Linear(in_features=self.hidden_units, hidden_size=hidden_size,out_features=3)
        self.fc_1 =  nn.Linear(proj_size, 64)
        self.fc = nn.Linear(64, out_features) #fully connected last layer 128
        self.softmax = nn.Softmax(dim = 1)
        self.sig = nn.Sigmoid()


    def forward(self, x):

        output,hn= self.lstm(x)
        out = output.view(-1, self.out_features)
        out = self.fc_1(out) #first Dense
        out = self.sig(out)
        out = self.softmax(self.fc(out))

        return out, hn



        

   
'''Function implementing the ODE'''
''' p_0: initial value of the dynamic system (alpha_0, beta_0)
    f_: neural net function, can be f_alpha or f_beta here. Depending the value of u it can be the fully connect nn or the lstm
    u: current time instant
    mov_U: size of the moving window [G_d(t) in the paper]
    Q: number of (row/column) cluster, corresponding to the number of columns of tau/eta'''

def get_next(p_0, f_,prev_, hidden, u,mov_U, Q):
    min_u =  max(0,u-mov_U)
    max_u = u
    softmax = nn.Softmax(0)
    if u ==0:
        out = softmax(p_0)
        return out, p_0
    elif u > 0 and u < mov_U:
        out = torch.zeros(len(range(min_u,max_u+1)),Q)
        for j in range(len(range(min_u,max_u+1))):
            next_ = prev_ + f_(prev_)  #this line represents the ODE system: a(t) = a(t-1) + f(a(t-1)), here f_ is the fully connected nn
            out[j,:] = softmax(next_.reshape(-1,))
            prev_ = next_.clone()
        return out, prev_
    else:
        next_, hn =  f_.forward(prev_) #Here f_ is the LSTM function
        prev_ = next_.clone()
    return hn, prev_
    
'''Computing the loss of the parameters: we want to minimize it'''
'''
vprob: latent variable tau/eta
p_0:  initial value of the dynamic system (alpha_0, beta_0)
prev_par: parameter values estimated in the previous time steps
u: current time instant
mov_U: size of the moving window [G_d(t) in the paper]
f_: neural net function, can be f_alpha or f_beta here. Depending the value of u it can be the fully connect nn or the lstm
param: either "alpha" or "beta", the "pi" part has been commented because in the pharmacovigilance application we use 
#       the point estimate of pi to avoid numerical problems due to the large number of zeros in the data.
'''
def loss(vprob, p_0, prev_par, u,mov_U, f_, param):
    softmax = nn.Softmax(0)
    min_u =  max(0,u-mov_U)
    max_u = u
    if param =="alpha" or param =="beta":
        if u ==0:
            prop_0 = softmax(p_0)
            out = torch.sum(torch.sum(vprob, 0)*torch.log(prop_0 + 1e-16))
            return -out
        elif u > 0 and u < mov_U:
            out = 0 
            for j in range(min_u,max_u+1):
                next_ = prev_par + f_(prev_par)
                propt_next_ = softmax(next_.reshape(-1,))
                log_next_ = torch.log(propt_next_ + 1e-16)
                out = out + torch.sum(torch.sum(vprob[:,:,j], 0)*log_next_)
                prev_par = next_
            return -out
        else:
            out = 0
            next_, hn =  f_.forward(prev_par)
            log_next_ = torch.log(next_ + 1e-16)    
            out = torch.sum(np.transpose(torch.sum(vprob[:,:,min_u:max_u], 0))*log_next_)
            prev_par = next_    
            return -out
    if param =="pi":
        if u ==0:
            prop_0 = softmax(p_0)
            out = torch.sum(vprob*torch.log(prop_0[0] + 1e-16))+torch.sum((1-vprob)*torch.log(1-prop_0[0] + 1e-16))
            return -out
        elif u > 0 and u < mov_U:
            out = 0 
            for j in range(min_u,max_u+1):
               next_ = prev_par + f_(prev_par)
               propt_next_ = softmax(next_.reshape(-1,))
               log_next_ = torch.log(propt_next_+ 1e-16)
               out = out + torch.sum(vprob[:,:,j]*torch.log(propt_next_[0]))+torch.sum((1-vprob[:,:,j])*torch.log(1-propt_next_[0] + 1e-16))
               prev_par = next_
            return -out
        else:
            out = 0
            softmax = nn.Softmax(0)
            next_, hn =  f_.forward(prev_par)
            log_next_ = torch.log(next_ + 1e-16)    
            out = out+ torch.sum((vprob[:,:,min_u:max_u]*torch.log(next_[:,0]+ 1e-16))+((1-vprob[:,:,min_u:max_u])*torch.log(1-next_[:,0]+ 1e-16)))
            prev_par = next_   
            return -out
'''Training the loop over the epochs'''
''' vprob: multinomial latent variable tau/eta, with parameter alpha/beta,
    p_0: initial value of the dynamic system, here alpha_0/beta_0,
    f_: neural net function, fully connected for the first mov_U time instants, LSTM later on
    opt_alpha: Adam optimizer
    param: alpha or beta
    epochs = number of epochs over which we train the model'''

def Mupdate(vprob, p_0, f_, prev_par, u, mov_U, opt, param="alpha", epochs = 100):
    vprob_ = torch.tensor(vprob, dtype = torch.float32, requires_grad = False)
    store_l = [i for i in range(0, epochs)]
    for epoch in range(epochs):
        opt.zero_grad()
        l = loss(vprob_, p_0, prev_par, u, mov_U, f_, param) 
        store_l[epoch] = l.item()     
        l.backward(retain_graph=False)
        opt.step()
        if epoch % 100 == 0:
            print('epoch :', epoch, 'loss :', l.item())
            sys.stdout.flush()

    return store_l  

''' onlineDLBM_noLstm is the main function:'''
'''
  X = our count data. MxPxU array cointaining the number of interaction between every row and column pair at any given time instants,
  Q_icl = optimal number of row clusters obtained through the ICL computation on the first time instant (this computation is in the R script, before we run the main function)
  L_icl = optimal number of column clusters obtained through the ICL computation on the first time instant (this computation is in the R script, before we run the main function)
  Q = number of row clusters over which we run the algorithm (Q>Q_icl),
  L = number of column clusters over which we run the algorithm (L>L_icl),
  max_iter = number of iteration of the algorithm 
  
  '''
def onlineDLBM(X, Q_icl, L_icl, Q, L, max_iter):
    M = X.shape[0]
    P = X.shape[1]
    U = X.shape[2]
    
    a_1 = np.random.uniform(0,1,Q_icl)
    b_1 = np.random.uniform(0,1,L_icl)
    alpha_init_1 = a_1/sum(a_1)
    beta_init_1 = b_1/sum(b_1)
    Lambda_init = np.asarray(random.sample(range(1,Q_icl*L_icl+1),Q_icl*L_icl)).reshape(Q_icl,L_icl)
    alpha = np.zeros((U, Q))
    alpha_res = np.zeros((U, Q_icl))
    beta =  np.zeros((U, L))
    beta_res =np.zeros((U, L_icl))
    pi = np.zeros((U, 2))
    pi_res = np.zeros((U))
    pi_1 = np.random.uniform(0,1,1)
    mov_U = 5 #Size of the moving window for the LSTM
    num_hidden_units = 8 #Number of hidden units in the LSTM
    
    # Storage of the parameters
    alpha_prev_ = np.zeros(shape=(U,Q))
    beta_prev_ = np.zeros(shape=(U,L))
    pi_prev_ = np.zeros(shape=(U,2))
    start_time_tot = time.time()
    features_alpha = list(range(Q)) 
    features_beta = list(range(L)) 
    features_pi = list(range(2))
    
    
    max_it = 3 #number of iteration of the Stream Zip-sLBM algorithm
    e_it = 3 #number of iteration of the E-step
    alpha[alpha<1e-16] = 1e-16
    beta[beta<1e-16] = 1e-16

    tau = np.zeros((M,Q, U))
    eta = np.zeros((P,L, U))
    delta = np.ones((M,P, U))
    q_ij = np.zeros((M, P, U))
    mat_R = np.zeros((M, Q, U))
    mat_S = np.zeros((P, L, U))
    est_alpha = torch.tensor(np.zeros((U, Q)), dtype=torch.float32)
    est_beta = torch.tensor(np.zeros((U, L)), dtype=torch.float32)
    est_pi = torch.tensor(np.zeros((U, 2)), dtype=torch.float32)
    
    #Latent variables initialization

    tau[:,:,0] =np.random.multinomial(1, alpha[0,:], M)
    eta[:,:,0] =np.random.multinomial(1, beta[0,:], P)
    delta[:,:,0][X[:,:,0]>0] = 0 
    tau[tau<1e-16] = 1e-16
    eta[eta<1e-16] = 1e-16
    delta[delta<1e-16] = 1e-16
    q_ij = np.zeros(shape=(M, P, U))
    mat_R = np.zeros(shape=(M, Q, U))
    mat_S = np.zeros(shape=(P, L, U))
    pi_sum = np.zeros((U))
    
    Den_old = np.zeros((Q,L, U))
    Den_new =np.zeros((Q,L, U))
    Num_new =np.zeros((Q,L, U))
    X_delta = np.zeros((M,P, U))
    Q_diff = Q-Q_icl #Difference between the maximum number of row clusters and the optimal number of row clusters
    L_diff = L-L_icl #Difference between the maximum number of column clusters and the optimal number of column clusters
    
    
    #Parameter inizialization through the LBM_ZIP function at the beginning of each time step. See Section 3.5

    for u in range(0,U):
        pi_res[pi_res<1e-16] = 1e-16
        alpha_res[alpha_res<1e-16] = 1e-16
        beta_res[beta_res<1e-16] = 1e-16
        if (u==0):
            out_cascata = LBM_ZIP(X[:,:,0], Q_icl, L_icl, max_it, alpha_init_1,beta_init_1, Lambda_init, pi_1)
            alpha_res[0,:] = out_cascata[3]
            beta_res[0,:] = out_cascata[4]
            Lambda_init = out_cascata[7]
            pi_res[0] = out_cascata[5]
            
            Lambda= np.pad(Lambda_init, [(0, Q_diff), (0, L_diff)], mode='constant')

        else:
            out_cascata = LBM_ZIP(X[:,:,u],Q_icl, L_icl, max_it, alpha_res[u-1,:], beta_res[u-1,:], Lambda_init, pi_res[u-1])

            alpha_res[u,:] = out_cascata[3]
            beta_res[u,:] = out_cascata[4]
            pi_res[u] = out_cascata[5]
            Lambda_init = out_cascata[7]
        Lambda_old = Lambda.copy() 
    
        alpha[u,:] = np.pad(alpha_res[u,:], (0, Q_diff), 'constant')

        beta[u,:] = np.pad(beta_res[u,:], (0, L_diff), 'constant')
        pi[u,0] = pi_res[u]
        pi[u,1] = 1-pi_res[u]
        Lambda=np.pad(Lambda_init, [(0, Q_diff), (0, L_diff)], mode='constant')
        Lambda[Lambda<1e-16] = 1e-16

        
        '''Change point detection algorithm'''
        # The first time we execute the CPD algorithm is offline to inizialize the algorithm, then it runs online when t>G_d(t) [see Section 3.6]
        if (u==mov_U):
            res_alpha = onlineCPD(datapt=alpha[0,:Q_icl])
            nb_alpha = 0
            res_beta = onlineCPD(datapt=beta[0,:L_icl])
            nb_beta = 0
            res_pi = onlineCPD(datapt=pi[0,:])
            nb_pi = 0
        if (u>mov_U):
            k =  u-mov_U
            print(">", k)
            res_alpha = onlineCPD(datapt = alpha[k,:Q_icl], oCPD = res_alpha.copy())
            if len(res_alpha["changes"]) > nb_alpha:
                print("!!! Change point for ALPHA detected at time stamp ", res_alpha["changes"][nb_alpha], "!!!\n", sep="")
                nb_alpha += 1
            res_beta = onlineCPD(datapt = beta[k,:L_icl], oCPD = res_beta.copy())
            if len(res_beta["changes"]) > nb_beta:
                print("!!! Change point for BETA detected at time stamp ", res_beta["changes"][nb_beta], "!!!\n", sep="")
                nb_beta += 1
            res_pi = onlineCPD(datapt = pi[k,:], oCPD = res_pi.copy())
            if len(res_pi["changes"]) > nb_pi:
                print("!!! Change point for PI detected at time stamp ", res_pi["changes"][nb_pi], "!!!\n", sep="")
                nb_pi+= 1
        
        '''Ve Step'''
        #Section 3.3 of the paper
        for i in range(0,max_iter):
            print("iter:",i)
            tau[tau<1e-16] = 1e-16
            eta[eta<1e-16] = 1e-16
            delta[delta<1e-16] = 1e-16
    
            for e in range(0,e_it):
                '''Delta Estimation'''
                # Equation 10 in the paper
                q_ij[:,:,u] = pi[u,0]*(X[:,:,u]==0)*np.exp(-np.matmul(np.matmul(tau[:,:,u], np.log(Lambda)), np.transpose(eta[:,:,u]))*X[:,:,u] +np.matmul(np.matmul(tau[:,:,u], Lambda), np.transpose(eta[:,:,u])) + np.log(factorial(X[:,:,u])) -np.log(1-pi[u,0]))
                delta[:,:,u] = q_ij[:,:,u]/(1+q_ij[:,:,u])
                delta[(X>0)]=0 ##Corect! Checked!
                delta[delta<1e-16] = 1e-16
                ''' Tau Estimation '''
                # Equation 11 in the paper    
                mat_R[:,:,u] = np.matmul(np.matmul((1-delta[:,:,u])*X[:,:,u], eta[:,:,u]), np.transpose(np.log(Lambda))) - np.matmul(np.matmul((1-delta[:,:,u]),eta[:,:,u]), np.transpose(Lambda)) + np.log(alpha[u,:]) 
                z_q = logsumexp(mat_R[:,:,u], axis = 1)
                log_r_iq = mat_R[:,:,u]-np.array([z_q,]*Q).transpose()
                tau[:,:,u] = np.exp(log_r_iq)
                    
                ''' Eta Estimation '''
                # Equation 12 in the paper
                mat_S[:,:,u] = np.matmul(np.matmul(np.transpose((1-delta[:,:,u])*X[:,:,u]),tau[:,:,u]),np.log(Lambda)) -np.matmul(np.matmul(np.transpose((1-delta[:,:,u])),tau[:,:,u]),Lambda)+np.log(beta[u,:]) 
                w_l = logsumexp(mat_S[:,:,u], axis = 1)
                log_s_jl = mat_S[:,:,u]-np.array([w_l,]*L).transpose()
                eta[:,:,u] = np.exp(log_s_jl)
         
            tau[tau<1e-16] = 1e-16
            eta[eta<1e-16] = 1e-16
            delta[delta<1e-16] = 1e-16
    
            pi[pi<1e-16] = 1e-16
            alpha[alpha<1e-16] = 1e-16
            beta[beta<1e-16] = 1e-16
            Lambda[Lambda<1e-16] = 1e-16
        
            ''' M - Step: Mixture and Sparsity parameters '''
            # Equation 18 in the paper
            Den_old = np.sum(Den_new[:, :, 0:u+1], 2)
            X_delta[:, :, u] = X[:, :, u] * delta[:, :, u]
            Num_new[:, :, u] = np.matmul(np.transpose(tau[:,:,u]),np.matmul(X[:,:,u],eta[:,:,u])) - np.matmul(np.transpose(tau[:,:,u]),np.matmul(X_delta[:,:,u],eta[:,:,u]))
            Den_new[:, :, u] = np.matmul(np.sum(tau[:,:,u],axis=0)[..., None], np.sum(eta[:,:,u],axis=0).reshape(1,L))-  np.matmul(np.transpose(tau[:,:,u]),np.matmul(delta[:,:,u], eta[:,:,u]))
    
            Lambda = Lambda_old *( Den_old / (Den_old + Den_new[:, :, u])) + Num_new[:, :, u] / (Den_old + Den_new[:, :, u])
            start_time_tot = time.time()
            Lambda[Lambda<1e-16] = 1e-16
            
            if u > 5:
                print('******', 'time :', u, '******')    
            if u == 0 :
                #when t = 0: for alpha and beta we just do the softmax to inizialize the model parameters. For pi we just compute the mean of the latent variable in t=0
              
                device = torch.device('cpu')
                mean_tau = np.pad(alpha_res[u,:], (0, Q_diff), 'constant')
                alpha_init = np.log(mean_tau+1e-16)
                hidden_alpha = torch.zeros(2, 1, Q)
                f_alpha =  ShallowRegressionLSTM(out_features =Q,num_sensors=len(features_alpha), hidden_size=num_hidden_units,  proj_size = Q,u = u, mov_U = mov_U)
                alpha_0 = torch.tensor(alpha_init, dtype=torch.float32, requires_grad=True)
                opt_alpha = torch.optim.Adam(params= [alpha_0]+ list(f_alpha.parameters()), lr = 1e-4)
                PATH = "model_lstm_alpha.pt"
                torch.save(f_alpha.state_dict(), PATH)
                start_time = time.time()
                store_l_alpha = Mupdate(tau[:,:,u], alpha_0,  f_alpha,est_alpha[u-1,:], u,mov_U, opt_alpha, param="alpha", epochs = 500)
                print("--- %s seconds ---" % (time.time() - start_time), "time for 1 run")
                alpha_u = get_next(alpha_0, f_alpha, est_alpha[u-1,:],hidden_alpha, u, mov_U, Q)
                alpha_u[0][alpha_u[0]<1e-16] = 1e-16
                est_alpha[u,:] = alpha_u[0]
                alpha_prev_[u,:] = alpha_u[1].detach().numpy()
                res_alpha = onlineCPD(datapt=est_alpha[u,:].detach().numpy())
                nb_alpha = 0
                alpha[u,:] = est_alpha[u,:].detach().numpy()
                
                mean_eta = np.pad(beta_res[u,:], (0, L_diff), 'constant')
                beta_init = np.log(mean_eta+1e-16)
                hidden_beta = torch.zeros(2, 1, L)
                f_beta =  ShallowRegressionLSTM(out_features =L,num_sensors=len(features_beta), hidden_size=num_hidden_units,  proj_size = L,u = u, mov_U = mov_U)
                beta_0 = torch.tensor(beta_init, dtype=torch.float32, requires_grad=True)
                opt_beta = torch.optim.Adam(params= [beta_0]+ list(f_beta.parameters()), lr = 1e-4)
                PATH = "model_lstm_beta.pt"
                torch.save(f_beta.state_dict(), PATH)
                start_time = time.time()
                store_l_beta = Mupdate(eta[:,:,u], beta_0,  f_beta,est_beta[u-1,:], u,mov_U, opt_beta, param="beta", epochs = 500)
                print("--- %s seconds ---" % (time.time() - start_time), "time for 1 run")
                beta_u = get_next(beta_0, f_beta, est_beta[u-1,:],hidden_beta, u, mov_U, L)
                beta_u[0][beta_u[0]<1e-16] = 1e-16
                est_beta[u,:] = beta_u[0]
                beta_prev_[u,:] = beta_u[1].detach().numpy()
                beta[u,:] = est_beta[u,:].detach().numpy()
                
                mean_delta = pi_res[u], 1-pi_res[u]
                pi_init = np.log(mean_delta)
                hidden_pi = torch.zeros(2, 1, 2)
                f_pi =  ShallowRegressionLSTM(out_features =2,num_sensors=len(features_pi), hidden_size=num_hidden_units,  proj_size = 2,u = u, mov_U = mov_U)
                pi_0 = torch.tensor(pi_init, dtype=torch.float32, requires_grad=True)
                opt_pi = torch.optim.Adam(params= [pi_0]+ list(f_pi.parameters()), lr = 1e-4)
                PATH = "model_lstm_pi.pt"
                torch.save(f_pi.state_dict(), PATH)
                start_time = time.time()
                store_l_pi = Mupdate(delta[:,:,u], pi_0,  f_pi,est_pi[u-1,:], u,mov_U, opt_pi, param="pi", epochs = 500)
                print("--- %s seconds ---" % (time.time() - start_time), "time for 1 run")
                pi_u = get_next(pi_0, f_pi, est_pi[u-1,:],hidden_pi, u, mov_U, 2)
                pi_u[0][pi_u[0]<1e-16] = 1e-16
                est_pi[u,:] = pi_u[0]
                pi_prev_[u,:] = pi_u[1].detach().numpy()
                pi[u,:] = est_pi[u,:].detach().numpy()
    
            elif u >0 and u<mov_U:
                #when 0<t<mov_U we use the fully connected neural network to estimate the parameter alpha and beta, while for pi we compute the mean of the related latent variable (no neural network to avoid numerical pronblems due to the values of pi very close to 1)
              
                min_u =  max(0,u-mov_U)
                max_u = u 
                mean_tau = np.pad(alpha_res[u,:], (0, Q_diff), 'constant')
                alpha_init = np.log(mean_tau+1e-16)
                PATH = "model_fc_alpha.pt"
                f_alpha =  Two_layer_nn(Q, 100,50, Q)
                device = torch.device('cpu')
                torch.save(f_alpha.state_dict(), PATH)
                f_alpha.load_state_dict(torch.load(PATH, map_location=device))
               
                alpha_0 = torch.tensor(alpha_init, dtype=torch.float32, requires_grad=True)
                opt_alpha = torch.optim.Adam(params= [alpha_0]+ list(f_alpha.parameters()), lr = 1e-5)
                torch.save(f_alpha.state_dict(), PATH)
                start_time = time.time()
                log_alpha = est_alpha[u-1,:].clone().detach()
                t_log_alpha = log_alpha.clone().detach().requires_grad_(True)
                store_l_alpha = Mupdate(tau, alpha_0, f_alpha,t_log_alpha, u, mov_U, opt_alpha, param="alpha", epochs = 2000)
                print("--- %s seconds ---" % (time.time() - start_time), "time for 1 run")
                alpha_u = get_next(alpha_0, f_alpha, t_log_alpha, hidden_alpha, u, mov_U, Q)
                alpha_u[0][alpha_u[0]<1e-16] = 1e-16
                est_alpha[min_u:max_u+1,:] = alpha_u[0]
                alpha_prev_[u,:] = alpha_u[1].detach().numpy()
                alpha[min_u:max_u+1,:] = est_alpha[min_u:max_u+1,:].detach().numpy()
    
                mean_eta= np.pad(beta_res[u,:], (0, L_diff), 'constant')
                beta_init = np.log(mean_eta+1e-16)
                PATH = "model_fc_beta.pt"
                f_beta =  Two_layer_nn(L, 100,50, L)
                device = torch.device('cpu')
                torch.save(f_beta.state_dict(), PATH)
                f_beta.load_state_dict(torch.load(PATH, map_location=device))
                beta_0 = torch.tensor(beta_init, dtype=torch.float32, requires_grad=True)
                opt_beta = torch.optim.Adam(params= [beta_0]+ list(f_beta.parameters()), lr = 1e-5)
                torch.save(f_beta.state_dict(), PATH)
                start_time = time.time()
                log_beta = est_beta[u-1,:].clone().detach()
                t_log_beta = log_beta.clone().detach().requires_grad_(True)
                store_l_beta = Mupdate(eta, beta_0, f_beta,t_log_beta, u, mov_U, opt_beta, param="beta", epochs = 2000)
                print("--- %s seconds ---" % (time.time() - start_time), "time for 1 run")
                beta_u = get_next(beta_0, f_beta, t_log_beta, hidden_beta, u, mov_U, L)
                beta_u[0][beta_u[0]<1e-16] = 1e-16
                est_beta[min_u:max_u+1,:] = beta_u[0]
                beta_prev_[u,:] = beta_u[1].detach().numpy()
                beta[min_u:max_u+1,:] = est_beta[min_u:max_u+1,:].detach().numpy()
    
                mean_delta = pi_res[u], 1-pi_res[u]
                pi_init = np.log(mean_delta)
                PATH = "model_fc_pi.pt"
                f_pi =  Two_layer_nn(2,100,50, 2)
                device = torch.device('cpu')
                torch.save(f_pi.state_dict(), PATH)
                f_pi.load_state_dict(torch.load(PATH, map_location=device))
                pi_0 = torch.tensor(pi_init, dtype=torch.float32, requires_grad=True)
                opt_pi = torch.optim.Adam(params= [pi_0]+ list(f_pi.parameters()), lr = 1e-5)
                torch.save(f_pi.state_dict(), PATH)
                start_time = time.time()
                log_pi = est_pi[u-1,:].clone().detach()
                t_log_pi = log_pi.clone().detach().requires_grad_(True)
                store_l_pi = Mupdate(delta, pi_0, f_pi,t_log_pi, u, mov_U, opt_pi, param="pi", epochs = 3000)
                print("--- %s seconds ---" % (time.time() - start_time), "time for 1 run")
                pi_u = get_next(pi_0, f_pi, t_log_pi, hidden_pi, u, mov_U, 2)
                pi_u[0][pi_u[0]<1e-16] = 1e-16
                est_pi[min_u:max_u+1,:] = pi_u[0]
                pi_prev_[u,:] = pi_u[1].detach().numpy()
                pi[min_u:max_u+1,:] = est_pi[min_u:max_u+1,:].detach().numpy()
                fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
                ax = fig.add_subplot(3, 1, 1)
                ax.plot(alpha)
                
                plt.title('Estimated alpha')
                plt.ylim(0, 1)
                
                ax = fig.add_subplot(3, 1, 2, sharex=ax)
                ax.plot(beta)
                
                plt.title('Estimated beta')
                plt.ylim(0, 1)
                
                ax = fig.add_subplot(3, 1, 3, sharex=ax)
                ax.plot(pi[:,0])
                
                plt.title('Estimated pi')
                plt.ylim(0, 1)
                plt.show()
        
            else:          
                #when t>mov_U we use the LSTM neural network to estimate the parameter alpha and beta, while for pi we compute the mean of the related latent variable (no neural network to avoid numerical pronblems due to the values of pi very close to 1)
                min_u =  max(0,u-mov_U)
                max_u = u 
                alpha_old = est_alpha[(min_u):(max_u),:].detach().numpy()
                t_log_alpha = torch.tensor(alpha_old, dtype=torch.float32)
                t_log_alpha = torch.reshape(t_log_alpha, ( 1,t_log_alpha.shape[0], t_log_alpha.shape[1]))
                mean_tau = np.pad(alpha_res[u,:], (0, Q_diff), 'constant')

                alpha_init = np.log(mean_tau+1e-16)
                PATH = "model_lstm_alpha.pt"
                f_alpha =  ShallowRegressionLSTM(out_features =Q, num_sensors=len(features_alpha), hidden_size=num_hidden_units, proj_size  = Q, u = u, mov_U = mov_U)
                device = torch.device('cpu')
                if u > mov_U:
                    f_alpha.load_state_dict(torch.load(PATH, map_location=device))
                alpha_0 = torch.tensor(alpha_init, dtype=torch.float32, requires_grad=True)
                opt_alpha = torch.optim.AdamW(params=  list(f_alpha.parameters()), lr = 1e-3) 
                start_time = time.time()         
                store_l_alpha = Mupdate(tau, alpha_0, f_alpha, t_log_alpha, u, mov_U, opt_alpha, param="alpha",  epochs =5000)            
                print("--- %s seconds ---" % (time.time() - start_time), "time for 1 run")
                tmp =  torch.tensor(alpha_prev_[min_u:max_u,:], dtype=torch.float32)            
                tmp1= torch.reshape(tmp,(1,tmp.shape[0],  tmp.shape[1]))
                alpha_u = get_next(alpha_0, f_alpha,  tmp1,hidden_alpha, u, mov_U, Q)    
                hidden_alpha = alpha_u[0]
                est_alpha[min_u+1:max_u+1,:] =  alpha_u[1].clone().detach()   
                alpha_prev_[(min_u+1):(max_u+1),:] = alpha_u[1].detach().numpy()
                alpha[min_u+1:max_u+1,:] = est_alpha[min_u+1:max_u+1,:].detach().numpy()
                    
                beta_old = est_beta[(min_u):(max_u),:].detach().numpy()
                t_log_beta = torch.tensor(beta_old, dtype=torch.float32)
                t_log_beta = torch.reshape(t_log_beta, ( 1,t_log_beta.shape[0], t_log_beta.shape[1]))
                mean_eta = np.pad(beta_res[u,:], (0, L_diff), 'constant')

                
                beta_init = np.log(mean_eta+1e-16)
                PATH = "model_lstm_beta.pt"
                f_beta =  ShallowRegressionLSTM(out_features =L, num_sensors=len(features_beta), hidden_size=num_hidden_units, proj_size  = L, u = u, mov_U = mov_U)
                device = torch.device('cpu')
                if u > mov_U:
                    f_beta.load_state_dict(torch.load(PATH, map_location=device))
                beta_0 = torch.tensor(beta_init, dtype=torch.float32, requires_grad=True)
                opt_beta = torch.optim.AdamW(params=  list(f_beta.parameters()), lr =   1e-3)   
                start_time = time.time()         
                store_l_beta = Mupdate(eta, beta_0, f_beta, t_log_beta, u, mov_U, opt_beta, param="beta",  epochs =5000)            
                print("--- %s seconds ---" % (time.time() - start_time), "time for 1 run")
                tmp =  torch.tensor(beta_prev_[min_u:max_u,:], dtype=torch.float32)            
                tmp1= torch.reshape(tmp,(1,tmp.shape[0],  tmp.shape[1]))
                beta_u = get_next(beta_0, f_beta,  tmp1,hidden_beta, u, mov_U, L)    
                hidden_beta = beta_u[0]
                est_beta[min_u+1:max_u+1,:] =  beta_u[1].clone().detach()   
                beta_prev_[(min_u+1):(max_u+1),:] = beta_u[1].detach().numpy()
                torch.save(f_beta.state_dict(), PATH)
                beta[min_u+1:max_u+1,:] = est_beta[min_u+1:max_u+1,:].detach().numpy()
                
                min_u =  max(0,u-mov_U)
                max_u = u 
                pi_old = est_pi[(min_u):(max_u),:].detach().numpy()
                t_log_pi = torch.tensor(pi_old,dtype=torch.float32)
                t_log_pi = torch.reshape(t_log_pi, ( 1,t_log_pi.shape[0], t_log_pi.shape[1]))

                mean_delta = np.mean(delta[:,:,u]), 1- np.mean(delta[:,:,u])
                pi_init = np.log(mean_delta[0])
                PATH = "model_lstm_pi.pt"
                    
                f_pi =  ShallowRegressionLSTM(out_features =2, num_sensors=len(features_pi), hidden_size=num_hidden_units, proj_size  = 2, u = u, mov_U = mov_U)

                device = torch.device('cpu')
                if u > mov_U:
                    f_pi.load_state_dict(torch.load(PATH, map_location=device))
                pi_0 = torch.tensor(pi_init, dtype=torch.float32, requires_grad=True)
                opt_pi= torch.optim.AdamW(params=  list(f_pi.parameters()), lr =   5e-4) #128 nuroni 5e-5

                start_time = time.time()
                

                store_l_pi= Mupdate(delta, pi_0, f_pi, t_log_pi, u, mov_U, opt_pi, param="pi", epochs =2500)
                
                print("--- %s seconds ---" % (time.time() - start_time), "time for 1 run")
                tmp =  torch.tensor(pi_prev_[min_u:max_u,:], dtype=torch.float32)
                
                tmp1= torch.reshape(tmp,(1,tmp.shape[0],  tmp.shape[1]))
                pi_u = get_next(pi_0, f_pi,  tmp1,hidden_pi, u, mov_U, 2)

                hidden_pi = pi_u[0]
                est_pi[min_u+1:max_u+1,:] =  pi_u[1].clone().detach()
                print("est_pi:", est_pi[min_u+1:max_u+1,:])
                pi_prev_[(min_u+1):(max_u+1),:] = pi_u[1].detach().numpy()  
                pi[min_u+1:max_u+1,:] = est_pi[min_u+1:max_u+1,:].detach().numpy()
    
                torch.save(f_pi.state_dict(), PATH)
                fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
                ax = fig.add_subplot(3, 1, 1)
                ax.plot(alpha)
                for c in range(1, len(res_alpha['changes'])):
                    plt.axvline(x=res_alpha['changes'][c], color='red', linestyle='--')        

                plt.title('Estimated alpha')
                plt.ylim(0, 1)
                
                ax = fig.add_subplot(3, 1, 2, sharex=ax)
                ax.plot(beta)
                for c in range(1, len(res_beta['changes'])):
                    plt.axvline(x=res_beta['changes'][c], color='red', linestyle='--')        
                plt.title('Estimated beta')
                plt.ylim(0, 1)
                
                ax = fig.add_subplot(3, 1, 3, sharex=ax)
                ax.plot(pi[:,0])
                for c in range(1, len(res_pi['changes'])):
                    plt.axvline(x=res_pi['changes'][c], color='red', linestyle='--')        
                plt.title('Estimated pi')
                plt.ylim(0, 1)
                plt.show()
            
    time_run = time.time() - start_time_tot        #4796.93
    print("--- %s seconds ---" % (time.time() - start_time_tot)) 
    result = {'X': X, 'alpha':alpha,'beta': beta, 'pi': pi,'tau': tau, 'eta': eta, 
              'Lambda': Lambda, 'delta': delta, 'res_alpha': res_alpha, 'res_beta': res_beta, 
              'res_pi':res_pi, 'time_run':time_run}

    return result

# data = SimulateData_Py(350,300, 200)
# X = data["X"]
# M = data["X"].shape[0]
# P = data["X"].shape[1]
# U = data["X"].shape[2]
# alpha_sim = data["alpha"]
# beta_sim = data["beta"]
# pi_sim = data["pi"]
# Lambda_sim = data["Lambda"]
# Q = 3
# L = 2
# test = onlineDLBM(X, Q, L, max_iter =3)
# test['res_alpha']['changes'] = np.delete(test['res_alpha']['changes'], [4])


# fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
# ax = fig.add_subplot(3, 1, 1)
# ax.plot(data["alpha"])
# plt.figure(figsize = [10,5])  
# plt.ylim(0, 1)
# plt.title("Simulated alpha") 
# plt.plot(data['alpha'])
# for c in range(0, len(data["cp_alpha"])):
#     plt.axvline(x=np.cumsum(data["cp_alpha"])[c], color='red', linestyle='--')        

# ax = fig.add_subplot(3, 1, 2)
# ax.plot(data["beta"])

# plt.figure(figsize = [10,5])  
# plt.ylim(0, 1)
# plt.title("Simulated beta") 
# plt.plot(data['beta'])
# for c in range(0, len(data["cp_beta"])):
#     plt.axvline(x=np.cumsum(data["cp_beta"])[c], color='red', linestyle='--')        

# ax = fig.add_subplot(3, 1, 3)
# ax.plot(data["pi"][:,0])
# plt.figure(figsize = [10,5])  
# plt.ylim(0, 1)
# plt.title("Simulated pi") 
# plt.plot(data["pi"][:,0])

# for c in range(0, len(data["cp_pi"])):
#     plt.axvline(x=np.cumsum(data["cp_pi"])[c], color='red', linestyle='--')        
# plt.show()
# # fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
# # ax = fig.add_subplot(3, 1, 1)
# plt.figure(figsize = [10,5])  
# plt.ylim(0, 1)
# plt.title("Estimated alpha") 
# plt.plot(test['alpha'])
    
# for c in range(1, len(test['res_alpha']['changes'])):
#     plt.axvline(x=test['res_alpha']['changes'][c], color='red', linestyle='--')        

# # ax = fig.add_subplot(3, 1, 2)
# # ax.plot(test['beta'])
# plt.figure(figsize = [10,5]) 
# plt.title("Estimated beta") 
  
# plt.plot(test['beta'])
# plt.ylim(0, 1)

# for c in range(1, len(test['res_beta']['changes'])):
#     plt.axvline(x=test['res_beta']['changes'][c], color='red', linestyle='--')        

# # ax = fig.add_subplot(3, 1, 3)
# # ax.plot(test['pi'][:,0])

# plt.figure(figsize = [10,5 ])
# plt.title("Estimated pi") 
  
# plt.plot(test['pi'][:,0])

# plt.ylim(0, 1)    
# for c in range(1, len(test['res_pi']['changes'])):
#     plt.axvline(x=test['res_pi']['changes'][c], color='red', linestyle='--')        
# plt.show()
# # Open a csv file for writing
# # with open("test.csv", "w", newline="") as fp:
# #     # Create a writer object
# #     writer = csv.DictWriter(fp, fieldnames=test.keys())

# #     # Write the header row
# #     writer.writeheader()

# #     # Write the data rows
# #     writer.writerow(test)
# #     print('Done writing dict to a csv file')
# # with open("data.csv", "w", newline="") as fp:
# #     # Create a writer object
# #     writer = csv.DictWriter(fp, fieldnames=data.keys())

# #     # Write the header row
# #     writer.writeheader()

# #     # Write the data rows
# #     writer.writerow(data)
# #     print('Done writing dict to a csv file')

# # data = pd.DataFrame(data)

# # data.to_csv('data.csv', index=False)
# # #Works well with 16 neurons, lr = 1e-5 and 5000 epochs
