
library(slam)
library(Rlab)
#--------------------------------------------------------------------
# softmax function: Recursive formula
#--------------------------------------------------------------------


SimulateData_ZIP <- function(M = 200, P = 300, U = 100, scenario = "easy", ...){
  
  if (scenario == "hard"){
    ###ORIGINAL - 1 Experiment
    Q = 3
    alpha = matrix(NA, nrow = U, ncol = Q)
    alpha[,1] = seq(from = 25, to = 6, length = U)
    alpha[,2] = seq(from = 6, to = 25, length = U)
    alpha[,3] = seq(from = 2, to = 25, length = U)
    alpha = exp(alpha)/rowSums(exp(alpha))
    # 
    # 
    L=2
    beta = matrix(NA, nrow = U, ncol = L)
    beta[,1] = seq(from = 5, to = 2, length = U)
    beta[,2] = seq(from = 1, to = 6, length = U)
    beta = exp(beta)/rowSums(exp(beta))
    
    pi = matrix(NA, nrow = U, ncol = 2)
    
    # pi[,1] = seq(from = 2, to = 5, length = U)
    # pi[,2] = seq(from = 0.2, to = 0.01, length = U)
    pi[,2] = seq(from = 5, to = 2, length = U)
    pi[,1] = seq(from = 1, to = 6, length = U)
    
    # pi[,1] = sin(seq(from = 4, to =10, length = U))
    # pi[,2] = sin(seq(from = 3, to = 8, length = U))
    pi = exp(pi)/rowSums(exp(pi))
    pi = 0.8+ pi/9 #0.85+ pi/7 before
    #matplot(pi[,1], type = "l")
    
    
    Lambda = rbind(c(6, 4),
                   c(1, 2),
                   c(7, 3))
  }else if(scenario == "online"){
      ###ORIGINAL - 1 Experiment
      Q = 3
      cp_alpha = c(U%/%9, U%/%3, U%/%4)
      alpha = matrix(NA, nrow = U, ncol = Q)
      alpha[,1] = c(rnorm(cp_alpha[1]+cp_alpha[2], 0, 0.12), 
                    rnorm(cp_alpha[3], 0.3, 0.12), 
                    rnorm(U-cumsum(cp_alpha)[3], 0.8, 0.12))
      alpha[,2] = c(rnorm(cp_alpha[1], 0.7,0.12), 
                    rnorm(cp_alpha[2], 0.2, 0.12), 
                    rnorm(cp_alpha[3], 0.4, 0.12), 
                    rnorm(U-cumsum(cp_alpha)[3], 0.6, 0.12))
      alpha[,3] = rep(1, U)-(alpha[,1]+alpha[,2])
      alpha = exp(alpha)/rowSums(exp(alpha))
      # 
      # 
      L=2
      beta = matrix(NA, nrow = U, ncol = L)
      cp_beta = c(U%/%6, U%/%4, U%/%3)
      
      beta[,1] = c(rnorm(cp_beta[1], 0.1, 0.1),
                   rnorm(cp_beta[2], 0.7, 0.1),
                   rnorm(cp_beta[3], 0.3, 0.1),
                   rnorm(U-cumsum(cp_beta)[3], 0.6, 0.1))
      
      beta[,2] = rep(1,U)-(beta[,1])
      beta = exp(beta)/rowSums(exp(beta))
      
      pi = matrix(NA, nrow = U, ncol = 2)
      cp_pi = c(U%/%9, U%/%3, U%/%4)
      
      pi[,1] = c(rnorm(cp_pi[1], 0.35, 0.1),
                 rnorm(cp_pi[2], 0.7, 0.1),
                 rnorm(cp_pi[3], 0.25, 0.1),
                 rnorm(U-cumsum(cp_pi)[3], 0.6, 0.1))
      #pi[,1]  = rep(0.3, U)
      pi[,2] = rep(1,U)-(pi[,1])
      
      # pi[,1] = sin(seq(from = 4, to =10, length = U))
      # pi[,2] = sin(seq(from = 3, to = 8, length = U))
      pi = exp(pi)/rowSums(exp(pi))
      #matplot(pi[,1], type = "l")
      
      # Lambda = rbind(c(6, 1, 1, 6),
      #                c(1, 6, 1, 1),
      #                c(6, 1, 6, 1))
      matplot(alpha, type = "l", main = "Simulated alpha")
      matplot(beta, type = "l", main = "Simulated beta")
      matplot(pi[,1], type = "l", main = "Simulated pi")
      
      Lambda = rbind(c(6, 4),
                     c(1, 2),
                     c(7, 3))
      # Lambda = rbind(c(1,4,1,3),
      #       c(1, 2,4,8),
      #       c(5,1, 7, 3))
    
  }else if(scenario == "easy"){
    alpha = matrix(NA, nrow = U, ncol = 3)
    alpha[,1] = log(seq(from = 0.65, to = 0.2, length = U))^2
    alpha[,2] = log(seq(from = 0.95, to = 0.4, length = U))^2
    alpha[,3] = log(seq(from = 0.35, to = 0.85, length = U))^4
    alpha = exp(alpha)/rowSums(exp(alpha))
    matplot(alpha, type ="l")
    # 
    beta = matrix(NA, nrow = U, ncol = 2)
    beta[,1] = seq(from = 5, to = 2, length = U)
    beta[,2] = seq(from = 1, to = 6, length = U)
    beta = exp(beta)/rowSums(exp(beta))
    matplot(beta, type ="l")
    
    par(mfrow=c(1,2))
    #matplot(alpha, type = "l")
    #matplot(beta, type = "l")
    
    X = array(NA,c(M, P, U))
    A = array(0,c(M, P, U))
    
    #  pi = runif(U,0,1)
    pi = matrix(NA, nrow = U, ncol = 2)
    
    # pi[,1] = seq(from = 2, to = 5, length = U)
    # pi[,2] = seq(from = 0.2, to = 0.01, length = U)
    pi[,2] = seq(from = 5, to = 2, length = U)
    pi[,1] = seq(from = 1, to = 6, length = U)
    
    # pi[,1] = sin(seq(from = 4, to =10, length = U))
    # pi[,2] = sin(seq(from = 3, to = 8, length = U))
    pi = exp(pi)/rowSums(exp(pi))
    pi = 0.8+ pi/9  #7 before
    matplot(pi[,1], type = "l")
    
    
    Lambda = rbind(c(6, 4),
                   c(1, 2),
                   c(7, 3))
  }else if(scenario == "exp4"){
    alpha = matrix(NA, nrow = U, ncol = 5)
    alpha[,1] = seq(from = 0.65, to = 0.2, length = U)
    alpha[,2] = seq(from = 0.95, to = 0.7, length = U)
    alpha[,3] = seq(from = 0.35, to = 0.65, length = U)
    alpha[,4] = seq(from = 0.35, to = 0.1, length = U)
    alpha[,5] = seq(from = 0.5, to = 0.8, length = U)
    
    alpha = exp(alpha)/rowSums(exp(alpha))
    matplot(alpha, type ="l")
    # 
    beta = matrix(NA, nrow = U, ncol = 4)
    beta[,1] = seq(from = 5, to = 2, length = U)
    beta[,2] = seq(from = 1, to = 6, length = U)
    beta[,3] = seq(from = 3, to = 4, length = U)
    beta[,4] = seq(from = 4, to = 1, length = U)
    
    beta = exp(beta)/rowSums(exp(beta))
    
    par(mfrow=c(1,2))
    matplot(alpha, type = "l")
    matplot(beta, type = "l")
    
    X = array(NA,c(M, P, U))
    A = array(0,c(M, P, U))
    
    #  pi = runif(U,0,1)
    pi = matrix(NA, nrow = U, ncol = 2)
    
    # pi[,1] = seq(from = 2, to = 5, length = U)
    # pi[,2] = seq(from = 0.2, to = 0.01, length = U)
    pi[,2] = seq(from = 5, to = 2, length = U)
    pi[,1] = seq(from = 1, to = 6, length = U)
    
    # pi[,1] = sin(seq(from = 4, to =10, length = U))
    # pi[,2] = sin(seq(from = 3, to = 8, length = U))
    pi = exp(pi)/rowSums(exp(pi))
    pi = 0.8+ pi/9  #7 before
    matplot(pi[,1], type = "l")
    
    
    Lambda = rbind(c(6, 4, 3, 2),
                   c(1, 2, 5, 3),
                   c(7, 3, 4, 2),
                   c(1, 5, 7, 4),
                   c(4, 3, 1, 6))
  }else{
    alpha = matrix(NA, nrow = U, ncol = 3)
    alpha[,1] = log(seq(from = 0.3, to = 0.7, length = U))
    alpha[,2] = log(seq(from = 0.75, to = 0.45, length = U))
    alpha[,3] = log(seq(from = 0.35, to = 0.55, length = U))
    alpha = exp(alpha)/rowSums(exp(alpha))
    matplot(alpha, type ="l")
    # 
    beta = matrix(NA, nrow = U, ncol = 2)
    beta[,1] = seq(from = 5, to = 2, length = U)
    beta[,2] = seq(from = 1, to = 6, length = U)
    beta = exp(beta)/rowSums(exp(beta))
    
    par(mfrow=c(1,2))
    matplot(alpha, type = "l")
    matplot(beta, type = "l")
    
    X = array(NA,c(M, P, U))
    A = array(0,c(M, P, U))
    
    #  pi = runif(U,0,1)
    pi = matrix(NA, nrow = U, ncol = 2)
    
    # pi[,1] = seq(from = 2, to = 5, length = U)
    # pi[,2] = seq(from = 0.2, to = 0.01, length = U)
    pi[,2] = seq(from = 5, to = 2, length = U)
    pi[,1] = seq(from = 1, to = 6, length = U)
    
    # pi[,1] = sin(seq(from = 4, to =10, length = U))
    # pi[,2] = sin(seq(from = 3, to = 8, length = U))
    pi = exp(pi)/rowSums(exp(pi))
    pi = 0.8+ pi/11 #7 before
    matplot(pi[,1], type = "l")
    
    
    Lambda = rbind(c(6, 4),
                   c(1, 2),
                   c(7, 3))
  }
  Q = ncol(alpha)
  L = ncol(beta)
  Zinit = array(NA, c(M, Q, U))
  Winit = array(NA, c(P, L, U))
  Z = matrix(NA, nrow = M, ncol = U)
  W = matrix(NA, nrow = P, ncol = U)
  X = array(NA,c(M, P, U))
  A = array(0,c(M, P, U))
  for (u in 1:U) {
    
    
    Zinit[,,u] = t(rmultinom(M,1,alpha[u,]))
    Winit[,,u] = t(rmultinom(P,1,beta[u,]))
    Z[,u] = max.col(Zinit[,,u])
    W[,u] = max.col(Winit[,,u])
    
    A[,,u] = rbern(M*P, pi[u,1])
    for (q in 1:Q) {
      for (l in 1:L) {
        sel_z =which(Z[,u]==q)
        sel_w = which(W[,u]==l)
        X[sel_z,sel_w,u] = rpois(length(sel_z)*length(sel_w), Lambda[q,l])
        # X[sel_z,sel_w,u] = matrix(rpois(length(sel_z)*length(sel_w), Lambda[q,l]), nrow = length(sel_z))
        #  X[sel_z,sel_w,u] = matrix(rpois(M*P, Lambda[q,l]), nrow = M, ncol = P)
        
        
      }
    }
  }
  
  X[which(A==1)]=0
  
  
  list(X=X,row_clust=Z,col_clust=W, A=A, pi = pi,alpha = alpha, beta= beta, Zinit = Zinit, Winit = Winit, Lambda = Lambda)
}

#Checking!
# which(A[1,,2]==0)
# which(X[1,,2]!=0)


