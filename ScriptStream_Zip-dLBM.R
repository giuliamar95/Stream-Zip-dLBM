
library(reticulate)
# In terminal :
# conda create r-reticulate
# conda activate r-reticulate
# conda install pytorch::pytorch torchvision torchaudio -c pytorch
# conda install numpy matplotlib pandas math random scipy
Sys.setenv("r-reticulate" =  path.expand("~/r-reticulate/envs/gain_env"))
use_python(path.expand("~/opt/anaconda3/envs/r-reticulate/bin/python"))
use_condaenv(path.expand("~/opt/anaconda3/envs/r-reticulate"))

py_config()

pandas <- import("pandas")
numpy <- import("numpy")
scipy <- import("scipy")
torch <- reticulate::import("torch") 
sklearn <- import("sklearn")
matplotlib <- import("matplotlib")
math <- import("math")
random <- import("random")
source_python("online_dLBM_py.py")
source("SimulateData_ZIP.R")


# Simulation setup:

data = SimulateData_ZIP(M  = 400, P = 300, U = 50, "online")
Q = 3L
L = 2L
X = data$X
M  = dim(X)[1]
P = dim(X)[2]
U = dim(X)[3]

Q_stream = 7L
L_stream = 7L

##Computing the ICL executing static LBM on the first time instant
max_iter = 20L
Q  = 3:7
L = 3:7
mod = expand.grid(Q=Q, L=L)
icl = rep(NA,dim(mod)[1])
out_test = list()
best_mod = matrix(NA, nrow=dim(mod)[1], ncol=3)
library(raster)
data_sim = 10
best_mod = matrix(NA, nrow=data_sim, ncol=3)

R = matrix(as.vector(X[,,1]), nrow = M, ncol = P)
for (n in 1:data_sim) {
  for(i in 1:dim(mod)[1]){
    a_1 = runif(mod[i,1])
    b_1 = runif(mod[i,2])
    
    alpha = a_1/sum(a_1)
    beta = b_1/sum(b_1)
    pi_init = runif(1)
    Lambda = matrix(sample.int(mod[i,1]*mod[i,2]+1, mod[i,1]*mod[i,2], replace = FALSE), mod[i,1],mod[i,2], byrow = T)
    
    out_test[[i]] = LBM_ZIP(R, mod[i,1], mod[i,2], max_iter, alpha, beta, Lambda, pi_init)
    print(i)
    print(out_test[[i]][[9]])
    icl[i] = out_test[[i]][[9]]
  }  
  mod[,3] = unlist(icl)
  mod_ord = as.matrix(mod[order(mod$V3,decreasing = TRUE),])
  best_mod[n,] = as.vector(mod_ord[1,])
}

par(mfrow = c(1,1))

plot.default(unlist(icl), type = "b")
Mode <-
  function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }

#We Computed 10 times the icl, now we choose the mode
# Q = Mode(best_mod[,1])
# L = Mode(best_mod[,2])

Q = 3L
L = 2L
Q_stream = 7L
L_stream = 7L

# Execution of the online Zip-dLBM: 
# Q,L : values obtained as mode of the ICL 
# Q_Stream, L_Stream: maximum number of clusters
test = onlineDLBM(X,Q, L, Q_stream, L_stream, max_iter =3L)

# Results:
tau = test["tau"]$tau
eta =test["eta"]$eta
delta = test["delta"]$delta
alpha = test["alpha"]$alpha
beta = test["beta"]$beta
pi = test["pi"]$pi
Lambda =test["Lambda"]$Lambda
res_alpha =test["res_alpha"]$res_alpha
res_beta =test["res_beta"]$res_beta
res_pi =test["res_pi"]$res_pi

res_pi$changes = res_pi$changes[-1]
res_alpha$changes = res_alpha$changes[-1]
res_beta$changes = res_beta$changes[-1]

# Plotting the results:
# Alpha:
NewColors = c("#A6CEE3","#1F78B4","#FF00FF","#33A02C","#FB9A99","#E31A1C","#FDBF6F","#FF7F00","#33EEFF","#6A3D9A","#00FF00","#B15928", "#C39BD3","#F7A6DD","#CAB2D6","#CF33FF", "#C70039","#3498DB", "#33FFB0", "#2AAE7B", "#B2BABB", "#EDAFC4", "#EDC0AF")
par(mfrow=c(2,2))
layout(matrix(c(1,2),nrow=1), width=c(6,1)) 
par(mar=c(4,4,4,0)) #No margin on the right side
matplot(alpha, type = "l", xaxt = "n", ylab = " ", col=NewColors[1:ncol(alpha)],lty=1:ncol(alpha), lwd = 3, ylim = c(0,1))
abline(v=res_alpha$changes+1,lty=4, lwd=2.75, col = "red")
axis(1, at =seq(1, U, by = 1)  , labels =unique(X$month) )

par(mar=c(5,0,4,2)) #No margin on the left side
plot.default(c(0,1),type="n", axes=FALSE, xlab="", ylab="")
legend("center", paste("D", 1:ncol(alpha)),col=NewColors[1:ncol(alpha)],cex = 1.5, lty=1:ncol(alpha), lwd  =3, title = "Cluster:")
dev.copy2pdf(file = "Drug_estim.pdf", width = 15, height = 12)


# Beta:
layout(matrix(c(1,2),nrow=1), width=c(6,1)) 
par(mar=c(4,4,4,0)) #No margin on the right sideU%/%6, U%/%4, U%/%3
matplot(beta, type = "l", cex.main =2.5, xaxt = "n", ylab = " ", lty=1:ncol(beta), col = NewColors[1:ncol(alpha)+7], lwd = 3, ylim = c(0,1))
abline(v=res_beta$changes+1,lty=4, lwd=2.75, col = "red")
axis(1, at =seq(1, U, by = 1)  , labels =unique(X$month) )
par(mar=c(5,0,4,2)) #No margin on the left side
plot.default(c(0,1),type="n", axes=FALSE, xlab="", ylab="")
legend("center", paste("A",1:ncol(beta)),col=NewColors[1:ncol(alpha)+7], cex =1.5,lty=1:ncol(beta), lwd = 3,  title = "Cluster:")
dev.copy2pdf(file = "B_estim.pdf", width = 15, height = 12)

par(mfrow = c(1,1))
par(mar=c(4,4,4,4)) #No margin on the right side
matplot(pi[,1], type = "l", cex.main =2.5, xaxt = "n", ylab = " ", col = "blueviolet", ylim = c(0.98,1), lwd = 3)
axis(1, at =seq(1, U, by = 1)  , labels =unique(X$month) )
abline(v=res_pi$changes[-1],lty=4, lwd=2.75, col = "red")
dev.copy2pdf(file = "pi_eestim.pdf", width = 15, height = 12)


# Lambda with all the clusters:
image( 1:(L_stream+1), 1:(Q_stream+1),t(Lambda)^0.3,col=colorRampPalette(c('white',NewColors[2]),alpha=TRUE)(255), axes=FALSE,
       useRaster = FALSE, xlab = "Clusters of adv effects",  ylab = "Clusters of drugs")
text(y=seq(1+0.5,Q_stream+0.5,1), labels = paste(seq(1,Q_stream), sep = ""),par("usr")[1], pos=2, cex=0.5, adj = 0,xpd=TRUE)
text(seq(1+0.5,L_stream+0.5,1), labels = paste(seq(1,L_stream), sep = ""),par("usr")[1], pos=1, cex=0.5, adj = 0,xpd=TRUE)
box()
abline(v=seq(1:L_stream+1),lty=2, lwd=0.25)
abline(h=seq(1:Q_stream+1),lty=2, lwd=0.25)
box()

# Lambda with only the used clusters
Lambda_ = Lambda[1:Q, 1:L]
par(mfrow = c(1,1))
par(mar=c(6,6,6,6)) 
#Lambda
j = 1:3
i = 1:3
image( 1:(L+1),1:(Q+1),t(Lambda_)^0.3,col=colorRampPalette(c('white',NewColors[2]),alpha=TRUE)(255), axes=FALSE,
       useRaster = TRUE, xlab = " ", line = 1, ylab = " ")
text(y=seq(1+0.5,Q+0.5,1),  x=rep(0.9,Q),labels = paste("D", seq(1,Q), sep = ""),par("usr")[1], pos=2, cex= 1, adj = 0,xpd=TRUE, col = NewColors[1:3])
points(y=seq(1+0.5,Q+0.5,1), x=rep(0.9,L), cex= 2.5, xpd=TRUE, pch = 16,col = NewColors[1:3])
text(x=seq(1+0.5,L+0.5,1),  y=rep(0.7, Q),labels = paste("A", seq(1,L), sep = ""),par("usr")[1], pos=1, cex= 1, adj = 0,xpd=TRUE, col = NewColors[8:11])
points(x=seq(1+0.5,L+0.5,1), y=rep(0.8,L), cex= 2.5, xpd=TRUE, pch = 16,col =NewColors[8:11])
ij <- expand.grid(i=i, j=j)
text (x=ij$j+0.5, y=ij$i+0.5, paste(round(Lambda[j,i], 2), sep = " "))
abline(v=seq(1:L+1),lty=2, lwd=0.25)
abline(h=seq(1:Q+1),lty=2, lwd=0.25)
box()

# ARI and CARI:
library(mclust)
library(bikm1)

ARI_Z = rep(NA, U)
ARI_W = rep(NA, U)
CARI = rep(NA, U)
ARI_delta = rep(NA, U)
library(mclust)
for (u in 1:U) {
  ARI_Z[u] = adjustedRandIndex(max.col(tau[,,u]), data$row_clust[,u])  
  ARI_W[u] = adjustedRandIndex(max.col(eta[,,u]), data$col_clust[,u])  
  CARI[u] = CARI(max.col(tau[,,u]), max.col(eta[,,u]), data$row_clust[,u], data$col_clust[,u])$cari
}
par(mfrow=c(1,3))

plot.default(ARI_Z, type = "b", ylim = c(0,1), col = 5, ylab = "ARI values", xlab = "Time (t)", main = "ARI - row clusters")
plot.default(ARI_W, type = "b", ylim = c(0,1), col = 5, ylab = "ARI values", xlab = "Time (t)", main = "ARI - column clusters")
plot.default(CARI, type = "b", ylim = c(0,1), col = 5, ylab = "CARI values", xlab = "Time (t)", main = "CARI")


# Plotting the number of clusters:
# 
# barplot(rep(1,40), 
#         axes = FALSE, 
#         space = 0, 
#         xlim = c(0,5),
#         width=0.12, 
#         col=colors()[grep("blue", colors())])
# #plot(ARI_delta, type = "b", ylim = c(0,1), col = 5, ylab = "ARI values", xlab = "Time(U)", main = "ARI - column clusters")
# mean(ARI_Z)
# sd(ARI_Z)
# mean(ARI_W)
# sd(ARI_W)
# mean(CARI)
# sd(CARI)
# incidence = function(X, u){
#   NewColors = c("#7F5AA2", "#04F6FF","#00CC66","#FF5804","#FFBC00","#E600FF","#7650F3","#FF7F00", "#B2DF8A")
#   ColorRamp <- rgb(seq(0.99,0.4,length=50),seq(0.99,0.6,length=50),seq(0.99,0.99,length=50))
#   M = dim(X)[1]; P = dim(X)[2]
#   N = matrix(X[,,u],nrow = M, ncol = P)
#   Zord = order(max.col(tau[,,u]))
#   Word = order(max.col(eta[,,u]))
#   N = N[Zord,Word]
#   # image(1:(P+1),1:(N+1),log(t(M)),col=colorRampPalette(NewColors[c],alpha=TRUE)(255),cex=100,axes= TRUE,
#   #       main=paste('Time cluster #',c,sep=''),useRaster = FALSE,  xlab="Adversarial Effects", ylab="Medical Products")
#   #"springgreen2", "deeppink"
#   image(1:(P+1),1:(M+1), log(t(N)),col=colorRampPalette(c('white',"deeppink"),alpha=TRUE)(255),axes= TRUE,
#         main=paste('Time instant t = ',u,sep=''),useRaster = TRUE,  xlab="Clusters of columns", ylab="Clusters of rows") 
#   #points( 5:12, 5:12, pch="X", cex=3)
#   
#   box()
#   abline(h=((which(diff(max.col(tau[,,u])[Zord])==1)+1)),lty=2, lwd=0.75)
#   abline(v=((which(diff(max.col(eta[,,u])[Word])==1)+1)),lty=2, lwd=0.75)
# }
# incidence(X, 10)
# incidence(X, 30)
# save(data, out1, file ="Exp_1.Rdata")
