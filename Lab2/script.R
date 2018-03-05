install.packages("factoextra")
library(factoextra)
names=unique(NAm2$Pop)
npop=length(names)
coord=unique(NAm2[,c("Pop","long","lat")]) #coordinates for each pop
colPalette=rep(c("black","red","cyan","orange","brown","blue","pink","purple","darkgreen"),3)
pch=rep(c(16,15,25),each=9)
NAm2 = read.table("NAm2.txt", header = TRUE)
longi = NAm2$long
NAaux = NAm2[,-c(1:8)]
dataframe2 = data.frame(NAaux)

pcaNAm2 <- prcomp(dataframe, scale = FALSE)
lmlat <- lm(NAm2$lat~pcaNAm2$x[,1:250])
lmlong <- lm(NAm2$long~pcaNAm2$x[,1:250])
A <- pcaNAm2$rotation
eigvalues <- get_eigenvalue(pcaNAm2)
plot(1:494,eigvalues$eigenvalue)

#summary(pcaNAm2)
#axes <- pcaNAm2$x[,-c(251:494)]
#lmlong <- lm(longi~axes)
#summary(lmlong)
err <- c()
for (i in 1:npop) {
  x1 = matrix(c(lmlong$fitted.values[which(NAm2[,3]==names[i])],lmlat$fitted.values[which(NAm2[,3]==names[i])]),ncol = 2)
  x2 = matrix(coord[,2][i],nrow = nrow(x1), ncol = 2)
  x2[,2] = coord[,3][i]
  err <- cbind(err, rdist.earth(x2,x1,miles = FALSE)[1,])
}
  
  
mean(diag(err))