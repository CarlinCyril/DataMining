
ellipses = function(m, CV, probs)
{
  # Compute and plot an ellipse region for bivariate Gaussians, i.e., some ellipse that 
  # has probability probs to contain a 2D Gaussian vector with given parameters.
  # Inputs: 
  #   - m: means
  #   - CV: covariance matrix
  #   - probs: probabilities
  # Source : https://waterprogramming.wordpress.com/2016/11/07/plotting-probability-ellipses-for-bivariate-normal-distributions/
  
  # Coordinates of mean
  b1 = m[1]
  b2 = m[2]
  
  eg = eigen(CV)
  Evec = eg$vectors
  Eval = eg$values
  
  theta = seq(0,2*pi,0.01) # angles used for plotting ellipses
  
  vec.norm = function(v) { sqrt(t(v) %*% v)}
  # compute angle for rotation of ellipse
  # rotation angle will be angle between x axis and first eigenvector
  x.vec = c(1, 0) # vector along x-axis
  cosrotation = t(x.vec) %*% Evec[,1]/(vec.norm(x.vec)*vec.norm(Evec[,1]))
  rotation = pi/2-acos(cosrotation) # rotation angle
  #create a rotation matrix
  R  = matrix(c(sin(rotation), cos(rotation), -cos(rotation), sin(rotation)), 
              nrow=2, ncol=2, byrow = TRUE)
  
  # Create chi squared vector
  chisq = qchisq(probs,2) # percentiles of chi^2 dist df=2
  
  # size ellipses for each quantile
  xRadius = rep(0, length(chisq))
  yRadius = rep(0, length(chisq))
  x = list()
  y = list()
  x.plot = list()
  y.plot = list()
  rotated_Coords = list()
  for (i in 1:length(chisq)) {
    # calculate the radius of the ellipse
    xRadius[i]=(chisq[i]*Eval[1])^.5; # primary axis
    yRadius[i]=(chisq[i]*Eval[2])^.5; # secondary axis
    # lines for plotting ellipse
    x[[i]] = xRadius[i]* cos(theta);
    y[[i]] = yRadius[i] * sin(theta);
    # rotate ellipse
    rotated_Coords[[i]] = R %*% matrix(c(x[[i]], y[[i]]), nrow=2, byrow=TRUE)
    # center ellipse
    x.plot[[i]] = t(rotated_Coords[[i]][1,]) + b1
    y.plot[[i]] = t(rotated_Coords[[i]][2,]) + b2}
  
  xlim = range(x.plot[[i]])
  ylim = range(y.plot[[i]])
  plot(b1,b2, xlab = "X1", ylab = "X2", xlim = xlim, ylim = ylim, cex=1)
  abline(h=0)
  abline(v=0)
  # Plot contours
  for (j in 1:length(chisq)) {
    points(x.plot[[j]],y.plot[[j]], cex=0.1)}
  
  legend("bottomright", c('Ellipse region', paste(probs)))}
