set.seed(3)
n=10
X1=rnorm(n)
X2=3*X1+rnorm(n)
Y=2+X1+X2+rnorm(n)
predictor1 <- lm(Y~X1)
predictor2 <- lm(Y~X2)
X=X1+X2
predictor <- lm(Y~X)
b01=predictor1$coefficients[1]
b11=predictor1$coefficients[2]
b02=predictor2$coefficients[1]
b12=predictor2$coefficients[2]
summary(predictor1)
summary(predictor2)
summary(predictor)
list(beta0=b01,beta1=b11,beta0=b02,beta1=b12)

