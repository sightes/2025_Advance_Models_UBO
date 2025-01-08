### PARA CALCULAR LAS DERIVADAS DE ORDEN 1 Y 2 ###
#install.packages("fda.usc")
library(fda.usc)

TCM.fD1 <- Data2fd(1:62, y=LATAM_MA, basisobj=bspline.basis)
fdataobj <- fdata(TCM.fD1)
class(fdataobj)

TCM.D1 <- fdata.deriv(fdataobj, nderiv = 1)
plot(TCM.D1, xlab="Time(year)", lty = 1:20, col = 1:20, main = "", ylim = c(-0.3,0.2), lwd = 1.5)
plot(TCM.D1, xlab="Time(year)", lty = 1:20, col = 1:20, main = "", ylim = c(-0.5,0.3), lwd = 1.5)
abline(h = 0, col = "black", lty = 5, lwd = 1.0) # línea y = 0

legend(x = 0, y = -0.1, legend = Paislatam1, col = 1:5, bty = "n", cex = 0.5, lty = 1:5, lwd = 1.5, horiz = TRUE)
legend(x = 0, y = -0.15, legend = Paislatam2, col = 6:10, bty = "n", cex = 0.5, lty = 6:10, lwd = 1.5, horiz = TRUE)
legend(x = 0, y = -0.2, legend = Paislatam3, col = 11:15, bty = "n", cex = 0.5, lty = 11:15, lwd = 1.5, horiz = TRUE)
legend(x = 0, y = -0.25, legend = Paislatam4, col = 16:20, bty = "n", cex = 0.5, lty = 16:20, lwd = 1.5, horiz = TRUE)




### Cálculo func derivadas orden 1 y 2 VENEZUELA ###
TCM.fD1_VEN <- Data2fd(1:62, y=LATAM_MA[,20], basisobj=bspline.basis)
fdataobj_ven <- fdata(TCM.fD1_VEN)

TCM.D1_ven <- fdata.deriv(fdataobj_ven, nderiv = 1)
plot(TCM.D1_ven, xlab="Time(year)", lty = 1:20, col = 1:20, main = "", ylim = c(-0.6,0.3), lwd = 1.5)
abline(h = 0, lty = 5, lwd = 1.0, col="red") # línea y = 0

TCM.D2_ven <- fdata.deriv(fdataobj_ven, nderiv = 2)
plot(TCM.D2_ven, xlab="Time(year)", lty = 1:20, col = 1:20, main = "", ylim = c(-0.2,0.3), lwd = 1.5)
abline(h = 0, lty = 5, lwd = 1.0, col="red") # línea y = 0




# AJUSTE DE LAS CURVAS ##
# CALCULANDO RSM A CADA PAÍS
# CÁLCULO DE RMSE ESTIMANDO CON 15 FUNCIONES SPLINE BASE, PAÍS CHILE ##
bspline.basisn <- create.bspline.basis(c(1,62), nbasis=15)
TCD.SCHIn <- smooth.basis(y=LATAM_MA[,6], fdParobj=bspline.basisn, returnMatrix=TRUE)
xfds15 <- TCD.SCHIn$fd
dom <- c(1:62)

plotfit.fd(LATAM_MA[,6], dom, xfds15, xlab="Time(year)", main="AJUSTE CON 15 FUNCIONES BASE")
lines(xfds15, col="red")

RMSE_15 <- sqrt(mean((eval.fd(dom, xfds15) - LATAM_MA[,6])^2))
RMSE_15


## CREANDO DATOS FUNCIONALES ##
## Se debe crear de esta manera los DATOS funcionales para poder obtener el FPCAL ## PCA ##
bspline.basis <- create.bspline.basis(rangeval=c(1,62), nbasis=35)
TCM.fd <- Data2fd(1:62, y=LATAM_MA, basisobj=bspline.basis)


# ANÁLISIS DE COMPONENTES PRINCIPALES FUNCIONALES ##
# Cuatro primeras componentes
pca_result <- pca.fd(TCM.fd, nharm = 4)
### GRÁFICO DE LAS CUATRO PRIMERAS COMPONENTES PRINCIPALES ###
plot(pca_result$harmonics, main = "", xlab = "Time(year)", ylab = "Funciones Propias")
abline(h = pca_result$scores[1,1], col = "orange", lty = 5, lwd = 1.0) # línea y = 0

pca_result <- pca.fd(TCM.fd, nharm = 2)
### GRÁFICO DE LAS CUATRO PRIMERAS COMPONENTES PRINCIPALES ###
plot(pca_result$harmonics, main = "", xlab = "Time(year)", ylab = "Funciones Propias")
abline(h = pca_result$scores[1,1], col = "blue", lty = 5, lwd = 1.0) # línea y = 0



# VARIANZA EXPLICADA POR LAS CUATRO PRIMERAS COMPONENTES
pca_result$varprop
barplot(pca_result$varprop, col = "blue", ylab = "var explicada", xlab = "Funciones principales")

# VARIANZA EXPLICADA POR LOS DOS PRIMEROS COMPONENTES
pca_result$varprop


# Obtener los scores de los primeros dos componentes principales
scores <- pca_result$scores
print(scores)



### GRAFICA DE LAS COORDENADAS DE LAS DOS PRIMERAS COMPONENTES PRINCIPALES ###
plot(pca_result$scores[,1], pca_result$scores[,2])
for (i in 1:20) {
  points(pca_result$scores[i,1], pca_result$scores[i,2])
  print(pca_result$scores[i,1], pca_result$scores[i,2])
}



# GRÁFICA DE LOS MARCADORES ASOCIADOS A LOS DISTINTOS DATOS FUNCIONALES (INDIVIDUOS)

Paiseslatam <- c(Paislatam1, Paislatam2, Paislatam3, Paislatam4)
plot(pca_result$scores[,1], pca_result$scores[,2], pch = 3, cex = 0.2,
     reg.line = FALSE, smooth = FALSE,
     spread = FALSE, id.method = 'mahal', id.n = 54,
     boxplots = FALSE, span = 0.3, cex.lab = 1, xlab = "Primera Componente", ylab = "Segunda Componente")
abline(h = 0, col = "black", lty = 5, lwd = 1.0) # línea y = 0
text(pca_result$scores[,1], pca_result$scores[,2], Paiseslatam, font = 0.6, col = 1:20)



### GRÁFICO DE LA FUNCIÓN DE COVARIANZA ###
# LA FUNCIÓN DE COVARIANZA ES BIVARIADA SU GRÁFICA ES UNA SUPERFICIE EN R3 #
heat.colors(20)
persp(dom, dom, F.covar, xlab = "", ylab = "", zlab = "Covarianza TCD",
      expand = 0.55, theta = 30, phi = 20,
      border = NA, shade = 0.15, ticktype = 'detailed', col = "#FF8800")




## GRÁFICO DE CONTORNOS ##
## CURVAS DE NIVEL Y VALORES DE LAS COVARIANZAS ##
gv <- contour(dom, dom, F.covar, lwd = 2, col = "brown")




## GENERANDO LA MATRIZ DE LAS CORRELACIONES ##
cor_matrix <- cor.fd(dom, FTCD$fd)
library(corrplot)
corrplot(cor_matrix, method = "color", tl.cex = 0.55)


