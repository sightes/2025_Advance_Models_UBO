#grafico de los valores observados de la TCD para los países de Latinoamérica#
library(fda)
library(corrplot)

LATAM=TCDM
matplot(LATAM[,2:21], type = "l", pch = 1, xlim = c(-1,63), ylim=c(-5,6),
        xlab = "Time(year)", ylab="TCD_LATAM", col = 1:20, cex = 0.7)
Paislatam1 = c("ARG", "BOL", "BRA", "CHT", "COL")
Paislatam2 = c("CRI", "CUB", "ECU", "SLV", "GTM")
Paislatam3 = c("HTI", "HND", "MEX", "NIC", "PAN")
Paislatam4 = c("PRY", "PER", "DOM", "URY", "VEN")
Paislatam5 = c("CHI", "COL", "PRY", "PER", "VEN")
legend(x = -1, y = -1.0, legend = Paislatam1, col = 1:5,bty = "n", cex = 0.7, lwd = 1,horiz = TRUE)
legend(x = -1, y = -2.0, legend = Paislatam2, col = 6:10,bty = "n", cex = 0.7,lty = 6:10, lwd = 1,horiz = TRUE)
legend(x = -1, y = -3.0, legend = Paislatam3, col = 11:15,bty = "n", cex = 0.7,lty = 11:15, lwd = 1,horiz = TRUE)
legend(x = -1, y = -4.0, legend = Paislatam4, col = 16:20,bty = "n", cex = 0.7,lty = 16:20, lwd = 1,horiz = TRUE)

#install.packages("fda")

## ANALISIS DE FUNCIONES ATÍPICAS##
## REQUIERE DE CREAR LAS FUNCIONES DE SERIES TEMPORALES##
## gráfico de las curvas como series temporales##
## sfts() ajusta un conjunto de datos a un modelo basado en funciones splines##
## (curvas suavizadas) en el contexto de series temporales ##
## Requiere de que los datos esten en formato matriz ##

dim(LATAM[,2:21])
LATAM_MA <- as.matrix(LATAM[,2:21])
LATAM_NU <- as.numeric(LATAM_MA)

#install.packages("rainbow")
#library(rainbow)
PMTs = sfts(ts(LATAM_NU, start = c(1,1), frequency = 62), xname = "Time(year)",
            yname = "TCD_LATAM")

plot(PMTs)



#legend pos usa librería Commt#
#ANALISIS DE FUNCIONES ATÍPICAS##
##REQUIERE DE CREAR LAS FUNCIONES DE SERIES TEMPORALES##
fboxplot(data= PMTs, plot.type = "functional", type = "bag", projmethod = "PCAproj",
         ylim = c(-3,4), xlim = c(-1.5,60), nrow=1)
legend(x = -0.1 , y = -2.0,legend = c("HND","URY","VEN"), col = c("red","green","blue")
       ,bty = "L", cex = 0.45, lty = 1, lwd = 2 ,horiz = TRUE)






##CREANDO LAS FUNCIONES BASE##
#install.packages("fda")
#library(fda)

###ES NECESARIO USAR LOS DATOS EN FORMA DE MATRIZ###
##A MODO DE EJEMPLO TOMAMOS 30 FUNCIONES SPLINE BASE###
bspline.basis = create.bspline.basis(rangeval = c(0,62), nbasis = 30)

##VISUALIZANDO LAS FUNCIONES BASE SPLINE##
plot(bspline.basis)




##ESTIMANDO TODAS LOS DATOS FUNCIONES CON LA BASE DE FUNCIONES SPLINE##
FTCD = smooth.basis(y = LATAM_MA, fdParobj = bspline.basis)
plot(FTCD, xlab = "Time(year)", ylab = "DF_LATAM", lty = 1:20, col = 1:20)

legend(x = -1, y = 0.0, legend = Paislatam1, col = 1:5, bty = "n", cex = 0.55, lwd = 1, horiz = TRUE)
legend(x = -1, y = -0.5, legend = Paislatam2, col = 6:10, bty = "n", cex = 0.55, lty = 6:10, lwd = 1, horiz = TRUE)
legend(x = -1, y = -1.0, legend = Paislatam3, col = 11:15, bty = "n", cex = 0.55, lty = 11:15, lwd = 1, horiz = TRUE)
legend(x = -1, y = -1.5, legend = Paislatam4, col = 16:20, bty = "n", cex = 0.55, lty = 16:20, lwd = 1, horiz = TRUE)


##ESTADÍSTICOS DESCRIPTIVOS DE DATOS FUNCIONALES##
##MEDIA Y DESVIACIÓN ESTÁNDAR DE LOS VALORES DE LAS FUNCIONES EN CADA PERÍODO OBSERVADO##
TCD.mean = mean.fd(FTCD$fd)
TCD.sd = std.fd(FTCD$fd)
lines(TCD.sd, lwd=3, col="red")
lines(TCD.mean, lty=3, lwd=3)
legend(x = 1, y = -1.5, legend = "Desviación Estándar", col = "red", bty = "n", cex = 0.55, lwd=3, horiz = TRUE)
legend(x = 1, y = -1, legend = "Media", bty = "n", cex = 0.55, lty=3, lwd=3, horiz = TRUE)




###OBTENCIÓN DE LA FUNCIÓN DE LA COVARIANZA###
## LA FUNCIÓN DE LA COVARIANZA ES BIVARIADA##
dom = c(1:62)
covar = var.fd(FTCD$fd)

###GENERANDO LA FUNCIÓN DE COVARIANZA EVALUABLE###
F.covar = eval.bifd(dom, dom, covar)
##VALORANDO ALGUNOS PUNTOS
F.covar[60,60]
F.covar[60,61]
F.covar[62,61]
F.covar[59,61]

## HISTOGRAMA DE FRECUENCIAS DE LAS COVARIANZAS##
hist(F.covar)
#CONOCIENDO LOS VALORES EXTREMOS DE LA COVARIANZA SE
##AJUSTAR MEJOR LOS LÍMITES DEL EJE X PARA EL HISTOGRAMA##
min(F.covar)
max(F.covar)





##GENERANDO UNA TABLA DE FRECUENCIAS PARA LAS COVARIANZAS CALCULADAS##
#H <- hist(F.covar, plot=FALSE);# Lista
#Tabla <- table.freq(H)
#Tabla

##GENERANDO UNA TABLA DE FRECUENCIAS PARA LAS COVARIANZAS CALCULADAS##
H <- hist(F.covar, plot=FALSE)
Tabla <- table.freq(H)
Tabla


hist(F.covar,
     ylab = "Frecuencia",
     xlab = "Covarianzas",
     col = rgb(0, 0, 1, 0.5),   # Azul semitransparente
     xlim = c(-0.1, 1.1), main = "",
     breaks = 10,               # Número de intervalos en el histograma
     freq = TRUE)               # Para mostrar la frecuencia no la densidad





###GRÁFICO DE LA FUNCIÓN DE COVARIANZA###
#LA FUNCIÓN DE COVARIANZA ES BIVARIADA SU GRÁFICA ES UNA SUPERFICIE EN R3#
heat.colors(20)
persp(dom, dom, F.covar, xlab = "", ylab = "", zlab = "Covarianza TCD",
      expand = 0.55, theta = 30, phi = 20,
      border = NA, shade = 0.15, ticktype = 'detailed', col = "#FF8800")



##GRÁFICO DE CONTORNOS##
##CURVAS DE NIVEL Y VALORES DE LAS COVARIANZAS##
gv <- contour(dom, dom, F.covar, lwd = 2, col = "brown")




##GENERANDO LA MATRIZ DE LAS CORRELACIONES##
cor_matrix <- cor.fd(dom, FTCD$fd)
corrplot(cor_matrix, method = "color", tl.cex = 0.55)

