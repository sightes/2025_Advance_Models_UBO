#install.packages("readxl")
#install.packages("fda")
#install.packages("corrplot")
library(readxl)

file_path <-  file.path(getwd(),"Tarea_electivo_profundizacion_1", "data", "data.xlsx")
data <- read_excel(file_path, sheet = "Target",skip=0)
head(data)


png("grafica.png", width = 1000, height = 600)

matplot(data[,2:21], type = "l", pch = 1
        ,xlab = "Periodo", ylab="venta", col = 1:20, cex = 0.7)
#Paislatam1 = c("ARG", "BOL", "BRA", "CHT", "COL")
#Paislatam2 = c("CRI", "CUB", "ECU", "SLV", "GTM")
#Paislatam3 = c("HTI", "HND", "MEX", "NIC", "PAN")
#Paislatam4 = c("PRY", "PER", "DOM", "URY", "VEN")
#Paislatam5 = c("CHI", "COL", "PRY", "PER", "VEN")
#legend(x = -1, y = -1.0, legend = Paislatam1, col = 1:5,bty = "n", cex = 0.7, lwd = 1,horiz = TRUE)
#legend(x = -1, y = -2.0, legend = Paislatam2, col = 6:10,bty = "n", cex = 0.7,lty = 6:10, lwd = 1,horiz = TRUE)
#legend(x = -1, y = -3.0, legend = Paislatam3, col = 11:15,bty = "n", cex = 0.7,lty = 11:15, lwd = 1,horiz = TRUE)
#legend(x = -1, y = -4.0, legend = Paislatam4, col = 16:20,bty = "n", cex = 0.7,lty = 16:20, lwd = 1,horiz = TRUE)
