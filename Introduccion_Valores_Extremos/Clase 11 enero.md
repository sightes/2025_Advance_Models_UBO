Como yo encuentro un modelo estadistico que me permita predecir estos datos...

per a  veces tenemos datos  ue se encuentran fuera de rango 

la que representa mas datos atipicos es la de pareto....


GEV ,v
GPD , generalized pareto distribution


Metods de 3stimacion ,maxima verosimilitud momentos u estimado de hill


enfoaue de modeloado block maxima y peak over trhjeshold 


aplicaciones en series temporales , en r o python 



para las tareas es necesario , hacer un analisis previo 



que es un valor extremo ???

-- es un valor outlier, se encuentra significativamente alejado el rango tanto superior como inferior , ubicadad en las colas de la distribucion y representaa eventos raros o poco probables ....
 



 que da origen a los outliers ??
 -- errores en los datoas
 -- variabilidad natural 
 -- Fenomenos inusuales
 -- datos faltantes  




 consecuencias de los outliers 

 sesgo en las medidas estadisticas d
 impacto de los modelos predictivops 
 etc ( falta info )}



 metodos datos a tipicos ..
 Zscore -- > en el rango u +- sigma tendemos el 68% de los datos 
        -- > en el rango u +- 2 sigma tendemos el 95% de los datos 
        -- > en el rango u +- 3 sigma tendemos el 99.7% de los datos 


que pasa si los datos no distrbuyen normal ?? , los rangos interquarttiles permiten calcular los outliers en aquellos casos donde la distribucion no es normal ....

Q1 - >  25% ok , los datos estan por debajo de este valor 



 
 
Metodos  manejo valores extremos 
- Eliminar valores extremos 
- transforma los datos 
- analiza por separado 
- asigna limites  (capping)
- metodo de eliminacion (trimming)
- metodo de recorte ( capping )
- probar con y sin valores extremos 


Metodos muktivariadfos 
- distacia de mahalanobis
- analisis de componentes principales PCA
- regresion robusta mahalanobis 
- local outlier factor 

visualizacion
- grafico de dispersion en 2D o 3D 
- Biplot del PCA 
- Grafico de mahalanovis
- Grafico de caras



Mahalanobis ers ua medida que nos ayuda a identificar valores extems y multibariados , considera varias dvarianles al mismo tiempos para medir
