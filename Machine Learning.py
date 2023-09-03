#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# ===================================================================
# Ampliación de Inteligencia Artificial, 2022-23
# PARTE I del trabajo práctico: Implementación de regresión logística
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: Martín Rodríguez
# NOMBRE: José David 
#
# Segundo(a) componente (si se trata de un grupo):
#
# APELLIDOS: Pérez Ortega
# NOMBRE: Pablo Antonio
# ----------------------------------------------------------------------------


# ****************************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen. La discusión 
# y el intercambio de información de carácter general con los compañeros se permite, 
# pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED o cualquier otro medio, se considerará plagio. En particular no se 
# permiten implementaciones obtenidas con HERRAMIENTAS DE GENERACIÓN AUTOMÁTICA DE CÓDIGO. 
# Si tienen dificultades para realizar el ejercicio, consulten con el profesor. 
# En caso de detectarse plagio (previamente con aplicaciones anti-plagio o durante 
# la defensa, si no se demuestra la autoría mediante explicaciones convincentes), 
# supondrá una CALIFICACIÓN DE CERO en la asignatura, para todos los alumnos involucrados. 
# Sin perjuicio de las medidas disciplinarias que se pudieran tomar. 
# *****************************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO SE PERMITE usar Scikit Learn (excepto las funciones que
#   se usan en carga_datos.py). 

# * SE RECOMIENDA y SE VALORA especialmente usar numpy. Las implementaciones 
#   saldrán mucho más cortas y eficientes, y se puntuarÁn mejor.   

import numpy as np
from carga_datos import X_credito, y_credito, X_iris, y_iris, X_votos, y_votos, X_cancer, y_cancer, X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aia.tgz y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn, por lo que para que la carga se haga sin
# problemas, deberá estar instalado el módulo sklearn). Todos los datos se
# cargan en arrays de numpy:

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.
  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 



# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# >>> Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#  (array([0, 1]), array([168, 267]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 
    
# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------
def particion_entr_prueba(X,y,test=0.20):
    
    diccionario = {}
    # Obtener los diferentes valores de y
    valores_unicos = np.unique(y)
    
    # Iterar sobre los valores únicos
    for valor in valores_unicos:
        # Obtener los índices donde y coincide con el valor actual de 'valor'
        indices = np.where(y == valor)[0]
        # Asignar los índices correspondientes en el diccionario
        diccionario[valor] = indices
    
    # Creamos listas para almacenar el tanto porciento de cada clase
    listTrain = []
    listTest = []
    for k,v in diccionario.items(): #Para cada clase dek dicc recorremos sus índices
        trainArray,testArray = divRandomEP(v,test) #Se eligen valores aleatorios
        listTrain.extend(trainArray)
        listTest.extend(testArray)
    
    #Esto lo hacemos para que los 0s y 1s se mezclen
    train_permutado = np.random.permutation(listTrain)
    test_permutado = np.random.permutation(listTest)
    
    # Se guardan los valores
    X_train = np.array(X)[train_permutado]
    X_test = np.array(X)[test_permutado]
    y_train = np.array(y)[train_permutado]
    y_test = np.array(y)[test_permutado]
    
    return X_train,X_test,y_train,y_test

    
def divRandomEP(array,test=0.20): 
    # Para cada array del diccionario calculamos el número de elementos teniendo en cuenta el porcentaje test
    divArray=int(np.round(len(array)*test))
    # Se reordenan los elementos del array
    array_permutado = np.random.permutation(array)
    # Se divide utilizando slicing dado el porcentaje.
    trainArray = array_permutado[divArray:]
    testArray = array_permutado[:divArray]

    return trainArray, testArray


def test_ej1():
    print("\nEJERCICIO 1:")
    
    
    print("\n Para el conjunto de votos:")
    Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)
    
    print("\n Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:")
    print(y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0])
    
    print("\n Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos:")
    print("  - " + str(np.unique(y_votos,return_counts=True)))
    print("  - " + str(np.unique(ye_votos,return_counts=True)))
    print("  - " + str(np.unique(yp_votos,return_counts=True)))
    
    
    
    print("\n\n Para el conjunto de cancer:\n")
        
    Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
    print("  - " + str(np.unique(y_cancer,return_counts=True)))
    print("  - " + str(np.unique(yev_cancer,return_counts=True)))
    print("  - " + str(np.unique(yp_cancer,return_counts=True)))
    
    print("\n Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en datos para validación.\n")
    Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
    print("  - " + str(np.unique(ye_cancer,return_counts=True)))
    print("  - " + str(np.unique(yv_cancer,return_counts=True)))
    
    
    
    print("\n\n Para el conjunto de credito:\n")
    Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)
    
    print("  - " + str(np.unique(y_credito,return_counts=True)))
    print("  - " + str(np.unique(ye_credito,return_counts=True)))
    print("  - " + str(np.unique(yp_credito,return_counts=True)))


# ------------------------------------------------------------------
# ===========================
# EJERCICIO 2: NORMALIZADORES
# ===========================

# En esta sección vamos a definir dos maneras de normalizar los datos. De manera 
# similar a como está diseñado en scikit-learn, definiremos un normalizador mediante
# una clase con un metodo "ajusta" (fit) y otro método "normaliza" (transform).


# ---------------------------
# 2.1) Normalizador standard
# ---------------------------

# Definir la siguiente clase que implemente la normalización "standard", es 
# decir aquella que traslada y escala cada característica para que tenga
# media 0 y desviación típica 1. 

# En particular, definir la clase: 


# class NormalizadorStandard():

#    def __init__(self):

#         .....
        
#     def ajusta(self,X):

#         .....        

#     def normaliza(self,X):

#         ......

# 


# donde el método ajusta calcula las corresondientes medias y desviaciones típicas
# de las características de X necesarias para la normalización, y el método 
# normaliza devuelve el correspondiente conjunto de datos normalizados. 

# Si se llama al método de normalización antes de ajustar el normalizador, se
# debe devolver (con raise) una excepción:

class NormalizadorNoAjustado(Exception):
    def __init__(self, mensaje):
        self.mensaje = mensaje
        super().__init__(self.mensaje)


# Por ejemplo:
    
    
# >>> normst_cancer=NormalizadorStandard()
# >>> normst_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, la media y desviación típica de Xe_cancer_n deben ser 
# 0 y 1, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_n, 
# ni con Xp_cancer_n. 

# ------ 

class NormalizadorStandard():

    def __init__(self):
        self.medias=None
        self.desviacion=None
        
    def ajusta(self,X):
        #Calculo la media y la desviacion estandar de cada característica(columna) de los datos.
        self.medias = np.mean(X,axis=0)
        self.desviacion = np.std(X,axis=0)
        
    def normaliza(self,X):
        if self.medias is None or self.desviacion is None:
            raise NormalizadorNoAjustado("El normalizador no ha sido ajustado.")
        return (X - self.medias) / self.desviacion


# ------------------------
# 2.2) Normalizador MinMax
# ------------------------

# Hay otro tipo de normalizador, que consiste en asegurarse de que todas las
# características se desplazan y se escalan de manera que cada valor queda entre 0 y 1. 
# Es lo que se conoce como escalado MinMax

# Se pide definir la clase NormalizadorMinMax, de manera similar al normalizador 
# del apartado anterior, pero ahora implementando el escalado MinMax.

# Ejemplo:

# >>> normminmax_cancer=NormalizadorMinMax()
# >>> normminmax_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_m=normminmax_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_m=normminmax_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_m=normminmax_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, los máximos y mínimos de las columnas de Xe_cancer_m
#  deben ser 1 y 0, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_m,
# ni con Xp_cancer_m. 


# ------ 

class NormalizadorMinMax():

    def __init__(self):
        self.min=None
        self.max=None
        
    def ajusta(self,X):
        #Calculo el valor mínimo y máximo de cada característica(columna) de los datos.
        self.min = np.min(X,axis=0)
        self.max = np.max(X,axis=0)
        
    def normaliza(self,X):
        if self.min is None or self.max is None:
            raise NormalizadorNoAjustado("El normalizador no ha sido ajustado.")
        return (X - self.min) / (self.max - self.min)


def test_ej2():
    print("\nEJERCICIO 2:")
    
    Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
    Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
    
    print("\nNormalizador Standard:\n")
    normst_cancer=NormalizadorStandard()
    normst_cancer.ajusta(Xe_cancer)
    Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
    Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
    Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)
    
    print("Media: " + str(np.mean(Xe_cancer_n)))
    print("Desviación Estandar: " + str(np.std(Xe_cancer_n)))
        
    print("\nNormalizador Standard:\n")
    normminmax_cancer=NormalizadorMinMax()
    normminmax_cancer.ajusta(Xe_cancer)
    Xe_cancer_m=normminmax_cancer.normaliza(Xe_cancer)
    Xv_cancer_m=normminmax_cancer.normaliza(Xv_cancer)
    Xp_cancer_m=normminmax_cancer.normaliza(Xp_cancer)
    
    print("Máximos: " + str(np.max(Xe_cancer_m)))
    print("Mínimos: " + str(np.min(Xe_cancer_m)))    
    
# ------ 

# ===========================================
# EJERCICIO 3: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# En este ejercicio se propone la implementación de un clasificador lineal 
# binario basado regresión logística (mini-batch), con algoritmo de entrenamiento 
# de descenso por el gradiente mini-batch (para minimizar la entropía cruzada).


# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
#                 batch_tam=64):

#         .....
        
#     def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
#                     early_stopping=False,paciencia=3):

#         .....        

#     def clasifica_prob(self,ejemplos):

#         ......
    
#     def clasifica(self,ejemplo):
                        
#          ......



# * El constructor tiene los siguientes argumentos de entrada:



#   + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#     durante todo el aprendizaje. Si rate_decay es True, rate es la
#     tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#   + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#     cada epoch. En concreto, si rate_decay es True, la tasa de
#     aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#     con la siguiente fórmula: 
#        rate_n= (rate_0)*(1/(1+n)) 
#     donde n es el número de epoch, y rate_0 es la cantidad introducida
#     en el parámetro rate anterior. Su valor por defecto es False. 
#  
#   + batch_tam: tamaño de minibatch


# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente. Las dos clases del problema 
#        son las que aparecen en el array y, y se deben almacenar en un atributo 
#        self.clases en una lista. La clase que se considera positiva es la que 
#        aparece en segundo lugar en esa lista.
#     
#     + Otros dos arrays Xv,yv, con los datos del conjunto de  validación, que se 
#       usarán en el caso de activar el parámetro early_stopping. Si son None (valor 
#       por defecto), se supone que en el caso de que early_stopping se active, se 
#       consideraría que Xv e yv son resp. X e y.

#     + n_epochs es el número máximo de epochs en el entrenamiento. 

#     + salida_epoch (False por defecto). Si es True, al inicio y durante el 
#       entrenamiento, cada epoch se imprime  el valor de la entropía cruzada 
#       del modelo respecto del conjunto de entrenamiento, y su rendimiento 
#       (proporción de aciertos). Igualmente para el conjunto de validación, si lo
#       hubiera. Esta opción puede ser útil para comprobar 
#       si el entrenamiento  efectivamente está haciendo descender la entropía
#       cruzada del modelo (recordemos que el objetivo del entrenamiento es 
#       encontrar los pesos que minimizan la entropía cruzada), y está haciendo 
#       subir el rendimiento.
# 
#     + early_stopping (booleano, False por defecto) y paciencia (entero, 3 por defecto).
#       Si early_stopping es True, dejará de entrenar cuando lleve un número de
#       epochs igual a paciencia sin disminuir la menor entropía conseguida hasta el momento
#       en el conjunto de validación 
#       NOTA: esto se suele hacer con mecanismo de  "callback" para recuperar el mejor modelo, 
#             pero por simplificar implementaremos esta versión más sencilla.  
#        



# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Un método clasifica_prob, que recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY con las probabilidades que el modelo 
#   asigna a cada ejemplo de pertenecer a la clase positiva.       
    

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception):
    def __init__(self, mensaje):
        self.mensaje = mensaje
        super().__init__(self.mensaje)
  

# RECOMENDACIONES: 


# + IMPORTANTE: Siempre que se pueda, tratar de evitar bucles for para recorrer 
#   los datos, usando en su lugar funciones de numpy. La diferencia en eficiencia
#   es muy grande. 

# + Téngase en cuenta que el cálculo de la entropía cruzada no es necesario
#   para el entrenamiento, aunque si salida_epoch o early_stopping es True,
#   entonces si es necesario su cálculo. Tenerlo en cuenta para no calcularla
#   cuando no sea necesario.     

# * Definir la función sigmoide usando la función expit de scipy.special, 
#   para evitar "warnings" por "overflow":

#   from scipy.special import expit    
#
#   def sigmoide(x):
#      return expit(x)

# * Usar np.where para definir la entropía cruzada. 

# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama (los resultados pueden variar):


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)

# >>> lr_cancer.clasifica(Xp_cancer_n[24:27])
# array([0, 1, 0])   # Predicción para los ejemplos 24,25 y 26 

# >>> yp_cancer[24:27]
# array([0, 1, 0])   # La predicción anterior coincide con los valores esperado para esos ejemplos

# >>> lr_cancer.clasifica_prob(Xp_cancer_n[24:27])
# array([7.44297196e-17, 9.99999477e-01, 1.98547117e-18])



# Para calcular el rendimiento de un clasificador sobre un conjunto de ejemplos, usar la 
# siguiente función:
    
def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/y.shape[0]

# Por ejemplo, los rendimientos sobre los datos (normalizados) del cáncer:
    
# >>> rendimiento(lr_cancer,Xe_cancer_n,ye_cancer)
# 0.9824561403508771

# >>> rendimiento(lr_cancer,Xp_cancer_n,yp_cancer)
# 0.9734513274336283




# Ejemplo con salida_epoch y early_stopping:

# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

# Inicialmente, en entrenamiento EC: 155.686323940485, rendimiento: 0.873972602739726.
# Inicialmente, en validación    EC: 43.38533009881579, rendimiento: 0.8461538461538461.
# Epoch 1, en entrenamiento EC: 32.7750241863029, rendimiento: 0.9753424657534246.
#          en validación    EC: 8.4952918658522,  rendimiento: 0.978021978021978.
# Epoch 2, en entrenamiento EC: 28.0583715052223, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.665719133490596, rendimiento: 0.967032967032967.
# Epoch 3, en entrenamiento EC: 26.857182744289368, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.09511082759361, rendimiento: 0.978021978021978.
# Epoch 4, en entrenamiento EC: 26.120803184993328, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.327991940213478, rendimiento: 0.967032967032967.
# Epoch 5, en entrenamiento EC: 25.66005010760342, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.376171724729662, rendimiento: 0.967032967032967.
# Epoch 6, en entrenamiento EC: 25.329200890122557, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.408704771704937, rendimiento: 0.967032967032967.
# PARADA TEMPRANA

# Nótese que para en el epoch 6 ya que desde la entropía cruzada obtenida en el epoch 3 
# sobre el conjunto de validación, ésta no se ha mejorado. 


# -----------------------------------------------------------------

from scipy.special import expit    

def sigmoide(x):
    return expit(x)

def entropia_cruzada(X,y,pesos):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    matriz_pred = np.dot(X,pesos)
    y_pred = sigmoide(matriz_pred)
    return np.sum(np.where(y == 1, -np.log(y_pred), -np.log(1 - y_pred)))

class RegresionLogisticaMiniBatch():

    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.batch_tam = batch_tam
        self.pesos = None
        self.clases = None
        self.entrenado = False

    def entrena(self,X,y,Xv=None,yv=None,n_epochs=100, salida_epoch=False, early_stopping=False,paciencia=3):
        self.entrenado = True
        self.clases = np.unique(y)
        
        num_datos, num_caracteristicas = X.shape
        
        # Inicializamos los pesos:
        self.pesos = np.random.uniform(-1, 1, (num_caracteristicas + 1))
        
        if early_stopping and (Xv is None or yv is None):
            Xv = X
            yv = y
        
        if salida_epoch:
            print(f"Inicialmente, en entrenamiento EC: {entropia_cruzada(X, y, self.pesos):.15f}, rendimiento: {rendimiento(self,X,y):.15f}.")
            print(f"Inicialmente, en validación  EC: {entropia_cruzada(Xv, yv, self.pesos):.15f}, rendimiento: {rendimiento(self,Xv,yv):.15f}.")
        
        best_entropy = np.inf
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            # Mezclamos aleatoriamente los datos
            indices = np.random.permutation(num_datos)
            X = X[indices]
            y = y[indices] 
            
            #Si rate_decay es True se irá decrementando la tasa de aprendizaje cada epoch
            if self.rate_decay:
                rate_epoch = (self.rate)*(1/(1+epoch))
            else:
                rate_epoch = self.rate
            
            for i in range(0, num_datos, self.batch_tam):
                
                # Obtenemos el mini-batch actual
                X_batch = X[i:i+self.batch_tam]
                y_batch = y[i:i+self.batch_tam]
                
                #Agregamos una columnas de 1s al comienzo para el bias
                X_batch = np.hstack((np.ones((X_batch.shape[0], 1)), X_batch))
                
                # Calculamos y_pred con la multiplicación de matrices de los datos por los pesos. Y aplicarle sigmoide.
                matriz_pred = np.dot(X_batch,self.pesos)
                y_pred = sigmoide(matriz_pred)
                
                # Calculamos el gradiente
                error = y_pred - y_batch
                gradiente = np.dot(X_batch.T, error)

                # Actualizamos los parámetros
                self.pesos -= rate_epoch * gradiente

            
            if salida_epoch:
                entropia_train = entropia_cruzada(X, y,self.pesos)
                rendimiento_train = rendimiento(self,X,y)
                print(f"Epoch {epoch+1}: en entrenamiento EC: {entropia_train:.15f}, rendimiento: {rendimiento_train:.15f}.")
                if early_stopping:
                    entropia_val = entropia_cruzada(Xv, yv,self.pesos)
                    rendimiento_val = rendimiento(self,Xv,yv)
                    print(f"         en validación    EC: {entropia_val:.15f}, rendimiento: {rendimiento_val:.15f}.")
                    
            
            if early_stopping:
                entropia_val = entropia_cruzada(Xv, yv, self.pesos)
                if entropia_val < best_entropy:
                    best_entropy = entropia_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= paciencia:
                    print("PARADA TEMPRANA")
                    break


    def clasifica_prob(self, ejemplos):
        if not self.entrenado:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado.")
        ejemplos = np.hstack((np.ones((ejemplos.shape[0], 1)), ejemplos))
        matriz_pred = np.dot(ejemplos,self.pesos)
        prob = sigmoide(matriz_pred)
        return prob

    def clasifica(self, ejemplos):
        if not self.entrenado:
           raise ClasificadorNoEntrenado("El modelo no ha sido entrenado.")
        probs = self.clasifica_prob(ejemplos)
        clases_pred = np.where(probs >= 0.5, self.clases[1], self.clases[0])
        return clases_pred


def test_ej3():

    print("\nEJERCICIO 3: \n")
    
    Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
    Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
    
    normst_cancer=NormalizadorStandard()
    normst_cancer.ajusta(Xe_cancer)
    Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
    Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
    Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)
    
    lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
    
    lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)
    
    print("  - Predicción obtenida para los ejemplos 24,25 y 26: " + str(lr_cancer.clasifica(Xp_cancer_n[24:27])))
    print("  - Predicción real: " + str(yp_cancer[24:27]))
    print("  - Probabilidades para dichos ejemplos: " + str(lr_cancer.clasifica_prob(Xp_cancer_n[24:27])))
    
    print("\n  - Rendimiento para el conjunto de entreno: " + str(rendimiento(lr_cancer,Xe_cancer_n,ye_cancer)))
    print("  - Rendimiento para el conjunto de test: " + str(rendimiento(lr_cancer,Xp_cancer_n,yp_cancer)))
    
    print("\n\n Ejemplo con salida_epoch y early_stopping: \n")
    lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
    lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)


# ------------------------------------------------------------------------------



# =================================================
# EJERCICIO 4: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================



# Este ejercicio puede servir para el ajuste de parámetros en los ejercicios posteriores, 
# pero si no se realiza, se podrían ajustar siguiendo el método "holdout" 
# implementado en el ejercicio 1


# Definir una función: 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,Xv=None,yv=None,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador (como por ejemplo 
# la clase RegresionLogisticaMiniBatch). El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)  

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cancer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad, 
# no tiene por qué coincidir el resultado):

# >>> rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":True},
#                                 Xe_cancer_n,ye_cancer,n=5)

# Partición: 1. Rendimiento:0.9863013698630136
# Partición: 2. Rendimiento:0.958904109589041
# Partición: 3. Rendimiento:0.9863013698630136
# Partición: 4. Rendimiento:0.9726027397260274
# Partición: 5. Rendimiento:0.9315068493150684
# >>> 0.9671232876712328




# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones DEBEN SER ALEATORIAS Y ESTRATIFICADAS. 
 
# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> lr16=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
# >>> lr16.entrena(Xe_cancer_n,ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(lr16,Xp_cancer_n,yp_cancer)
# 0.9646017699115044

#------------------------------------------------------------------------------
def generar_particiones_estratificadas(y, n):
    particiones = []
    clases = np.unique(y)

    # Para cada clase se crean n arrays de indices
    for clase in clases:
        indices_clase = np.where(y == clase)[0]
        np.random.shuffle(indices_clase) # Aleatoridad

        # Dividir en n parte los indices de los datos de cada clase
        particiones_clase = np.array_split(indices_clase, n)
        particiones.extend(particiones_clase)

    # Nos quedamos con n arrays uniendo las particiones de las clases
    resultado=[]
    # Teniendo en cuenta el numero de particiones y numeros de clases
    # cogemos el array i-ésimo de cada clase y los concatenamos
    for i in range(n):
        particion=[]
        for j in range(len(clases)):
            particion.extend(particiones[(j*n)+i])
        np.random.shuffle(particion) # Mezclar 0s y 1s
        resultado.append(np.array(particion))
        
    return resultado



def rendimiento_validacion_cruzada(clase_clasificador, params, X, y, Xv=None, yv=None, n=5):
    rendimientos = []
    particiones = generar_particiones_estratificadas(y, n)

    # Para cada particion vamos a seleccionar un conjunto de validacion
    for i in range(n):
        # Conjunto de indices de validación con la particion i
        indices_validacion = np.array(particiones[i])
    
        # Conjunto de indices de entrenamiento con las particiones restantes
        indices_entreno = np.concatenate(np.delete(particiones, i))
    
        # Se divide en entrenamiento y validacion
        X_train, y_train = X[indices_entreno], y[indices_entreno]
        X_val, y_val = X[indices_validacion], y[indices_validacion]

        # Creamos una instancia del clasificador con los parámetros dados
        clasificador = clase_clasificador(**params)
        
        # Entrenamos el clasificador con los datos de entrenamiento
        clasificador.entrena(X_train, y_train)

        # Calculamos el rendimiento de cada particion
        rd = rendimiento(clasificador, X_val, y_val)
        print(f'Partición: {i+1}. Rendimiento:{rd}')
        rendimientos.append(rd)
   
    return np.mean(rendimientos)  

    
def test_ej4():
    print("\nEJERCICIO 4: \n")
    
    Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
    Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
    
    normst_cancer=NormalizadorStandard()
    normst_cancer.ajusta(Xe_cancer)
    Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
    Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
    Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)
    
    print(rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                                    {"batch_tam":16,"rate":0.01,"rate_decay":True},
                                     Xe_cancer_n,ye_cancer,n=5))
    
    print("\n Si decidimos que es es un buen rendimiento finalmente entrenaríamos con el conjunto de entrenamiento completo:")
    lr16=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
    lr16.entrena(Xe_cancer_n,ye_cancer)
    print("  - Rendimiento obtenido: " + str(rendimiento(lr16,Xp_cancer_n,yp_cancer)))
#------------------------------------------------------------------------------


# ===================================================
# EJERCICIO 5: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================



# Usando la regresión logística implementada en el ejercicio 3, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros (tasa, rate_decay, batch_tam) para mejorar el rendimiento 
# (no es necesario ser muy exhaustivo, tan solo probar algunas combinaciones). 
# Si se ha hecho el ejercicio 4, usar validación cruzada para el ajuste 
# (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos
# sobre un conjunto de prueba.     

# Mostrar también, para cada conjunto de datos, un ejemplo con salida_epoch, 
# en el que se vea cómo desciende la entropía cruzada y aumenta el 
# rendimiento durante un entrenamiento.     

# ------------------------------------------------------------------------------------

def test_ej5():
    
    print("\nEJERCICIO 5: \n")
    
    print("\n Para el conjunto de VOTOS: \n")
    y_votos_binario = np.where(y_votos == 'republicano', 0, 1)
    Xev_votos,Xp_votos,yev_votos,yp_votos=particion_entr_prueba(X_votos,y_votos_binario,test=0.2)
    Xe_votos,Xv_votos,ye_votos,yv_votos=particion_entr_prueba(Xev_votos,yev_votos,test=0.2)
    
    normst_votos=NormalizadorStandard()
    normst_votos.ajusta(Xe_votos)
    Xe_votos_n=normst_votos.normaliza(Xe_votos)
    Xv_votos_n=normst_votos.normaliza(Xv_votos)
    Xp_votos_n=normst_votos.normaliza(Xp_votos)
    
    lr_votos=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
    lr_votos.entrena(Xe_votos_n,ye_votos,Xv_votos_n,yv_votos,salida_epoch=True,early_stopping=True)
    
    print("  - Rendimiento a priori: " + str(rendimiento(lr_votos,Xp_votos_n,yp_votos)))
    
    print("\n Se han probado diferentes pesos del mini batch (8,10,12,16,32,64), de tasa de aprendizaje se ha probado con (0.1, 0.01, 0.01) y poniendo el rate_decay tanto True como False. Y la mejor combinación en la que se consigue un pequeño aumento ha sido la siguente: ")
    print("\n Estos han sido los parametros elegidos: batch_tam:10, rate:0.1, rate_decay:True \n")
    print("  - Rendimiento de validacion cruzada: " + str(rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,{"batch_tam":10,"rate":0.1,"rate_decay":True}, Xe_votos_n,ye_votos,n=5)))
    
    print("\n Entonces ahora vamos a entrenar: ")
    lr16_votos=RegresionLogisticaMiniBatch(batch_tam=10,rate=0.1,rate_decay=True)
    lr16_votos.entrena(Xe_votos_n,ye_votos)
    print("  - Rendimiento obtenido: " + str(rendimiento(lr16_votos,Xp_votos_n,yp_votos)))
    
    
    
    print("\n\n\n Para el conjunto de CANCER: \n")
    Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
    Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
    
    normst_cancer=NormalizadorStandard()
    normst_cancer.ajusta(Xe_cancer)
    Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
    Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
    Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)
    
    lr_cancer_2=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
    lr_cancer_2.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)
    
    
    print("  - Rendimiento a priori: " + str(rendimiento(lr_cancer_2,Xp_cancer_n,yp_cancer)))
    
    print("\n Se han probado diferentes pesos del mini batch (8,10,12,16,32,64), de tasa de aprendizaje se ha probado con (0.1, 0.01, 0.01) y poniendo el rate_decay tanto True como False. Y la mejor combinación en la que se consigue un pequeño aumento ha sido la siguente: ")
    print("\n Estos han sido los parametros elegidos: batch_tam:16, rate:0.1, rate_decay:True \n")
    print("  - Rendimiento de validacion cruzada: " + str(rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,{"batch_tam":16,"rate":0.1,"rate_decay":True}, Xe_cancer_n,ye_cancer,n=5)))
    
    print("\n Entonces ahora vamos a entrenar: ")
    lr16_cancer_2=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.1,rate_decay=True)
    lr16_cancer_2.entrena(Xe_cancer_n,ye_cancer)
    print("  - Rendimiento obtenido: " + str(rendimiento(lr16_cancer_2,Xp_cancer_n,yp_cancer)))
    
    
    
    print("\n\n\n Para el conjunto de CRÍTICAS PELICULAS IMDB: \n")
    Xe_imdb,Xv_imdb,ye_imdb,yv_imdb=particion_entr_prueba(X_train_imdb,y_train_imdb,test=0.2)
    
    normst_criticas=NormalizadorStandard()
    normst_criticas.ajusta(Xe_imdb)
    Xe_imdb_n=normst_criticas.normaliza(Xe_imdb)
    Xv_imdb_n=normst_criticas.normaliza(Xv_imdb)
    X_test_imdb_n=normst_criticas.normaliza(X_test_imdb)
    
    lr_criticas=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
    lr_criticas.entrena(Xe_imdb_n,ye_imdb,Xv_imdb_n,yv_imdb,salida_epoch=True,early_stopping=True)
    
    
    print("  - Rendimiento a priori: " + str(rendimiento(lr_criticas,X_test_imdb_n,y_test_imdb)))
    
    print("\n Se han probado diferentes pesos del mini batch (8,10,12,16,32,64), de tasa de aprendizaje se ha probado con (0.1, 0.01, 0.01) y poniendo el rate_decay tanto True como False. Y la mejor combinación en la que se consigue un pequeño aumento ha sido la siguente: ")
    print("\n Estos han sido los parametros elegidos: batch_tam:32, rate:0.1, rate_decay:True \n")
    print("  - Rendimiento de validacion cruzada: " + str(rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,{"batch_tam":32,"rate":0.1,"rate_decay":True}, Xe_imdb_n,ye_imdb,n=5)))
    
    print("\n Entonces ahora vamos a entrenar: ")
    lr16_criticas=RegresionLogisticaMiniBatch(batch_tam=32,rate=0.1,rate_decay=True)
    lr16_criticas.entrena(Xe_imdb_n,ye_imdb)
    print("  - Rendimiento obtenido: " + str(rendimiento(lr16_criticas,X_test_imdb_n,y_test_imdb)))
    
# ------------------------------------------------------------------------------------

# =====================================================
# EJERCICIO 6: CLASIFICACIÓN MULTICLASE CON ONE vs REST
# =====================================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica One vs Rest. 


#  Para ello, implementar una clase  RL_OvR con la siguiente estructura, y que 
#  implemente un clasificador OvR (one versus rest) usando como base el
#  clasificador binario RegresionLogisticaMiniBatch


# class RL_OvR():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica(self,ejemplos):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, aunque ahora referido a cada uno de los k entrenamientos a 
#  realizar (donde k es el número de clases).
#  Por simplificar, supondremos que no hay conjunto de validación ni parada
#  temprana.  

 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=8)

# >>> rl_iris_ovr.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris_ovr,Xe_iris,ye_iris)
# 0.8333333333333334

# >>> rendimiento(rl_iris_ovr,Xp_iris,yp_iris)
# >>> 0.9
# --------------------------------------------------------------------
        
class RL_OvR():
    def __init__(self, rate=0.1, rate_decay=False, batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.clasificadores = {}  # Diccionario para los clasificadores utilizados
        self.clases = None

    def entrena(self, X, y, n_epochs=100, salida_epoch=False):
        self.clases = list(np.unique(y))

        for c in self.clases:
            y_binary = np.where(y == c, 1, 0)  # La clase que se mira 1, el resto 0
            clasificador = RegresionLogisticaMiniBatch(rate=self.rate, rate_decay=self.rate_decay,
                                                     n_epochs=n_epochs, batch_tam=self.batch_tam)
            
            clasificador.entrena(X, y_binary, n_epochs=n_epochs, salida_epoch=salida_epoch)
            self.clasificadores[c] = clasificador

    def clasifica(self, ejemplos):
        # Matriz para las probabilidades de cada clase
        num_ejemplos = ejemplos.shape[0]
        num_clases = len(self.clases)
        probs = np.zeros((num_ejemplos, num_clases))

        # Llamamos a clasifica_prob para que nos de las probabilidades de
        # todos los datos por columna (clase que se esta mirando)
        for i, c in enumerate(self.clases):
            clasificador = self.clasificadores[c]
            # Actualizamos la matriz de 0s con la clasificacion obtenida por columna
            probs[:, i] = clasificador.clasifica_prob(ejemplos)

        # Sacamos el argumento maximo de cada fila
        clases_pred = np.argmax(probs, axis=1)
        # Cambiamos los datos binarios por los nombres de las clases
        clases_pred = [self.clases[pred] for pred in clases_pred]

        return clases_pred

def test_ej6():
    print("\nEJERCICIO 6: \n")
    
    Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)
    
    rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=8)
    rl_iris_ovr.entrena(Xe_iris,ye_iris)
    
    print("  - Rendimiento conjunto de entreno: " + str(rendimiento(rl_iris_ovr,Xe_iris,ye_iris)))
    print("  - Rendimiento conjunto de test: " + str(rendimiento(rl_iris_ovr,Xp_iris,yp_iris)))

# --------------------------------------------------------------------


# =================================
# EJERCICIO 7: CODIFICACIÓN ONE-HOT
# =================================


# Los conjuntos de datos en los que algunos atributos son categóricos (es decir,
# sus posibles valores no son numéricos, o aunque sean numéricos no hay una 
# relación natural de orden entre los valores) no se pueden usar directamente
# con los modelos de regresión logística, o con redes neuronales, por ejemplo.

# En ese caso es usual transformar previamente los datos usando la llamada
# "codificación one-hot". Básicamente, cada columna se reemplaza por k columnas
# en los que los valores psoibles son 0 o 1, y donde k es el número de posibles 
# valores del atributo. El valor i-ésimo del atributo se convierte en k valores
# (0 ...0 1 0 ...0 ) donde todas las posiciones son cero excepto la i-ésima.  

# Por ejemplo, si un atributo tiene tres posibles valores "a", "b" y "c", ese atributo 
# se reemplazaría por tres atributos binarios, con la siguiente codificación:
# "a" --> (1 0 0)
# "b" --> (0 1 0)
# "c" --> (0 0 1)    

# Definir una función:    
    
#     codifica_one_hot(X) 

# que recibe un conjunto de datos X (array de numpy) y devuelve un array de numpy
# resultante de aplicar la codificación one-hot a X.Por simplificar supondremos 
# que el array de entrada tiene todos sus atributos categóricos, y que por tanto 
# hay que codificarlos todos.

# Aplicar la función para obtener una codificación one-hot de los datos sobre
# concesión de prestamo bancario.     
  
# >>> Xc=np.array([["a",1,"c","x"],
#                  ["b",2,"c","y"],
#                  ["c",1,"d","x"],
#                  ["a",2,"d","z"],
#                  ["c",1,"e","y"],
#                  ["c",2,"f","y"]])
   
# >>> codifica_one_hot(Xc)
# 
# array([[1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
#        [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.],
#        [0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
#        [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.]])

# En este ejemplo, cada columna del conjuto de datos original se transforma en:
#   * Columna 0 ---> Columnas 0,1,2
#   * Columna 1 ---> Columnas 3,4
#   * Columna 2 ---> Columnas 5,6,7,8
#   * Columna 3 ---> Columnas 9, 10,11     


# ------------------------------------------------------------------------

def codifica_one_hot(X):
    # Diccionario para los valores unicos de cada columna
    valores_unicos = {}

    # Para cada columna cogemos los valores unicos
    for i in range(X.shape[1]):
        valores_unicos[i] = np.unique(X[:, i])

    # Para almacenar las columnas
    matriz = []

    # Para cada columna de X le añadimos tantas columnas como categorias
    for i in range(X.shape[1]):
        matriz_columna = np.zeros((X.shape[0], len(valores_unicos[i])))

        #Para cada valor unico buscamos donde deben de ir los 1s
        for j, valor in enumerate(valores_unicos[i]):
            matriz_columna[:, j] = np.where(X[:, i] == valor, 1, 0)
    
        matriz.append(matriz_columna)

    # Concatenar todas las columnas en un solo array
    res = np.concatenate(matriz, axis=1)

    return res

def test_ej7():
    print("\nEJERCICIO 7: \n")
    Xc=np.array([["a",1,"c","x"],
                 ["b",2,"c","y"],
                 ["c",1,"d","x"],
                 ["a",2,"d","z"],
                 ["c",1,"e","y"],
                 ["c",2,"f","y"]])
       
    print(codifica_one_hot(Xc))

# ------------------------------------------------------------------------

# =====================================================
# EJERCICIO 8: APLICACIONES DEL CLASIFICADOR MULTICLASE
# =====================================================


# ---------------------------------------------------------
# 8.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación OvR Y one-hot de los ejercicios anteriores,
# para obtener un clasificador que aconseje la concesión, 
# estudio o no concesión de un préstamo, basado en los datos X_credito, y_credito. 

# Ajustar adecuadamente los parámetros (nuevamente, no es necesario ser demasiado 
# exhaustivo)

# ----------------------
def test_ej8_1():
    print("\nEJERCICIO 8.1: ")
    
    Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito)
    
    Xe_credito_oh = codifica_one_hot(Xe_credito)
    Xp_credito_oh = codifica_one_hot(Xp_credito)
    
    rl_credito_ovr=RL_OvR(rate=0.001,batch_tam=8)
    rl_credito_ovr.entrena(Xe_credito_oh,ye_credito)
    
    print("\n Rendimientos a priori: ")
    print("  - Rendimiento conjunto de entreno: " + str(rendimiento(rl_credito_ovr,Xe_credito_oh,ye_credito)))
    print("  - Rendimiento conjunto de test: " + str(rendimiento(rl_credito_ovr,Xp_credito_oh,yp_credito)))
    
    print("\n Se ha probado la tasa de aprendizaje (0.1, 0.01, 0,001) y tamaño de batch valores como (4,8,16,32) y el mejor valor obtenido ha sido: ")
    
    rl_credito_ovr_2=RL_OvR(rate=0.01,batch_tam=16)
    rl_credito_ovr_2.entrena(Xe_credito_oh,ye_credito)
    
    print("\n Rendimientos tras pruebas: ")
    print("  - Rendimiento conjunto de entreno: " + str(rendimiento(rl_credito_ovr_2,Xe_credito_oh,ye_credito)))
    print("  - Rendimiento conjunto de test: " + str(rendimiento(rl_credito_ovr_2,Xp_credito_oh,yp_credito)))


# ---------------------------------------------------------
# 8.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación OvR anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 


# --------------------------------------------------------------------------


def cargar_imagenes(ruta_imagenes):
    with open(ruta_imagenes, "r") as file:
        lines = file.readlines()

    num_imagenes = len(lines) // 28  # Cada imagen tiene 28 filas
    imagenes = np.zeros((num_imagenes, 784))  # 28x28 = 784 píxeles

    # Para cada imagen
    for i in range(num_imagenes):
        # Para cada "linea" de la imagen
        for j in range(28):
            line = lines[i * 28 + j]
            # Para cada pixel de la linea de la imagen
            for k in range(28):
                if line[k] == " " :
                    imagenes[i, j * 28 + k] = 0  # Pixel blanco
                elif line[k] == "#" or line[k] == "+":
                    imagenes[i, j * 28 + k] = 1  # Pixel negro

    return imagenes

def procesar_etiquetas(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    labels = [int(line.strip()) for line in lines]
    return np.array(labels)


# Cargar los datos de entrenamiento
Xe_digit = cargar_imagenes('datos/trainingimages')
ye_digit = procesar_etiquetas('datos/traininglabels')

# Cargar los datos de validación
Xv_digit = cargar_imagenes('datos/validationimages')
yv_digit = procesar_etiquetas('datos/validationlabels')

# Cargar los datos de test
Xt_digit = cargar_imagenes('datos/testimages')
yt_digit = procesar_etiquetas('datos/testlabels')

def test_ej8_2():
    print('\nEJERCICIO 8.2:')
    
    rl_digit_ovr=RL_OvR(rate=0.001,batch_tam=8)
    
    rl_digit_ovr.entrena(Xe_digit,ye_digit)
    
    print('  - Rendimiento conjunto de entreno: ' + str(rendimiento(rl_digit_ovr,Xe_digit,ye_digit)))
    
    print('  - Rendimiento conjunto de test: ' + str(rendimiento(rl_digit_ovr,Xt_digit,yt_digit)))


# =========================================================================
# EJERCICIO OPCIONAL PARA SUBIR NOTA: 
#    CLASIFICACIÓN MULTICLASE CON REGRESIÓN LOGÍSTICA MULTINOMIAL
# =========================================================================


#  Se pide implementar un clasificador para regresión
#  multinomial logística con softmax (VERSIÓN MINIBATCH), descrito en las 
#  diapositivas 55 a 57 del tema de "Complementos de Aprendizaje Automático". 

# class RL_Multinomial():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica_prob(self,ejemplos):

#        ......
 

#     def clasifica(self,ejemplos):

#        ......
   

 
# Los parámetros tiene el mismo significado que en el ejercicio 7 de OvR. 

# En eset caso, tiene sentido definir un clasifica_prob, ya que la función
# softmax nos va a devolver una distribución de probabilidad de pertenecia 
# a las distintas clases. 


# NOTA 1: De nuevo, es muy importante para la eficiencia usar numpy para evitar
#         el uso de bucles for convencionales.  

# NOTA 2: Se recomienda usar la función softmax de scipy.special: 

    # from scipy.special import softmax   
#

    
# --------------------------------------------------------------------

# Ejemplo:

# >>> rl_iris_m=RL_Multinomial(rate=0.001,batch_tam=8)

# >>> rl_iris_m.entrena(Xe_iris,ye_iris,n_epochs=50)

# >>> rendimiento(rl_iris_m,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris_m,Xp_iris,yp_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------

# --------------- 
















