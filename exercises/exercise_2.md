## Taller de predicción de precios mediante modelos de regresión

Machine Learning, Tensor Flow, Regresión Lineal Simple, Regresión Lineal Múltiple

## Ejercicio 2 - Creación de un una regresión lineal simple mediante algoritmos matemáticos

El objetivo de este ejercicio es construir un modelo de regresión lineal simple mediante la utilización de la regresión matemática. 

**Paso 1: Instación de paquetes**

Los cuadernos (notebooks) son entidades independientes que permiten la utilización de cualquier tipo de páquete python y para ellos nos ofrece la posibilidad de instalar paquete mediante la utilización de la sistema de instalación de paquetes pip. Para la instalación de los diferentes paquetes que utilizaremos para la realización de nuestro paquetes tenemos que ejecutar el siguiente comando:

```
!pip install pandas scikit-learn numpy seaborn matplotlib tensorflow
```

Como podemos observar, es necesario incluir el caracter __!__ antes del comando de instalación. A continuación hay que seleccionar el fragmento y pulsar la tecla play para ejecutar el código contenido en el fragmento. Siendo el resultado de la ejecución de esta linea, el siguiente:

```
Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (1.0.5)
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (0.22.2.post1)
Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (1.18.5)
Requirement already satisfied: seaborn in /usr/local/lib/python3.6/dist-packages (0.10.1)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (3.2.2)
Requirement already satisfied: tensorflow in /usr/local/lib/python3.6/dist-packages (2.2.0)
Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas) (2018.9)
Requirement already satisfied: python-dateutil>=2.6.1 in /usr/local/lib/python3.6/dist-packages (from pandas) (2.8.1)
Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn) (1.4.1)
Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn) (0.15.1)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib) (0.10.0)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib) (2.4.7)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib) (1.2.0)
Requirement already satisfied: tensorboard<2.3.0,>=2.2.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (2.2.2)
Requirement already satisfied: wrapt>=1.11.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.12.1)
Requirement already satisfied: gast==0.3.3 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.3.3)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (3.2.1)
Requirement already satisfied: astunparse==1.6.3 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.6.3)
Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.12.0)
Requirement already satisfied: absl-py>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.9.0)
Requirement already satisfied: protobuf>=3.8.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (3.10.0)
Requirement already satisfied: wheel>=0.26; python_version >= "3" in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.34.2)
Requirement already satisfied: tensorflow-estimator<2.3.0,>=2.2.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (2.2.0)
Requirement already satisfied: google-pasta>=0.1.8 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.2.0)
Requirement already satisfied: h5py<2.11.0,>=2.10.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (2.10.0)
Requirement already satisfied: keras-preprocessing>=1.1.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.1.2)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.1.0)
Requirement already satisfied: grpcio>=1.8.6 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.30.0)
Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (1.0.1)
Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (47.3.1)
Requirement already satisfied: google-auth<2,>=1.6.3 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (1.17.2)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (2.23.0)
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (0.4.1)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (3.2.2)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (1.6.0.post3)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.6/dist-packages (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow) (0.2.8)
Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow) (4.1.0)
Requirement already satisfied: rsa<5,>=3.1.4; python_version >= "3" in /usr/local/lib/python3.6/dist-packages (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow) (4.6)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow) (1.24.3)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow) (2020.6.20)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.3.0,>=2.2.0->tensorflow) (1.3.0)
Requirement already satisfied: importlib-metadata; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from markdown>=2.6.8->tensorboard<2.3.0,>=2.2.0->tensorflow) (1.6.1)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.6/dist-packages (from pyasn1-modules>=0.2.1->google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow) (0.4.8)
Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.3.0,>=2.2.0->tensorflow) (3.1.0)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata; python_version < "3.8"->markdown>=2.6.8->tensorboard<2.3.0,>=2.2.0->tensorflow) (3.1.0)
```

En este caso no se ha realizado la instalación de ningún paquete debido a que todos ya estaban instalados en Collaborate. 

**Paso 2: Despliegue de librerías**

Una vez que se ha realizado la instalación de los diferentes paquetes python, en el caso de que no estuvieran instalados en el sistema, es necesario importar aquellas clases y métodos necesarios para la realización del ejercicio. 

```
import os
import random
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from os import getcwd
from sklearn.model_selection import train_test_split
```

Para el desarrollo de este ejercicio vamos a necesitar un conjunto de liberías que servirán para lo siguiente:

* os: Nos ofrece funciones para la manipulación de recursos del sistema operativo.
* random: Nos ofrece funciones para la generación de números aleatorios.
* zipfile: Nos ofrece funciones para la manipulación de archivos comprimidos.
* pandas: Nos ofrece funciones para la manipulación de los datos.
* numpy: Nos ofrece funciones para la manipulación de arrays y conjunto de datos.
* matplotlib: Nos ofrece funciones para la visualización de datos.
* tensorflow: Nos ofrece funciones para la construacción de procesos de entrenamiento.

**Paso 3: Descarga de datos**

A continuación vamos a realizar la descarga de los datos. Para el desarrollo de este ejercicio vamos a utilizar los datos del dataset sobre [precios de vivienda de Kaggle](https://www.kaggle.com/c/neolen-house-price-prediction). 

En primer lugar deberemos crear un directorio, que denominaremos data, para almacenar y manipular la información que utilizaremos para la realización del ejercicio. Para ello deberemos incluir el siguiente fragmento de código:

```
data_path = 'data'

try:
  os.makedirs(data_path, mode=0o777, exist_ok=False)
except OSError:
    print ("El directorio %s no hay podido ser creado o ya existe." % (data_path))
else:
    print ("El directorio %s ha sido creado correctamente." % (data_path))
```

Una vez hayamos creado el directorio __data__ podemos proceder a descargar el archivo comprimido donde se encuentra toda la información necesaria para el desarrollo de este tutorial mediante el siguiente fragmento de código:

```
!wget --no-check-certificate --content-disposition \
    https://github.com/momartinm/linear_regression_tf/tree/master/data/neolen-house-price-prediction-kaggle.zip \
    -O /content/data/neolen-house-price-prediction-kaggle.zip
```

Al igual que en el paso 1 hemos utilizado un comando para la descarga de los datos, por ese motivo hemos tenido que incluir el caracter especial !.

**Paso 4 - Descompresión y carga de archivos**

Una vez que hemos descargado correctamente nuestro datos, podremos descomprimir el archivo con el objetivo de utilizar los diferentes archivos que contiene. Para ello deberemos utilizar el siguiente fragmento de código:

local_zip = '/content/data/neolen-house-price-prediction-kaggle.zip'

```
try:
  zip_ref = zipfile.ZipFile(local_zip, 'r')
  zip_ref.extractall(data_path)
  zip_ref.close()
except:
    print ("El archivo no ha podido ser descomprimido." % (local_zip))
```

**Paso 5 - Preparación de los conjuntos de datos**

Una vez que hemos descomprimido el archivo que descargamos previamente podremos comenzar a trabajar con los datos. Tras el proceso de descompresión tendremos tres ficheros de datos:

* data_description.txt: Fichero con información referente a cada uno atributos (features) de las instancias. 
* train.csv (entrenamiento/validacion): Conjunto de ejemplos de información para los procesos de entrenamiento y validación.
* test.csv (test): Conjunto de ejemplos de información para los procesos de test.

Para poder trabajar con estos conjuntos de datos vamos a utilizar la librería Pandas que importamos previamente. Mediante esta librería podemos transformar/cargar los datos contenidos en un archivo csv a un DataFrame. Los DataFrame son conjuntos de series de datos sobre los que se pueden realizar ciertos proceso de transformación. Para poder cargar los datos deberemos utilizar el siguiente fragmento de código:

```
file_train = os.path.join(getcwd(), data_path, 'train.csv')
file_test = os.path.join(getcwd(), data_path, 'test.csv')

data_train = pd.read_csv(file_train)
data_test = pd.read_csv(file_test)
```
Una vez que hemos cargado los mediante la función __read_csv__ los podemos inspeccionar mediante las funciones _head_ o __tail__ que nos permiten visualizar un conjunto de elementos del comienzo o el final del fichero respectivamente (por defecto se muestrarán 5 instancias. Además podemos visualizar los tamaños de ambos conjuntos mediante la propiedad __shape__. 

```
print(data_train.head())
print(data_train.shape)
```

Una vez que hemos analizado los datos que tenemos en nuestros conjuntos de entrenamiento y test podemos crear los conjuntos reales que vamos a utilizar. Como estamos trabajando con una regresión lineal simple sólo tendremos un valor en X y un valor en Y. Es decir, sólo tendremos una feature para cada una de nuestras instancias y entrenamiento y una etiqueta. Para este ejemplo vamos a utilizar el número de habitaciones (TotRmsAbvGrd) como feature y el precio de venta como etiqueta (SalePrice). 

```
features_train = data_train['TotRmsAbvGrd']
features_test = data_test['TotRmsAbvGrd']

labels_train = data_train['SalePrice']
```

**Paso 6 - Construcciones de funciones de perdida (loss)**

Una vez que hemos preparado nuestros datos podemos empezar a definir los diferentes elementos del proceso de entrenamiento. El primero que vamos a definir es la función de pérdida (loss) que nos permite evaluar la "calidad" del modelo que estamos construyendo durante el proceso de entrenamiento. En este caso vamos a utilizar el estimador [Mean Square Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) que nos mide la media del cuadrado de los errores. Para ello tendremos que definir la siguiente función:

```
def mean_squared_error_loss( y_pred , y_true ):
    return tf.reduce_mean( tf.square( y_pred - y_true ) )
```

Cómo se puede observar la función ha sido construida mediante la utilización de la funciones que ofrece tensorflow para la realización de operaciones. 

**Paso 7 - Construcción del modelo**

Una vez que hemos definido la función de loss a utilizar, vamos a construir una clase para la manipulación de nuestro modelo. Para ello crearemos una clase en python denominada Model que tendrá tres atributos:

* Pesos (weights) que serán los multiplicadores de las variables independientes. En este caso sólo tendremos una. 
* Pendiente (bias) que será la pendiente de nuestra recta de regresión.
* Variables (vars) que serán los diferentes valores que recogeremos durante el proceso de entrenamiento para evalur visualmente la evolución del proceso. 

```
class Model:
    
    def __init__(self):
      self.weights = tf.Variable( 0. , dtype=tf.float32 )
      self.bias = tf.Variable( 0. , dtype=tf.float32 )
      self.vars = dict()
      
      self.vars['epoch'] = list()
      self.vars['loss'] = list()
      self.vars['loss_val'] = list()
```

Además crearemos una función denominada ____call____ que nos permitirá calcular el valor de y para cada una de las x en base a los valores calculados. 

```
    def __call__(self, x):
      return self.weights * x + self.bias
```

y una función denominada add que nos permitirá añadir una nuevo valor a las diferentes métricas almacenadas en vars indicando el nombre de la métrica y el valor. 

```
    def add(self, variable, value):
      self.vars[variable].append(value)
```

**Paso 8 - Definición de la función de preparación de los datos**

Una vez definidos nuestro modelo generaremos la función que nos permitirá construir conjuntos de datos a partir de una previo. Esta función nos servirá para construir el conjunto de validación o de test a partir de una previo. Para ellos definiremos una función que tomará tres parámetros de entrada:

* Conjunto de caractéristicas (X)
* Conjunto de etiquetas (Y)
* El tamaño del conjunto de menor tamaño (split_size). Este tendrá un valor comprendido entre 0 y 1 y tomará por defecto el valor de 0.2 que se corresponde con el 20%.

Para la construcción de esta función utilizaremos la función que nos ofrece la libreria sklearn, que tiene una función denominada __train_test_split__ que nos permite dividir un conjunto en dos, por lo que la utilizaremos para construir los conjuntos que necesitemos.

```
def data_generation( X, Y, split_size=0.2):
  return train_test_split(X, Y, test_size=split_size)
```

**Paso 9 - Definición de la función de evaluación**

Además de una función de generación de conjuntos necesitaremos una función para la evaluación de nuestro modelo con el objetivo de conocer el error sobre un determinado conjuntos. Para ello creamos una función denominada __evaluate__ que tomará tres parámetros de entrada 

* El modelo que hemos construido o que estamos construyendo (modelo)
* Conjunto de caractéristicas (features)
* Conjunto de etiquetas (features)

```
def evaluate( model, features, labels):
  y_predicted = model( features )
  loss = mean_squared_error_loss( y_predicted , labels )
  return loss
```

**Paso 10 - Definición de la función de entrenamiento**

Una vez que hemos definido todas la funciones y clases necesarias para el funcionamiento de nuestro proceso de entrenamiento, vamos a definir nuestra función de entrenamiento con el objetivo de poder realizar diferentes procesos de entrenamiento con diferentes parámetros. Para ello definiremos una función denominada __fit__ que tomará cuatro parámetros de entrada:

* Conjunto de caractéristicas de entrenamiento (features_train)
* Conjunto de etiquetas de entrenamiento (labels_train)
* Número de epocas del proceso de entrenamiento (num_epochs)
* Tasa de aprendizaje del proceso de entrenaminto (Learning_rate)

En primer lugar definiremos los elementos básicos para el proceso de entrenamiento, que se corresponden con los datos referentes al número de ejemplos, al número de características de nuestras instancias, el model que vamos a aprender y los conjuntos de entrenamiento y validación.  

```
def fit( features_train, labels_train, num_epochs, learning_rate=0.001):

  num_samples = features_train.shape[0]
  num_features = 1 if len(features_train.shape) == 1 else features_train.shape[1]

  model = Model()

  X_train, X_validate, Y_train, Y_validate = data_generation(features_train, labels_train, 0.1)

```

A continuación tenemos que definir el bucle de entrenamiento que se ejecutará durante un conjunto de iteraciones (épocas) que hemos definido como parámetro previamente en la función. 

```
  for epoch in range( num_epochs ):
      
```

Para intentar minimizar el coste del error de nuestro modelo vamos a utilizar el [algoritmo de descenso del Gradiente] para ello deberemos incluir la siguiente linea que nos permitirá calcular los gradientes de los dos parámetros que tiene que calcular nuestro algoritmo. 

```
      with tf.GradientTape() as tape:

        y_predicted = model( X_train ) 
        current_loss = mean_squared_error_loss( y_predicted , Y_train )
```

En cada iteración calcularemos el valor predecido por el modelo actual y calcularemos el error con el objetivo de ir minimizándolo en cada iteración

```

        gradients = tape.gradient(current_loss, [model.weights, model.bias])
        model.weights.assign_sub(gradients[0]*learning_rate)
        model.bias.assign_sub(gradients[1]*learning_rate)

```

Al final de cada iteración calcularemos el valor de error (loss) sobre el conjunto de validación con el objetivo de comprobar como va evolucionando nuestro modelo. 

```
      validation_loss = np.array(evaluate(model, X_validate, Y_validate))

      model.add('loss', current_loss)
      model.add('loss_val', validation_loss)
      loss = np.array( model.vars['loss'] ).mean()
      model.add('epoch', epoch +1 )

      print( 'Epoch ' + str(epoch+1) + '/' + str(num_epochs) )
      print( 'loss: ' + str(loss) + ' - val_loss: ' + str(validation_loss) )
```

Una vez finalizada el proceso de entrenamiento devolveremos nuestro modelo 

```
  return model
```

**Paso 11 - Entrenamiento de modelos**

Una vez construidas nuestras funciones podemos ejecutar nuestro proceso de aprendizaje de la siguiente manera, ejecutando el proceso de aprendizaje durante 100 iteraciones con una tasa de aprendizaje del 0.001. Si queremos podemos construir diferentes modelos y ver el resultado de cada uno de ellos. 

```
model = fit(features_train, labels_train, 100)
```

En este caso yo he construido un modelo tras 100 iteraciones. 

**Paso 12 - Visualización de la evolución del loss**

Una vez finalizado el proceso de entrenamiento podemos visualizar la evolusión de nuestros valores de loss sobre el conjunto de entrenamiento y validación mediante el siguiente fragmento de código:

```
plt.plot( model.vars['epoch'] , model.vars['loss'] ) 
plt.plot( model.vars['epoch'] , model.vars['loss_val'] ) 
plt.legend(['epoch', 'loss'])
plt.show()
```

Este fragmento de código nos mostrará la evolución de los valores de loss para el conjunto de entrenamiento y el de validación. 

**Paso 13 - Visualización de la recta de regresión para los conjuntos de entrenamiento y test**

Para finalizar podremos comprobar el resultado de nuestra recta de regresión en dos dimensiones mediante el siguiente fragmento de código 

```
plt.scatter(features_train, labels_train, label="true")
plt.scatter(features_train, model(features_train), label="predicted")
plt.legend(['true', 'predicted'])
plt.show()
```

**Congratulations Ninja!**

Has aprendido como construir un modelo de regresión mediante Machine Learning utilizando TensorFlow 2.0. Has conseguido aprender:

1. Como instalar paquetes y desplegar/importar librerías. 
2. Como definir los conjuntos de entrenamiento y test.
3. Como crear un función de loss (pérdida).  
4. Como crear una variable de tipo Variable en TensorFlow. 
5. Como crear un modelo que puedar ser reutilizado en el futuro. 
5. Como crear conjunto de validación.
7. Como construir el bucle de entrenamiento. 
8. Como realizar una predicción. 

<img src="../img/ejercicio_2_congrats.png" alt="Congrats ejercicio 2" width="800"/>
