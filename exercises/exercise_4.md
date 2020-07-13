## Taller de predicción de precios mediante modelos de regresión

Machine Learning, Tensor Flow, Regresión Lineal Simple, Regresión Lineal Múltiple

## Ejercicio 4 - Creación de un una regresión lineal múltiple mediante redes de neuronas

El objetivo de este ejercicio es construir un modelo de regresión lineal mediante la utilización de una red de neuronas. 

**Paso 1: Instación de paquetes**

Los cuadernos (notebooks) son entidades independientes que permiten la utilización de cualquier tipo de páquete python y para ellos nos ofrece la posibilidad de instalar paquete mediante la utilización de la sistema de instalación de paquetes pip. Para la instalación de los diferentes paquetes que utilizaremos para la realización de nuestro paquetes tenemos que ejecutar el siguiente comando:

```
!pip install pandas scikit-learn numpy seaborn matplotlib tensorflow h5py keras
```

Como podemos observar, es necesario incluir el caracter __!__ antes del comando de instalación. A continuación hay que seleccionar el fragmento y pulsar la tecla play para ejecutar el código contenido en el fragmento. Siendo el resultado de la ejecución de esta linea, el siguiente:

```
Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (1.0.5)
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (0.22.2.post1)
Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (1.18.5)
Requirement already satisfied: seaborn in /usr/local/lib/python3.6/dist-packages (0.10.1)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (3.2.2)
Requirement already satisfied: tensorflow in /usr/local/lib/python3.6/dist-packages (2.2.0)
Requirement already satisfied: h5py in /usr/local/lib/python3.6/dist-packages (2.10.0)
Requirement already satisfied: keras in /usr/local/lib/python3.6/dist-packages (2.3.1)
Requirement already satisfied: python-dateutil>=2.6.1 in /usr/local/lib/python3.6/dist-packages (from pandas) (2.8.1)
Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas) (2018.9)
Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn) (0.15.1)
Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn) (1.4.1)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib) (2.4.7)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib) (0.10.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib) (1.2.0)
Requirement already satisfied: absl-py>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.9.0)
Requirement already satisfied: wrapt>=1.11.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.12.1)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.1.0)
Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.12.0)
Requirement already satisfied: wheel>=0.26; python_version >= "3" in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.34.2)
Requirement already satisfied: google-pasta>=0.1.8 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.2.0)
Requirement already satisfied: tensorboard<2.3.0,>=2.2.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (2.2.2)
Requirement already satisfied: protobuf>=3.8.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (3.10.0)
Requirement already satisfied: gast==0.3.3 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (0.3.3)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (3.2.1)
Requirement already satisfied: astunparse==1.6.3 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.6.3)
Requirement already satisfied: keras-preprocessing>=1.1.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.1.2)
Requirement already satisfied: tensorflow-estimator<2.3.0,>=2.2.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (2.2.0)
Requirement already satisfied: grpcio>=1.8.6 in /usr/local/lib/python3.6/dist-packages (from tensorflow) (1.30.0)
Requirement already satisfied: keras-applications>=1.0.6 in /usr/local/lib/python3.6/dist-packages (from keras) (1.0.8)
Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from keras) (3.13)
Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (47.3.1)
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (0.4.1)
Requirement already satisfied: google-auth<2,>=1.6.3 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (1.17.2)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (1.6.0.post3)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (3.2.2)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (2.23.0)
Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.6/dist-packages (from tensorboard<2.3.0,>=2.2.0->tensorflow) (1.0.1)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.3.0,>=2.2.0->tensorflow) (1.3.0)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.6/dist-packages (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow) (0.2.8)
Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow) (4.1.0)
Requirement already satisfied: rsa<5,>=3.1.4; python_version >= "3" in /usr/local/lib/python3.6/dist-packages (from google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow) (4.6)
Requirement already satisfied: importlib-metadata; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from markdown>=2.6.8->tensorboard<2.3.0,>=2.2.0->tensorflow) (1.6.1)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow) (1.24.3)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3,>=2.21.0->tensorboard<2.3.0,>=2.2.0->tensorflow) (2020.6.20)
Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.3.0,>=2.2.0->tensorflow) (3.1.0)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.6/dist-packages (from pyasn1-modules>=0.2.1->google-auth<2,>=1.6.3->tensorboard<2.3.0,>=2.2.0->tensorflow) (0.4.8)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata; python_version < "3.8"->markdown>=2.6.8->tensorboard<2.3.0,>=2.2.0->tensorflow) (3.1.0)
```

En este caso no se ha realizado la instalación de ningún paquete debido a que todos ya estaban instalados en Collaborate. 

Además vamos a incluir un comando que permite cargar la extensión de TensorFlow Board dentro de los cuadernos de tipo Jupyter, de forma que se despligue de manera embebida en el entorno. 

```
%load_ext tensorboard
```

**Paso 2: Despliegue de librerías**

Para la realización de este ejercicio tenemos que importar nuevas librerías relacionadas con keras. Para ello es necesario modificar los paquetes importados que vamos a utilizar con respecto al ejercicio anterior. 

```
import os
import random
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt

from os import getcwd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from tensorflow import keras
from time import time

from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras import optimizers
from keras.utils import plot_model
from keras.models import model_from_json

```

Para el desarrollo de los diferentes ejercicios vamos a necesitar un conjunto de liberías que servirán para lo siguiente:

* numpy: Nos ofrece funciones para la manipulación de arrays y conjunto de datos. 
* matplotlib: Nos ofrece funciones para la visualización de datos. 
* tensorflow: Nos ofrece funciones para la construacción de procesos de entrenamiento.
* random: Nos ofrece funciones para la generación de números aleatorios.
* zipfile: Nos ofrece funciones para la manipulación de archivos comprimidos.
* pandas: Nos ofrece funciones para la manipulación de los datos.en o
* os: Nos ofrece funciones para la manipulación de recursos del sistema operativo. 
* os.path: Nos ofrece funciones para la manipulación del sistema de ficheros del sistema operativo.
* requests: Nos ofrece funciones para la descarga de archivos.
* math: Nos ofrece funciones para la realización de operaciones matemáticas complejos (no elementales).
* time: Nos ofrece funciones para la obtención de information referente al tiempo, para crear contadores o archivos de log. 
* Keras.model: Nos permite utilizar diferentes tipo de modelos, en este caso vamos a utilizar el modelo secuencial. 
* Keras.layers: Nos permite utilizar diferentes tipo de capas para incluir en una red de neuronas.
* optimizers from keras: Nos permite utilizar diferentes tipos de algoritmos de optimización, en nuestro caso utilizaremos el optimizador de Adams. 
* Keras.utils: Nos ofrece diferentes funciones para obtener información de la red construida. 
* TensorBoard: Nos ofrece diferentes funciones para cargar información en tensorborad y poder visualizar la evoluación de nuestros algoritmos. 


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
!wget --no-check-certificate \
    https://github.com/momartinm/linear_regression_tf/raw/master/data/neolen-house-price-prediction-kaggle.zip \
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

Una vez que hemos analizado los datos que tenemos en nuestros conjuntos de entrenamiento y test podemos crear los conjuntos reales que vamos a utilizar. Como estamos trabajando con una regresión lineal simple sólo tendremos un valor en X y un valor en Y. Es decir, sólo tendremos una feature para cada una de nuestras instancias y entrenamiento y una etiqueta. Para este ejemplo vamos a utilizar más campos (número de habitaciones, el año de venta y el mes de venta) como features y el precio de venta como etiqueta (SalePrice). 

```
features_train = data_train[['TotRmsAbvGrd', 'MoSold', 'YrSold', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr']]

features_train['TotRmsAbvGrd'] = pd.to_numeric(features_train['TotRmsAbvGrd'], downcast='float')
features_train['MoSold'] = pd.to_numeric(features_train['MoSold'], downcast='float')
features_train['YrSold'] = pd.to_numeric(features_train['YrSold'], downcast='float')
features_train['OverallQual'] = pd.to_numeric(features_test['OverallQual'], downcast='float')
features_train['OverallCond'] = pd.to_numeric(features_test['OverallCond'], downcast='float')
features_train['YearBuilt'] = pd.to_numeric(features_test['YearBuilt'], downcast='float')
features_train['YearRemodAdd'] = pd.to_numeric(features_test['YearRemodAdd'], downcast='float')
features_train['1stFlrSF'] = pd.to_numeric(features_test['1stFlrSF'], downcast='float')
features_train['2ndFlrSF'] = pd.to_numeric(features_test['2ndFlrSF'], downcast='float')
features_train['FullBath'] = pd.to_numeric(features_test['FullBath'], downcast='float')
features_train['HalfBath'] = pd.to_numeric(features_test['HalfBath'], downcast='float')
features_train['BedroomAbvGr'] = pd.to_numeric(features_test['BedroomAbvGr'], downcast='float')
features_train['KitchenAbvGr'] = pd.to_numeric(features_test['KitchenAbvGr'], downcast='float')

labels_train = data_train['SalePrice']

features_test = data_test[['TotRmsAbvGrd', 'MoSold', 'YrSold', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr']]

features_test['TotRmsAbvGrd'] = pd.to_numeric(features_test['TotRmsAbvGrd'], downcast='float')
features_test['MoSold'] = pd.to_numeric(features_test['MoSold'], downcast='float')
features_test['YrSold'] = pd.to_numeric(features_test['YrSold'], downcast='float')
features_test['OverallQual'] = pd.to_numeric(features_test['OverallQual'], downcast='float')
features_test['OverallCond'] = pd.to_numeric(features_test['OverallCond'], downcast='float')
features_test['YearBuilt'] = pd.to_numeric(features_test['YearBuilt'], downcast='float')
features_test['YearRemodAdd'] = pd.to_numeric(features_test['YearRemodAdd'], downcast='float')
features_test['1stFlrSF'] = pd.to_numeric(features_test['1stFlrSF'], downcast='float')
features_test['2ndFlrSF'] = pd.to_numeric(features_test['2ndFlrSF'], downcast='float')
features_test['FullBath'] = pd.to_numeric(features_test['FullBath'], downcast='float')
features_test['HalfBath'] = pd.to_numeric(features_test['HalfBath'], downcast='float')
features_test['BedroomAbvGr'] = pd.to_numeric(features_test['BedroomAbvGr'], downcast='float')
features_test['KitchenAbvGr'] = pd.to_numeric(features_test['KitchenAbvGr'], downcast='float')

```

**Paso 6. Generación de la red**

Una vez definadas la variables de entrada y salida con su formato (shape) podemos construir nuestra red de neuronas que estará compuesta de tres 5 capas: 

- Capa Fully Connected (Dense): Es la capa básica de una red de neuronas convencionales donde cada neurona de la capa está conectada con todas las neuronas de la capa anterior, de este modelo de conexión proviene su nombre __fully connected__. En este caso creamos una capa completamente connectada con 13 neuronas que se corresponde con los 13 parámetros de entrada que hemos seleccionado. 
- Capa Fully Connected (Dense): Es la capa básica de una red de neuronas convencionales donde cada neurona de la capa está conectada con todas las neuronas de la capa anterior, de este modelo de conexión proviene su nombre __fully connected__. En este caso creamos una capa completamente connectada con 9 neuronas.
- Capa Fully Connected (Dense): Es la capa básica de una red de neuronas convencionales donde cada neurona de la capa está conectada con todas las neuronas de la capa anterior, de este modelo de conexión proviene su nombre __fully connected__. En este caso creamos una capa completamente connectada con 9 neuronas.
- Capa Fully Connected (Dense): Es la capa básica de una red de neuronas convencionales donde cada neurona de la capa está conectada con todas las neuronas de la capa anterior, de este modelo de conexión proviene su nombre __fully connected__. En este caso creamos una capa completamente connectada con 9 neuronas.
- Capa Fully Connected de salida (Dense): Es la capa básica de una red de neuronas convencionales donde cada neurona de la capa está conectada con todas las neuronas de la capa anterior, de este modelo de conexión proviene su nombre __fully connected__. En esta caso creamos una capa de tipo densa donde el número de neuronas de salida será 1, que se corresponderá con el valor numérico que queremos definir.

```

net = Sequential(name='Linear Regresion multiple')
net.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
net.add(Dense(39, activation='relu'))
net.add(Dense(117, activation='relu'))
net.add(Dense(39, activation='relu'))
net.add(Dense(1, kernel_initializer='normal'))
```

**Paso 7. Definición de función de optimización**

A continuación tenemos que definir la función de obtimización que utilizaremos para minimizar el valor de función de coste. Para este ejecicio vamos a utilizar el algoritmo de [Adam](https://arxiv.org/abs/1412.6980https://arxiv.org/abs/1412.6980) con el fin de minimizar el coste del error mediante la función __optimizers.Adam__. 

```
optimizer_fn = optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
```

**Paso 8. Compilación de la red**

A continuación debemos compilar nuestra red utilizando un algoritmo de optimización, una función de loss, que en este caso utilizaremos la función que calcula el error cuadrático medio y por último definimos la metrica que utilizaremos para el proceso de entrenamiento que será el __accuracy__. 

```
net.compile(optimizer=optimizer_fn, loss='mean_squared_error', metrics=['accuracy'])
net.summary()
```

Además una vez compilada la red, utilizaremos la función __summary__ que nos presenta un resumen de la estructura de la red que hemos creado (capas, orden de las capas, variables a entrenar, etc). 

**Paso 9. Definición de bucle de entrenamiento (Función)**

Una vez que se han definido todas las variables y funciones necesarias para el proceso de aprendizaje, podemos crear la función de entrenamiento. En este caso la función es muy sencilla y formada por tres parámetros:

- net: Que se corresponde con la red secuencial que hemos definido previamente.
- training_iters: Que se corresponde con el número de iteraciones del proceso de entrenamiento.
- batch_size: Que se corresponde con el tamaño de los conjuntos de entrenamiento que se utilizarán. 
- validation_split: Que se corresponde con el tamaño del cojunto de validación. Es decir, el conjunto de entrada (x_shaped_array) se divirá en dos conjunto: (1) el conjunto de entrenamiento que contendrá el 90% de los ejemplos; y (2) el conjunto de validación que contendrá el 10% de los ejemplos. El primero ser utilizada para cada iteración de entrenamiento, mientras que el segundo será utilizado para validar el modelo después de cada iteración. 

Esta función realiza una reestructuración de los datos de los conjuntos de entrenamiento y test para ajustarlos al formato y tamaño de las imágenes que hemos definido en caso de que existe alguna discrepancia y ejecuta el proceso de entrenamiento mediante la utilización del método __fit__ que ejecuta un proceso similar al que definimos en el ejercicio anterior. Además en este caso incluimos un __callback__ con el objeto de recolectar información que nos permita visualizar la evolución del proceso de entrenamiento mediante TensorBoard. 

```
logdir = "./logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def train(net, training_iters, batch_size = 10, validation_split=0.1):
    
    tensorboard_callback = TensorBoard(log_dir=logdir)
    
    net.fit(
        features_train, 
        labels_train,
        batch_size=batch_size,
        epochs=training_iters,
        validation_split=validation_split,
        callbacks=[tensorboard_callback]
        )
    
    return net
```

Además tras el proceso de entrenamiento podemos calcular el resultado de nuestro modelo sobre el conjunto de test con el objetivo de conocer la capacidad real de nuestro modelo. Para ello utilizaremos la función __evaluate__ que como como parámetros obligatorios son el conjunto de características (x_test_shaped_array) y el de etiquetas (test_y). Además vamos a incluir el mismo tamaño de bacth que utilizamos en el proceso de entrenamiento. 

**Paso 10. Ejecución del proceso de entrenamiento**

Una vez construidas nuestras funciones podemos ejecutar nuestro proceso de aprendizaje de la siguiente manera, ejecutando el proceso de aprendizaje durante 100 iteraciones con una tasa de aprendizaje del 0.001 y un tamaño de batch de 128 imágenes. 

```
model = train(net, 10, 128)
```

**Paso 11. Visualización de los resultados con TensorFlowBoard**

Es posible visualizar la información mediante TensorFlow Board con el objetivo de poder obtener toda la información sobre el proceso de aprendizaje. Para ello es necesario incluir el siguiente comando y ejercutar el fragmento del cuarderno. TensorBoard utilizar los ficheros de logs que se han generado en el fichero que indiquemos como valor del parámetro __logdir__, que en este caso se corresponde con la carpeta logs que hemos utilizado para almacenzar los logs generados en el proceso de entrenamiento del paso 10. 

```
%tensorboard --logdir logs
```

Tras la ejecución podremos ver a través del interfaz web, embevida en nuestro cuaderno, el resultado de nuestro proceso de aprendizaje, como se muestra en la siguiente imagen:

<img src="../img/tensorboard_1.png" alt="Resultado de un proceso de aprendizaje mediante TensorBoard" width="800"/>

Si ejecutamos este comando antes del proceso de aprendizaje podremos ver en tiempo real la evolución del proceso, ya que TensorBoard tiene un sistema de refresco de 30 segundos. 

**Paso 12: Almacenamiento de nuestro modelo**

Una vez que hemos construido nuestro modelo, podemos almacenarlo con dos objetivos: (1) utilizar para realizar inferencia sobre nuevos datos; y (2) cargarlo para seguir aprendiendo en el futuro con un nuevo conjunto de datos. Para ello es necesario almacenar la información del modelo mediante dos ficheros:

* Fichero de tipo json que almacena la estructura de la red que hemos construido.
* Fichero de tipo h5 que almacena la información de los pesos de las neuronas de nuestro red. 

Para poder generaro estos dos ficheros debemos utilizar el siguiente fragmento de código:

```

model_folder = "models"

try:
    os.mkdir(model_folder)
except OSError:
    print ("El directorio %s no hay podido ser creado" % (model_path))
else:
    print ("El directorio %s ha sido creado correctamente" % (model_path))


model_path = './models/'
model_name = 'model'

model_json = model.to_json()

with open(model_path + model_name + '.json', "w") as json_file:
    json_file.write(model_json)

model.save_weights(model_path + model_name + ".h5")
```

Tras la ejecución de este fragmento de código habremos generado nuestro los dos ficheros que describen la estructua de nuestra red de neuronas y los valos de los pesos de las diferentes capas. 

**Paso 13 - Visualización de la recta de regresión para los conjuntos de entrenamiento y test**

Una vez que hemos almacenado nuestro modelo, podemos cargarlo con el objetivo de poder ejecutar el proceso de inferencia en la aplicación en la cual queremos desplegar el modelo. Para ello tendremos que cargar el modelo que hemos almacenado previamente mediante el siguiente fragmento de código.

```
json_file = open(model_path + model_name + '.json', 'r')

loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_path + model_name + '.h5')
```

Una vez que hemos cargado el modelo podemos realizar la inferencia sobre el modelo mediante la función __predict__ que nos permite predecir el valor de salida mediante un valor de entrada. Para comprobar el funcionamiento del modelo de predicción vamos a evaluarlo mediante el siguiente código:

```
predicted_labels = net.predict(features_test).flatten()

plt.scatter(features_train, labels_train, label="true")
plt.scatter(features_train, predicted_labels, label="predicted")
plt.legend(['true', 'predicted'])
plt.show()
```

**Congratulations Ninja!**

Has aprendido como construir un modelo de regresión mediante Machine Learning utilizando TensorFlow 2.0 y Keras. Has conseguido aprender:

1. Como desplegar TensorBoard en un entorno Notebook.
2. Como definir los conjuntos de entrenamiento y test.
3. Como crear una variable de tipo Variable en TensorFlow. 
4. Como construir una red mediante Keras.
5. Como definir la función de optimización. 
6. Como construir el bucle de entrenamiento. 
7. Como visualizar datos referentes al proceso de entrenamiento mediante TensorBoard. 
8. Como guardar un modelo para poder utilizarlo en el futuro.
9. Como cargar un modelo previamente guardado y realizar una predicción. 

<img src="../img/ejercicio_4_congrats.png" alt="Congrats ejercicio 4" width="800"/>
