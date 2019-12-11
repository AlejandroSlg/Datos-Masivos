# Datos-Masivos

<details>
<summary>Contenido</summary>

- [Introduccion](#introduccion)
- [Marco teórico](#marco-teorico)
  * [SVM](#support-vector-machine) 
  * [Decision Tree](#decision-tree)
  * [Logistic Regression](#multilayer-perceptron)
- [Implementación](#implementacion)
- [Resultados](#resultados)
- [Conclusiones](#conclusiones)
- [Referencias](#referencias)
  
</details>

---
## Marco teórico

### Support Vector Machine
Es un clasificador discriminativo definido formalmente por un hiperplano de separación. En otras palabras, dados los datos de entrenamiento etiquetados ( aprendizaje supervisado ), el algoritmo genera un hiperplano óptimo que categoriza nuevos ejemplos. En dos espacios dimensionales, este hiperplano es una línea que divide un plano en dos partes donde en cada clase se encuentra a cada lado.

#### ¿Qué hace SVM?

Dado un conjunto de ejemplos de entrenamiento, cada uno marcado como perteneciente a una u otra de dos categorías, un algoritmo de entrenamiento SVM construye un modelo que asigna nuevos ejemplos a una categoría u otra, convirtiéndolo en un clasificador lineal binario no probabilístico.

Support Vector Machine (SVM) es principalmente un método más clásico que realiza tareas de clasificación mediante la construcción de hiperplanos en un espacio multidimensional que separa los casos de diferentes etiquetas de clase. SVM admite tareas de regresión y clasificación y puede manejar múltiples variables continuas y categóricas. Para las variables categóricas, se crea una variable ficticia con valores de casos como 0 o 1. Por lo tanto, una variable dependiente categórica que consta de tres niveles, digamos (A, B, C), está representada por un conjunto de tres variables ficticias: A: {1 0 0}, B: {0 1 0}, C: {0 0 1}

Para construir un hiperplano óptimo, SVM emplea un algoritmo de entrenamiento iterativo, que se utiliza para minimizar una función de error. Según la forma de la función de error, los modelos SVM se pueden clasificar en cuatro grupos distintos:

* Clasificación SVM Tipo 1 (también conocida como clasificación C-SVM)
* Clasificación SVM tipo 2 (también conocida como clasificación nu-SVM)
* Regresión SVM tipo 1 (también conocida como regresión epsilon-SVM)
* Regresión SVM tipo 2 (también conocida como regresión nu-SVM)

#### Parámetros de ajuste: Kernel, Regularización, Gamma y Margen

* Kernel
El aprendizaje del hiperplano en SVM lineal se realiza transformando el problema utilizando algo de álgebra lineal. Aquí es donde el kernel juega un papel.
Para el kernel lineal, la ecuación para la predicción de una nueva entrada utilizando el producto de punto entre la entrada (x) y cada vector de soporte (xi) se calcula de la siguiente manera:
f (x) = B (0) + suma (ai * (x, xi))
Esta es una ecuación que implica calcular los productos internos de un nuevo vector de entrada (x) con todos los vectores de soporte en los datos de entrenamiento. El coeficiente de aprendizaje debe estimar los coeficientes B0 y ai (para cada entrada) a partir de los datos de entrenamiento.

El núcleo polinomial se puede escribir como K (x, xi) = 1 + sum (x * xi) ^ d y exponencial como K (x, xi) = exp (-gamma * sum ((x - xi²)).

* Regularización
El parámetro Regularización (a menudo denominado parámetro C en la biblioteca sklearn de python) le dice a la optimización SVM cuánto desea evitar clasificar erróneamente cada ejemplo de entrenamiento.
Para valores grandes de C, la optimización elegirá un hiperplano de menor margen si ese hiperplano hace un mejor trabajo al clasificar correctamente todos los puntos de entrenamiento. Por el contrario, un valor muy pequeño de C hará que el optimizador busque un hiperplano de separación de mayor margen, incluso si ese hiperplano clasifica erróneamente más puntos.

* Gama
El parámetro gamma define qué tan lejos llega la influencia de un solo ejemplo de entrenamiento, con valores bajos que significan "lejos" y valores altos que significan "cerca". En otras palabras, con gama baja, los puntos alejados de la línea de separación plausible se consideran en el cálculo de la línea de separación. Donde como gama alta significa que los puntos cercanos a la línea plausible se consideran en el cálculo.


---

## Resultados

Iteracion | Decision Tree| Logistic Regression| SVM
------------ | -------------| -------------| -------------
1 | 89.83% | 89.29% | 88.90%
2 | 89.83% | 89.29% | 88.90%
3 | 89.83% | 89.29% | 88.90%
4 | 89.83% | 89.29% | 88.90%
5 | 89.83% | 89.29% | 88.90%
6 | 89.83% | 89.29% | 88.90%
7 | 89.83% | 89.29% | 88.90%
8 | 89.83% | 89.29% | 88.90%
9 | 89.83% | 89.29% | 88.90%
10 | 89.83% | 89.29% | 88.90%
11 | 89.83% | 89.29% | 88.90%
12 | 89.83% | 89.29% | 88.90%
13 | 89.83% | 89.29% | 88.90%
14 | 89.83% | 89.29% | 88.90%
15 | 89.83% | 89.29% | 88.90%
Promedio | 89.83% | 89.29% | 88.90%
