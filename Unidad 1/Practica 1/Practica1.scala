
/*1. Desarrollar un algoritmo en scala que calcule el radio de un circulo*/
val p=20 //Se declara el valor del perímetro
val pi=3.1416 //Se declara el valor de Pi
val r=p/(2*pi) //Despejando la formula para calcular el perímetro de un circulo obtenemos la formula para calcular el radio
println(r) //Imprimimos el valor del radio

/*2. Desarrollar un algoritmo en scala que me diga si un numero es primo*/
def espar(num:Int):Boolean ={return (num%2==0)}
var num = 0
println(espar(num))

/*3. Dada la variable bird = "tweet", utiliza interpolacion de string para mprimir "Estoy ecribiendo un tweet"*/
var bird = "tweet"
println(s"Estoy escribiendo un $bird")

/*4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la secuencia "Luke"*/
val sw="Hola Luke soy tu padre"
sw slice (5,9)

/*5. Cual es la diferencia en value y una variable en scala?*/
/*Value se utiliza para asignar un valor inmutable, en cambio Variable es para que el valor que se asigna pueda cambiar*/

/*6. Dada la tupla ((2,4,5),(1,2,3),(3.1416,23))) regresa el numero 3.1416 */
val Tupla=((2,4,5),(1,2,3),(3.1416,23))
println(Tupla._3._1)