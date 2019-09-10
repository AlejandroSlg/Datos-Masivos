/*Muestras de datis correspondientes a dos de las temporadas de Maria*/
val muestra0 = List(10,5,20,20,4, 5, 2, 25, 1)
val muestra1 = List(3,4,21,36,10,28,35,5,24,42)

/*Definición de la función*/
def breakingRecords(score:List[Int]): Unit =//Se define la variable de la funcion y el tipo de dato del parametro de entrada, la cual es en esta caso una lista
{
    var min = score(0) //*"min" tomara el primer elemento de la lista a analizar
    var max = score(0) //*"max" tomara el primer elemento de la lista a analizar
    var cont_min = 0
    var cont_max = 0

    /*Declaración de ciclo for para comparar los scores de manera iterativa*/
     for (i <- score) //Para cada valor de i de la variable "Num" de la funcion "breakingRecords"
    {
        if (i<min)//Si el valor de "i" es menor a "min"
        {
            min = i //se establece el valor de "min" igual al que tiene "i"
            cont_min = cont_min +1 //se suma 1 al contador minimo
        }
        else if (i>max) //Si el valor de "i" es mayor a "max"
        {
            max = i //se establece el valor de "max" igual al que tiene "i"
            cont_max = cont_max + 1 //se suma 1 al contador maximo
        }
    }
    println("Resultados:")
    println ("Maximo:"+cont_max, " Minimo:"+cont_min)
}


/*Saldia de la muestra 0: 2 4*/
breakingRecords(muestra0)

/*Salida de la muestra 1: 4 0*/
breakingRecords(muestra1)
