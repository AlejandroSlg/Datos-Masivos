/* 1. Crea una lista llamada "lista" con los elementos "rojo", "blanco", "negro"*/
var lista = List("rojo", "blanco", "negro")

/* 2. Añadir 5 elementos mas a "lista": "verde" ,"amarillo", "azul", "naranja", "perla"*/
lista = "verde":: "amarillo"::"azul"::"naranja"::"perla" :: lista

/* 3. Traer los elementos de "lista" "verde", "amarillo", "azul"*/
lista slice(0,3)

/* 4. Crea un arreglo de numero en rango del 1-1000 en pasos de 5 en 5*/
import scala.language.postfixOps
val r= 1 to 1000 by 5 toArray

/* 5. Cuales son los elementos unicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversion a conjuntos*/
val unic_list= List(1,3,3,4,6,7,3,7)
unic_list.toSet

/* 6. Crea una mapa mutable llamado nombres que contenga los siguiente "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"*/
val nombres=collection.mutable.Map(("Jose",20),("Luis",24),("Ana",23),("Susana",27))

/* 7 . Imprime todas la llaves del mapa*/
nombres.keys

/* 8 . Agrega el siguiente valor al mapa("Miguel", 23)*/
nombres +=("Miguel"->23)