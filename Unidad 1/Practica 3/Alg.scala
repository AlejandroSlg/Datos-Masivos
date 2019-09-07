/*Sucesión de Fibonacci*/

/*Primer Algoritmo*/
def fib(n:Int):Int = if(n<2) 1 else fib(n-1)+fib(n-2)
println(fib(8))
res1: Int = 34

/*Segundo Algoritmo*/
def fib2(n:Double): Double ={
     |     if(n<2){
     |         return n
     |     }
     |     else {
     |         var m = ((1+math.sqrt(5))/2)
     |         var j = ((math.pow(m,n)-math.pow((1-m),n))/math.sqrt(5))
     |         return j
     |     }
     | }

/*Tercer Algoritmo*/
def fib3(x : Int):Int={
     |     var a = 0
     |     var b = 1
     |     var c =0
     |     var k = 0
     |     for(k <- 1 to x){
     |         println(a)
     |         c=b+a
     |         a=b
     |         b=c            
     |     }
     |     return(x)
     | }

/*Cuarto Algoritmo*/
def fib4(x : Int):Int={
     |     var a = 0
     |     var b = 1
     |     var k = 0
     |     for(k <- 1 to x){
     |         println(a)
     |         b=b+a
     |         a=b-a        
     |     }
     |     return(x)
     | }

/*Quinto Algoritmo*/
def fib5(n:Int): Double = {
     | if( n < 2) {
     | return n
     | }
     | else {
     | var v = new Array[Double](n + 1)
     | v(0) = 0
     | v(1) = 1
     | for(k <- Range(2,n + 1)){
     | v(k) = v(k-1) + v(k-2)
     | }
     | return v(n)
     | }
     | }

/*Sexto Algoritmo*/
def fib6(n:Double):Double = {
     | if (n <= 0 ) {
     | return 0
     | }
     | var i = n - 1
     | var auxOne = 0.00
     | var auxTwo = 1.00
     | var a_b = (auxTwo,auxOne)
     | var c_d = (auxOne,auxTwo)
     | while( i > 0 ){
     | if((i%2) != 0 ) {
     | auxOne = ( c_d._2 * a_b._2) + (c_d._1 * a_b._1)
     | auxTwo = ( c_d._2*(a_b._2 + a_b._1) + (c_d._1 * a_b._2))
     | a_b = (auxOne,auxTwo)
     | }
     | auxOne = pow(c_d._1,2) + pow(c_d._2,2)
     | auxTwo = ( c_d._2 * ((2 * c_d._1) + c_d._2))
     | c_d = (auxOne,auxTwo)
     | i = i / 2
     | }
     | return ((a_b._1)+ (a_b._2))
     | }

