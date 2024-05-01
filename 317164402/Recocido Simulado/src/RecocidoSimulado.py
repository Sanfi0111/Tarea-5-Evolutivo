import numpy as np 
import random
import math
import matplotlib.pyplot as plt
from tabulate import tabulate


class RecocidoSimulado:
    nBits = 0
    dim = 0
    lim_Inf = 0
    lim_Sup = 0
    numIters = 0
    temperatura = 400
    #temperatura = 100
    temperaturainicial = 0
    solucion = []
    fitnessActual = 0
    alpha = 0.8
    #alpha = 0.7
    fitnessPorGeneracion = []
    entraProb = 0
    noEntraProb =0

    '''
     * Inicializamos la población teniendo en cuenta el número de bits que ocuparemos, el tamaño de la población que se tendrá
     * y la función de optimización que se ocupará
     * @param tamPob el tamaño de la poblacion por generación
     * @param nBits el número de bits por población
     * @param dim la dimensión
     * @param funcionDeOptimizacion la funcion de optimización que se utilizará
    '''
    def inicializaSolucion(self, nBits, dim, funcionDeOptimizacion):
        self.nBits  = nBits
        if(dim ==1):
            self.dim = 1
        else:
            self.dim = dim -1
        print("Dimension; ",self.dim)       
        self.funcionDeOptimizacion = funcionDeOptimizacion
        self.temperaturainicial = self.temperatura
        if(funcionDeOptimizacion ==1):
            self.lim_Inf = -5.12
            self.lim_Sup = 5.12
            self.solucion = self.inicializa()      
            print("Se usará Sphere \n")
            sol1 = self.decodifica_aux(self.solucion)
            sol = [0]
            sol[0] = sol1
            self.fitnessActual = self.sphere(sol)
        if(funcionDeOptimizacion ==2):
            print("Se usará Ackley")
            self.lim_Inf = -30
            self.lim_Sup = 30
            self.solucion = self.inicializa()      
            sol1 = self.decodifica_aux(self.solucion)
            sol = [0]
            sol[0] = sol1
            self.fitnessActual = self.ackley(np.array(sol))
        if(funcionDeOptimizacion == 3):
            print("Griewank")
            self.lim_Inf = -600
            self.lim_Sup = 600
            self.solucion = self.inicializa()      
            sol1 = self.decodifica_aux(self.solucion)
            sol = [0]
            sol[0] = sol1
            self.fitnessActual = self.griewank(np.array(sol))

        if(funcionDeOptimizacion ==4):
            print("Se usará rastrigin")
            self.lim_Inf = -5.12
            self.lim_Sup = 5.12
            self.solucion = self.inicializa()      
            sol1 = self.decodifica_aux(self.solucion)
            sol = [0]
            sol[0] = sol1
            self.fitnessActual = self.rastrigin(np.array(sol))

        if(funcionDeOptimizacion == 5):
            print("Se usará Rosenbrock")
            self.lim_Inf = -2.04
            self.lim_Sup = 2.04
            self.solucion = self.inicializa()      
            sol1 = self.decodifica_aux(self.solucion)
            sol = [0]
            sol1 = self.deletePoint(str(sol1))
            sol1 = "."+str(sol1)
            sol1 = float(sol1)
            sol[0] = sol1
            self.fitnessActual = self.rosenbrock(np.array(sol))

        print("solucion inicial: ")
        print(self.solucion)
        print("Fitness inicial: ",self.fitnessActual)

    '''
     * Inicializa una solución dando puntos al azar del tamaño de nbits.
     * @return una solución inicial
    '''
    def inicializa(self):
        pob = []
        s =0
        while(s < self.dim):
            ini = []    
            for i in range(self.nBits):
                ini.append(random.randint(0,1))
            if(self.decodifica_aux1Dim(ini)>self.lim_Sup):
                print("Más grande que el  límite superior")
            else: 
                pob.append(ini)
                s+=1
        print("pob: ",pob)
        return pob
    '''
     * Convierte un número binario a decimal.
     * @param binario un arreglo de un número en binario
    '''
    def decodifica_aux1Dim(self, binario):
        n = 0
        for bit in binario:
            n = n * 2 + bit
        return n
    
    '''
     * Realiza el enfriamiento lineal
     * Multiplica la temperatura actual por el decamento 
    '''
    def enfriamientoLineal(self,i):
        self.temperatura = self.temperaturainicial - self.alpha* i
        print("Nueva temperatura: ",self.temperatura)

    '''
     * Cambia el bit de un indice aleatorio en la solución actual y lo devuelve
     * como candidato
    '''
    def generaCandidato(self):
        candidato = self.solucion
        print("Candidado en genera:", candidato)
        indiceAl = random.randint(0, len(candidato[0])-1)
        candidato[0][indiceAl] = 1-candidato[0][indiceAl]
        return candidato
    
    '''
     * Genera el algoritmo de recocido simulado
    '''
    def recocidoSimulado(self):
        candidato = self.generaCandidato()
        fitnessCandidato = self.calculaNuevoFitness(candidato)
        print("Fitness del candidato: ",fitnessCandidato, ". Fitness anterior: ",self.fitnessActual)
        if(fitnessCandidato > self.fitnessActual):
            print("Como el fitness es mayor, se cambiará")
            self.solucion =  candidato
            self.fitnessActual = fitnessCandidato
        else:
            exponente = self.fitnessActual- fitnessCandidato
            probabilidad1 = (-exponente/self.temperatura)
            print("Exponente: ",exponente, "Temp:", self.temperatura, "Probabildidad 1: ",probabilidad1)
            probabilidad = np.exp(probabilidad1)
            r = random.uniform(0, 1)
            print("proba: ", probabilidad," r: ",r)
            if probabilidad > r:
                self.entraProb+=1
                print("entre a la proba, se cambia")
                self.solucion =  candidato
                self.fitnessActual = fitnessCandidato
            else:
                self.noEntraProb+=1
  
        self.fitnessPorGeneracion.append(self.fitnessActual)
        return self.solucion 

        
    '''
     * Calcula un fitness para una población nueva dependiendo de la función de optimización
     * @param lista la poblacipon inicial
    '''
    def calculaNuevoFitness(self, lista):
        nuevoFitness = 0
        if(self.funcionDeOptimizacion ==1):
            sol1 = self.decodifica_aux(lista)
            sol = [0]
            sol[0] = sol1
            nuevoFitness = self.sphere(sol)
        if(self.funcionDeOptimizacion ==2):
            sol1 = self.decodifica_aux(lista)
            sol = [0]
            sol[0] = sol1
            nuevoFitness = self.ackley(np.array(sol))
        if(self.funcionDeOptimizacion == 3):
            sol1 = self.decodifica_aux(lista)
            sol = [0]
            sol[0] = sol1
            nuevoFitness = self.griewank(np.array(sol))
        if(self.funcionDeOptimizacion ==4):
            sol1 = self.decodifica_aux(lista)
            sol = [0]
            sol[0] = sol1
            nuevoFitness = self.rastrigin(np.array(sol))
        if(self.funcionDeOptimizacion == 5):
            sol1 = self.decodifica_aux(lista)
            sol = [0]
            sol1 = self.deletePoint(str(sol1))
            sol1 = "."+str(sol1)
            sol1 = float(sol1)
            sol[0] = sol1
            nuevoFitness = self.rosenbrock(np.array(sol))
        return nuevoFitness
    '''
     * Elimina el punto decimal de un número double.
     * @param n el número
    '''
    def deletePoint(self,n):
        ret = ""
        for i in range(len(n)):
            if(not n[i] == '.'):
                ret = ret+n[i]
        return ret

    '''
     * Convierte un número binario a decimal.
     * @param binario un arreglo de un número en binario
    '''
    def decodifica_aux(self, binario):
        binar = []
        for i in range(self.dim):
            n = 0
            for k in range(self.nBits):
                n = n * 2 + binario[i][k]
            binar.append(n)
        print(binar)
        return binar
    

    '''
     * Función Ackley para un conjunto de puntos x
     * @param x una liasta de puntos x
    '''
    def ackley(self,x):
        n = len(x)
        sum_term = np.sum(np.square(x))
        cos_term = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(1/n * sum_term)) - np.exp(1/n * cos_term) + 20 + np.exp(1)

    '''
     * Función Sphere para un conjunto de puntos x
     * @param x una liasta de puntos x
    '''
    def sphere(self,x):
        return np.sum(np.square(x))
    '''
     * Función Griewank para un conjunto de puntos x
     * @param x una liasta de puntos x
    '''
    def griewank(self,x):
        sum_term = np.sum(np.square(x))/4000
        prod_term = np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))
        return 1 + sum_term - prod_term
    '''
     * Función Rastrigin para un conjunto de puntos x
     * @param x una liasta de puntos x
    '''
    def rastrigin(self,x):
        return 10 * len(x) + np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x))
    '''
     * Función Rosenbrock para un conjunto de puntos x
     * @param x una liasta de puntos x
    '''
    def rosenbrock(self, x):
        a = 1
        b = 100
        y = 1  
        valor= (a - x)**2 + b * (y - x**2)**2
        print("Fitness", valor)
        return valor
    '''
     * Grafica los mejores resultados por generación
     * @param el número de generaciones  
    '''
    def plot(self,n):
        x = np.arange(0,n,1)
        y = self.fitnessPorGeneracion
        print("x:", x)
        print("y",y)
        plt.plot(x, y)
        plt.xlabel('Iteraciones')
        plt.ylabel('Fitness')
        plt.title('Fitness por iteración')
        plt.show()

def main():
    n = 500
    resultadosFinales = []
    for i in range(1):
        alg = RecocidoSimulado()
        alg.solucion = []
        alg.fitnessPorGeneracion = []
        alg.inicializaSolucion(10,10,3)
        alg.fitnessPorGeneracion.append(alg.fitnessActual)
        print("fitness inicial: ", alg.fitnessActual)
        for i in range(n):
            alg.recocidoSimulado()
            alg.enfriamientoLineal(i)
            print("Fitness en generacion ",i,": ",alg.fitnessActual)
        print("Fitness por generaciones: ", alg.fitnessPorGeneracion)    
        alg.plot(n+1)
        sorted = np.sort(alg.fitnessPorGeneracion)
        print("Mejor resultado = ",sorted[len(sorted)-1])
        resultadosFinales.append(sorted[len(sorted)-1])
    print("Resultados finales: ",resultadosFinales)
    x = np.arange(0,20,1)
    print("x:", x)
    plt.plot(x, resultadosFinales)
    plt.xlabel('Ejecuciones')
    plt.ylabel('Fitness por ejecución')
    plt.title('Mejor fitness por ejecución')
    plt.show()
if __name__ == '__main__':
    main()
    