import numpy as np 
import random
import math
import matplotlib.pyplot as plt
from tabulate import tabulate


class AlgoritmoGenetico():

    nBits = 0
    dim = 0
    lim_Inf = 0
    lim_Sup = 0
    numIters = 0
    tamPob = 0
    # Probabilidad de mutación
    pm = .05
    f_x = ""
    poblacion = []
    fitnessPorSolucion =[]
    mejorsolucion = []
    funcionDeOptimizacion = 0
    mejorFitnesPorGeneracion = []

    '''
     * Inicializamos la población teniendo en cuenta el número de bits que ocuparemos, el tamaño de la población que se tendrá
     * y la función de optimización que se ocupará
     * @param tamPob el tamaño de la poblacion por generación
     * @param nBits el número de bits por población
     * @param dim la dimensión
     * @param funcionDeOptimizacion la funcion de optimización que se utilizará
    '''
    def inicializaPoblacion(self, tamPob, nBits, dim, funcionDeOptimizacion):
        self.funcionDeOptimizacion = funcionDeOptimizacion
        self.nBits  = nBits
        self.tamPob = tamPob
        if(dim ==1):
            self.dim = 1
        else:
            self.dim = dim -1
        self.fitnessPorSolucion = np.zeros(self.tamPob)
        if(funcionDeOptimizacion ==1):
            self.lim_Inf = -5.12
            self.lim_Sup = 5.12
            for i in range(tamPob):
                self.poblacion.append(self.inicializaSolucion())
            print("poblacion: ",self.poblacion)
            print("Se usará Sphere \n")
            for i in range(self.tamPob):
                sol1 = self.decodifica_aux(self.poblacion[i])
                sol = [0]
                sol[0] = sol1
                self.fitnessPorSolucion[i] = self.sphere(sol)
        if(funcionDeOptimizacion ==2):
            print("Se usará Ackley")
            self.lim_Inf = -30
            self.lim_Sup = 30
            for i in range(tamPob):
                self.poblacion.append(self.inicializaSolucion())
            print("poblacion: ",self.poblacion)
            for i in range(self.tamPob):
                sol1 = self.decodifica_aux(self.poblacion[i])
                sol = [0]
                sol[0] = sol1
                self.fitnessPorSolucion[i] = self.ackley(np.array(sol))
        if(funcionDeOptimizacion == 3):
            print("Griewank")
            self.lim_Inf = -600
            self.lim_Sup = 600
            for i in range(tamPob):
                self.poblacion.append(self.inicializaSolucion())
            print("poblacion: ",self.poblacion)
            for i in range(self.tamPob):
                print("poblacion en i:",self.poblacion[i])
                sol1 = self.decodifica_aux(self.poblacion[i])
                sol = [0]
                sol[0] = sol1
                self.fitnessPorSolucion[i] = self.griewank(np.array(sol))
        if(funcionDeOptimizacion ==4):
            print("Se usará rastrigin")
            self.lim_Inf = -5.12
            self.lim_Sup = 5.12
            for i in range(tamPob):
                self.poblacion.append(self.inicializaSolucion())
            print("poblacion: ",self.poblacion)
            for i in range(self.tamPob):
                sol1 = self.decodifica_aux(self.poblacion[i])
                sol = [0]
                sol[0] = sol1
                self.fitnessPorSolucion[i] = self.rastrigin(np.array(sol))
        if(funcionDeOptimizacion == 5):
            print("Se usará Rosenbrock")
            self.lim_Inf = -2.04
            self.lim_Sup = 2.04
            for i in range(tamPob):
                self.poblacion.append(self.inicializaSolucion())
            print("poblacion: ",self.poblacion)
            for i in range(self.tamPob):
                print(i)
                sol1 = self.decodifica_aux(self.poblacion[i])
                sol = [0]
                sol[0] = sol1
                print("Valor:" ,sol)
                self.fitnessPorSolucion[i] = self.rosenbrock(np.array(sol))
        print("poblacion inicial: ")
        print(self.poblacion)
        print("Fitness iniciales de las soluciones:")
        print(self.fitnessPorSolucion)   
        #Da el indice de la mejor solucion     
        self.mejorsolucion = int(np.argsort(self.fitnessPorSolucion)[len(self.fitnessPorSolucion)-1])
        print("Mejor solucion: ", self.mejorsolucion)

    '''
     * Inicializa una solución dando puntos al azar del tamaño de nbits.
     * @return una solución inicial
    '''
    def inicializaSolucion(self):
        pob = []
        s =0
        while(s < self.dim):
            ini = []    
            for i in range(self.nBits):
                ini.append(random.randint(0,1))
            pob.append(ini)
            s+=1
        
        return pob

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
     * Convierte un número binario a decimal.
     * @param binario un arreglo de un número en binario
    '''
    def decodifica_aux1Dim(self, binario):
        n = 0
        for bit in binario:
            n = n * 2 + bit
        return n
    '''
     * Escoge un individuo al azar, teniendo en cuenta su probabilidad. Entre mayor sea su fitness, mayor probabilidad tendrá de ser escogido
    '''
    def seleccionDeRuleta(self):
        totalFitness = np.sum(self.fitnessPorSolucion)
        proba = self.fitnessPorSolucion / totalFitness
        indices  = np.random.choice(np.arange(len(self.fitnessPorSolucion)), size=2, p=proba)
        print("primer padre: ", indices[0])
        print("segundo padre: ", indices[1])
        return indices

    '''
     * Operador de cruza de n puntos.
     * A partir de n puntos, partirá el arreglo de los dos padres en n posiciones. 
     * Luego, producirá dos offsprings, que son resultado de intercambiar los valores del padre 1 con el padre 2 en las posciones anteriormente especificadas
     * Tiene una probabildad de .7 
     * Una vez que tengamos los offsprings, llevaremos a cabo la mutación flip (especificada en el método de mutación Flip)
    '''
    def operadorDeCruza(self, padres, n):
        prob = random.random()
        resultado = []
        resultado.append(self.poblacion[padres[0]][0])
        resultado.append(self.poblacion[padres[1]][1])
        if(prob< 0.75):
            print("Entre a la proba")
            padre1 = self.poblacion[padres[0]]
            print("Padre 1")
            print(padre1)
            padre2 = self.poblacion[padres[1]]
            print("Padre 2")
            print(padre2)
            offspring1 = np.copy(padre1[0])
            offspring2 = np.copy(padre2[0])
            indicesRandom = np.zeros(n)
            # tendrá los valores pero se irán eliminando
            for i in range(n):
                ran = random.randint(1,self.nBits-1)
                while(ran in indicesRandom):
                    ran = random.randint(1,self.nBits-1)
                indicesRandom[i] = ran
                indicesRandom[i] = ran
            indicesRandom = np.sort(indicesRandom)
            long = len(indicesRandom)
            while(len(indicesRandom) > 0):
                if(long == len(indicesRandom)):
                    k = 0
                    while(k <=  int(indicesRandom[0])):
                        offspring1[k] = padre2[0][k]
                        offspring2[k] = padre1[0][k]
                        k+=1
                    indicesRandom = np.delete(indicesRandom,0)
                else:
                    if(len(indicesRandom)>1):
                        for i in range(int(indicesRandom[1])- int(indicesRandom[0])):
                            offspring1[int(indicesRandom[0])+i] = padre2[0][int(indicesRandom[0])+i]
                            offspring2[int(indicesRandom[0])+i] = padre1[0][int(indicesRandom[0])+i]
                        indicesRandom = np.delete(indicesRandom,0)
                        indicesRandom = np.delete(indicesRandom,0)
                    else:
                        i = 0
                        ind = int(indicesRandom[0])-1
                        while((i+int(indicesRandom[0]))<=len(padre1)):
                            offspring1[ind+i] = padre2[0][ind+i]
                            offspring2[ind+i] = padre1[0][ind+i]
                            i+=1

                        indicesRandom = np.delete(indicesRandom,0)
            print("Primer offspring:")
            print(offspring1)
            print("Segundo offspring:")
            print(offspring2)
            # mutacion flip en ambos offsprings
            offspring1 = self.mutacionFlip(offspring1)
            offspring2 = self.mutacionFlip(offspring2)
            resultado[0] = (offspring1)
            resultado[1]= (offspring2)
            print("resultado: ", resultado)

        else:
            print("no se realizó operación de cruza debido a que a probabilidad que se obtuvo: ", prob," es mas alta que .75")
            print("resultado: ", resultado)
        return resultado

    '''
     * Realiza la mutación flip.
     * Dada una probabilidad de mutación de 0.05. Irá recorriendo cada uno de las posiciones de la solución, 
     * y en caso de cumplir con la probabilidad, cambiará el valor (si es 0, dará 1, si es 1, dará 0)
    '''
    def mutacionFlip(self, elem):
        for i in range(self.nBits):
            prob = np.random.rand()
            if  prob < self.pm:
                elem[i] = 1 -  elem[i]
                print("nueva mutacion: ",elem)
        return elem
    
    '''
     * Primero guardamos la mejor solucion y la ponemos en el lugar de que tenga peor lugar
     * El mejor padre se mantiene.
     * Reemplacemos a los peores con los mejores
     * @param hijos los nuevos hijos
    '''
    def reemplazoPorElitismo(self, hijos):
        nuevaPoblacion = []
        if(hijos != []):
            poblacionConHijos = np.concatenate((self.poblacion,hijos))
            print("poblacion con hijos:", poblacionConHijos)
            fitnessConHijos = self.calculaNuevoFitness(poblacionConHijos)
            print("fitness con hijos:", fitnessConHijos)
            # contiene los fitness con indices de mayor a menor
            fitnessOrdenado = np.argsort(fitnessConHijos)
            print("Fitness ordenado", fitnessOrdenado)
            i = 0
            m = 0
            while(i < len(poblacionConHijos)):
                if(not(i == fitnessOrdenado[0] or i == fitnessOrdenado[1])):
                    nuevaPoblacion.append(poblacionConHijos[i])
                    m +=1
                i = i+1
            print("nueva poblacion: ",nuevaPoblacion)
            self.poblacion[0] = nuevaPoblacion
            self.fitnessPorSolucion = self.calculaNuevoFitness(nuevaPoblacion)
            print("Nuevo fitness:", self.fitnessPorSolucion)
        return nuevaPoblacion
        
    '''
     * Se reemplazaran las soluciones con peor fitness por los nuevos resultados
     * @param hijos los nuevos hijos
    '''
    def reemplazoDeLosPeores(self, hijos):
        nuevaPoblacion = self.poblacion.copy()
        if(hijos!= []):
            fitnessOrdenado = np.argsort(self.fitnessPorSolucion)
            nuevaPoblacion[fitnessOrdenado[0]][0] = hijos[0]
            nuevaPoblacion[fitnessOrdenado[1]][1] = hijos[1]
            self.poblacion = nuevaPoblacion
            print("nueva poblacion:", self.poblacion)
            self.fitnessPorSolucion = self.calculaNuevoFitness(nuevaPoblacion)
        return nuevaPoblacion

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
        return np.sum(10 * len(x) + np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x)))

    '''
     * Función Rosenbrock para un conjunto de puntos x
     * @param x una liasta de puntos x
    '''
    def rosenbrock(self, x):
        a = 1
        b = 100
        y = 1  
        valor = np.sum((a - x)**2 + b * (y - x**2)**2)
        print("Fitness", valor)
        return valor

    '''
     * Calcula un fitness para una población nueva dependiendo de la función de optimización
     * @param lista la poblacipon inicial
    '''
    def calculaNuevoFitness(self, lista):
        resultado = np.zeros(len(lista))
        if(self.funcionDeOptimizacion == 1):
            for i in range(len(lista)):
                sol1 = self.decodifica_aux(lista[i])
                sol = [0]
                sol[0] = sol1
                resultado[i] = self.sphere(sol)
        if(self.funcionDeOptimizacion ==2):
            for i in range(len(lista)):
                sol1 = self.decodifica_aux(lista[i])
                sol = [0]
                sol[0] = sol1
                resultado[i] = self.ackley(np.array(sol))
        if(self.funcionDeOptimizacion ==3):
            for i in range(len(lista)):
                sol1 = self.decodifica_aux(lista[i])
                sol = [0]
                sol[0] = sol1
                resultado[i] = self.griewank(np.array(sol))
        if(self.funcionDeOptimizacion == 4):
            for i in range(len(lista)):
                sol1 = self.decodifica_aux(lista[i])
                sol = [0]
                sol[0] = sol1
                resultado[i] = self.rastrigin(np.array(sol))
        if(self.funcionDeOptimizacion == 5):
            for i in range(self.tamPob):
                sol1 = self.decodifica_aux(self.poblacion[i])
                sol = [0]
                sol[0] = sol1
                print("Valor:" ,sol)
                self.fitnessPorSolucion[i] = self.rosenbrock(np.array(sol))
        return resultado
    '''
     * Grafica los mejores resultados por generación
     * @param el número de generaciones  
    '''
    def plot(self,n):
        x = np.arange(0,n,1)
        y = self.mejorFitnesPorGeneracion.tolist()
        print("x:", x)
        print("y",y)
        plt.plot(x, y)
        plt.xlabel('Ejecuciones')
        plt.ylabel('Fitness')
        plt.title('Fitness por generación')
        plt.show()
    '''
     * Tabula los mejores resultados por generación
     * @param n el número de generaciones
    '''
    def tabula(self, n):
        data = []
        data.append(["Iteración","Mejor Fitness"])
        for i in range(1,n):
            data.append([i,self.mejorFitnesPorGeneracion[i]])
        print(tabulate(data))

def main():
    alg = AlgoritmoGenetico()
    n = 500
    resultadosFinales = []
    for i in range(1):
        alg = AlgoritmoGenetico()
        alg.poblacion = []
        alg.mejorFitnesPorGeneracion = np.zeros(n)
        alg.inicializaPoblacion(3,10,10,3)
        # Ejecuta el algoritmo n veces.
        for i in range(n):
            indices = alg.seleccionDeRuleta()
            offsprings = alg.operadorDeCruza(indices,3)
            # Nueva poblacion para la sigueinte generacion
            alg.reemplazoDeLosPeores(offsprings)
            alg.mejorFitnesPorGeneracion[i] = float(alg.fitnessPorSolucion[alg.mejorsolucion])
            print("Mejor solucion en iteracion : ",i, ": ", alg.mejorFitnesPorGeneracion[i])
        print("Mejor solucion: ",alg.poblacion[alg.mejorsolucion], "Fitness: ", alg.fitnessPorSolucion[alg.mejorsolucion])
        resultadosFinales.append((alg.poblacion[alg.mejorsolucion], alg.fitnessPorSolucion[alg.mejorsolucion]))
        alg.tabula(n)
        alg.plot(n)
    print(resultadosFinales)       

if __name__ == '__main__':
    main()
    

    
    