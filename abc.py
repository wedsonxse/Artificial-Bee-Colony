# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:35:52 2019

@author: wedson
"""
#-----------FUNCOES A OTIMIZAR--------------------
import random
import math
import numpy as np
import pandas as pd

results = pd.DataFrame()
resultsValue = pd.DataFrame()

def sphere(x):
    vec = np.array(x)
    return np.sum(vec ** 2)

def rastrigin(x):
        f_x = [xi ** 2 - 10 * math.cos(2 * math.pi * xi) + 10 for xi in x]
        return sum(f_x)

def rosenbrock (x):
    total = 0
    for i in range(0,(len(x)-1)):
        total += 100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    
    return total

class FoodSource:
    def __init__(self, pDimensions, lUp, lInf):
        self.solution = -1
        self.bestSolution = -1
        self.foodSourcePosition = []
        self.trials = 0
        self.fit = -1
        
        self.initialSuperiorBound = lUp
        self.initialInferiorBound = int(lUp/2)
        
        for i in range (0,pDimensions):
            r = random.uniform(0,1)
            pos = self.initialInferiorBound + r * (self.initialSuperiorBound - self.initialInferiorBound)
            self.foodSourcePosition.append(pos)
            
    def evaluate (self, costFunc):
        self.solution = costFunc(self.foodSourcePosition)
        
    def calculate_fitness(self):
        if self.solution >= 0:
            result = 1.0 / (1.0 + self.solution)
        else:
            result = 1.0 + abs(self.solution)
            
        self.fit = result
        
        
class Bee:
    def __init__(self):
        self.position = []
        self.newPosition = [None] * 30
        self.foodSolution = -1
        self.solution = -1
        self.fit = -1
        self.foodFit = -1
        
    def stepAside(self,Min,Max):
        #passo pro lado pra verificar o valor de fitness da redondeza
        
        i = random.randint(0,29)
        k = random.randint(0, 29)
        r = random.uniform(-1,1)    
        
        while True:
            if (k == i):
                k = random.randint(0, 29)
            else:
                break
            
        self.newPosition[i] = float(self.position[i] + r * (self.position[i] - self.position[k]))
        
        if self.newPosition[i] < Min:
            self.newPosition[i] = Min
        elif self.newPosition[i] > Max:
            self.newPosition[i] = Max

    def evaluate(self, costFunc):
        self.solution = costFunc(self.position)
    
    def evaluateNew(self, costFunc):
        self.newSolution = costFunc(self.newPosition)
    
    def calculate_fitness(self):
        if self.solution >= 0:
            result = 1.0 / (1.0 + self.solution)
        else:
            result = 1.0 + abs(self.solution)
            
        self.fit = result
        
class ABC:
    
    global sphereFitVector
    sphereFitVector = [None] * 16667
    
    global rosenbrockFitVector
    rosenbrockFitVector = [None] * 16667
    
    global rastriginFitVector
    rastriginFitVector = [None] * 16667
    
    global limitIt
    limitIt = 500000
    
    def __init__(self, pCostName, pCostFunc, pSwarmSize, pDimensions,lUp, lInf):
        self.foodSources = []
        self.swarmEmployees = []
        self.swarmOnlookers = []
        self.fitnessIt = 0
        self.trialsLimit = 100
        self.bestFit = -1
        self.DimensionInfLimit = lInf
        self.DimensionSupLimit = lUp
        
        print(">starting abc...")
        
        #inicializa os food sources
        for i in range (0, int (pSwarmSize/2)):
            self.foodSources.append(FoodSource(pDimensions, lUp, lInf))
            
        #Atribui o fitness de cada food source    
        for i in range (0, int (pSwarmSize/2)):
            self.foodSources[i].evaluate(pCostFunc)
            self.foodSources[i].calculate_fitness()
        
        #inicializa as abelhas employee
        for i in range (0, int (pSwarmSize/2)):
            self.swarmEmployees.append(Bee())
            
        #inicializa as abelhas onlookers  
        for i in range (0, int (len(self.swarmEmployees))):
            self.swarmOnlookers.append(Bee())
        
        #Início das atividades das abelhas
        j = 0
        while True:
            #print("ITERAÇÃO GERAL: " + str(j))
            
        #loop para as atividades das employees
            for i in range (0, len(self.swarmEmployees)):
                self.swarmEmployees[i].position = self.foodSources[i].foodSourcePosition
                self.swarmEmployees[i].foodFit = self.foodSources[i].fit
                self.swarmEmployees[i].stepAside(lInf,lUp)
                self.swarmEmployees[i].evaluate(pCostFunc)
                self.swarmEmployees[i].calculate_fitness()
                self.fitnessIt += 1
                
                if (self.swarmEmployees[i].fit >= self.swarmEmployees[i].foodFit):
                    self.foodSources[i].trials +=1
                    
                elif (self.swarmEmployees[i].fit < self.swarmEmployees[i].foodFit):
                    self.foodSources[i].foodSourcePosition = self.swarmEmployees[i].newPosition
                    self.foodSources[i].solution = self.swarmEmployees[i].solution
                    self.foodSources[i].fit = self.swarmEmployees[i].fit
                    self.foodSources[i].trials = 0
            
            if(self.fitnessIt > limitIt):
                break
                
            probabilities = [self.probability(fs) for fs in self.foodSources]
            t = s = 0
                
            while t < len(self.swarmOnlookers):
                s = (s + 1) % len(self.swarmOnlookers)
                r = np.random.uniform()
                  
                if r < probabilities[s]:
                   t += 1
                   #selectedIndex = self.selection(range(len(self.foodSources)), probabilities)
                   self.swarmOnlookers[s].position = self.foodSources[s].foodSourcePosition
                   self.swarmOnlookers[s].foodFit = self.foodSources[s].fit
                   self.swarmOnlookers[s].stepAside(lInf,lUp)
                   self.swarmOnlookers[s].evaluate(pCostFunc)
                   self.swarmOnlookers[s].calculate_fitness()
                   self.fitnessIt += 1
                    
                if (self.swarmOnlookers[i].fit >= self.swarmOnlookers[i].foodFit):
                    self.foodSources[s].trials +=1
                        
                elif (self.swarmOnlookers[s].fit < self.swarmOnlookers[s].foodFit):
                    self.foodSources[s].foodSourcePosition = self.swarmOnlookers[s].newPosition
                    self.foodSources[s].solution = self.swarmOnlookers[s].solution
                    self.foodSources[s].fit = self.swarmOnlookers[s].fit
                    self.foodSources[s].trials = 0
              
            if(self.fitnessIt > limitIt):
                break    
                
        #loop para as atividades das scouts            
            for i in range (0, len(self.swarmEmployees)):
                if (self.foodSources[i].trials > self.trialsLimit):
                    self.foodSources[i] = self.createFoodSource(pCostFunc, pDimensions)
                
        #salvar o melhor valor de fit entre os foodsources
            for i in range (0, len(self.foodSources)):
                if(self.foodSources[i].solution < self.bestFit or self.bestFit == -1):
                    self.bestFit = self.foodSources[i].solution
                    
            if(pCostName == "sphere"):
                #print("best: ", self.bestFit)
                sphereFitVector[j] = self.bestFit
            if(pCostName == "rosenbrock"):
                rosenbrockFitVector[j] = self.bestFit
            if(pCostName == "rastrigin"):
                rastriginFitVector[j] = self.bestFit
             
            j += 1
        
        #calculo da probabilidade de cada food source
    def probability(self, solut):
        solutionSum = sum([fs.fit for fs in self.foodSources])
        probability = solut.fit / solutionSum
        
        return probability

    def selection(self, solutions, weights):
        return random.choices(solutions, weights)[0] 
    
    #criação de um novo food source usado pela abelha scout
    def createFoodSource(self, pCostFunc, pDimensions):
        position = [None] * pDimensions

        for i in range (0, pDimensions):
            position[i] = self.candidatePosition(self.DimensionInfLimit, self.DimensionSupLimit)
        
        solution = pCostFunc(position)
        
        if solution >= 0:
            result = 1.0 / (1.0 + solution)
        else:
            result = 1.0 + abs(solution)
            
        fitn = result
        
        Fs = FoodSource(pDimensions, self.DimensionSupLimit, self.DimensionInfLimit)
        Fs.foodSourcePosition = position
        Fs.solution = solution
        Fs.fit = fitn
        
        return Fs
    
    #pegar um aposição na redondeza para testar um novo possivel food source
    def candidatePosition(self, lb, ub):
        r = random.uniform(0,1)
        position = lb + (ub - lb) * r

        return np.around(position, decimals=4)
    
if __name__ == "__ABC__":
    main()
    
dimensions = 30
swarmSize = 30
DimensionInfLimit = -100
DimensionSupLimit = 100
n_sim = 30

print('>a colonia começa a se movimentar...')

for j in range (0, n_sim):
    print("simulação esfera : " + str(j))
    ABC("sphere", sphere, swarmSize, dimensions, DimensionSupLimit, DimensionInfLimit)
    results[f'sim_{j}'] = sphereFitVector #vetor com os valores de fitness específicos

results.to_csv(f'abc_sphere_{n_sim}.csv', index=False)   

DimensionInfLimit = -5.12
DimensionSupLimit = 5.12

for k in range (0, n_sim):
    print("simulação rastrigin : " + str(k))
    ABC("rastrigin", rastrigin, swarmSize, dimensions, DimensionSupLimit, DimensionInfLimit)
    results[f'sim_{k}'] = rastriginFitVector #vetor com os valores de fitness específicos

results.to_csv(f'abc_rastrigin_{n_sim}.csv', index=False)    

DimensionInfLimit = -30
DimensionSupLimit = 30

for l in range (0, n_sim):
    print("simulação rosenbrock : " + str(l))
    ABC("rosenbrock", rosenbrock, swarmSize, dimensions, DimensionSupLimit, DimensionInfLimit)    
    results[f'sim_{l}'] = rosenbrockFitVector #vetor com os valores de fitness específicos
            
results.to_csv(f'abc_rosenbrock_{n_sim}.csv', index=False)

print(">A colônia se recolheu...")
    
    
    
    
    
    
    
    
    
    
    
    
    
    