#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import time
set_matplotlib_formats('svg')

start = time.time()
# In[2]:
# data = np.load('att48.npz', allow_pickle=True)
# lst = data.files
# mp=data[lst[1]]

class City:
    def __init__(self, n):
        self.n = n
    
    def distance(self, city):
        mp = [
        [15000, 266.6538, 15000, 15000, 15000, 15000, 388.4219,427.6574, 15000, 262.287, 295.6253]
          ,[266.6538, 15000, 320.6497, 15000, 15000, 15000,15000, 15000, 15000, 15000, 15000]
          ,[15000, 320.6497, 15000, 219.4997, 15000, 15000, 15000, 15000, 15000,15000, 170.118]
          ,[15000, 15000, 219.4997, 15000, 166.0354, 15000, 15000 ,15000 ,15000 ,15000, 15000]
          ,[15000, 15000, 15000, 166.0354, 15000, 138.238, 15000, 15000, 15000, 15000, 15000]
          ,[15000, 15000, 15000, 15000,138.283, 15000, 152.842,15000, 15000, 15000, 15000]
          ,[388.4219,15000, 15000, 15000, 15000, 152.842, 15000, 182.6261,15000, 15000, 193.4845]
          ,[427.6574, 15000, 15000, 15000, 15000, 15000, 182.6261, 15000, 192.3555, 15000, 232.7199]
          ,[15000, 15000, 15000, 15000, 15000, 15000, 15000,192.3555, 15000, 107.5493, 15000]  
          ,[262.2874, 15000, 15000, 15000, 15000, 15000, 15000, 15000,107.5493, 15000, 15000]
          ,[295.6253, 15000, 170.118, 15000, 15000, 15000, 193.4845,232.7199,15000, 15000, 15000],
        ]
        distance=mp[self.n][city.n]
        return distance
    
    def __repr__(self):
        return str(self.n)


# In[3]:


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


# In[4]:


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    #route.insert(0,City(n=1))
    #print(route)
    return route


# In[5]:


def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


# In[6]:


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


# In[7]:


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


# In[8]:


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


# In[9]:


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


# In[10]:


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children


# In[11]:


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


# In[12]:


def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


# In[13]:


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


# In[14]:


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


# In[15]:


city=[0,1,2,3,4,5,6,7,8,9,10]
cityList = []

for i in city:
    cityList.append(City(n=i))


# In[16]:


geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=100)


# In[17]:


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


# In[22]:


geneticAlgorithmPlot(population=cityList, popSize=150, eliteSize=30, mutationRate=0.01, generations=100)


# In[ ]:

end = time.time()

print("The time of execution of above program is :", end-start)
