# Andrew Donate
# 6/20/2023 to 6/25/2023
# Project 2

import numpy as np
import random as rand
import pandas as pd
import matplotlib.pyplot as plt

# For print colors
RED = '\033[91m'
RESET = '\033[0m'


# The different points a salesman can go to
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDistance = abs(self.x - city.x)
        yDistance = abs(self.y - city.y)
        distance = np.sqrt((xDistance ** 2) + (yDistance ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


# Similar to the chess ai, this uses a point systems where
# the larger the point value is the more likely it will
# be used as one of the parents for evolution
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = np.sum([fromCity.distance(toCity) for fromCity, toCity in zip(
                self.route, np.roll(self.route, -1))])
            # for i in range(0, len(self.route)):
            #     fromCity = self.route[i]
            #     toCity = None
            #     if i + 1 < len(self.route):
            #         toCity = self.route[i + 1]
            #     else:
            #         toCity = self.route[0]
            #     pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


# Creating individuals and randomly picking the current path they take to vist each city once.
def createRoute(cityList):
    route = rand.sample(cityList, len(cityList))
    return route


# Creating the initial full population (given the size)
def initialPopulation(populationSize, cityList):
    population = []

    for i in range(0, populationSize):
        population.append(createRoute(cityList))
    return population


# Sorting our population by which individual has a higher "Fitness" score
def sortRoutes(population):
    # fitnessResults = {}
    # for i in range(0, len(population)):
    #     fitnessResults[i] = Fitness(population[i]).routeFitness()
    # return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)
    fitnessResults = np.array([Fitness(route).routeFitness()
                              for route in population])
    sortedIndices = np.argsort(-fitnessResults)  # Sort in descending order
    return [(i, fitnessResults[i]) for i in sortedIndices]


# Selecting our individuals who have the highest score in amounts of the "amountOfElite"
def selectParents(populationRanked, amountOfElite):
    selectionResults = []
    dataFrame = pd.DataFrame(np.array(populationRanked),
                             columns=["Index", "Fitness"])
    dataFrame['cum_sum'] = dataFrame.Fitness.cumsum()
    dataFrame['cum_percent'] = 100 * dataFrame.cum_sum/dataFrame.Fitness.sum()

    for i in range(0, amountOfElite):
        selectionResults.append(populationRanked[i][0])

    for j in range(0, len(populationRanked) - amountOfElite):
        pick = 100 * rand.random()
        for k in range(0, len(populationRanked)):
            if pick <= dataFrame.iat[k, 3]:
                selectionResults.append(populationRanked[k][0])
                break

    return selectionResults


# The amount of availible individuals who can become parents to reproduce
def matingPool(population, selectionResults):
    matingPool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingPool.append(population[index])
    return matingPool


# Creates a child between two individuals and mixes their genes together
def breedOffspring(parent1, parent2):
    child = []
    childPart1 = []
    childPart2 = []

    geneA = int(rand.random() * len(parent1))
    geneB = int(rand.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childPart1.append(parent1[i])

    childPart2 = [gene for gene in parent2 if gene not in childPart1]

    child = childPart1 + childPart2
    return child


# With each selected "elite" have them breed to produce children
def breedEntirePopulation(matingPool, amountOfElite):
    children = []
    length = len(matingPool) - amountOfElite
    pool = rand.sample(matingPool, len(matingPool))

    for i in range(0, amountOfElite):
        children.append(matingPool[i])

    for j in range(0, length):
        child = breedOffspring(pool[j], pool[len(matingPool) - j - 1])
        children.append(child)
    return children


# Change the locations of cities to be our mutation
def changeCity(individual, mutationRate):
    for swapped in range(len(individual)):
        if (rand.random() < mutationRate):
            swapWith = int(rand.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1

    return individual


# Change the whole population based on mutation percentage
def changeCityWholePopulation(population, mutationRate):
    mutatedPopulation = []

    for individual in range(0, len(population)):
        mutatedIndvidual = changeCity(population[individual], mutationRate)
        mutatedPopulation.append(mutatedIndvidual)
    return mutatedPopulation


# Create the next generation that came from those in the previous that reproduced
def nextGeneration(currentGeneration, amountOfElite, mutationRate):
    populationRanked = sortRoutes(currentGeneration)
    selectionResults = selectParents(populationRanked, amountOfElite)
    matingpool = matingPool(currentGeneration, selectionResults)
    children = breedEntirePopulation(matingpool, amountOfElite)
    nextGeneration = changeCityWholePopulation(children, mutationRate)
    return nextGeneration


# THE MOST IMPORTANT PART OF THIS PROJECT
def geneticAlgorithm(population, populationSize, amountOfElite, mutationRate, numberOfGenerations, stopOnStagnation, stagnationThreshold):
    # Debug
    # print(RED + "Population Size: " + str(populationSize))
    # print("Elite Size: " + str(amountOfElite))
    # print("Mutation Rate: " + str(mutationRate))
    # print("Number of Generations: " + str(numberOfGenerations))
    # print("Stop on stagnation?: " + str(stopOnStagnation))
    # print("Population Size: " + str(stagnationThreshold) + RESET)

    pop = initialPopulation(populationSize, population)
    print("Initial distance: " + str(round(1 / sortRoutes(pop)[0][1], 3)))

    stagnationCount = 0
    bestDistance = round(1 / sortRoutes(pop)[0][1], 3)
    bestRoute = None

    for i in range(0, int(numberOfGenerations)):
        pop = nextGeneration(pop, amountOfElite, mutationRate)
        print("Gen " + str(i + 1) + " | Minimum Total Distance: " +
              str(round(1 / sortRoutes(pop)[0][1], 3)))

        # Stagnation settings
        if stopOnStagnation:
            currentDistance = round(1 / sortRoutes(pop)[0][1], 3)
            if currentDistance >= bestDistance:
                stagnationCount += 1
            else:
                stagnationCount = 0
                bestDistance = currentDistance
                bestRoute = pop[sortRoutes(pop)[0][0]]

            if stagnationCount >= int(stagnationThreshold):
                break

    print("Final distance: " + str(1 / sortRoutes(pop)[0][1]))
    if bestRoute is None:
        bestRoutePerson = sortRoutes(pop)[0][0]
        bestRoute = pop[bestRoutePerson]
    return bestRoute


# PLOTTING
def geneticAlgorithmPlot(population, populationSize, amountOfElite, mutationRate, numberOfGenerations, stopOnStagnation, stagnationThreshold):
    pop = initialPopulation(populationSize, population)
    print("Initial distance: " + str(round(1 / sortRoutes(pop)[0][1], 3)))
    progress = []
    progress.append(1 / sortRoutes(pop)[0][1])

    stagnationCount = 0
    bestDistance = round(1 / sortRoutes(pop)[0][1], 3)

    for i in range(int(numberOfGenerations)):
        pop = nextGeneration(pop, amountOfElite, mutationRate)
        print("Gen " + str(i + 1) + " | Minimum Total Distance: " +
              str(round(1 / sortRoutes(pop)[0][1], 3)))
        progress.append(1 / sortRoutes(pop)[0][1])

        # Stagnation settings
        if stopOnStagnation:
            currentDistance = round(1 / sortRoutes(pop)[0][1], 3)
            if currentDistance >= bestDistance:
                stagnationCount += 1
            else:
                stagnationCount = 0
                bestDistance = currentDistance
                bestRoute = pop[sortRoutes(pop)[0][0]]

            if stagnationCount >= int(stagnationThreshold):
                break

    print("Final distance: " + str(1 / sortRoutes(pop)[0][1]))
    bestRoutePerson = sortRoutes(pop)[0][0]
    bestRoute = pop[bestRoutePerson]

    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(progress)
    ax1.set_title('Generational Change')
    ax1.set_ylabel('Distance')
    ax1.set_xlabel('Generation')

    ax2.set_title('Initial Route (Gen 1)')
    plotRoute(population, ax2)

    ax3.set_title('Best Route (Gen ' + str(len(progress) - 1) + ')')
    plotRoute(bestRoute, ax3)

    plt.tight_layout()
    plt.show()

    return bestRoute


# Plot given route
def plotRoute(route, ax):
    x = [city.x for city in route]
    y = [city.y for city in route]
    ax.plot(x, y, 'o-')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')

    startPoint = route[0]
    endPoint = route[-1]
    ax.plot(startPoint.x, startPoint.y, 'go', label='Start')
    ax.plot(endPoint.x, endPoint.y, 'ro', label='End')
    ax.plot([startPoint.x, endPoint.x], [startPoint.y, endPoint.y], 'r--')
    ax.legend()


# User settings
def userSettings():
    cityList = []

    print("Parameters for Genetic Algorithm")

    # Number of cities
    numberOfCities = input(
        "Enter the integer number of cities in travelling salesman problem (default is 25): ")
    if not checkUserInputInt(numberOfCities):
        print(RED + "Not valid number, defaulting to 25." + RESET)
        numberOfCities = 25

    for i in range(0, int(numberOfCities)):
        cityList.append(City(x=int(rand.random() * 200),
                        y=int(rand.random() * 200)))

    # Population size
    populationSize = input(
        "Enter the integer population size (default is 100): ")
    if not checkUserInputInt(populationSize):
        print(RED + "Not valid number, defaulting to 100" + RESET)
        populationSize = 100

    # Mutation rate
    mutationRate = input("Enter the float mutation rate (default is 0.01): ")
    if not checkUserInputFloat(mutationRate):
        print(RED + "Not valid number, defaulting to 0.01" + RESET)
        mutationRate = 0.01

    # Number of elite
    amountOfEliteFloat = input(
        "Enter the proportion of new children in each generation (default is 0.2): ")
    if not checkUserInputFloat(amountOfEliteFloat):
        print(RED + "Not valid number, defaulting to 0.2" + RESET)
        amountOfEliteFloat = 0.2
    amountOfElite = int(populationSize) * float(amountOfEliteFloat)

    # Stagnation and generations
    stagnationThreshold = 0
    numberOfGenerations = 0
    stopOnStagnationYorN = input(
        "Do you want to stop on stagnation? (default is N) (Y/N): ").lower()
    if stopOnStagnationYorN.lower() == 'y':
        stopOnStagnation = True
        stagnationThreshold = input(
            "Enter number of consecutive generations with no improvement to use as a stopping condition (default is 40): ")
        if not checkUserInputInt(stagnationThreshold):
            print(RED + "Not valid number, defaulting to 40" + RESET)
            stagnationThreshold = 40
        numberOfGenerations = 500
    else:
        stopOnStagnation = False
        stagnationThreshold = 0
        # Number of generations
        numberOfGenerations = input(
            "Enter the integer number of generations (default 500): ")
        if not checkUserInputInt(numberOfGenerations):
            print(RED + "Not valid number, defaulting to 500" + RESET)
            numberOfGenerations = 500

    populationSize = int(populationSize)
    amountOfElite = int(amountOfElite)
    mutationRate = float(mutationRate)

    # Plot or no plot
    plotOption = input(
        "Would you like the output to be graphed? (default is N) (Y/N): ")

    if plotOption.lower() != 'y':
        geneticAlgorithm(cityList, populationSize, amountOfElite,
                         mutationRate, numberOfGenerations, stopOnStagnation, stagnationThreshold)
    else:
        geneticAlgorithmPlot(cityList, populationSize, amountOfElite,
                             mutationRate, numberOfGenerations, stopOnStagnation, stagnationThreshold)


def checkUserInputInt(input):
    try:
        return int(input)
    except ValueError:
        return False


def checkUserInputFloat(input):
    try:
        return float(input)
    except ValueError:
        return False


# MAIN
if __name__ == '__main__':
    userSettings()
