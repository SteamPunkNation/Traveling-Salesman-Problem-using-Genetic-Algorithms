# Traveling-Salesman-Problem-using-Genetic-Algorithms

## Introduction
In this Python program, the objective was to develop an AI solution using genetic algorithms to efficiently solve the Traveling Salesman Problem (TSP). By employing genetic operators such as selection, crossover, and mutation, the program aims to evolve a population of candidate solutions iteratively, gradually improving their fitness until an optimal or near-optimal route is obtained.

## Functions
1. **distance**: calculates and returns the distance between the current city and the input city. It uses the difference in x and y coordinates to compute the distance, applying the Pythagorean theorem and returning the value.
2. **routeDistance**: calculates and returns the total distance of a given route. It checks if the distance was already calculated, if not, it then calculates the distance by summing up the distances between consecutive cities in the route.
3. **routeFitness**: calculates the fitness of an individual route. If not calculated already it will do so by taking the reciprocal of the route distance (using the function routeDistance).
4. **createRoute**: generates a random route by shuffling the given cities and returns the route.
5. **initialPopulation**: creates an initial population of routes by repeatedly calling createRoute(4) and appending the generated routes to the population list until the population size is reached.
6. **sortRoutes**: calculates the fitness of each route in the population using the Fitness class and returns a sorted list of tuples, where each one contains the index of a route and its fitness value.
7. **selectParents**: selects a subset of routes as “parents” for the next generation based on their ranking in the population. It first adds the top routes (amountOfElite) based on ranking to the selection results list. Then adds the remaining by randomly selecting routes with a higher cumulative percentage fitness, favoring those with higher values.
8. **matingPool**: creates a mating pool by selecting routes from the population based on the indices provides in the selectionResults list. It goes over selectionResults and moves the corresponding routes to the mating pool.
9. **breedOffSpring**: takes two parents, randomly selects a gene range from one parent and creates a child by combining the selected gene range from the first to the remaining in the second.
10. **breedEntirePopulation**: breeds the entire population by selecting a certain number of “elites,” randomly selecting the remaining, and creating children by breeding pairs. The new children are then part of the next generation.
11. **changeCity**: mutates an individual by randomly swapping cities in a route based on mutation rate.
12. **changeCityWholePopulation**: calls changeCity(11) to apply mutation to the entire population.
13. **nextGeneration**: generates the next generation of the population by calling the following, selectParents(7), matingPool(8), breedEntirePopulation(10), and changeCityWholePopulation(12). The results of the previous functions create the next generation of the population.
14. **geneticAlgorithm**: calls initialPopulation(5), nextGeneration(13), and sortRoutes(6) to run the genetic algorithm based on the number of generations and “evolves” the population and tracks the best route found.
15. **geneticAlgorihtmPlot**: has the same functionality as geneticAlgorithm(14) but also calls plotRoute(16).
16. **plotRoute**: plots out the progression chart (distance/generations), the initial route, and the best route.
17. **userSettings**: prompts the user to enter various parameters such as number of cities, population size, mutation rate, number of “elites”, enable/disable stagnation, plotting preferences, generation amount, and calls checkUserInputInt(18) and checkUserInputFloat(19) to ensure invalid inputs are addressed.
18. **checkUserInputInt**: takes input from user and ensures the value is an integer, if not returns false.
19. **checkUserInputFloat**: takes input from user and ensures the value is a float, if not returns false.
