import numpy as np
import math
import random
import statistics
import matplotlib.pyplot as plt

def initialization(n, mu, max_value):
    individuals = np.random.uniform(low=-1 * max_value, high=max_value, size=(mu, n))
    return individuals

def fitness(individuals, function):
    individuals_fitness = np.apply_along_axis(function, axis=1, arr=individuals)
    return individuals_fitness

def change_velocities(individuals, velocities, bests, whole_best, w, phi1, phi2):
    whole_best = whole_best.reshape(-1, 1).T
    whole_best1 = whole_best
    for i in range(individuals.shape[0]-1):
        whole_best = np.concatenate((whole_best, whole_best1), axis=0)

    u1 = np.random.uniform(low=0, high=1, size=(individuals.shape[0], individuals.shape[1]))
    u2 = np.random.uniform(low=0, high=1, size=(individuals.shape[0], individuals.shape[1]))
    
    velocities = w * velocities + phi1 * np.multiply(u1, bests - individuals) + phi2* np.multiply(u2, whole_best - individuals)
    return velocities

def mutate(individuals, velocities):
    return np.add(individuals, velocities)

def update_bests(individuals, bests, individuals_fitness, bests_fitness):
    for i in range(individuals_fitness.shape[0]):
        if(individuals_fitness[i]<bests_fitness[i]):
            bests[i] = individuals[i]
    return bests

def ackley(x):
	firstSum = 0.0
	secondSum = 0.0
	for c in x:
		firstSum += c**2.0
		secondSum += math.cos(2.0*math.pi*c)
	n = float(len(x))
	return -20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e

def rastrigin(x):
    sum = 0
    for c in x:
        if (c>5.12 or c<-5.12):
            sum += 10* (c**2.0)
        else:
            sum += (c**2.0) - 10*math.cos(2*math.pi*c)
    n = float(len(x))
    return 10*n + sum

def schwefel(x):
    sum = 0
    for c in x:
        if (c>500 or c<-500):
            sum += 0.02* (c**2.0)
        else:
            sum += -1 * c * math.sin(math.sqrt(math.fabs(c)))
    n = float(len(x))
    return 418.9829*n + sum


# Ackley
function = ackley
max_range = 32.768

# Rastrigin
# function = rastrigin
# max_range = 5.12

# Schwefel
# function = schwefel
# max_range = 500

n = 30
mu = 50
w = 0.6
phi1 = 0.01
phi2 = 0.1

individuals = initialization(n=n, mu=mu, max_value=max_range)
# velocities = initialization(n=n, mu=mu, max_value=0.01)
velocities = np.zeros((mu, n))
bests = individuals
whole_best =bests[np.argmin(fitness(individuals=bests, function=function))]

gen = 0
gens = []
best_mins = []
while(gen < 150):
    gen = gen + 1

    velocities = change_velocities(individuals, velocities, bests, whole_best, w, phi1, phi2)

    individuals = mutate(individuals, velocities)

    individuals_fitness = fitness(individuals=individuals, function=function)
    bests_fitness = fitness(individuals=bests, function=function)
    bests = update_bests(individuals, bests, individuals_fitness, bests_fitness)

    whole_best =bests[np.argmin(bests_fitness)]

    gens.append(gen)
    best_mins.append(np.min(fitness(individuals=individuals, function=function)))


print('last best individual',best_mins[-1])
plt.plot(gens, best_mins)
plt.ylabel('best individual fitness')
plt.xlabel('generation')
plt.title(label='Ackley, w='+str(w)+' ,phi1='+str(phi1)+' ,phi2='+str(phi2) +' , mu='+str(mu)+' , last best '+str(best_mins[-1]))
plt.savefig('PSO.png')