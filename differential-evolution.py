import numpy as np
import math
import random
import matplotlib.pyplot as plt


def initialization(n, mu, max_value):
    individuals = np.random.uniform(low=-1 * max_value, high=max_value, size=(mu, n))
    return individuals

def fitness(individuals, function):
    individuals_fitness = np.apply_along_axis(function, axis=1, arr=individuals)
    return individuals_fitness

def crossover(individuals, mutants, cr):
	trials = np.zeros_like(individuals)
	for i in range(individuals.shape[0]):
		for a in range(individuals.shape[1]):
			random_parent = random.normalvariate(0, 1)
			if(random_parent > cr):
				trials[i,a] = individuals[i, a]
			else: trials[i,a] = mutants[i, a]
		random_position = random.randint(0, individuals.shape[1]-1)
		trials[i, random_position] = individuals[i, a]
	return trials

def make_mutants(individuals, f_val):
	mutants = np.zeros_like(individuals)
	for i in range(individuals.shape[0]):
		choosed = individuals[np.random.choice(individuals.shape[0], 3, replace=False), :]
		mutants[i] = choosed[0] + f * (choosed[1] - choosed[2]) 
	return mutants

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
# function = ackley
# max_range = 32.768

# Rastrigin
# function = rastrigin
# max_range = 5.12

# Schwefel
function = schwefel
max_range = 500

n = 30
cr = 0.5
mu = 50
f = 0.2

individuals = initialization(n=n, mu=mu, max_value=max_range)


gen = 0
gens = []
best_mins = []
while(gen<700):
	gen = gen+1
	mutants = make_mutants(individuals=individuals, f_val=f)
	trials = crossover(individuals, mutants, cr)

	individuals_fitness = fitness(individuals, function)
	trials_fitness = fitness(trials, function)

	next_generation = np.zeros_like(individuals)
	for i in range(individuals.shape[1]):
		if(individuals_fitness[i]<trials_fitness[i]):
			next_generation[i] = individuals[i]
		else: next_generation[i] = trials[i]
	
	individuals = next_generation
	best_mins.append(np.mean(fitness(individuals=individuals, function=function)))
	gens.append(gen)
    # best_mins.append(np.min(fitness(individuals=individuals, function=ackley)))

print('last best individual',best_mins[-1])
plt.plot(gens, best_mins)
plt.ylabel('best individual fitness')
plt.xlabel('generation')
plt.title(label='Schwefel, F='+str(f)+' ,cr='+str(cr)+' , mu='+str(mu)+' , last best '+str(best_mins[-1]))
plt.savefig('DEshww.png')