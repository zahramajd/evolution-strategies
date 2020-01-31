import numpy as np
import math
import random
import statistics
import matplotlib.pyplot as plt

# 145
# Sect. 8.2.1


def initialization(n, mu, max_value):
    individuals = np.random.uniform(low=-1 * max_value, high=max_value, size=(mu, n))
    return individuals

def fitness(individuals, function):
    individuals_fitness = np.apply_along_axis(function, axis=1, arr=individuals)
    return individuals_fitness

def discrete_recombination(individuals, landa, number):
    offsprings = []
    for l in range(landa):
        parents = individuals[np.random.choice(individuals.shape[0], number, replace=False), :]
        offspring = np.zeros((individuals.shape[1]))
        for a in range(offspring.size):
            list_alleles = [parent[a] for parent in parents]
            offspring[a] = random.choice(list_alleles)
            
        offsprings.append(offspring)
    _offsprings = np.array(offsprings)        
    return _offsprings

def intermediary_recombination(individuals, landa, number):
    offsprings = []
    for l in range(landa):
        parents = individuals[np.random.choice(individuals.shape[0], number, replace=False), :]
        offspring = np.zeros((individuals.shape[1]))
        for a in range(offspring.size):
            list_alleles = [parent[a] for parent in parents]
            offspring[a] = statistics.mean(list_alleles)
            
        offsprings.append(offspring)
    _offsprings = np.array(offsprings)        
    return _offsprings

def mutate_offsprings(offsprings, sigma):
    mutated_offsprings = offsprings + sigma
    return mutated_offsprings

def update_sigma(sigma, c, ps):

    if ps == 1/5:
        return sigma
    if ps > 1/5:
        return sigma/c
    if ps < 1/5:
        return sigma*c

    return sigma

def compute_successful_mutation(offsprings, mutated_offsprings, function):
    offspring_fitness = fitness(individuals=offsprings, function=function)
    mutated_offsprings_fitness = fitness(individuals=mutated_offsprings, function=function)
    return np.count_nonzero(np.greater(mutated_offsprings_fitness,offspring_fitness))

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
# ra and shew remained
# Rastrigin
# function = rastrigin
# max_range = 5.12

# Schwefel
function = schwefel
max_range = 500

n = 30
c = 0.90
k = 3
sigma = 0.01
landa = 1400
mu = 200

individuals = initialization(n=n, mu=mu, max_value=max_range)

gen = 0
gens = []
best_mins = []
current_k = 0
total_mutation = 0
successful_mutation = 0
while(gen < 300):
    gen = gen +1

    """
        number if 2: local recombination
        number if 2<: global recombination
    """
    offsprings = discrete_recombination(individuals=individuals, landa=landa, number=2)

    
    """
        handle K iterations
    """
    current_k = current_k + 1
    if(current_k % k == 0):
        ps = successful_mutation/total_mutation
        sigma = update_sigma(sigma, c, ps) 

    mutated_offsprings = mutate_offsprings(offsprings, sigma) 
    total_mutation = total_mutation + offsprings.shape[0]
    successful_mutation = successful_mutation + compute_successful_mutation(offsprings, mutated_offsprings, function)

    """
        mu + landa
    """
    all_ind_offspring = np.concatenate((individuals, mutated_offsprings), axis=0)
    all_ind_offspring_fitness = fitness(individuals=all_ind_offspring, function=function)
    best_indices = np.argsort(all_ind_offspring_fitness)[:mu]
    new_generation = all_ind_offspring[best_indices,:]

    """
        mu, landa
    """
    # all_offspring_fitness = fitness(individuals=mutated_offsprings, function=function)
    # best_indices = np.argsort(all_offspring_fitness)[:mu]
    # new_generation = mutated_offsprings[best_indices,:]

    individuals = new_generation

    gens.append(gen)
    print(gen)
    best_mins.append(np.min(fitness(individuals=individuals, function=function)))
    # print(np.min(fitness(individuals=individuals, function=ackley)))

print('last best individual',best_mins[-1])
plt.plot(gens, best_mins)
plt.ylabel('best individual fitness')
plt.xlabel('generation')
plt.title(label='Schwefel, C='+str(c)+' ,k='+str(k)+' , mu='+str(mu)+' , last best '+str(best_mins[-1]))
plt.savefig('ADSh.png')