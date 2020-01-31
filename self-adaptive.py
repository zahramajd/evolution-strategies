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

def mutate_sigmas_n_step(sigmas, tau, tau_prime):

    sigma_mul = np.ones((1, sigmas.shape[1]))
    rand_power = random.normalvariate(0, 1)
    mul_val = math.pow(math.e, tau_prime * rand_power)
    for i in range(sigma_mul.shape[1]):
        rand_power_i = random.normalvariate(0, 1)
        mul_val_i = math.pow(math.e, tau * rand_power_i)
        sigma_mul[0,i] = sigma_mul[0,i] * mul_val * mul_val_i * rand_power_i

    sigmas *= sigma_mul
    return sigmas

def mutate_sigmas_one_step(sigmas, tau, tau_prime):

    sigma_mul = np.ones((1, sigmas.shape[1]))
    rand_power = random.normalvariate(0, 1)
    mul_val = math.pow(math.e, tau_prime * rand_power)
    for i in range(sigma_mul.shape[1]):
        rand_power_i = random.normalvariate(0, 1)
        sigma_mul[0,i] = sigma_mul[0,i] * mul_val * rand_power_i

    sigmas *= sigma_mul
    return sigmas

def mutate_offsprings(offsprings, sigmas):
    mutated_offsprings = np.add(offsprings, sigmas)
    return mutated_offsprings

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
landa = 1400 
mu = 200
tau = 1 / math.sqrt(2*math.sqrt(n))
tau_prime = 1/ math.sqrt(2*n)
number = 2
# one or n

individuals = initialization(n=n, mu=mu, max_value=max_range)
sigmas = initialization(n=n, mu=mu, max_value=0.01)

gen = 0
gens = []
best_mins = []
while(gen < 500):
    gen = gen +1

    """
        number if 2: local recombination
        number if 2<: global recombination
    """
    offsprings = discrete_recombination(individuals=individuals, landa=landa, number=2)
    new_sigmas = intermediary_recombination(individuals=sigmas, landa=landa, number=2)


    """
        uncorrelated mutation with one step sizes
        uncorrelated mutation with n step sizes
    """
    mutated_sigmas = mutate_sigmas_one_step(new_sigmas, tau, tau_prime)
    mutated_offsprings = mutate_offsprings(offsprings, mutated_sigmas)


    """
        mu + landa
    """
    all_ind_offspring = np.concatenate((individuals, mutated_offsprings), axis=0)
    all_ind_offspring_fitness = fitness(individuals=all_ind_offspring, function=function)
    best_indices = np.argpartition(all_ind_offspring_fitness, mu)
    new_generation = all_ind_offspring[best_indices,:]
  
    individuals = new_generation
    sigmas = mutated_sigmas

    gens.append(gen)
    print(gen)
    best_mins.append(np.min(fitness(individuals=individuals, function=function)))
    # print(np.min(fitness(individuals=individuals, function=ackley)))

print('last best individual',best_mins[-1])
plt.plot(gens, best_mins)
plt.ylabel('best individual fitness')
plt.xlabel('generation')
plt.title(label='Ackley, one step, both local, mu='+str(mu)+' , last best '+str(best_mins[-1]))
plt.savefig('SA.png')