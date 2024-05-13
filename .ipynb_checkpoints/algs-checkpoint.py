import numpy as np
from numpy import pi,sin,cos,sqrt,sinc
import matplotlib.pyplot as plt

# Defauld numpy rng
rng = np.random.default_rng()

#------------------------------------------------------------------------ 
#                           Global Functions
#------------------------------------------------------------------------ 

#----------------------- Initialization Functions ----------------------- 

# Initializes population within specified bounds
def initialize(dims, bounds, fit_eval, rng=rng):
    pop_size = dims[0]
    ind_size = dims[1]
    pop_init = np.zeros((pop_size,ind_size))

    # Initial xi population
    for i in range(pop_size):
        pop_init[i,:] = rand_vec(ind_size,bounds)

    # compute fitness of initial population
    fit = fit_eval(pop_init)

    return pop_init,fit

# generates a random vector with each coordinate uniformly 
# sampled, with different bounds on each coordinate
def rand_vec(size, bounds, rng=rng):
    vec = np.zeros(size)
    for i in range(size):
        lb = bounds[i][0]
        ub = bounds[i][1]
        vec[i] = rng.uniform(low=lb,high=ub)
    return vec

#---------------------- Parent Selection Operators ----------------------

# Uniform selection 
def unif_select(pop_size, pool_size=None, rng=rng):
    if pool_size is None:
        # Default mating pool size is population size
        pool_size = pop_size
    return rng.choice(pool_size, size=pool_size, replace=True)

# tournament selection
def tournament_select(fit, t_size, pool_size=None, rng=rng):
    pop_size = len(fit)
    if pool_size is None:
        # Default mating pool size is population size
        pool_size = pop_size
        
    pool = np.zeros(pool_size, dtype=int)
    
    for i in range(pool_size):
        participants = rng.choice(pop_size, size=t_size, replace=False)
        best_ind = np.argmax(fit[participants])
        pool[i] = participants[best_ind]

    return pool

#------------------ Crossover Operators (real vectors) ------------------ 

# Uniform Crossover (traditional)
def unif_cross1(p1, p2, param=0.5, rng=rng):
    n = len(p1)
    c1 = np.zeros(n)
    c2 = np.zeros(n)
    rand = rng.uniform(size=n)

    for i in range(n):
        if rand[i] < param:
            c1[i] = p1[i]
            c2[i] = p2[i]
        else:
            c1[i] = p2[i]
            c2[i] = p1[i]

    return c1, c2

# Uniform Crossover, prevents duplicating a parent,
# returns a single child.
def unif_cross2(p1, p2, param=0.5, rng=rng):
    n = len(p1)
    child = np.zeros(n)
    rand = rng.uniform(size=n)

    # uniformly inherit alleles from either parent
    for i in range(n):
        if rand[i] < param:
            child[i] = p1[i]
        else:
            child[i] = p2[i]

    # if child dim > 1 inherit at least one allele from each
    # parent to prevent duplicates
    if n > 1:
        g = rng.choice(n,2,replace=False)    # Indices for deterministic inheritance
        child[g[0]] = p1[g[0]]
        child[g[1]] = p2[g[1]]

    return child

# Intermediary crossover, whole arithmetic
# recombination
def inter_cross(p1, p2, alpha):
    c1 = alpha*p1 + (1-alpha)*p2
    c2 = alpha*p2 + (1-alpha)*p1

    return c1, c2

# Blend crossover (blx-a)
def blend_cross(p1, p2, a=0.5, rng=rng):
    n = len(p1)
    c = np.zeros(n)

    for i in range(n):
        d = np.abs(p2[i] - p1[i])            # coordinate distance
        lb = np.min([p1[i], p2[i]]) - a*d    # child range lower bound
        ub = np.max([p1[i], p2[i]]) + a*d    # child range upper bound
        
        # Create offspring
        c[i] = rng.uniform(low=lb, high=ub)
        
    return c

#------------------ Mutation Operators (real vectors) ------------------

# Gaussian perturbation
# m_steps is a list of standard deviations, one for each
# coordinate of the c vector. This is to account for the different 
# bounds on each coordinate.
def gaussian_mutation(c, m_steps, rng=rng):
    n = len(c)
    for i in range(n):
        m_step = m_steps[i]
        noise = rng.normal(scale=m_step)
        c[i] += noise
    return c


#------------------------------------------------------------------------ 
#                        Real-coded Genetic Algorithm
#------------------------------------------------------------------------ 
# Main function for a real coded genetic algorithm
# Parent selection = tournament selection
# Crossover method = blx-a
# mutation method = Gaussian noise
# Survivor selection uses elitism
def evolve1(dims, bounds, maxIter, fit_eval, m_steps, p_mutate=0.2, p_cross=0.7, tmnt_size=3, elitism=0.1):
    pop, fit = initialize(dims,bounds,fit_eval)
    pop_size, sol_dim = np.shape(pop)
    ind = np.arange(0,pop_size,1)
    
    best_fit_vector = np.zeros((maxIter, sol_dim))
    global_best_fit = np.zeros(maxIter)
    mean_fit = np.zeros(maxIter)

    # Start main loop
    for iter in range(maxIter):
        # Sort population by fitness
        temp = np.column_stack((fit, ind))
        # Indices sorted by fitness in ascending order (for maximization)
        temp = temp[np.argsort(temp[:,0])]
    
        # Get Indices for elitism. Elitism
        # parameter is a percentage of the population.
        if elitism != False:
            elitism_cap = int(pop_size - np.ceil(pop_size*elitism)) # minimum fitness index
            elite_ind = range(elitism_cap, pop_size)
            elite_ind = np.array(temp[elite_ind, 1], dtype=int) # best indices

        # Parent selection, by tournament
        parent_pool = tournament_select(fit, t_size=tmnt_size)

        # Reproduction, get children
        children = reproduction1(pop,parent_pool,fit,m_steps,bounds=bounds,p_cross=p_cross,p_mutate=p_mutate)

        # Evaluate fitness
        print(f"Running fit eval {iter+1}")
        fit_children = fit_eval(children)

        # Replace worst children with best parents
        if elitism != False:
            temp = np.column_stack((fit_children, ind))
            # Indices sorted by fitness in descending order (for maximization)
            # This way the worst children are at the bottom.
            temp = temp[np.argsort(-temp[:,0])]
            ind_weakest = np.array(temp[elitism_cap:pop_size, 1], 
                                      dtype=int) # Indices of worst children

            # Replace worst with previous best
            children[ind_weakest] = pop[elite_ind]
            fit_children[ind_weakest] = fit[elite_ind]
            
        # Setup next iteration
        pop = children
        fit = fit_children
        # Record mean and best fitness
        mean_fit[iter] = np.mean(fit)
        global_best_fit[iter] = np.max(fit)
        best_fit_vector[iter] = pop[np.argmax(fit)]

    #----------------------------------------------------------------------------------------------------------------------------
    # Plot best and mean fitness
    plt.clf()
    plt.plot(range(maxIter), mean_fit, label="mean fit")
    plt.plot(range(maxIter), global_best_fit, label="best fit")
    plt.title("Fitness over time")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("fit_es2.png")

    return best_fit_vector, global_best_fit


# Reproduction function, applies crossover and mutation.
# Returns a list of children of the same size as the previous generation
def reproduction1(pop, parents, fit, m_steps, bounds=None, p_cross=0.7, p_mutate=0.2, rng=rng):
    pop_size, sol_dim = np.shape(pop)
    rand = rng.uniform(size=2)
    children = np.zeros((pop_size,sol_dim))

    # Main loop, one child per iteration
    for i in range(pop_size):
        # Select parents
        p1_ind = None
        p2_ind = None
        while p1_ind == p2_ind:
            # Ensure Parents are distinct
            p1_ind,p2_ind = rng.choice(parents, 2, replace=False)

        p1,p2 = pop[[p1_ind,p2_ind]]    # Actual parents
        f1,f2 = fit[[p1_ind,p2_ind]]    # Parent's fitness
        # Apply crossover
        if rand[0] <= p_cross:
            child = blend_cross(p1,p2)
        else:
            # Copy fittest parent
            child = np.copy(p1) if (f1>f2) else np.copy(p2)

        # Apply mutation
        if rand[1] <= p_mutate:
            # m_steps are the standard deviations of the distributions
            child = gaussian_mutation(child,m_steps)

        # Enforce Bounds
        if bounds is not None:
            for k in range(sol_dim):
                lb = bounds[k][0]    # k-th coordinate lower bound
                ub = bounds[k][1]    # k-th coordinate upper bound
                if child[k] > ub:
                    child[k] = ub
                elif child[k] < lb:
                    child[k] = lb
        
        children[i] = child

    return children
    
#------------------------------------------------------------------------ 
#                       Differential evolution
#------------------------------------------------------------------------ 
# Main function, includes initialization. dims is a tuple (pop_size, sol_dim),
# bounds is a list of tuples.
# pop is a 2D array where the dim 0 is the population size
# and dim 1 is the solution's domain dimensions.
def diff_evolve(dims, bounds, maxIter, F, c_rate, fit_eval):
    pop, fit = initialize(dims,bounds,fit_eval)
    dims = np.shape(pop)
    best_fit_vector = np.zeros((maxIter, dims[1]))
    global_best_fit = np.zeros(maxIter)
    mean_fit = np.zeros(maxIter)

    # Main loop
    for iter in range(maxIter):
        # Get new vector population
        children = diff_reproduction(pop,bounds,F,c_rate)

        # Evaluate children fitness
        fit_child = fit_eval(children)

        # Elitist replacement
        for i in range(dims[0]):
            # This is for maximization, inequality must be
            # inverted for minimization
            if fit[i] < fit_child[i]:
                pop[i] = children[i]
                fit[i] = fit_child[i]

        # Update mean and best fitness
        mean_fit[iter] = np.mean(fit)
        best_ind = np.argmax(fit)
        best_fit_vector[iter] = pop[best_ind]
        global_best_fit[iter] = fit[best_ind]

    # Plot fitness
    plt.clf()
    plt.plot(range(maxIter), mean_fit, label="mean fit")
    plt.plot(range(maxIter), global_best_fit, label="best fit")
    plt.title("Fitness over time")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("diff_fit.png")

    return best_fit_vector, global_best_fit


def diff_reproduction(pop, bounds, F, c_rate):
    # Create mutant population
    mutants = diff_mutation(pop,F)

    # Apply crossover to get next generation
    dims = np.shape(pop)
    children = np.zeros(dims)
    for i in range(dims[0]):
        ci = unif_cross2(pop[i], mutants[i], param=c_rate)
        
        # Enforce bounds
        for k in range(dims[1]):
            lb = bounds[k][0]    # k-th coordinate upper bound
            ub = bounds[k][1]    # k-th coordinate lower bound
            if ci[k] > ub:
                ci[k] = ub
            elif ci[k] < lb:
                ci[k] = lb

        children[i] = ci

    return children


# Creates mutant population for differential evolution
def diff_mutation(pop, F, rng=rng):
    pop_size = len(pop)
    ind_size = len(pop[0])
    mutants = np.zeros((pop_size, ind_size))

    for i in range(pop_size):
        inds = rng.choice(pop_size,size=3,replace=False)
        p = F*(pop[inds[1]] - pop[inds[2]])
        mutants[i] = pop[inds[0]] + p

    return mutants

#------------------------------------------------------------------------ 
#                       Particle Swarm Optimization
#------------------------------------------------------------------------ 
# Main function, initializes and runs the algorithm. Bounds is specified
# as a list of tuples. dims is a tuple specifying population size, and 
# individual vector dimension.
def pso_alg(dims, bounds, maxIter, w, phi1, phi2, fit_eval):
    p_init,fit = pso_init(dims,bounds,fit_eval)
    sol,fit = pso_evolve(p_init,fit,bounds,maxIter, w, phi1, phi2, fit_eval)
    return sol,fit

#------------------------------------------------------------------------
# p_init is a 3D array where dim 0 is the population size,
# dim 1 is the given vector and dim 2 are the vector's dimensions.

def pso_evolve(p_init, fit, bounds, maxIter, w, phi1, phi2, fit_eval):
    # It first needs to be initialized, calculating  also the fitness of the
    # initial population.
    pop = p_init
    dims = np.shape(pop)
    best_fit_vector = np.zeros((maxIter, dims[2]))
    global_best_fit = np.zeros(maxIter)
    mean_fit = np.zeros(maxIter)
    personal_best_fit = fit

    # main loop
    for iter in range(maxIter):
        # Calculate mean fitness
        mean_fit[iter] = np.mean(fit)

        # Update global best
        best_ind = np.argmax(fit)
        global_best_fit[iter] = fit[best_ind]
        best_fit_vector[iter] = pop[best_ind,0,:]

        c_vec = best_fit_vector[iter]
        # Updates population (xi,vi) -> (xi',vi')
        pso_mutation(pop,bounds,c_vec,w,phi1,phi2)
        
        # Update fitness
        individuals = pop[:,0,:]
        fit = fit_eval(individuals)

        # Update personal bests
        for i in range(dims[0]):
            # This condition is for maximization problems, the 
            # inequality has to be inverted for minimization
            if personal_best_fit[i] < fit[i]:
                # bi' -> xi' 
                pop[i,2,:] = pop[i,0,:]
                personal_best_fit[i] = fit[i]
        
        # Solution plots, mostly for debugging
        # if (iter%1 == 0):
        #     plt.clf()
        #     ang = np.linspace(-2*pi,2*pi, 200)
        #     plt.plot(ang, sinc(ang))
        #     plt.scatter(individuals,fit)
        #     plt.savefig("pop{num}".format(num = iter//1))
        
        #     print("best fit vector:",best_fit_vector[iter])
        #     print("best fitness:",global_best_fit[iter])
        #     print("mean fit:",mean_fit[iter])


    # Plot best and mean fitness
    plt.clf()
    plt.plot(range(maxIter), mean_fit, label="mean fit")
    plt.plot(range(maxIter), global_best_fit, label="best fit")
    plt.title("Fitness over time")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("fit.png")
    
    # save data
    #np.savetxt("meanFit.csv",mean_fit,delimiter=",",fmt='%.6e')
    #np.savetxt("bestFit.csv",mean_fit,delimiter=",",fmt='%.6e')
    return best_fit_vector,global_best_fit


# The triplet order is (xi, vi, bi)
# Modifies population in place (only vi and xi)
def pso_mutation(pop, bounds, c_vec, w, phi1, phi2, rng=rng):
    pop_size = np.shape(pop)[0]
    n = np.shape(pop)[2]               # Individual's dimension
    U1 = rng.uniform(size=(n,n))
    U2 = rng.uniform(size=(n,n))
        
    for i in range(pop_size):
        xi = pop[i,0,:]    # candidate solution
        vi = pop[i,1,:]    # velocity
        bi = pop[i,2,:]    # personal best
        pv = bi - xi       # points to personal best
        sv = c_vec - xi    # points to global best
        
        # Update velocity
        vi_new = w*vi + phi1*np.matmul(U1,pv) + phi2*np.matmul(U2,sv)
        pop[i,1,:] = vi_new
        # Update candidate solution
        pop[i,0,:] += vi_new
        # Enforce bounds
        for k in range(n):
            lb = bounds[k][0]    # k-th coordinate upper bound
            ub = bounds[k][1]    # k-th coordinate lower bound
            if pop[i,0,:][k] > ub:
                pop[i,0,:][k] = ub
            elif pop[i,0,:][k] < lb:
                pop[i,0,:][k] = lb

#------------------------------------------------------------------------
# Initializes population within specified bounds
def pso_init(dims, bounds, fit_eval, rng=rng):
    pop_size = dims[0]
    ind_size = dims[1]
    pop_init = np.zeros((pop_size,3,ind_size))

    # Initial xi population
    for i in range(pop_size):
        pop_init[i,0,:] = rand_vec(ind_size,bounds)

    # set bi initial to xi
    individuals = pop_init[:,0,:]
    pop_init[:,2,:] = individuals

    # compute fitness of initial population
    fit = fit_eval(individuals)

    return pop_init,fit

#------------------------------------------------------------------------

            