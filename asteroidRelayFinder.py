from  deap import base, creator, tools, algorithms
from scipy.spatial.distance import pdist
import numpy as np
import random
import matplotlib.pyplot as plt


inputFileName = './data/example.txt'
outputFileName = './data/output.txt'

def readData(inputFileName: str):
    lines = []
    with open(inputFileName, "r") as f:
        lines = f.readlines()
    
    xDim, yDim, relayNum, maxDist = map(int, lines[0].split())
    coordinates = np.array([list(map(float, line.split())) for line in lines[1:]])
    
    return xDim, yDim, relayNum, maxDist, coordinates

def readOutput(outputFileName: str):
    with open(outputFileName, "r") as f:
        lines = f.readlines()
    
    relay_positions = np.array([list(map(float, line.split())) for line in lines])
    return relay_positions
    
    
def evaluate(individual, coordinates, dmax):
    relay_positions = np.array(individual).reshape(-1, 2)
    distances = np.linalg.norm(coordinates[:, np.newaxis, :] - relay_positions[np.newaxis, :, :], axis=2)
    min_distances = np.min(distances, axis=1)

    if np.any(min_distances > dmax):
        return float('inf'), float('inf')

    max_distance = np.max(min_distances)

    relay_distances = pdist(relay_positions)
    total_distance = np.sum(relay_distances)
    avg_distance = total_distance / len(relay_distances) if len(relay_distances) > 0 else 0

    return max_distance, avg_distance


def plot_positions(inputFileName: str, outputFileName: str):
    k, a, b, dmax, team_positions = readData(inputFileName)
    relay_positions = readOutput(outputFileName)
    
    desired_positions = np.loadtxt('./data/desired.txt')

    plt.figure(figsize=(10, 6))
    plt.scatter(team_positions[:, 0], team_positions[:, 1], c='blue', label='Teams')
    plt.scatter(relay_positions[:, 0], relay_positions[:, 1], c='red', label='Relay Stations')
    plt.scatter(desired_positions[:, 0], desired_positions[:, 1], c='green', label='Desired Relay Positions')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Positions of Teams, Relay Stations, and Desired Relay Positions')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, a)
    plt.ylim(0, b)
    plt.savefig('./data/plot')
    
def generate_near_teams(coordinates, max_offset, k, a, b):
    relay_positions = []
    for _ in range(k):
        team_x, team_y = random.choice(coordinates)
        offset_x = random.uniform(-max_offset, max_offset)
        offset_y = random.uniform(-max_offset, max_offset)
        relay_x = max(0, min(team_x + offset_x, a))
        relay_y = max(0, min(team_y + offset_y, b))
        relay_positions.extend([relay_x, relay_y])
    return relay_positions

def check_bounds(individual, a, b):
    for i in range(0, len(individual), 2):
        if individual[i] < 0:
            individual[i] = 0
        elif individual[i] > a:
            individual[i] = a
        if individual[i + 1] < 0:
            individual[i + 1] = 0
        elif individual[i + 1] > b:
            individual[i + 1] = b
    return individual
   
def setup_toolbox(k, a, b, coordinates, dmax):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("individual_near_teams", generate_near_teams, coordinates=coordinates, max_offset=10, k=k, a=a, b=b)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.individual_near_teams)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", evaluate, coordinates=coordinates, dmax=dmax)
    toolbox.decorate("mate", tools.DeltaPenalty(lambda ind: check_bounds(ind, a, b), float('inf')))
    toolbox.decorate("mutate", tools.DeltaPenalty(lambda ind: check_bounds(ind, a, b), float('inf')))
    return toolbox

def run_evolution(toolbox, max_generations, max_no_improve_epochs, improvement_threshold, mu, lambda_, cxpb, mutpb):
    population = toolbox.population(n=100)
    best_fitness = float('inf')
    no_improve_counter = 0
    max_distance_history = []

    for gen in range(max_generations):
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population[:] = toolbox.select(population + offspring, mu)
        current_best_individual = tools.selRandom(population, 1)[0]
        current_best = current_best_individual.fitness.values[0]

        if abs(best_fitness - current_best) > improvement_threshold:
            best_fitness = current_best
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        max_distance_history.append(best_fitness)

        if gen % 10 == 0:
            print(f"Generacja {gen + 1}, najlepszy wynik: {best_fitness}")

        if no_improve_counter >= max_no_improve_epochs:
            print(f"Algorytm zatrzymany po {gen + 1} epokach (brak znaczącej poprawy).")
            break
        
    return tools.selBest(population, 1)[0]

def save_results(best_individual, outputFileName):
    relay_positions = [(best_individual[i], best_individual[i+1]) for i in range(0, len(best_individual), 2)]
    with open(outputFileName, 'w') as f:
        for x, y in relay_positions:
            f.write(f"{x} {y}\n")

def main():
    k, a, b, dmax, coordinates = readData(inputFileName)
    toolbox = setup_toolbox(k, a, b, coordinates, dmax)
    best_individual = run_evolution(toolbox, max_generations=1000, max_no_improve_epochs=10, improvement_threshold=1e-6, mu=100, lambda_=200, cxpb=0.7, mutpb=0.2)
    save_results(best_individual, outputFileName)
            
    plot_positions(inputFileName, outputFileName)

if __name__ == "__main__":
    main()