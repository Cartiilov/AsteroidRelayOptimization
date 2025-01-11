from  deap import base, creator, tools, algorithms
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
    max_distance = 0
    total_distance = 0
    num_pairs = 0

    for x, y in coordinates:
        distances = [np.sqrt((x - px)**2 + (y - py)**2) for px, py in zip(individual[::2], individual[1::2])]
        min_distance = min(distances)
        if min_distance > dmax:
            return float('inf'), float('inf')
        max_distance = max(max_distance, min_distance)
    
    for i in range(0, len(individual), 2):
        for j in range(i + 2, len(individual), 2):
            total_distance += np.sqrt((individual[i] - individual[j])**2 + (individual[i+1] - individual[j+1])**2)
            num_pairs += 1

    avg_distance = total_distance / num_pairs if num_pairs > 0 else 0
    return max_distance, avg_distance


def plot_positions(inputFileName: str, outputFileName: str):
    k, a, b, dmax, team_positions = readData(inputFileName)
    relay_positions = readOutput(outputFileName)
    
    # Read desired positions
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

            
def main():
    random.seed(42)
    k, a, b, dmax, coordinates = readData(inputFileName)

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, max(a, b))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2*k)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate, coordinates=coordinates, dmax=dmax)
    
    
    def check_bounds(individual):
        for i in range(0, len(individual), 2):
            if individual[i] < 0:
                individual[i] = 0
            elif individual[i] > a:
                individual[i] = a
            if individual[i+1] < 0:
                individual[i+1] = 0
            elif individual[i+1] > b:
                individual[i+1] = b
        return individual

    toolbox.decorate("mate", tools.DeltaPenalty(check_bounds, float('inf')))
    toolbox.decorate("mutate", tools.DeltaPenalty(check_bounds, float('inf')))

    population = toolbox.population(n=100)
    algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.2, ngen=50, verbose=True)

    best_individual = tools.selBest(population, 1)[0]
    relay_positions = [(best_individual[i], best_individual[i+1]) for i in range(0, len(best_individual), 2)]

    with open(outputFileName, 'w') as f:
        for x, y in relay_positions:
            f.write(f"{x} {y}\n")
            
    plot_positions(inputFileName, outputFileName)

if __name__ == "__main__":
    main()