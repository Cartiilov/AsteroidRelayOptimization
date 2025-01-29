from deap import base, creator, tools, algorithms
from scipy.spatial.distance import pdist
import numpy as np
import random
import matplotlib.pyplot as plt

inputFileName = './data/input.txt'
outputFileName = './data/output.txt'

def readData(inputFileName: str):
    lines = []
    with open(inputFileName, "r") as f:
        lines = f.readlines()
    
    relayNum, xDim, yDim, maxDist = map(float, lines[0].split())
    coordinates = np.array([list(map(float, line.split())) for line in lines[1:]])
    
    return int(relayNum), xDim, yDim, maxDist, coordinates

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
    print("dmax ", dmax)
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the team positions
    plt.scatter(team_positions[:, 0], team_positions[:, 1], c='blue', label='Zespoły')
    
    # Plot the relay station positions
    plt.scatter(relay_positions[:, 0], relay_positions[:, 1], c='red', label='Nadajniki')
    
    # Draw circles with radius dmax around relay positions
    for relay in relay_positions:
        circle = plt.Circle((relay[0], relay[1]), dmax, color='red', fill=False, linestyle='--')
        plt.gca().add_patch(circle)
    
    # Set plot labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Finalne koordynaty nadajników dla zadanego rozmieszczenia zespołów')
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True)
    
    # Set limits for the plot to match the area
    plt.xlim(0, a)
    plt.ylim(0, b)
    
    # Save the plot to a file
    plt.savefig('./data/plot')


def plot_fitness_over_generations(avg_f1_values, avg_f2_values):
    generations = range(len(avg_f1_values))
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_f1_values, label='Maksymalna odległość od najbliższego przekaźnika')
    plt.plot(generations, avg_f2_values, label='Średnia odległość między wszystkimi parami przekaźników')
    plt.xlabel('Pokolenie')
    plt.ylabel('Wartość dostosowania')
    plt.title('Średnia wartość dostosowania osobników z populacji przez pokolenia')
    plt.legend()
    plt.grid(True)
    plt.savefig('./data/plot_fitness')

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
        individual[i] = max(0, min(individual[i], a))
        individual[i + 1] = max(0, min(individual[i + 1], b))
    return individual

def setup_toolbox(k, a, b, coordinates, dmax):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("individual_near_teams", generate_near_teams, coordinates=coordinates, max_offset=dmax, k=k, a=a, b=b)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.individual_near_teams)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate, coordinates=coordinates, dmax=dmax)
    toolbox.decorate("mutate", tools.DeltaPenalty(lambda ind: check_bounds(ind, a, b), float('inf')))
    return toolbox

def is_pareto_equal(front1, front2, threshold):
    if len(front1) != len(front2):
        return False
    for ind1, ind2 in zip(front1, front2):
        for v1, v2 in zip(ind1.fitness.values, ind2.fitness.values):
            if abs(v1 - v2) > threshold:
                return False
    return True

def run_evolution(toolbox, max_generations, mu, lambda_, cxpb, mutpb):
    population = toolbox.population(n=mu)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pareto_fronts_history = []
    avg_f1_values = []
    avg_f2_values = []

    for gen in range(max_generations):
        # Generate offspring
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        fits = list(toolbox.map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        # Combine parents and offspring, and select the next generation
        population[:] = toolbox.select(population + offspring, mu)

        # Ensure all fitness values are valid
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # Get the first Pareto front
        pareto_fronts = tools.sortNondominated(population, len(population), first_front_only=True)
        current_pareto_front = pareto_fronts[0]
        pareto_fronts_history.append(current_pareto_front)

        # Check if the entire population belongs to the first Pareto front
        all_in_front = len(current_pareto_front) == len(population)

        # Print progress every 10 generations
        if gen % 10 == 0:
            overall_fitness1 = np.mean([ind.fitness.values[0] for ind in population])
            overall_fitness2 = np.mean([ind.fitness.values[1] for ind in population])
            print(f"Generation {gen}")
            print(f"Population Fitness Averages: {overall_fitness1}, {overall_fitness2}")
        
        avg_f1 = np.mean([ind.fitness.values[0] for ind in population])
        avg_f2 = np.mean([ind.fitness.values[1] for ind in population])
        avg_f1_values.append(avg_f1)
        avg_f2_values.append(avg_f2)
        # Stop if all individuals are in the first Pareto front
        
    if all_in_front:
        print(f"Wszystkie osobniki należą do pierwszego frontu pareto, względem znalezionych rozwiązań po pokoleniu {gen}.")
            
    # Update the Hall of Fame with the best solutions
    hall_of_fame.update(current_pareto_front)

    return hall_of_fame[0], avg_f1_values, avg_f2_values


def save_results(best_individual, outputFileName):
    relay_positions = [(best_individual[i], best_individual[i+1]) for i in range(0, len(best_individual), 2)]
    with open(outputFileName, 'w') as f:
        for x, y in relay_positions:
            f.write(f"{x} {y}\n")

def main():
    k, a, b, dmax, coordinates = readData(inputFileName)
    toolbox = setup_toolbox(k, a, b, coordinates, dmax)
    best_individual, avg_f1_values, avg_f2_values = run_evolution(
        toolbox,
        max_generations=100,  # Set an upper limit to prevent infinite loops
        mu=100,
        lambda_=200,
        cxpb=0.7,
        mutpb=0.2,
    )
    save_results(best_individual, outputFileName)
    plot_positions(inputFileName, outputFileName)
    plot_fitness_over_generations(avg_f1_values, avg_f2_values)


if __name__ == "__main__":
    main()
