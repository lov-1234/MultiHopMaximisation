from base_solver import BaseSolver
import networkx as nx
import numpy as np


class SimulatedAnnealing(BaseSolver):
    EPS = 1e-3

    def __init__(self, graph: nx.Graph, initial_temperature, num_d, num_k, cooling_rate, max_iters):
        super().__init__(graph)
        self.temp = initial_temperature
        self.num_d = num_d
        self.num_k = num_k
        self.cooling_rate = cooling_rate
        self.max_iters = max_iters

    def calculate_fitness(self, candidates):
        nodes_influenced = set(candidates)  # initially only the seed nodes are influenced, a set used because a node
        # can be influenced twice.
        for candidate in candidates:
            influenced = nx.single_source_dijkstra_path_length(self.g, candidate, cutoff=self.num_d).keys()
            # I used dijkstra because we might want to give the edge a weight too to display edge strength in some cases
            nodes_influenced.update(influenced)

        return len(nodes_influenced)  # set length returned because we want to know the number of nodes influenced
        # by some seed node.

    def create_initial_solution(self):
        return np.random.randint(self.g.number_of_nodes(), size=self.num_k)

    def initialze_neighbourhood(self, sol):
        neighbour = sol[:]
        replacement_idx = np.random.randint(len(neighbour))
        new_seed = np.random.randint(self.g.number_of_nodes())
        while new_seed in neighbour:
            new_seed = np.random.randint(self.g.number_of_nodes())
        neighbour[replacement_idx] = new_seed
        return neighbour

    def update_temperature(self, iteration):
        self.temp = self.temp * self.cooling_rate ** (iteration + 1)

    def acceptance_probability(self, neighbour, current):
        if self.calculate_fitness(neighbour) > self.calculate_fitness(current):
            return 1.
        return np.exp((self.calculate_fitness(neighbour) - self.calculate_fitness(current)) / self.temp)

    def solve(self):
        current_solution = self.create_initial_solution()
        best_sol = current_solution
        best_fitness = self.calculate_fitness(current_solution)
        for iters in range(self.max_iters):
            neighbour = self.initialze_neighbourhood(current_solution)
            neighbour_fitness = self.calculate_fitness(neighbour)
            if self.acceptance_probability(neighbour, current_solution) > 0.5:
                current_solution = neighbour

            if neighbour_fitness > best_fitness:
                best_sol = neighbour
                best_fitness = neighbour_fitness

            self.update_temperature(iters)

            if self.temp < self.EPS:
                break

        return best_sol


if __name__ == '__main__':
    pass
