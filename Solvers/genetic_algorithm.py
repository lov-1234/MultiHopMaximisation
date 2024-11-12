import networkx as nx
import numpy as np
import random
from base_solver import BaseSolver

class GeneticAlgorithmSolver(BaseSolver):

    def __init__(self, graph: nx.Graph, num_k: int, num_d: int, num_subset_for_selection: int, mutation_rate: float,
                 num_generations: int, population_size: int):
        super().__init__(graph)
        # self.g = graph
        self.k = num_k  # Number of k influential nodes
        self.num_d = num_d  # max number of hops permitted
        self.num_sub = num_subset_for_selection     # Subset of the nodes for selection from the population
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.population_size = population_size

    def create_initial_population(self):
        return [np.random.randint(self.g.number_of_nodes(), size=self.k) for _ in range(self.population_size)]

    # I will define fitness by reachability of nodes from a seed node.
    def calculate_fitness(self, candidates):
        nodes_influenced = set(candidates)  # initially only the seed nodes are influenced, a set used because a node
        # can be influenced twice.
        for candidate in candidates:
            influenced = nx.single_source_dijkstra_path_length(self.g, candidate, cutoff=self.num_d).keys()
            # I used dijkstra because we might want to give the edge a weight too to display edge strength in some cases
            nodes_influenced.update(influenced)

        return len(nodes_influenced)  # set length returned because we want to know the number of nodes influenced
        # by some seed node.

    def select_subset_from_population(self, population):
        # Mimicing natural selection here. Therefore, not just choosing the fittest but giving more weightage to
        # higher fitness score
        fitnesses = np.array([self.calculate_fitness(candidates=candidates) for candidates in population])
        subset_selected = random.choices(population, fitnesses, k=self.num_sub)
        return subset_selected

    @staticmethod
    def crossover(parent1, parent2):
        # I will randomly choose between the two lists of parents to produce the offspring
        assert len(parent1) == len(parent2), (f"Length of Candidates differ, parent 1 length : {len(parent1)}, "
                                              f"parent 2 length: {len(parent2)}")

        offspring = list()
        for i in range(len(parent1)):
            if np.random.uniform() <= np.random.uniform():  # Randomly choosing the parent gene
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])

        return offspring

    def mutate(self, candidate):
        num_genes_mutated = int(len(candidate)*self.mutation_rate)   # Given a mutation rate, these many genes in a
        # node will have to be mutated. that is if k = 10, and mutation rate is .3 then 3 nodes will be mutated
        random_idx = np.random.randint(len(candidate), size=num_genes_mutated)
        for i in random_idx:
            node = np.random.randint(self.g.number_of_nodes())
            while node in candidate:
                node = np.random.randint(self.g.number_of_nodes())
            candidate[i] = node

    def pair_idx(self, fit_population):
        return [np.random.choice(len(fit_population), size=2, replace=False) for _ in range(self.population_size)]
        # Generates pairs for reproduction

    def solve(self):
        population = self.create_initial_population()
        for generation in range(self.num_generations):
            subset_for_reproduction = self.select_subset_from_population(population)

            new_population = list()
            for parents in self.pair_idx(subset_for_reproduction):
                offspring = self.crossover(subset_for_reproduction[parents[0]], subset_for_reproduction[parents[1]])
                new_population.append(offspring)

            num_nodes_mutated = int(len(new_population) * self.mutation_rate)  # Given a mutation rate, these many
            # candidates will have a mutation
            random_idx = np.random.randint(len(new_population), size=num_nodes_mutated)
            for i in random_idx:
                self.mutate(new_population[i])

            population = new_population

        final_fitness = [self.calculate_fitness(candidates=candidates) for candidates in population]
        best_solution = population[final_fitness.index(max(final_fitness))]
        return best_solution


if __name__ == '__main__':
    # g = nx.erdos_renyi_graph(200, 0.05)
    g = nx.barabasi_albert_graph(200, 4)
    solver = GeneticAlgorithmSolver(g, 2, 3, 4, .3, 100, 20)
    best_sol = solver.solve()

