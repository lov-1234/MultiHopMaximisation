from base_solver import BaseSolver
import networkx as nx
import numpy as np
import heapq


class LazyGreedy(BaseSolver):

    def __init__(self, graph: nx.Graph, num_d, num_k, max_iters=1000):
        super().__init__(graph)
        self.num_d = num_d,
        self.num_k = num_k
        self.max_iters = max_iters  # cutoff value for the greedy algo

    def calculate_fitness(self, candidates):
        nodes_influenced = set(candidates)  # initially only the seed nodes are influenced, a set used because a node
        # can be influenced twice.
        for candidate in candidates:
            influenced = nx.single_source_dijkstra_path_length(self.g, candidate, cutoff=self.num_d).keys()
            # I used dijkstra because we might want to give the edge a weight too to display edge strength in some cases
            nodes_influenced.update(influenced)

        return len(nodes_influenced)  # set length returned because we want to know the number of nodes influenced
        # by some seed node.

    def solve(self):
        best_sol = set()
        influence = 0

        priority_queue = list()

        for node in range(self.g.number_of_nodes()):
            marginal_gain = -self.calculate_fitness({node})  # negative because heapq implements minheap
            heapq.heappush(priority_queue, (marginal_gain, node))  # Pushing each node to the heap with its marginal
            # gain

        iters = 0

        while len(best_sol) < self.num_k and iters < self.max_iters:
            gain, node = heapq.heappop(priority_queue)

            seed_set_influence = self.calculate_fitness(best_sol | {node})
            marginal_gain = seed_set_influence - influence

            if marginal_gain == -gain:
                best_sol.add(node)
                influence = seed_set_influence
            else:
                heapq.heappush(priority_queue, (-marginal_gain, node))

            iters += 1

        return list(best_sol)
