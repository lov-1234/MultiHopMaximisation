import matplotlib.pyplot as plt
import networkx as nx

class BaseSolver:
    def __init__(self, graph):
        self.g = graph

    def solve(self):
        '''
        To Be overridden
        :return: None
        '''
        pass

    def visualise_graph(self):
        nx.draw(self.g)
        plt.show()
