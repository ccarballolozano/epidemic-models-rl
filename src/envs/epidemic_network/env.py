import random

import gymnasium as gym
import networkx as nx


class SISNetworkEnv(gym.Env):

    def __init__(self, size: int = 20, density: float = 0.1):
        self.size = size
        self.density = density

        self._graph = self._build_graph()
        self._state = 

    def _build_graph(self) -> nx.Graph:
        graph = nx.Graph()
        for i in range(self.size):
            for j in range(self.size - i):
                if i != j and random.random() < self.density:
                    graph.add_edge(i, j)
        return graph

