from queue import PriorityQueue

from manim import *
import random as rand
import math

from broad_overview import is_position_valid, distance, determine_num_of_edges, generate_edge_config


# Redundant code
def generate_layout(vertices, bounds, min_distance=0.5):
    n = len(vertices)

    width = 10
    height = 6
    assert(width*height==n)

    x_bound, y_bound = bounds

    x = np.linspace(x_bound, -x_bound, width)
    y = np.linspace(y_bound, -y_bound, height)

    xs, ys = np.meshgrid(x, y)
    points = [(x, y) for x, y in zip(xs.flatten(), ys.flatten())]

    layout = {}

    org_min_dist = min_distance

    for v in vertices:
        x,y = points[v-1]
        placed = False
        attempts = 0

        min_distance = org_min_dist

        while not placed and attempts < 1000:
            x += rand.uniform(-0.3, 0.3)
            y += rand.uniform(-0.3, 0.3)

            if is_position_valid(x, y, [pos for pos in layout.values()], min_distance):
                layout[v] = [x, y, 0]
                placed = True

            attempts += 1

            if attempts % 100 == 0:
                min_distance += 0.5

    return layout


def generate_edges(layout, bounds, min_edges, max_edges, max_edge_length=4.0):

    vertices = list(layout.keys())

    edges = []
    neg_edges = []

    for v in vertices:
        distances = [(u, distance(v, u, layout)) for u in vertices if u != v]
        distances.sort(key=lambda x: x[1])

        isNeg = rand.choices([0, 1], [0.8, 0.2], k=1)[0]

        if not isNeg:
            num_edges = determine_num_of_edges(v, layout, bounds, min_edges, max_edges)
            for u, dist in distances:
                if dist <= max_edge_length:
                    if len([(e1, e2) for e1, e2 in edges if (e1 == v and e2 == u) or (e1 == u and e2 == v)]) == 0:
                        edge = (u, v) if layout[u][0] < layout[v][0] else (v, u)
                        edges.append(edge)

                        if len([e for e in edges if v in e]) >= num_edges:
                            break
        else:
            segment = [u[0] for u in distances[:8]]
            u = rand.choices(segment)[0]
            edges.append((v,u))
            neg_edges.append((v,u))

    return edges, neg_edges


def select_subset(vertices, tau, c):
    size_of_subset = c * tau * math.ceil(math.log(len(vertices)))
    return rand.sample(vertices, size_of_subset)

def compute_weights(edges, neg_edges):
    weights = {}
    possible_pos_weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    possible_neg_weights = [-1, -2, -3, -4, -5]

    for (u,v) in edges:
        if (u,v) not in neg_edges:
            length = rand.sample(possible_pos_weights, 1)[0]
        else:
            length = rand.sample(possible_neg_weights, 1)[0]
        weights[(u,v)] = length

    return weights


def compute_SSSP(source, beta, graph, weights_dict, neg_edges):

    def dijkstra():
        pq = PriorityQueue()
        for (v, d) in dist.items():
            pq.put((d, v))

        visited = set()

        while not pq.empty():
            d, u = pq.get()

            if u in visited or d > dist[u]:
                continue

            visited.add(u)

            for edge in graph.edges.keys():
                if edge in neg_edges or edge[0] != u:
                    continue

                weight = weights_dict[edge]
                if dist[edge[1]] > dist[u] + weight:
                    dist[edge[1]] = dist[u] + weight
                    pq.put((dist[edge[1]], edge[1]))

    def bellman_ford():
        for (u,v) in neg_edges:
            if dist[v] > dist[u] + weights_dict[(u,v)]:
                dist[v] = dist[u] + weights_dict[(u,v)]


    # init distance dict
    dist = {v: math.inf for v in graph.vertices}
    dist[source] = 0

    for rounds in range(beta):
        dijkstra()
        bellman_ford()
    dijkstra()

    return dist


def transpose_data(g: DiGraph, weights_dict, neg_edges):
    transposed_edges = []
    transposed_neg_edges = []
    for (u,v) in g.edges.keys():
        if (u,v) in neg_edges:
            transposed_neg_edges.append((v,u))
        transposed_edges.append((v,u))
    transposed_g = DiGraph(g.vertices, transposed_edges)

    transposed_weights_dict = {}
    for ((u,v), d) in weights_dict.items():
        transposed_weights_dict[(v,u)] = d

    return transposed_g, transposed_weights_dict, transposed_neg_edges

def construct_h_edges(vertices, t_set):
    edges = []
    for v in vertices:
        for t in t_set:
            edges.append((v, t))
            edges.append((t, v))

    return edges

class BetweennessReduction(Scene):
    def construct(self):
        n = 60
        vertices = [i for i in range(n)]

        bounds = (5, 2.5)
        layout = generate_layout(vertices, bounds)

        edges, neg_edges = generate_edges(layout, bounds, 2, 7)

        weights_dict = compute_weights(edges, neg_edges)

        edge_config = generate_edge_config(neg_edges)
        edge_config["tip_config"] = {"tip_length": 0.2, "tip_width": 0.2}

        g = DiGraph(vertices, edges, layout = layout, edge_config=edge_config)#, labels=True)

        self.play(Create(g))
        self.wait()

        # https://www.geeksforgeeks.org/randomly-select-n-elements-from-list-in-python/
        T = select_subset(vertices, 2, 2)

        self.play([FadeToColor(g.vertices[v], BLUE) for v in T])
        self.wait()

        beta = 1
        SSSPs = {}
        STSPs = {}

        for v in T:
            SSSPs[v] = compute_SSSP(v, beta, g, weights_dict, neg_edges)
            transposed_graph, transposed_weights_dict, transposed_neg_edges = transpose_data(g, weights_dict, neg_edges)
            STSPs[v] = compute_SSSP(v, beta, transposed_graph, transposed_weights_dict, transposed_neg_edges)

        #print(SSSPs)
        #print(STSPs)

        self.play(Uncreate(g))
        self.wait()

        h_edges = construct_h_edges(vertices, T)
        h = DiGraph(vertices, h_edges, layout = layout, edge_config={"tip_config": {"tip_length": 0.2, "tip_width": 0.2}})

        self.play(Create(h))
        self.wait(2)