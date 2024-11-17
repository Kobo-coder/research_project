from manim import *
import random as rand
from math import pow, log2

from manim.utils.color.XKCD import REDWINE


def generate_edge_config(neg_edges):
    edge_config = {}
    for e in neg_edges:
        edge_config[e] = {"stroke_color": RED }
    return edge_config

def is_position_valid(x, y, existing_positions, min_dist):
    for pos in existing_positions:
        if np.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2) < min_dist:
            return False
    return True

def generate_vertex_layout(vertices, boundaries, min_distance=0.5):
    x_bound, y_bound = boundaries

    layout = {}

    x = np.linspace(x_bound, -x_bound, 15)
    y = np.linspace(y_bound, -y_bound, 10)

    xs, ys = np.meshgrid(x, y)
    points = [(x, y) for x, y in zip(xs.flatten(), ys.flatten())]

    for v in vertices:
        x,y = points[v-1]
        placed = False
        attempts = 0

        while not placed and attempts < 1000:
            x += rand.uniform(-0.3, 0.3)
            y += rand.uniform(-0.3, 0.3)

            if is_position_valid(x, y, [pos for pos in layout.values()], min_distance):
                layout[v] = [x, y, 0]
                placed = True

            attempts += 1

            if attempts % 100 == 0:
                min_distance *= 0.95

    return layout

def generate_edges(layout, bounds, min_edges_per_vertex=2, max_edges_per_vertex=6, max_edge_length=4.0):
    edges = []
    neg_edges = []
    pos_edges = []
    vertices = list(layout.keys())

    def distance(v1, v2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(layout[v1], layout[v2])))

    for v in vertices:
        distances = [(u, distance(v, u)) for u in vertices if u != v]
        distances.sort(key=lambda x: x[1])

        if not -bounds[0]-1 <= layout[v][0] <= bounds[0]-1 or not -bounds[1]-1 <= layout[v][0] <= bounds[1]-1:
            num_edges = np.random.randint(min_edges_per_vertex, max_edges_per_vertex-1)
        elif not -bounds[0]-0.5 <= layout[v][0] <= bounds[0]-0.5 or not -bounds[1]-0.5 <= layout[v][0] <= bounds[1]-0.5:
            num_edges = np.random.randint(min_edges_per_vertex, max_edges_per_vertex//2)
        else:
            num_edges = np.random.randint(min_edges_per_vertex, max_edges_per_vertex + 1)
        for u, dist in distances:
            if dist <= max_edge_length:
                if len([(e1, e2) for e1, e2 in edges if (e1 == v and e2 == u) or (e1 == u and e2 == v)]) == 0:
                    edge = (min(u,v), max(u,v))
                    edges.append(edge)

                    if rand.choices([0, 1], [0.75, 0.25], k=1)[0]:
                        neg_edges.append(edge)
                    else:
                        pos_edges.append(edge)

                    if len([e for e in edges if v in e]) >= num_edges:
                        break
    return edges, pos_edges, neg_edges

# def partition_graph(layout, edges, y_bound):
#     partitions = [[]] * y_bound*2
#
#     for e in edges:
#         print(e)
#         v1 = layout[e[0]]
#         v2 = layout[e[1]]
#         partitions[int(min((v1[1], int(v2[1]))))+y_bound].append(e[0] if int(v1[1]) <= int(v2[1]) else e[1])
#     return partitions


class BroadOverview(Scene):

    def construct(self):
        num_of_vertices = 150
        vertices = [v for v in range(1, num_of_vertices+1)]

        bounds = (5, 2.5)
        layout = generate_vertex_layout(vertices, bounds)
        edges, pos_edges, neg_edges = generate_edges(layout, bounds)

        graph = Graph(vertices, edges, edge_config=generate_edge_config(neg_edges), layout=layout)

        self.play(Create(graph))
        self.wait()


        self.play([Indicate(graph.edges[e], color=REDWINE) for e in neg_edges])
        self.wait()

        #partitions = partition_graph(layout, edges, bounds[0])

        past = None
        for i in range(int(log2(num_of_vertices))):
            num_in_pass = pow(len(neg_edges), 2/3)
            newly_pos = []
            for j in range(int(num_in_pass)):
                to_be_eliminated = pow(len(neg_edges), 1 / 3)
                cur = Text(f"{j+1}").to_corner(LEFT+UP)
                for e in neg_edges:
                    if rand.choices((0, 1), [(len(neg_edges) - to_be_eliminated) / len(neg_edges), to_be_eliminated], k=1)[0]:
                        neg_edges.remove(e)
                        pos_edges.append(e)
                        newly_pos.append(e)

                if past is None:
                    anim = Write(cur.to_corner(LEFT+UP))
                else:
                    anim = ReplacementTransform(past, cur)
                self.play(
                    AnimationGroup(
                        anim,
                        [FadeToColor(graph.edges[e], color=WHITE) for e in newly_pos],
                        run_time=1
                    )
                )
                self.wait()
                past = cur
