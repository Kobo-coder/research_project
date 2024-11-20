from manim import *
import random as rand
from math import pow, log2, floor
from manim.utils.color.XKCD import REDWINE


def generate_edge_config(neg_edges):
    edge_config = {}
    for e in neg_edges:
        edge_config[e] = { "stroke_color": RED_E }
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

def determine_num_of_edges(vertex, layout, bounds, min_edges_per_vertex, max_edges_per_vertex):
    if not -bounds[0] - 1 <= layout[vertex][0] <= bounds[0] - 1 or not -bounds[1] - 1 <= layout[vertex][0] <= bounds[1] - 1:
        return np.random.randint(min_edges_per_vertex, max_edges_per_vertex - 1)
    elif not -bounds[0] - 0.5 <= layout[vertex][0] <= bounds[0] - 0.5 or not -bounds[1] - 0.5 <= layout[vertex][0] <= bounds[
        1] - 0.5:
        return np.random.randint(min_edges_per_vertex, max_edges_per_vertex // 2)
    else:
        return np.random.randint(min_edges_per_vertex, max_edges_per_vertex + 1)


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

        num_edges = determine_num_of_edges(v, layout, bounds, min_edges_per_vertex, max_edges_per_vertex)

        for u, dist in distances:
            if dist <= max_edge_length:
                if len([(e1, e2) for e1, e2 in edges if (e1 == v and e2 == u) or (e1 == u and e2 == v)]) == 0:
                    edge = (u,v) if layout[u][0] < layout[v][0] else (v,u)
                    #edge = (min(u,v), max(u,v))
                    edges.append(edge)

                    if rand.choices([0, 1], [0.75, 0.25], k=1)[0]:
                        neg_edges.append(edge)
                    else:
                        pos_edges.append(edge)

                    if len([e for e in edges if v in e]) >= num_edges:
                        break
    return edges, pos_edges, neg_edges

def partition_graph(layout, edges, x_bound):
    partitions = [[] for _ in range(x_bound * 2)]
    for e in edges:
        v1 = layout[e[0]]
        v2 = layout[e[1]]

        partitions[int(floor(min(v1[0], v2[0]))) + x_bound].append(e)

    return partitions


def indicate_pass(self, graph, partitions):
    wave_animations = []
    for partition in partitions:
        edge_copies = [
            graph.edges[edge].copy()
            .set_z_index(graph.z_index + 1)
            for edge in partition
        ]

        partition_animations = AnimationGroup(
            *[ShowPassingFlash(
                edge_cop.set_color(BLUE_D).set_stroke(width=4),
                time_width=0.1
            ) for edge_cop in edge_copies],
            lag_ratio=0
        )
        wave_animations.append(partition_animations)

    self.play(
        AnimationGroup(
            *wave_animations,
            lag_ratio=0.05
        ),
        run_time=2
    )


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

        partitions = partition_graph(layout, edges, bounds[0]+1)

        past = None
        for i in range(int(log2(num_of_vertices))):
            num_in_pass = pow(len(neg_edges), 2/3)
            newly_pos = []
            for j in range(int(num_in_pass)):
                if not neg_edges:
                    break

                to_be_eliminated = pow(len(neg_edges), 1 / 3)
                rounds = Text("No. of elimination rounds: ", font_size=36).to_corner(LEFT+UP)
                cur = Text(f"{j+1}", font_size=36).next_to(rounds)
                text = VGroup(rounds, cur)

                for e in neg_edges:
                    if rand.choices((0, 1), [(len(neg_edges) - to_be_eliminated) / len(neg_edges), to_be_eliminated], k=1)[0]:
                        neg_edges.remove(e)
                        pos_edges.append(e)
                        newly_pos.append(e)

                if past is None:
                    anim = Write(text)
                else:
                    anim = ReplacementTransform(past, cur)

                self.play(anim)
                indicate_pass(self, graph, partitions)

                self.play(
                    [FadeToColor(graph.edges[e], color=WHITE) for e in newly_pos],
                    run_time=1
                )
                self.wait()
                past = cur
