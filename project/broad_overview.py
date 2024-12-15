import math
import random as rand
from math import pow, floor

from manim import *


def generate_edge_config(neg_edges):
    edge_config = {}
    for e in neg_edges:
        edge_config[e] = { "stroke_color": RED }
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

def determine_num_of_edges(vertex, layout, bounds, min_edges_per_vertex, max_edges_per_vertex):
    if not -bounds[0] - 1 <= layout[vertex][0] <= bounds[0] - 1 or not -bounds[1] - 1 <= layout[vertex][0] <= bounds[1] - 1:
        return np.random.randint(min_edges_per_vertex, max_edges_per_vertex - 1)
    elif not -bounds[0] - 0.5 <= layout[vertex][0] <= bounds[0] - 0.5 or not -bounds[1] - 0.5 <= layout[vertex][0] <= bounds[
        1] - 0.5:
        return np.random.randint(min_edges_per_vertex, max_edges_per_vertex // 2)
    else:
        return np.random.randint(min_edges_per_vertex, max_edges_per_vertex + 1)

def distance(v1, v2, layout):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(layout[v1], layout[v2])))

def generate_edges(layout, bounds, min_edges_per_vertex=2, max_edges_per_vertex=6, max_edge_length=4.0):
    edges = []
    neg_edges = []
    pos_edges = []
    vertices = list(layout.keys())

    for v in vertices:
        distances = [(u, distance(v, u, layout)) for u in vertices if u != v]
        distances.sort(key=lambda x: x[1])

        num_edges = determine_num_of_edges(v, layout, bounds, min_edges_per_vertex, max_edges_per_vertex)

        for u, dist in distances:
            if dist <= max_edge_length:
                if len([(e1, e2) for e1, e2 in edges if (e1 == v and e2 == u) or (e1 == u and e2 == v)]) == 0:
                    edge = (u,v) if layout[u][0] < layout[v][0] else (v,u)
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


def indicate_pass(self, graph, newly_pos, partitions):
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
            [FadeToColor(graph.edges[e], PURE_RED) for e in newly_pos],
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

def find_most_mid_vertex(layout, deviation=0.25):
    around_zero = []
    for v, pos in layout.items():
        if pos[0] < 0 and math.isclose(pos[1], 0, abs_tol=deviation):
            around_zero.append(v)

    if len(around_zero) == 0:
        new_deviation = deviation + 0.1
        find_most_mid_vertex(layout, new_deviation)

    return min(around_zero, key=lambda v: layout[v][0])


class BroadOverview(Scene):

    def indicate_source(self, graph, layout):
        source = find_most_mid_vertex(layout)
        source_text = MathTex(r"\text{s}").next_to(graph.vertices[source], LEFT*0.5)
        self.play(
            FadeToColor(graph.vertices[source], GREEN),
            Write(source_text)
        )
        self.wait(5)

        return source, source_text

    def display_title(self):
        title = Tex("Overview of the Algorithm")
        title.scale(1)
        h_line = Line(
            start=title.get_left(),
            end=title.get_right(),
        ).next_to(title, DOWN, buff=0.1)

        self.play(
            Write(title, run_time=2, rate_func=slow_into),
            Create(h_line, run_time=2, rate_func=slow_into),

        )
        self.wait(3)
        self.play(
            Unwrite(title, run_time=2, rate_func=slow_into),
            Uncreate(h_line, run_time=2, rate_func=slow_into)
        )
        self.wait()

    def run_dijkstra(self, source, graph, layout):
        distances = {v: float("inf") for v in graph.vertices}
        distances[source] = 0
        visited = set()

        while len(visited) < len(graph.vertices):
            current = min(
                (v for v in distances if v not in visited),
                key=lambda x: distances[x]
            )
            self.play(FadeToColor(graph.vertices[current], YELLOW), run_time=0.005)

            neighboring = [(v, 0) for (u, v) in graph.edges.keys() if u == current] + [(u, 1) for (u, v) in graph.edges.keys() if v == current]

            for neighbor, is_cur_dest in neighboring:
                if neighbor in visited:
                    continue
                edge_weight = abs(distance(current, neighbor, layout))
                new_distance = distances[current] + edge_weight

                if is_cur_dest:
                    edge = graph.edges[(neighbor, current)]
                else:
                    edge = graph.edges[(current, neighbor)]

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                cute_green = rgb_to_color([124, 228, 129])
                self.play(ShowPassingFlash(edge.copy().set_color(cute_green).set_stroke(width=4).set_z_index(edge.z_index+1)),
                        FadeToColor(edge, GREY), time_width=1, run_time=0.005)

            visited.add(current)
            if current != source:
                self.play(FadeToColor(graph[current], cute_green), run_time=0.005)


        self.wait()



    def construct(self):


        title = Text("Presentation of Fineman's algorithm", font_size=50)
        subtitle = MathTex(r"\text{ solving the SSSP problem in }\tilde{O}(mn^{8/9})\text{ time}", font_size=40)
        title.move_to(ORIGIN).shift(UP*0.3)
        subtitle.move_to(ORIGIN).shift(DOWN*0.8)

        h_line = Line(
            start=title.get_left(),
            end=title.get_right(),
        ).next_to(title, DOWN, buff=0.1)

        self.play(
            Write(title, run_time=2, rate_func=slow_into),
            Create(h_line, run_time=2, rate_func=slow_into),
            Write(subtitle, run_time=2, rate_func=slow_into)
        )

        self.wait(6)

        self.play(
            Unwrite(title, run_time=2, rate_func=slow_into),
            Uncreate(h_line, run_time=2, rate_func=slow_into),
            Unwrite(subtitle, run_time=2, rate_func=slow_into)
        )



        num_of_vertices = 150
        vertices = [v for v in range(1, num_of_vertices+1)]

        bounds = (5, 2.5)
        layout = generate_vertex_layout(vertices, bounds)
        edges, pos_edges, neg_edges = generate_edges(layout, bounds)

        graph = Graph(vertices, edges, edge_config=generate_edge_config(neg_edges), layout=layout)
        [graph.vertices[vertex].set_z_index(graph.z_index + 1) for vertex in vertices]

        self.display_title()

        self.play(Create(graph))
        self.wait()

        self.play([Indicate(graph.edges[e], color=PURE_RED) for e in neg_edges])
        self.wait()

        partitions = partition_graph(layout, edges, bounds[0]+1)

        past = None
        num_of_rounds = 1

        font_size = 26
        rounds_text = Text("No. of elimination rounds: ", font_size=font_size).to_corner(LEFT + UP)
        cur = Text(f"{num_of_rounds}", font_size=font_size).next_to(rounds_text)
        text = VGroup(rounds_text, cur)

        while len(neg_edges) > 0:
            newly_pos = []
            to_be_eliminated = pow(len(neg_edges), 1 / 3)

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
            indicate_pass(self, graph, newly_pos, partitions)

            self.play(
                [FadeToColor(graph.edges[e], color=WHITE) for e in newly_pos],
                run_time=1
            )
            self.wait()
            past = cur
            cur = Text(f"{num_of_rounds}", font_size=font_size).next_to(rounds_text)
            num_of_rounds += 1

        self.play(Unwrite(text))
        self.wait()

        source, source_text = self.indicate_source(graph, layout)
        self.run_dijkstra(source, graph, layout)

        self.wait()
        self.play(FadeOut(text),
                  Uncreate(graph),
                  FadeOut(source_text))
        self.wait()
