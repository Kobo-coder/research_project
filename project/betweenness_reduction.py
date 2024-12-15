from collections import defaultdict
from queue import PriorityQueue
from manim import *
import random as rand
import math

from broad_overview import is_position_valid, distance, determine_num_of_edges, generate_edge_config


def generate_layout(vertices, bounds, min_distance=0.5):
    n = len(vertices)

    width = 6
    height = 4
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

def construct_small_example():
    vertices = [i for i in range(6)]
    edges = [(0,1), (0,4), (1,2), (2,3), (2,4), (3,5), (4,5)]

    neg_edges = [(1,2)]

    weights = {
        (0, 4): 5,
        (0,1): 0,
        (1,2): -7,
        (2,3): 2,
        (2,4): 3,
        (3,5): 4,
        (4,5): 2
    }

    vertex_layout = {
        0: [-4, 0, 0],
        1: [-2, 0, 0],
        2: [0, 0, 0],
        3: [2, 1, 0],
        4: [2, -1, 0],
        5: [4, 0, 0]
    }

    vertex_labels = {
        0: MathTex("0", font_size=30, color="black"),
        1: MathTex("1", font_size=30, color="white"),
        2: MathTex("2", font_size=30, color="black"),
        3: MathTex("3", font_size=30, color="black"),
        4: MathTex("4", font_size=30, color="black"),
        5: MathTex("5", font_size=30, color="black"),
    }

    vertex_config = {
        "radius": 0.3,
        1: {"fill_color": RED, "radius": 0.3}
    }

    edge_config = {"tip_config": {"tip_length": 0.2, "tip_width": 0.2},
                   (1,2): {"stroke_color": RED},
                   (0,4): {"stroke_color": BLACK}}

    return vertices, edges, neg_edges, weights, vertex_layout, vertex_labels, vertex_config, edge_config


def compute_new_weights(h_edges_removed_dups, T, SSSPs, STSPs):
    h_edge_weights = {}
    for (u, v) in h_edges_removed_dups:
        if u in T and v in T:
            assert (SSSPs[u][v] == STSPs[v][u])
            assert (SSSPs[v][u] == STSPs[u][v])

        if u in T:
            h_edge_weights[(u, v)] = SSSPs[u][v]
        elif v in T:
            h_edge_weights[(u, v)] = STSPs[v][u]
    return h_edge_weights

def create_curved_edges(graph, weight_dict):
    curved_edges = {}

    curved_edges_config = {
        (0, 1): (0.5, graph.vertices[0].get_center(), graph.vertices[1].get_boundary_point(LEFT)),
        (1, 0): (0.5, graph.vertices[1].get_center(), graph.vertices[0].get_boundary_point(RIGHT)),
        (4, 5): (0.5, graph.vertices[4].get_center(), graph.vertices[5].get_boundary_point(DOWN)),
        (5, 4): (0.5, graph.vertices[5].get_center(), graph.vertices[4].get_boundary_point(UP*LEFT)),
        (3, 5): (0.5, graph.vertices[3].get_center(), graph.vertices[5].get_boundary_point(LEFT)),
        (5, 3): (0.5, graph.vertices[5].get_center(), graph.vertices[3].get_boundary_point(DOWN*LEFT)),
        (0, 2): (0.75, graph.vertices[0].get_center(), graph.vertices[2].get_boundary_point(LEFT)),
        (2, 0): (0.75, graph.vertices[2].get_center(), graph.vertices[0].get_boundary_point(DOWN*RIGHT)),
        (2, 5): (0.75, graph.vertices[2].get_center(), graph.vertices[5].get_boundary_point(LEFT)),
        (5, 2): (0.75, graph.vertices[5].get_center(), graph.vertices[2].get_boundary_point(UP*RIGHT)),
        (3, 0): (1.1, graph.vertices[3].get_center(), graph.vertices[0].get_boundary_point(RIGHT)),
        (0, 3): (1.5, graph.vertices[0].get_center(), graph.vertices[3].get_boundary_point(LEFT)),
        (0, 4): (1.1, graph.vertices[0].get_center(), graph.vertices[4].get_boundary_point(DOWN)),
        (4, 0): (1.5, graph.vertices[4].get_center(), graph.vertices[0].get_boundary_point(DOWN)),
        (1, 5): (1.7, graph.vertices[1].get_center(), graph.vertices[5].get_boundary_point(DOWN)),
        (5, 1): (1.7, graph.vertices[5].get_center(), graph.vertices[1].get_boundary_point(UP)),
        (0, 5): (1.7, graph.vertices[0].get_center(), graph.vertices[5].get_boundary_point(DOWN)),
        (5, 0): (1.7, graph.vertices[5].get_center(), graph.vertices[0].get_boundary_point(UP)),
    }

    for (u,v), (a, s, e) in curved_edges_config.items():
        if weight_dict[(u,v)] < 0:
            color = RED
        elif weight_dict[(u,v)] == float('inf'):
            color = GREY
        else:
            color = WHITE

        custom_edge = CurvedArrow(
            s,  # Start vertex
            e,  # End vertex
            angle=a,  # Adjust angle to control curve
            tip_length=0.2,  # Optional buffer to prevent overlapping
            color=color
        )
        curved_edges[(u,v)] = custom_edge

    return curved_edges


def generate_new_layout():
    vertex_layout = {
        0: [-6, 0, 0],
        1: [-3, 0, 0],
        2: [0, 0, 0],
        3: [3, 1, 0],
        4: [3, -1, 0],
        5: [6, 0, 0]
    }
    return vertex_layout

def supersource_BFD(vertices, positive_edges, negative_edges, weights, h):
    graph = defaultdict(list[type(vertices[0])])
    pos_edges = positive_edges.copy()
    q = 6
    for vertex in vertices:
        for v, u in pos_edges:
            if v == vertex:
                graph[v].append(u)
        for v, u in negative_edges:
            if v == vertex:
                graph[v].append(u)
    for v in vertices:
        graph[q].append(v)
        pos_edges.append((q, v))
        weights[(q, v)] = 0

    dist = defaultdict(int)
    for v in vertices:
        dist[v] = math.inf
    dist[q] = 0
    print(graph)

    def dijkstra():
        pq = PriorityQueue()
        for v in graph.keys():
            pq.put((dist[v], v))
        while not pq.empty():
            current_dist, u = pq.get()
            if current_dist > dist[u]:
                continue
            for v in graph[u]:
                if (u, v) in negative_edges:
                    continue
                alt_dist = dist[u] + weights[(u, v)]
                if alt_dist < dist[v]:
                    dist[v] = alt_dist
                    pq.put((alt_dist, v))

    def bellman_ford_round():
        for u in vertices:
            for v in graph[u]:
                if (u, v) not in negative_edges:
                    continue
                alt_dist = dist[u] + weights[(u, v)]
                if alt_dist < dist[v]:
                    dist[v] = alt_dist

    all_distances = []

    for i in range(h + 1):
        dijkstra()
        round_i_dist = dist.copy()
        del round_i_dist[q]
        all_distances.append(round_i_dist)
        print(f"Dijkstra's pass {i + 1}")
        for key, value in dist.items():
            print(f"{key} : {value}")
        if not h == i:
            bellman_ford_round()
            print(f"BF pass {i + 1}")
            for key, value in dist.items():
                print(f"{key} : {value}")

    h_distances = dist.copy()

    ##### CYCLE CHECK #####
    bellman_ford_round()
    dijkstra()
    del h_distances[q]
    del dist[q]
    h1 = [value for value in h_distances.values()]
    h2 = [value for value in dist.values()]
    for i in range(len(h1)):
        if h2[i] < h1[i]:
            raise ValueError("NEGATIVE CYCLE DETECTED!")
    #######################
    return all_distances


class BetweennessReduction(Scene):
    def apply_weights(self, graph, weights, anim = True, font_size = 24, mid_point = 0):
        weight_labels = []

        for (u, v) in graph.edges.keys():
            u_pos = graph.vertices[u].get_center()
            v_pos = graph.vertices[v].get_center()
            mid = (u_pos + v_pos) / 2
            # https://docs.manim.community/en/stable/guides/using_text.html
            weight = Text(str(weights[(u, v)]), font_size=font_size)
            if (mid[1] < mid_point):
                mid[1] = mid[1] - 0.3
                weight.move_to(mid)
            elif (u_pos[0] == v_pos[0]):
                mid[0] = mid[0] + 0.3
                weight.move_to(mid)
            else:
                mid[1] = mid[1] + 0.3
                weight.move_to(mid)
            if (u,v) == (0,4):
                weight.next_to(mid, DOWN, buff=0.2)

            weight_labels.append(weight)

        if anim:
            self.play(
                *[Write(label, run_time=0.5) for label in weight_labels]
            )
            self.wait(2)

        return weight_labels


    def apply_weights_on_curved_edges(self, edges, curved_edges, weights, anim = True):
        weight_labels = []

        for (u,v) in edges:

            if (u,v) in {(0,0), (5,5)}:
                continue

            if weights[(u,v)] < 0:
                color = RED
            else:
                color = WHITE


            if weights[(u,v)] == float("inf"):
                label = MathTex("\infty", font_size=28, color=GREY).set_z_index(12)
            else:
                label = MathTex(str(weights[(u,v)]), font_size=24, color=color).set_z_index(12)


            if u < v:
                direction = DOWN
            else:
                direction = UP

            point = curved_edges[(u,v)].get_boundary_point(direction)

            if (u,v) in {(0,4), (3,0)}:
                point = (point[0]-1, point[1], point[2])

            elif (u,v) in {(0,3), (4,0)}:
                if (u,v) == (0,3):
                    point = (point[0] + (3.4+0.14), point[1] + 0.65, point[2])
                else:
                    point = (point[0] + 3.4, point[1] - 0.65, point[2])

            elif (u,v) in {(0,1), (1,0)}:
                if (u,v) == (0,1):
                    point = (point[0] + 1.3, point[1]+0.15, point[2])
                else:
                    point = (point[0] + 1.02, point[1]-0.15, point[2])

            elif (u,v) in {(2,5), (5,2)}:
                if (u,v) == (2,5):
                    point = (point[0] - 0.85, point[1], point[2])
                else:
                    point = (point[0] - (0.85+0.28), point[1], point[2])

            elif (u,v) in {(4,5), (5,3)}:
                if (u,v) == (4,5):
                    point = (point[0]+1, point[1]+0.1, point[2])
                else:
                    point = (point[0]+(1-0.28), point[1]-0.2, point[2])

            elif (u,v) in {(3,5), (5,4)}:
                if (u,v) == (3,5):
                    point = (point[0]-1.6, point[1]+0.5, point[2])
                else:
                    point = (point[0]-(1.6+0.28), point[1]-0.5, point[2])


            label.next_to(point, direction, buff=0.2)
            weight_labels.append(label)

        if anim:
            self.play(
                *[Write(label, run_time=0.5) for label in weight_labels]
            )
            self.wait(2)

        return weight_labels


    def compute_SSSP(self, source, beta, graph, weights_dict, neg_edges, curved_edge = None, weight_labels=None, animated = False):

        def dijkstra(animated = False, animation_graph = None):

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
                        if animated:
                            if edge == (0,4):
                                self.play(
                                    ShowPassingFlash(
                                        animated_custom_edge.copy().set_color(PURE_GREEN).set_z_index(100),
                                        time_width=2,
                                        run_time=2
                                    ),
                                )
                            else:
                                self.play(
                                    ShowPassingFlash(
                                        animation_graph.edges[edge].copy().set_color(PURE_GREEN).set_z_index(100),
                                        time_width=2,
                                        run_time=2
                                    ),
                                )
                        dist[edge[1]] = dist[u] + weight
                        pq.put((dist[edge[1]], edge[1]))

                        if animated:
                             self.update_dist_table(edge[1], dist[edge[1]], dist_table)



        def bellman_ford(animated = False, animation_graph = None):
            for (u, v) in neg_edges:

                if dist[v] > dist[u] + weights_dict[(u, v)]:
                    dist[v] = dist[u] + weights_dict[(u, v)]
                    if animated:
                        if (u,v) == (0, 4):
                            self.play(
                                ShowPassingFlash(
                                    animated_custom_edge.copy().set_color(PURE_GREEN).set_z_index(100),
                                    time_width=2,
                                    run_time=2
                                ),
                            )
                        else:
                            self.play(
                                ShowPassingFlash(
                                    animation_graph.edges[(u,v)].copy().set_color(PURE_GREEN).set_z_index(100),
                                    time_width=2,
                                    run_time=2
                                ),
                            )
                        self.wait()
                        self.update_dist_table(v, dist[v], dist_table)
                        self.wait()


        # init distance dict
        if animated:
            dist, dist_table, new_weight_labels, animation_graph, animated_custom_edge = self.prepare_for_bfd(graph, 0, curved_edge, weights_dict, weight_labels)
        else:
            dist = {v: math.inf for v in graph.vertices}
            dist[source] = 0

        header = Tex(r"Computing $\beta$-hop distances from vertex $0$, for $\beta = 1$", font_size=35).to_corner(LEFT + UP)
        text = Text("Running ", font_size=22).align_to(header, LEFT + DOWN).shift(DOWN*0.5)
        dijk = Text("Dijkstra", font_size=22).next_to(text, buff=0.2)
        bell = Text("Bellman-Ford", font_size=22).next_to(text, buff=0.2)

        if animated:
            self.play(Write(header),
                      Write(text),
                      Write(dijk))
            dijkstra(animated, animation_graph)
        else:
            dijkstra()

        for rounds in range(beta):

            if animated:
                self.play(ReplacementTransform(dijk, bell))
                self.wait()
                bellman_ford(animated, animation_graph)

                dijk = Text("Dijkstra", font_size=22).next_to(text, buff=0.2)
                self.play(ReplacementTransform(bell, dijk))
                self.wait()
                dijkstra(animated, animation_graph)

            else:
                bellman_ford()
                dijkstra()

        if animated:
            self.play(
                FadeOut(text),
                FadeOut(dijk),
                FadeOut(header)
            )
            self.wait(4)
            if dist_table is not None:
                background = SurroundingRectangle(dist_table.get_entries((6, 2)),
                                                  color=BLACK,
                                                  fill_color=BLACK,
                                                  fill_opacity=1).set_z_index(10)
                self.play(Uncreate(dist_table),
                          Create(background)
                          )
                self.wait()

            self.play(FadeOut(animation_graph),
                      [FadeOut(l) for l in new_weight_labels],
                      FadeOut(animated_custom_edge))

        return dist



    def prepare_for_bfd(self, graph, source, custom_edge, weights, weights_labels):
        dist = {v: math.inf for v in graph.vertices}
        dist[source] = 0

        graph_cop = graph.copy()
        custom_edge_copy = custom_edge.copy()
        new_graph = Group(graph_cop, custom_edge_copy)
        new_graph.shift(RIGHT * 2.2).scale(0.8)
        new_weight_labels = self.apply_weights(graph_cop, weights, anim=False, font_size=18)

        self.play(
            ReplacementTransform(Group(graph, custom_edge), new_graph),
            *[ReplacementTransform(weights_labels[i], new_weight_labels[i].set_z_index(10)) for i in
              range(len(new_weight_labels))]
        )
        self.wait(2)


        dist_table = MathTable(
            [[str(vertex), r"\infty"] if distance == float("inf") else [str(vertex), f"{distance}"] for vertex, distance
             in dist.items()],
            col_labels=[Text("v", slant=ITALIC), Text("dist", slant=ITALIC)],
            include_outer_lines=True,
        ).scale(0.55).next_to(self.camera.frame_center, LEFT, buff=2.5)

        self.play(Create(dist_table))

        return dist, dist_table, new_weight_labels, graph_cop, custom_edge_copy


    def update_dist_table(self, vertex, new_distance, dist_table: MathTable):
        if new_distance == -2:
            z_ind = 6
        else: z_ind = 4
        table_vertex= dist_table.get_entries((vertex + 2, 2))
        new_dist_text = MathTex(str(new_distance)).scale(0.55).move_to(table_vertex.get_center()).set_z_index(z_ind+1)
        background = SurroundingRectangle(new_dist_text,
                                          color=BLACK,
                                          fill_color=BLACK,
                                          fill_opacity=1).set_z_index(z_ind)

        self.play(
            ReplacementTransform(
                table_vertex[0],
                new_dist_text
            ),
            Create(background),
            FadeToColor(new_dist_text, PURE_GREEN)
        )
        self.wait(0.5)
        self.play(
            FadeToColor(new_dist_text, WHITE),
        )
        #dist_table.set_entries({(vertex + 2, 2): MathTex(str(new_distance))})



    def construct(self):

        ######################## TITLE ########################

        title = Tex("Betweenness Reduction")
        title.scale(1)
        h_line = Line(
            start=title.get_left(),
            end=title.get_right(),
        ).next_to(title, DOWN, buff=0.1)

        self.play(
            Write(title, run_time=2),
            Create(h_line, run_time=2),

        )
        self.wait(3)
        self.play(
            Unwrite(title, run_time=2),
            Uncreate(h_line, run_time=2)
        )


        ######################## LARGE EXAMPLE ########################

        n = 24
        vertices = [i for i in range(n)]

        bounds = (5, 2.5)
        layout = generate_layout(vertices, bounds)

        edges, neg_edges = generate_edges(layout, bounds, 2, 7)

        weights_dict = compute_weights(edges, neg_edges)

        edge_config = generate_edge_config(neg_edges)
        edge_config["tip_config"] = {"tip_length": 0.2, "tip_width": 0.2}

        g = DiGraph(vertices, edges, layout = layout, edge_config=edge_config)

        self.play(Create(g))
        self.wait()

        # https://www.geeksforgeeks.org/randomly-select-n-elements-from-list-in-python/
        # the algorithm sets c = 3
        T = select_subset(vertices, 1, 3)

        self.play([FadeToColor(g.vertices[v], BLUE) for v in T])
        self.wait()

        beta = 1
        SSSPs = {}
        STSPs = {}

        for v in T:
            SSSPs[v] = self.compute_SSSP(v, beta, g, weights_dict, neg_edges)
            transposed_graph, transposed_weights_dict, transposed_neg_edges = transpose_data(g, weights_dict, neg_edges)
            STSPs[v] = self.compute_SSSP(v, beta, transposed_graph, transposed_weights_dict, transposed_neg_edges)

        self.play(Uncreate(g))
        self.wait()

        h_edges = construct_h_edges(vertices, T)
        h = DiGraph(vertices, h_edges, layout = layout, edge_config={"tip_config": {"tip_length": 0.2, "tip_width": 0.2}})

        self.play(Create(h))
        self.wait(2)

        self.play(Uncreate(h))
        self.wait()




        ##########  SMALL EXAMPLE  ##########


        vertices, edges, neg_edges, weights, layout, vertex_labels, vertex_config, edge_config = construct_small_example()

        small_g = DiGraph(vertices, edges, layout = layout, vertex_config=vertex_config, edge_config=edge_config, labels = vertex_labels)

        pos_4 = layout[4]
        custom_edge = CurvedArrow(
            layout[0],  # Start vertex
            small_g.vertices[4].get_boundary_point(LEFT),  # End vertex
            angle=0.4,  # Adjust angle to control curve
            tip_length=0.2  # Optional buffer to prevent overlapping
        )
        small_g.edges[(0,4)].set_z_index(small_g.z_index - 1)
        small_g.edges[(0,1)].set_z_index(small_g.z_index + 1)
        [small_g.vertices[v].set_z_index(small_g.z_index + 2) for v in vertices]

        self.play(
            Create(small_g),
            Create(custom_edge)
        )
        self.wait()

        weight_labels = self.apply_weights(small_g, weights)
        self.wait()

        T = [0, 5]
        t_text = Text("T = {0, 5}", font_size=25).move_to(small_g.vertices[2].get_center()).shift(UP*2.1).shift(LEFT*0.8)
        self.play(
            *[Indicate(small_g.vertices[v], run_time=2) for v in T],
            Write(t_text)
        )
        self.wait(2)


        beta = 1
        SSSPs = {}
        STSPs = {}


        for v in T:
            SSSPs[v] = self.compute_SSSP(v, beta, small_g, weights, neg_edges, custom_edge, weight_labels, True if v == 0 else False)
            transposed_graph, transposed_weights_dict, transposed_neg_edges = transpose_data(small_g, weights, neg_edges)
            STSPs[v] = self.compute_SSSP(v, beta, transposed_graph, transposed_weights_dict, transposed_neg_edges)

        small_g_target = small_g.copy()
        small_g_target.scale(0.5)
        small_g_target.to_corner(UL)

        custom_edge_target = CurvedArrow(
            small_g_target.vertices[0].get_boundary_point(LEFT),
            small_g_target.vertices[4].get_boundary_point(LEFT),
            angle=0.4,
            tip_length=0.2*0.5
        )

        smaller_weights_copy = self.apply_weights(small_g_target, weights, False, 12, small_g.vertices[0].get_midpoint()[0])
        small_g.vertices[0].set_z_index(small_g.z_index +1)

        self.play(ReplacementTransform(small_g, small_g_target),
                    ReplacementTransform(custom_edge, custom_edge_target),
                  *[ReplacementTransform(weight_labels[i], smaller_weights_copy[i].shift(DOWN*0.12)) for i in range(len(weight_labels))],
                    FadeOut(t_text)
                    )

        self.wait()



        ####### CONSTRUCTION OF H ########

        # explanatory_text1 = Tex(r"$\forall v \in T$, we compute the $\beta$-hop SSSP and STSP to and from $v$")
        # explanatory_text2 = Tex("This is done by running BFD")
        explanatory_text3 = Tex(r"Based on the computed $\beta$-hop distances,\\ we construct an auxiliary graph $H$").set_z_index(11)


        self.add(explanatory_text3)
        self.wait(5)
        self.remove(explanatory_text3)
        self.wait()


        h_edges = construct_h_edges(vertices, T)
        h_edges_removed_dups = list(set(h_edges))
        new_layout = generate_new_layout()
        h = DiGraph(vertices, h_edges_removed_dups, layout = new_layout, vertex_config= {"fill_color": PURE_BLUE}, edge_config={"tip_config": {"tip_length": 0.2, "tip_width": 0.2}})
        [h.vertices[v].set_z_index(12) for v in vertices]

        h_edge_weights = compute_new_weights(h_edges_removed_dups, T, SSSPs, STSPs)

        curved_edges = create_curved_edges(h, h_edge_weights)
        [c.set_z_index(11) for c in curved_edges.values()]
        assert(len(curved_edges) + 2 == len(h_edges_removed_dups)) # without self-loops, hence the +2


        self.play([Create(h.vertices[v]) for v in vertices],
                  [Create(edge) for edge in curved_edges.values()]
                  )
        h_weight_labels = self.apply_weights_on_curved_edges(h_edges_removed_dups, curved_edges, h_edge_weights)
        self.wait()

        supersouce = ["s"]
        supersource_graph = Graph(supersouce, [], labels=True).shift(DOWN*3.25).shift(LEFT*4).set_z_index(13)

        self.play(
            Create(supersource_graph)
        )
        self.wait()
        ss_edges = [CurvedArrow(start_point=supersource_graph.vertices["s"].get_center(),
                                end_point=h.vertices[0].get_boundary_point(DOWN), angle=-0.4, tip_length=0.2, color=BLUE).set_z_index(12),
                    CurvedArrow(start_point=supersource_graph.vertices["s"].get_center(),
                                end_point=h.vertices[1].get_boundary_point(DOWN), angle=-0.2, tip_length=0.2, color=BLUE).set_z_index(12),
                    CurvedArrow(start_point=supersource_graph.vertices["s"].get_center(),
                                end_point=h.vertices[2].get_boundary_point(DOWN), angle=0.2, tip_length=0.2, color=BLUE).set_z_index(12),
                    CurvedArrow(start_point=supersource_graph.vertices["s"].get_center(),
                                end_point=h.vertices[3].get_boundary_point(DOWN), angle= 0.4, tip_length=0.2, color=BLUE).set_z_index(12),
                    CurvedArrow(start_point=supersource_graph.vertices["s"].get_center(),
                                end_point=h.vertices[4].get_boundary_point(DOWN), angle = 0.6, tip_length=0.2, color=BLUE).set_z_index(12),
                    CurvedArrow(start_point=supersource_graph.vertices["s"].get_center(),
                                end_point=h.vertices[5].get_boundary_point(DOWN), angle= 1.4, tip_length=0.2, color=BLUE).set_z_index(12)
                    ]
        self.play(
            [Create(a) for a in ss_edges]
        )
        self.wait(5)

        self.clear()

        cute_text = Tex(r"Computing $dist_H^{\ell}(V,v)$ for $\ell = 2|T| = 4$ for all $v \in V$ \\ gives us $\phi_1$ as $\phi_1(v) = dist_H^{\ell}(V,v)$")
        self.add(cute_text)
        self.wait(7)
        self.remove(cute_text)

        h_pos_edges = []
        h_neg_edges = []
        for edge in h_edges_removed_dups:
            if h_edge_weights[edge] < 0:
                h_neg_edges.append(edge)
            else:
                h_pos_edges.append(edge)

        price_function = supersource_BFD(vertices, h_pos_edges, h_neg_edges, h_edge_weights, 4)

        phi_k = [key for key in price_function[-1].keys()]
        phi_v = [value for value in price_function[-1].values()]
        print(phi_k)
        print(phi_v)

        phi_1 = MathTable(
            [phi_k, phi_v],
            row_labels=[MathTex("v"), MathTex("\phi_1")],
            include_outer_lines=True
        ).scale(0.8).move_to(ORIGIN)

        self.play(
            Create(phi_1)
        )

        self.wait(2)

        self.play(
            Uncreate(phi_1)
        )
        self.wait()

        juleskum = Tex(r"Thus, reweighting $G$ with $\phi_1$ by applying \\ $dist_\phi^\ell(u,v) = dist^\ell(u,v) + \phi_1(u) - \phi_1(v)$")
        self.add(juleskum)
        self.wait(7)
        self.remove(juleskum)


        phi_reweighted = {}
        for ((u,v), w) in weights.items():
            phi_reweighted[(u,v)] = (w + phi_v[u] - phi_v[v])

        # FRESH GRAPH

        vertices, edges, neg_edges, weights, layout, vertex_labels, _, _ = construct_small_example()
        vertex_config = {
            "radius": 0.3
        }
        edge_config = {"tip_config": {"tip_length": 0.2, "tip_width": 0.2},
                       (0, 4): {"stroke_color": BLACK}}
        small_g = DiGraph(vertices, edges, layout=layout, vertex_config=vertex_config, edge_config=edge_config,
                          labels=vertex_labels)
        custom_edge = CurvedArrow(
            layout[0],  # Start vertex
            small_g.vertices[4].get_boundary_point(LEFT),  # End vertex
            angle=0.4,  # Adjust angle to control curve
            tip_length=0.2  # Optional buffer to prevent overlapping
        )
        small_g.edges[(0, 4)].set_z_index(small_g.z_index - 1)
        small_g.edges[(0, 1)].set_z_index(small_g.z_index + 1)
        [small_g.vertices[v].set_z_index(small_g.z_index + 2) for v in vertices]




        self.play(
            Create(small_g),
            Create(custom_edge)
        )
        weight_labels = self.apply_weights(small_g, phi_reweighted)
        self.wait(8)

        self.play(Uncreate(small_g),
                  Uncreate(custom_edge))
        self.clear()

        self.wait(2)
