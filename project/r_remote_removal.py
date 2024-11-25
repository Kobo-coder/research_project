from manim import *
import math
from collections import defaultdict
from queue import PriorityQueue


class r_remote_removal(Scene):

    def supersource_BFD(self,vertices,positive_edges,negative_edges,weights,h):
        graph = defaultdict(list[str])
        q = "supersource"
        for vertex in vertices:
            for v,u in positive_edges:
                if v == vertex:
                    graph[v].append(u)
            for v,u in negative_edges:
                if v == vertex:
                    graph[v].append(u)
        for v in vertices:
            graph[q].append(v)
            positive_edges.append((q,v))
            weights[(q,v)] = 0

        dist = defaultdict(int)
        for v in vertices:
            dist[v] = math.inf
        dist[q] = 0


        def dijkstra():
            pq = PriorityQueue()
            for v in vertices:
                pq.put((dist[v],v))

            while not pq.empty():
                current_dist,u = pq.get()
                if current_dist > dist[u]:
                    continue
                for v in graph[u]:
                    if (u,v) in negative_edges:
                        continue
                    alt_dist = dist[u]+weights[(u,v)]
                    if alt_dist < dist[v]:
                        dist[v] = alt_dist
                        pq.put((alt_dist,v))

        def bellman_ford_round():
            for u in vertices:
                for v in graph[u]:
                    if (u,v) not in negative_edges:
                        continue
                    alt_dist = dist[u] + weights[(u,v)]
                    if alt_dist < dist[v]:
                        dist[v] = alt_dist
                
        for i in range(h+1):
            dijkstra()
            if not h == i:
                bellman_ford_round()
            for key,value in dist.items():
                print(f"{key} : {value}")

        h_distances = dist.copy()

        ##### CYCLE CHECK #####
        bellman_ford_round()
        dijkstra()
        del h_distances[q]
        del dist[q]
        h1= [value for value in h_distances.values()]
        h2= [value for value in dist.values()]
        for i in range(len(h1)):
            if h2[i]<h1[i]:
                raise ValueError("NEGATIVE CYCLE DETECTED!")
        #######################


    def create_r_removal_example(self):
        vertices = [i for i in range(1,10)]
        edges = [(1,2),(2,3),(2,4),(2,5),(3,6),(4,6),(5,6),(6,7),(7,9),(7,8),(8,9)]
        U = {3,4,5}
        negative_edges = {(3,6),(4,6),(5,6),(2,3),(2,4),(2,5)}
        out_U = {(3,6),(4,6),(5,6)}
        positive_edges = [e for e in edges if e not in out_U and e not in negative_edges]

        vertex_layout = {
            1: [-5,0,0],
            2: [-3,0,0],
            3: [-1,1,0],
            4: [-1,0,0],
            5: [-1,-1,0],
            6: [1,0,0],
            7: [3,0,0],
            8: [5,0,0],
            9: [5,1,0],
        }

        edge_weights = {
            (1,2): 2,
            (2,3): -2,
            (2,4): -5,
            (2,5): -9,
            (3,6): -6,
            (4,6): -8,
            (5,6): -4,
            (6,7): 12,
            (7,8): 3,
            (7,9): 5,
            (8,9): 1
        }

        return vertices,edges,U,out_U,vertex_layout,edge_weights,positive_edges

    def construct_h(self, g: DiGraph, neg_edges,positive_edges):


        k_hat = len(neg_edges)
        r = math.floor(pow(k_hat,1/9))
        R = [6]

        h_vertices = []

        for v in g.vertices:
            if v in R:
                    for i in range(r+1):
                        h_vertices.append(f"{v}_{i}")
            else:
                h_vertices.append(f"{v}_0")

        h_edges = []

        #Rule 1
        for (u,v) in positive_edges:
            if u in R and v in R:
                for i in range(r+1):
                    h_edges.append((str(u) + "_" + str(i), str(v) + "_" + str(i)))

        # Rule 2
        for (u,v) in neg_edges:
            if u in R and v in R:
                for i in range(r):
                    h_edges.append((str(u)+"_"+str(i), str(v)+ "_"+str(i+1)))

        # Rule 3
        for (u,v) in positive_edges:
            if u in R and v not in R:
                for i in range(r+1):
                    h_edges.append((str(u)+"_" + str(i), str(v) + "_0"))

        # Rule 4
        for (u,v) in neg_edges:
            if u in R and v not in R:
                for i in range(r):
                    h_edges.append((str(u)+"_" + str(i), str(v) + "_0"))

        # Rule 5
        for (u,v) in positive_edges:
            if u not in R and v in R:
                h_edges.append((str(u)+"_0", str(v)+"_0"))

        # Rule 6
        for (u,v) in neg_edges:
            if u not in R and v in R:
                h_edges.append((str(u)+"_0", str(v)+"_1"))

        # Rule 7
        for (u,v) in positive_edges:
            if u not in R and v not in R:
                h_edges.append((str(u)+"_0", str(v)+"_0"))

        # Rule 8
        for (u,v) in neg_edges:
            if u not in R and v not in R:
                h_edges.append((str(u)+"_0", str(v)+"_0"))

        # Rule 9
        for u in R:
            for i in range(r):
                h_edges.append((str(u)+"_"+str(i), str(u)+"_"+ str(i+1)))
            h_edges.append((str(u)+"_" + str(r), str(u)+"_0"))


        h_layout = {
            "1_0": [-5, 0, 0],
            "2_0": [-3, 0, 0],
            "3_0": [-1, 1, 0],
            "4_0": [-1, 0, 0],
            "5_0": [-1, -1, 0],
            "6_0": [1, 0, 0],
            "6_1": [1,1,0],
            "7_0": [3, 0, 0],
            "8_0": [4, 0, 0],
            "9_0": [5, 1, 0],
            "10_0": [5, -1, 0]
        }

        h = DiGraph(h_vertices, h_edges, layout=h_layout)
        self.play(Create(h))
        # TODO: Think of way to make layers visible

        self.wait(5)



    def construct(self):
        title = Tex("r-Remote Edge Elimination by Hop Reduction")
        title.scale(1)
        h_line = Line(
            start=title.get_left(), 
            end=title.get_right(), 
        ).next_to(title, DOWN, buff=0.1)

        # The run_time speed parameter alter the speed at which the title is written.
        # We use it to sync the creation of the title underline.
        self.play(
            Write(title,run_time=2), 
            Create(h_line,run_time=2),
            
        )
        self.wait(3)
        self.play(
            Unwrite(title,run_time=2),
            Uncreate(h_line,run_time=2)
        )


        vertices,edges,U,negative_edges,vertex_layout,edge_weights,positive_edges = self.create_r_removal_example()

        edge_config = {
            "tip_config": {"tip_length": 0.2, "tip_width": 0.2},
            (3, 6): {"stroke_color": RED},
            (4, 6): {"stroke_color": RED},
            (5, 6): {"stroke_color": RED},
            (2, 3): {"stroke_color": RED},
            (2, 4): {"stroke_color": RED},
            (2, 5): {"stroke_color": RED},
            }
        vertex_config = {
            # "radius":radius
            2: {"fill_color": RED},
            3: {"fill_color": RED},
            4: {"fill_color": RED},
            5: {"fill_color": RED}
            }

        g = DiGraph(vertices, edges, edge_config = edge_config,vertex_config=vertex_config, layout=vertex_layout)
        self.play(
            Create(g,run_time=4)
        )
        self.wait(2)

        weight_labels = []

        for (u,v) in edges:
            u_pos = g.vertices[u].get_center()
            v_pos = g.vertices[v].get_center()
            mid   = (u_pos + v_pos)/2
            # https://docs.manim.community/en/stable/guides/using_text.html
            weight = Text(str(edge_weights[(u,v)]),font_size=24)
            if (mid[1] < 0):
                mid[1] = mid[1] -0.3
                weight.move_to(mid)
            elif (u_pos[0] == v_pos[0]):
                mid[0] = mid[0] +0.3
                weight.move_to(mid)
            else:
                mid[1] = mid[1] +0.3
                weight.move_to(mid)
            weight_labels.append(weight)
        
        for label in weight_labels:
            self.play(Write(label,run_time=0.5))
        self.wait(2)


        # NOTE: The graph we create here tries to show the 1-hop relationship graph. Maybe not correct at this very moment
        # but it still shows in general how we find this negative sandwich of negative 1-hop paths between negative vertices.
        # In the paper this is omitted or at least not explained as he does in his video. His video explains this negative sandwich
        # and how it is found more intuitively.


        # Find bounds of vertices in U + x + y
        xs = [vertex_layout[v][0] for v in U] + [vertex_layout[2][0],vertex_layout[6][0]]
        ys = [vertex_layout[v][1] for v in U] + [vertex_layout[2][1],vertex_layout[6][1]]
        min_x = min(xs); max_x = max(xs)
        min_y = min(ys); max_y = max(ys)

        U_indicator = Rectangle(
            width= max_x - min_x + 1,
            height= max_y - min_y + 1,
            fill_opacity = 0.0,
            stroke_color = BLUE,
            stroke_width = 5
        )
        print(U_indicator.height)
        U_indicator.move_to([(max_x+min_x)/2,(max_y+min_y)/2,0])
        U_indicator_label = MathTex("(x, U, y)",font_size=48,color=BLUE)
        U_indicator_label.move_to([(max_x+min_x)/2,(max_y - min_y),0])

        self.play(
            Create(U_indicator),
            Write(U_indicator_label)
        )

        self.wait(2)

        #### SETUP SMALLER RECTANGLE FOR U

        xs = [vertex_layout[v][0] for v in U] 
        ys = [vertex_layout[v][1] for v in U]
        min_x = min(xs); max_x = max(xs)
        min_y = min(ys); max_y = max(ys)


        U_indicator_2 = Rectangle(
            width= max_x - min_x + 1,
            height= max_y - min_y + 1,
            fill_opacity = 0.0,
            stroke_color = BLUE,
            stroke_width = 5
        )
        U_indicator_2.move_to([(max_x+min_x)/2,(max_y+min_y)/2,0])
        U_indicator_label_2 = Text("U",font_size=48,color=BLUE)
        U_indicator_label_2.move_to([(max_x+min_x)/2,(max_y - min_y),0])



        #### REMOVE Sandwich RECTANGLE by Transforming to new
        self.play(
            ReplacementTransform(U_indicator,U_indicator_2),
            ReplacementTransform(U_indicator_label,U_indicator_label_2)
        )

        self.wait(2)


        ### MOVE U TO THE SIDE TO INDICATE WE MAKE NEW GRAPH G after it has been transformed.
        
        self.play(
            Uncreate(g),
            Uncreate(U_indicator_2),
            Unwrite(U_indicator_label_2),
        )

        self.wait(2)
        
        # TODO: remember to swap this for own text. This is directly copied from finemans paper.
        gn_definition = MathTex(
            r"{\text{For a subset }}N \subseteq E^- \text{ of negative edges on the input graph,}",
            r"\text{ we use }G^N \text{ to denote the subgraph: }",
            r"G^N = (V, E^+ \cup N, w).",
            r"\text{ Moreover, }G^N_\phi \text{ denotes the reweighted subgraph: }",
            r"G^N_\phi = (V, E^+ \cup N, w_\phi)."
        )
        
        gn_definition.arrange(DOWN, aligned_edge=ORIGIN, buff=0.3)
        gn_definition.move_to(ORIGIN)
        
        self.play(
            Write(gn_definition,run_time = 5)
            )
        
        self.wait(2)

        self.play(
            Unwrite(gn_definition)
        )


        input_description_1 = MathTex(
           r"{\text{As a product of previous steps of the algorithm,}",
           r"{\text{the negative vertices in }U\text{ are made } r\text{-remote.}",
            r"\text{Notationally this is written:}}",
            r"{\left| R^{r}_{\phi_1 + \phi_2}(U)\right| > n/r}",
        )

        input_description_2 = MathTex(
            r"{\text{We will use the hop-reduction technique on the graph}",
            r"{G^{out(U)}_{\phi_1 + \phi_2} \text{ to eliminate }out(U)\text{.}}",
            r"{\text{A price function }\phi \text{ is computed by this step.}}"
        )

        input_description_1.arrange(DOWN, aligned_edge=ORIGIN, buff=0.3)
        input_description_1.move_to(ORIGIN)
        input_description_2.arrange(DOWN, aligned_edge=ORIGIN, buff=0.3)
        input_description_2.move_to(ORIGIN)


        self.play(Write(input_description_1))
        self.wait(3)
        self.play(Unwrite(input_description_1))
        self.wait(1)
        self.play(Write(input_description_2))
        self.wait(3)
        self.play(Unwrite(input_description_2))


        self.construct_h(g, negative_edges,positive_edges)
