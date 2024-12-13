import copy
from manim import *
import math
from collections import defaultdict
from queue import PriorityQueue


class r_remote_removal(Scene):

    def supersource_BFD(self,vertices,positive_edges,negative_edges,weights,h):
        graph = defaultdict(list[type(vertices[0])])
        pos_edges = positive_edges.copy()
        q = 0 if isinstance(vertices[0], int) else "supersource" if isinstance(vertices[0], str) else None
        for vertex in vertices:
            for v,u in pos_edges:
                if v == vertex:
                    graph[v].append(u)
            for v,u in negative_edges:
                if v == vertex:
                    graph[v].append(u)
        for v in vertices:
            graph[q].append(v)
            pos_edges.append((q,v))
            weights[(q,v)] = 0

        dist = defaultdict(int)
        for v in vertices:
            dist[v] = math.inf
        dist[q] = 0
        print(graph)

        def dijkstra():
            pq = PriorityQueue()
            for v in graph.keys():
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

        all_distances = []
                
        for i in range(h+1):
            dijkstra()
            round_i_dist = dist.copy()
            del round_i_dist[q]
            all_distances.append(round_i_dist)
            print(f"Dijkstra's pass {i+1}")
            for key,value in dist.items():
                print(f"{key} : {value}")
            if not h == i:
                bellman_ford_round()
                print(f"BF pass {i+1}")
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
        return all_distances

    def create_r_removal_example(self):
        vertices = [i for i in range(1,9)]
        edges = [(1,2),(2,4),(2,3),(4,5),(3,6),(5,6),(5,7),(6,7),(7,8)]
        U = {3,4}
        negative_edges = {(1,2),(3,6),(4,5),(7,8)}
        out_U = {(3,6),(4,5)}
        positive_edges = [(2,4),(2,3),(5,6),(5,7),(6,7)]

        vertex_layout = {
            1: [-5,0,0],
            2: [-3,0,0],
            3: [-1,-1,0],
            4: [-1,1,0],
            5: [1,1,0],
            6: [1,-1,0],
            7: [3,0,0],
            8: [5,0,0],
        }

        edge_weights = {
            (1,2): -10,
            (2,3): 7,
            (2,4): 8,
            (4,5): -5,
            (3,6): -6,
            (5,6): 1,
            (5,7): 2,
            (6,7): 3,
            (7,8): -8,
        }

        ############# TEST ###################
        # vertices_2 = [i for i in range(1,10)]
        # edges_2 = [(1,2),(3,6),(4,6),(5,6),(6,7),(7,9),(7,8),(8,9)]
        # negative_edges_2 = {(3,6),(4,6),(5,6)}
        # positive_edges_2 = [e for e in edges_2 if e not in negative_edges]


        # edge_weights_2 = {
        #     (1,2): 2,
        #     (3,6): -5,
        #     (4,6): -3,
        #     (5,6): -2,
        #     (6,7): 10,
        #     (7,8): 4,
        #     (7,9): 3,
        #     (8,9): 3
        # }

        # self.supersource_BFD(vertices_2,positive_edges_2,negative_edges_2,edge_weights_2,1)

        #######################################

        return vertices,edges,U,out_U,vertex_layout,edge_weights,positive_edges,negative_edges

    def construct_h(self, g: DiGraph, neg_edges,positive_edges):
        k_hat = len(neg_edges)
        r = math.floor(pow(k_hat,1/9))
        R = [5,6,7]

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

        edge_config = {
            "tip_config": {"tip_length": 0.2, "tip_width": 0.2},
            ("6_1", "6_0"): {"stroke_color": BLACK},
            ("6_0", "6_1"): {"stroke_color": BLACK},
            ("5_1", "5_0"): {"stroke_color": BLACK},
            ("5_0", "5_1"): {"stroke_color": BLACK},
            ("7_1", "7_0"): {"stroke_color": BLACK},
            ("7_0", "7_1"): {"stroke_color": BLACK},
            }

        h_layout = {
            "1_0": [-5, -2, 10],
            "2_0": [-3, -2, 10],
            "3_0": [-1, -3, 10],
            "4_0": [-1, -1, 10],
            "5_0": [1, -1, 10],
            "5_1": [1, 3, 10],
            "6_0": [1, -3, 10],
            "6_1": [1, 1, 10],
            "7_0": [3, -2, 10],
            "7_1": [3, 1, 10],
            "8_0": [5, -1, 10],
        }

        vertex_config = {
            # "radius":radius
            "5_1": {"fill_color": BLUE},
            "6_1": {"fill_color": BLUE},
            "7_1": {"fill_color": BLUE},
            }

        h = DiGraph(h_vertices, h_edges, layout=h_layout,edge_config=edge_config,vertex_config=vertex_config,labels=True)
        return h_vertices,h_edges,h


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


        vertices,edges,U,out_U,vertex_layout,edge_weights,positive_edges,negative_edges = self.create_r_removal_example()

        edge_config = {
            "tip_config": {"tip_length": 0.2, "tip_width": 0.2},
            (1, 2): {"stroke_color": RED},
            (3, 6): {"stroke_color": RED},
            (4, 5): {"stroke_color": RED},
            #(5, 6): {"stroke_color": RED},
            (7,8): {"stroke_color": RED},
            # (2, 3): {"stroke_color": RED},
            # (2, 4): {"stroke_color": RED},
            # (2, 5): {"stroke_color": RED},
            }
        vertex_config = {
            # "radius":radius
            1: {"fill_color": RED},
            # 2: {"fill_color": RED},
            3: {"fill_color": RED},
            4: {"fill_color": RED},
            7: {"fill_color": RED},
            }

        g = DiGraph(vertices, edges, edge_config = edge_config,vertex_config=vertex_config, layout=vertex_layout)
        self.play(
            Create(g,run_time=4)
        )
        self.wait(2)

        weight_labels = []
        label_to_remove = []

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
            if (u,v) in [(1,2),(7,8)]:
                label_to_remove.append(weight)
            weight_labels.append(weight)
        
        self.play(
            *[Write(label,run_time=0.5) for label in weight_labels]
        )
        self.wait(2)


        # NOTE: The graph we create here tries to show the 1-hop relationship graph. Maybe not correct at this very moment
        # but it still shows in general how we find this negative sandwich of negative 1-hop paths between negative vertices.
        # In the paper this is omitted or at least not explained as he does in his video. His video explains this negative sandwich
        # and how it is found more intuitively.


        # Find bounds of vertices in U + x + y
        xs = [vertex_layout[v][0] for v in U] + [vertex_layout[2][0],vertex_layout[7][0],vertex_layout[1][0]]
        ys = [vertex_layout[v][1] for v in U] + [vertex_layout[2][1],vertex_layout[7][1]]
        min_x = min(xs); max_x = max(xs)
        min_y = min(ys); max_y = max(ys)

        U_indicator = Rectangle(
            width= max_x - min_x + 1,
            height= max_y - min_y + 1,
            fill_opacity = 0.0,
            stroke_color = BLUE,
            stroke_width = 5
        )

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
        U_indicator_label_2 = MathTex("U",font_size=48,color=BLUE)
        U_indicator_label_2.move_to([(max_x+min_x)/2,(max_y - min_y),0])

        x_indicator = Rectangle(
            width= 1,
            height=  1,
            fill_opacity = 0.0,
            stroke_color = BLUE,
            stroke_width = 5
        )
        x_indicator.move_to([vertex_layout[1][0],vertex_layout[1][1],0])
        x_indicator_label = MathTex("x",font_size=48,color=BLUE)
        x_indicator_label.move_to([vertex_layout[1][0],vertex_layout[1][1]-1,0])

        y_indicator = Rectangle(
            width= 1,
            height=  1,
            fill_opacity = 0.0,
            stroke_color = BLUE,
            stroke_width = 5
        )
        y_indicator.move_to([vertex_layout[7][0],vertex_layout[7][1],0])
        y_indicator_label = MathTex("y",font_size=48,color=BLUE)
        y_indicator_label.move_to([vertex_layout[7][0],vertex_layout[7][1]-1,0])
        



        # #### REMOVE Sandwich RECTANGLE by Transforming to new
        self.play(
            ReplacementTransform(U_indicator,U_indicator_2),
            ReplacementTransform(U_indicator_label,U_indicator_label_2),
            Create(x_indicator),
            Write(x_indicator_label),
            Create(y_indicator),
            Write(y_indicator_label)
        )

        self.wait(2)


        # ### MOVE U TO THE SIDE TO INDICATE WE MAKE NEW GRAPH G after it has been transformed.
        
        self.play(
            Uncreate(U_indicator_2),
            Unwrite(U_indicator_label_2),
            Uncreate(x_indicator),
            Unwrite(x_indicator_label),
            Uncreate(y_indicator),
            Unwrite(y_indicator_label)
        )

        self.wait(2)

        g_c = g.copy()
        w_c = weight_labels.copy()

        # self.play(
        #     FadeOut(g),
        #     *[FadeOut(label,run_time=0.5) for label in weight_labels]
        # )
        # self.wait(1)
        
        # TODO: remember to swap this for own text. This is directly copied from finemans paper.
        # gn_definition = MathTex(
        #     r"{\text{For a subset }}N \subseteq E^- \text{ of negative edges on the input graph,}",
        #     r"\text{ we use }G^N \text{ to denote the subgraph: }",
        #     r"G^N = (V, E^+ \cup N, w).",
        #     r"\text{ Moreover, }G^N_\phi \text{ denotes the reweighted subgraph: }",
        #     r"G^N_\phi = (V, E^+ \cup N, w_\phi)."
        # )

        # gn_definition.arrange(DOWN, aligned_edge=ORIGIN, buff=0.3)
        # gn_definition.move_to(ORIGIN)
        
        # self.play(
        #     Write(gn_definition,run_time = 5)
        #     )
        
        # self.wait(2)

        # self.play(
        #     Unwrite(gn_definition)
        # )


        # gn_definition = MathTex(
        #     r"{\text{Let }out(U)\text{ denote the set of outgoing edges from U}",
        #     r"\text{The input for \textit{r}-remote hop-reduction is:}",
        #     r"G^{out(U)}_{\phi_1 + \phi_2}",
        # )
        # gn_definition.arrange(DOWN, aligned_edge=ORIGIN, buff=0.3)
        # gn_definition.move_to(ORIGIN)
        # self.play(
        #     Write(gn_definition,run_time = 3)
        # )
        # self.wait(2)
        # self.play(
        #     Unwrite(gn_definition)
        # )
        g_c = copy.deepcopy(g)
        w_c = copy.deepcopy(weight_labels)

        
        # self.play(
        #     FadeIn(g),
        #     *[FadeIn(label,run_time=0.5) for label in weight_labels]
        # )
        gu = MathTex("G^{out(U)}_{\phi_1 + \phi_2}")
        gu.next_to(g.vertices[1],UP,buff=1)
        to_remove = [v for k,v in g.edges.items() if k in negative_edges and k not in out_U]
        for (u,v) in [(1,2),(7,8)]:
            del g.edges[(u,v)]
        for label in label_to_remove:
            weight_labels.remove(label)

        self.play(*[Uncreate(e,run_time=0.5) for e in to_remove],*[Unwrite(label) for label in label_to_remove])
        self.wait(2)
        self.play(Write(gu))
        self.wait(3)
        self.play(Uncreate(g),*[Unwrite(label) for label in weight_labels],Unwrite(gu))

        # input_description_1 = MathTex(
        #    r"{\text{As a product of previous steps of the algorithm,}",
        #    r"{\text{the negative vertices in }U\text{ are } r\text{-remote.}",
        #     r"\text{Notationally this is written:}}",
        #     r"{\left| R^{r}_{\phi_1 + \phi_2}(U)\right| > n/r}",
        # )

        # input_description_2 = MathTex(
        #     r"{\text{We will use the hop-reduction technique on the graph}",
        #     r"{G^{out(U)}_{\phi_1 + \phi_2} \text{ to eliminate }out(U)\text{.}}",
        #     r"{\text{A price function }\phi \text{ is computed by this step.}}"
        # )

        # description_3 = MathTex(
        #     r"{\text{We will use the hop-reduction technique on }G^{out(U)}_{\phi_1 + \phi_2}}",
        #     r"{\text{by computing all super-source distance }\delta_j = dist^j_{G^{out(U)}_{\phi_1 + \phi_2}}(V,v)}",
        #     r"\text{for all }v\text{ and }j, 0\leq j \leq r}\text{, and find } R:=\{v|\delta_r(v) < 0\}",
        #     r"\text{to construct our graph }H=(V_H,E_H,w_h)\text{ using }\delta_j(v)\text{ and }R"
        # )

        # input_description_1.arrange(DOWN, aligned_edge=ORIGIN, buff=0.3)
        # input_description_1.move_to(ORIGIN)
        # input_description_2.arrange(DOWN, aligned_edge=ORIGIN, buff=0.3)
        # input_description_2.move_to(ORIGIN)
        # description_3.arrange(DOWN, aligned_edge=ORIGIN, buff=0.3)
        # description_3.move_to(ORIGIN)


        # self.play(Write(input_description_1))
        # self.wait(3)
        # self.play(Unwrite(input_description_1))
        # self.wait(1)
        # self.play(Write(input_description_2))
        # self.wait(3)
        # self.play(Unwrite(input_description_2))
        # self.wait(1)
        # self.play(Write(description_3))
        # self.wait(3)
        # self.play(Unwrite(description_3))
        # self.wait(2)
        

        distance_maps = self.supersource_BFD(vertices,positive_edges,out_U,edge_weights,1)
        for k,distances in distance_maps[1].items():
            print(f"{k} with distance {distances}")

        h_weights = defaultdict(int)
        h_vertices,h_edges,h = self.construct_h(g, out_U,positive_edges)
        for (u_g,v_g) in h_edges:
            u = int(u_g[0])
            u_h = int(u_g[-1])
            v = int(v_g[0])
            v_h = int(v_g[-1])
            if u == v:
                h_weights[(u_g,v_g)] = 0 + distance_maps[u_h][u] - distance_maps[v_h][v]
            else:
                h_weights[(u_g,v_g)] = edge_weights[(u,v)] + distance_maps[u_h][u] - distance_maps[v_h][v]
        
        ################################################################################
        custom_6_edge = CurvedArrow(
            h.vertices["6_0"].get_center(),  # Start vertex
            h.vertices["6_1"].get_boundary_point(RIGHT),  # End vertex
            angle=1,  # Adjust angle to control curve
            tip_length=0.2    # Optional buffer to prevent overlapping
        ).set_z_index(0)
        custom_6_edge_2 = CurvedArrow(
            h.vertices["6_1"].get_center(),  # Start vertex
            h.vertices["6_0"].get_boundary_point(LEFT),  # End vertex
            angle=1,  # Adjust angle to control curve
            tip_length=0.2  
        ).set_z_index(0)
        ################################################################################
        custom_5_edge = CurvedArrow(
            h.vertices["5_0"].get_center(),  # Start vertex
            h.vertices["5_1"].get_boundary_point(RIGHT),  # End vertex
            angle=1,  # Adjust angle to control curve
            tip_length=0.2    # Optional buffer to prevent overlapping
        ).set_z_index(0)
        custom_5_edge_2 = CurvedArrow(
            h.vertices["5_1"].get_center(),  # Start vertex
            h.vertices["5_0"].get_boundary_point(LEFT),  # End vertex
            angle=1,  # Adjust angle to control curve
            tip_length=0.2  
        ).set_z_index(0)
        ################################################################################
        custom_7_edge = CurvedArrow(
            h.vertices["7_0"].get_center(),  # Start vertex
            h.vertices["7_1"].get_boundary_point(RIGHT),  # End vertex
            angle=1,  # Adjust angle to control curve
            tip_length=0.2
        ).set_z_index(0)

        custom_7_edge_2 = CurvedArrow(
            h.vertices["7_1"].get_center(),  # Start vertex
            h.vertices["7_0"].get_boundary_point(LEFT),  # End vertex
            angle=1,  # Adjust angle to control curve
            tip_length=0.2  ,
        ).set_z_index(0)
        edges_to_remove = [("6_1", "6_0"),("6_0", "6_1"),("5_1", "5_0"),("5_0", "5_1"),("7_1", "7_0"),("7_0", "7_1")]
        m_objects_remove = [h.edges[e] for e in edges_to_remove]
        for e in edges_to_remove:
            h.edges[e].set_z(-2)
        self.play(
            Create(h),
            *[Uncreate(e) for e in m_objects_remove],
            Create(custom_6_edge_2),
            Create(custom_6_edge),
            Create(custom_5_edge_2),
            Create(custom_5_edge),
            Create(custom_7_edge_2),
            Create(custom_7_edge)
        )
        # TODO: Think of way to make layers visible

        self.wait(3)

        weight_labels_h = []

        for (u,v) in h_edges:
            u_pos = h.vertices[u].get_center()
            v_pos = h.vertices[v].get_center()
            mid   = (u_pos + v_pos)/2
            weight = Text(str(h_weights[(u,v)]),font_size=24)
            if (mid[1] < 0):
                mid[1] = mid[1] -0.3
                weight.move_to(mid)
            elif (u_pos[0] == v_pos[0]):
                mid[0] = mid[0] +0.3
                weight.move_to(mid)
            else:
                mid[1] = mid[1] +0.3
                weight.move_to(mid)
            if (u,v) == ("6_1","6_0"):
                weight.next_to(custom_6_edge_2,RIGHT,buff = 0.1)
                weight.shift(DOWN*0.5)
                weight.shift(LEFT*0.6)
            if (u,v) == ("6_0","6_1"):
                weight.next_to(custom_6_edge,RIGHT,buff = 0.1)
            if (u,v) == ("5_1","5_0"):
                weight.next_to(custom_5_edge_2,RIGHT,buff = 0.1)
                weight.shift(DOWN*0.6)
            if (u,v) == ("5_0","5_1"):
                weight.next_to(custom_5_edge,LEFT,buff = 0.1)
                weight.shift(UP*0.6)
                weight.shift(LEFT*0.2)
            if (u,v) == ("7_1","7_0"):
                weight.next_to(custom_7_edge_2,RIGHT,buff = 0.1)
            if (u,v) == ("7_0","7_1"):
                weight.next_to(custom_7_edge,LEFT,buff = 0.1)
                weight.shift(LEFT*0.1)
            if (u,v) == ("5_0","6_0"):
                weight.shift(RIGHT*0.2)
                weight.shift(UP*0.3)
            weight_labels_h.append(weight)
        
        label_h = MathTex("H")
        label_h.next_to(h.vertices["1_0"],UP*4,buff=1)
        self.play(*[Write(label,run_time=0.5) for label in weight_labels_h])
        self.play(Write(label_h))
        self.wait(3)
        

        self.play(
            FadeOut(h),
            FadeOut(label_h),
            FadeOut(custom_6_edge),
            FadeOut(custom_6_edge_2),
            FadeOut(custom_5_edge),
            FadeOut(custom_5_edge_2),
            FadeOut(custom_7_edge),
            FadeOut(custom_7_edge_2),
            *[Unwrite(label,run_time=0.5) for label in weight_labels_h]
        )



        # description_4 = MathTex(
        #     r"{\text{On H, we compute super source distances}}",
        #     r"{d(v)=dist^{k}_H(V,v) \text{ and }d'(v)=dist^{k+1}_H(V,v) \text{ for all } v\in V_H}",
        #     r"{k=\lceil\hat{k}/r\rceil\text{, to check for cycles if }\exists v\in V_h\text{such that} d'(v) < d(v)}",
        #     r"{\text{if there exists a cycle the algorithm terminates, otherwise}}",
        #     r"{\text{we return price function }\phi : V \rightarrow \text{ with }\phi(v)=d(v)}"
        # )

        # description_4.arrange(DOWN, aligned_edge=ORIGIN, buff=0.3)
        # description_4.move_to(ORIGIN)

        # self.play(Write(description_4))
        # self.wait(3)
        # self.play(Unwrite(description_4))
        # self.wait(1)

        # TODO: check if k is the right value
        print("-------------------------------------------------")
        print(h_weights)
        print("-------------------------------------------------")

        h_neg_edges = [e for e in h_edges if h_weights[e] < 0]
        h_pos_edges = [e for e in h_edges if h_weights[e] >= 0]
        print(h_pos_edges)
        price_function = self.supersource_BFD(h_vertices,h_pos_edges,h_neg_edges,h_weights,3)
        for v in price_function:
            print(v)


        phi_k = [key for key in price_function[-1].keys()]
        phi_v = [value for value in price_function[-1].values()]
        dist_table = MathTable(
        [phi_k,phi_v],
        row_labels=[MathTex("v"), MathTex("\phi")],
        include_outer_lines=True,
        ).scale(0.6).move_to(ORIGIN)

        
        # phi_array = VGroup()
        # for i, (vertex, dist) in enumerate(phi):

        #     square = Square(side_length=1, color=WHITE, fill_opacity=0)

        #     number = Text(str(dist), font_size=24).move_to(square.get_center())

        #     vertex_label = Text(vertex, font_size=24, color=BLUE)
        #     vertex_label.next_to(square, UP)
        
        #     index = VGroup(vertex_label, square, number)
            
        #     index.move_to(RIGHT * i)
            
        #     phi_array.add(index)
        
        # phi_array.move_to(ORIGIN)
        # phi_label = MathTex("\phi = ", font_size=60).next_to(phi_array[0][1], LEFT)

        # TODO: check to see if we should remove 6_1 from the price function 
        self.play(
            Create(dist_table)
            # Write(phi_label),
            # Create(phi_array)
        )
        
        self.wait(2)

        self.play(
            # Unwrite(phi_label),
            # Uncreate(phi_array)
            Uncreate(dist_table)
        )



        

        self.wait(2)
        print(w_c)

        self.play(
            Create(g_c),
            *[Write(label) for label in w_c]
        )
        self.wait(2)

        

        new_labels = []
        print(price_function[-1])
        for (u,v) in edges:
            u_pos = g.vertices[u].get_center()
            v_pos = g.vertices[v].get_center()
            mid   = (u_pos + v_pos)/2
            
            u_s = str(u)+"_0"
            v_s = str(v)+"_0"
            
            weight_value =  edge_weights[(u,v)]+price_function[-1][u_s]-price_function[-1][v_s]

            weight = Text(str(weight_value),font_size=24)
            if (mid[1] < 0):
                mid[1] = mid[1] -0.3
                weight.move_to(mid)
            elif (u_pos[0] == v_pos[0]):
                mid[0] = mid[0] +0.3
                weight.move_to(mid)
            else:
                mid[1] = mid[1] +0.3
                weight.move_to(mid)
            new_labels.append(weight)

        print(new_labels)
        
        to_transform = zip(w_c,new_labels)

        vertices,edges,U,out_U,vertex_layout,edge_weights,positive_edges,negative_edges = self.create_r_removal_example()

        edge_config = {
            "tip_config": {"tip_length": 0.2, "tip_width": 0.2},
            (1, 2): {"stroke_color": RED},
            (7,8): {"stroke_color": RED},
            }
        vertex_config = {
            1: {"fill_color": RED},
            7: {"fill_color": RED},
            }

        g = DiGraph(vertices, edges, edge_config = edge_config,vertex_config=vertex_config, layout=vertex_layout)

        self.play(
            *[ReplacementTransform(l1,l2) for (l1,l2) in to_transform],
            ReplacementTransform(g_c,g)
        )   
        self.wait(4)