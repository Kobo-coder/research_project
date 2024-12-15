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
            if not h == i:
                bellman_ford_round()

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
            (7,8): {"stroke_color": RED},
            }
        vertex_config = {
            1: {"fill_color": RED},
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
        
        self.play(
            ReplacementTransform(U_indicator,U_indicator_2),
            ReplacementTransform(U_indicator_label,U_indicator_label_2),
            Create(x_indicator),
            Write(x_indicator_label),
            Create(y_indicator),
            Write(y_indicator_label)
        )

        self.wait(2)
        
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

        g_c = copy.deepcopy(g)
        w_c = copy.deepcopy(weight_labels)

        
        gu = MathTex("G^{out(U)}_{\phi_1 + \phi_2}")
        gu.next_to(g.vertices[1],UP*1.75,buff=1)
        to_remove = [v for k,v in g.edges.items() if k in negative_edges and k not in out_U]
        for (u,v) in [(1,2),(7,8)]:
            del g.edges[(u,v)]
        for label in label_to_remove:
            weight_labels.remove(label)

        self.play(Write(gu),
                  *[Uncreate(e,run_time=0.5) for e in to_remove],
                  *[Unwrite(label) for label in label_to_remove]
                  )
        self.wait(5)
        self.play(Uncreate(g),*[Unwrite(label) for label in weight_labels],Unwrite(gu))

        distance_maps = self.supersource_BFD(vertices,positive_edges,out_U,edge_weights,1)

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
            h.vertices["6_0"].get_center(),
            h.vertices["6_1"].get_boundary_point(RIGHT),
            angle=1,
            tip_length=0.2 
        ).set_z_index(0)
        custom_6_edge_2 = CurvedArrow(
            h.vertices["6_1"].get_center(),
            h.vertices["6_0"].get_boundary_point(LEFT), 
            angle=1,
            tip_length=0.2  
        ).set_z_index(0)
        ################################################################################
        custom_5_edge = CurvedArrow(
            h.vertices["5_0"].get_center(),
            h.vertices["5_1"].get_boundary_point(RIGHT),
            angle=1,
            tip_length=0.2
        ).set_z_index(0)
        custom_5_edge_2 = CurvedArrow(
            h.vertices["5_1"].get_center(),
            h.vertices["5_0"].get_boundary_point(LEFT), 
            angle=1, 
            tip_length=0.2  
        ).set_z_index(0)
        ################################################################################
        custom_7_edge = CurvedArrow(
            h.vertices["7_0"].get_center(),
            h.vertices["7_1"].get_boundary_point(RIGHT),
            angle=1,
            tip_length=0.2
        ).set_z_index(0)

        custom_7_edge_2 = CurvedArrow(
            h.vertices["7_1"].get_center(), 
            h.vertices["7_0"].get_boundary_point(LEFT),  
            angle=1, 
            tip_length=0.2  ,
        ).set_z_index(0)
        edges_to_remove = [("6_1", "6_0"),("6_0", "6_1"),("5_1", "5_0"),("5_0", "5_1"),("7_1", "7_0"),("7_0", "7_1")]
        m_objects_remove = [h.edges[e] for e in edges_to_remove]
        [h.edges[e].set_z_index(-2) for e in edges_to_remove]
        [h.vertices[v].set_z_index(10) for v in h_vertices]

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

        # TODO: check if k is the right value

        h_neg_edges = [e for e in h_edges if h_weights[e] < 0]
        h_pos_edges = [e for e in h_edges if h_weights[e] >= 0]

        price_function = self.supersource_BFD(h_vertices,h_pos_edges,h_neg_edges,h_weights,3)

        phi_k = [key for key in price_function[-1].keys()]
        phi_v = [value for value in price_function[-1].values()]
        dist_table = MathTable(
        [phi_k,phi_v],
        row_labels=[MathTex("v"), MathTex("\phi")],
        include_outer_lines=True,
        ).scale(0.6).move_to(ORIGIN)

        self.play(
            Create(dist_table)
        )
        
        self.wait(6)

        self.play(
            Uncreate(dist_table)
        )

        self.wait(2)

        self.play(
            Create(g_c),
            *[Write(label) for label in w_c]
        )
        self.wait(2)

        new_labels = []
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
        self.wait(2)