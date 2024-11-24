from manim import *
from collections import defaultdict
import random as rand
class r_remote_removal(Scene):

    # def generate_vertex_layout(self,vertices, boundaries, min_distance=0.5):
    #     x_bound, y_bound = boundaries

    #     layout = {}

    #     x = np.linspace(x_bound, -x_bound, 10)
    #     y = np.linspace(y_bound, -y_bound, 5)

    #     xs, ys = np.meshgrid(x, y)
    #     points = [(x, y) for x, y in zip(xs.flatten(), ys.flatten())]

    #     for v in vertices:
    #         x,y = points[v-1]
    #         placed = False
    #         attempts = 0

    #         while not placed and attempts < 1000:
    #             x += rand.uniform(-0.3, 0.3)
    #             y += rand.uniform(-0.3, 0.3)

    #             if self.is_position_valid(x, y, [pos for pos in layout.values()], min_distance):
    #                 layout[v] = [x, y, 0]
    #                 placed = True

    #             attempts += 1

    #             if attempts % 100 == 0:
    #                 min_distance *= 0.95

    #     return layout
    
    # def is_position_valid(self,x, y, existing_positions, min_dist):
    #     for pos in existing_positions:
    #         if np.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2) < min_dist:
    #             return False
    #     return True
    
    # def create_graph(self,n,m):
    #     vertices = [i for i in range(n)]
    #     edges = []
    #     negative_vertices = set()
    #     negative_edges = set()
    #     max_degree = 4*(m/n)
    #     edge_count = 0
    #     degrees = defaultdict(int)
    #     weights = defaultdict(lambda: 0.0)
    #     u = 0; v = 0

    #     while edge_count < m:
    #         # may seem weird at first but we always try to make new edges otherwise we can get stuck in infinite loop
    #         # in case of negative
    #         u = rand.randint(0,n-1)
    #         v = rand.randint(0,n-1)
        
    #         #diabolical double while-loop - but we want to avoid self-loops
    #         while (u == v or edges.count((u,v))):
    #             u = rand.randint(0,n-1)
    #             v = rand.randint(0,n-1)

    #         negative = True if rand.random() < 0.3 else False

    #         if negative:
    #             if u in negative_vertices or degrees.get(u):
    #                 # skip because a negative vertex may only have one outgoing edge.
    #                 continue
    #             # add to negative vertex set and negative edges set
    #             negative_vertices.add(u)
    #             negative_edges.add((u,v))
    #             # increase degree of both vertices
    #             degrees[u] = degrees[u] + 1
    #             degrees[v] = degrees[v] + 1

    #         # check degree of vertices
    #         if (degrees[u] >= max_degree or degrees[v] >= max_degree):
    #             #skip because one may not have any more edges
    #             continue
            
    #         # randly choose a negative or positive weight depending on sign. 
    #         # We use a larger range for negative weights to increase the reach of vertices
    #         # TODO: Check if range must be decrease for negative weights if r-reach is too large
    #         weights[(u,v)] = rand.uniform(0,50.0) if not negative else rand.uniform(-100,-1)
            
    #         edges.append((u,v))
    #         edge_count += 1

    #     # TODO: Consider what is worth returning.
    #     return vertices,edges,weights,negative_vertices,negative_edges
    
    def create_r_removal_example(self):
        vertices = [i for i in range(1,11)]
        edges = [(1,2),(2,3),(2,4),(2,5),(3,6),(4,6),(5,6),(6,7),(7,8),(8,10),(8,9),(10,9)]
        U = {3,4,5}
        negative_edges = {(3,6),(4,6),(5,6)}

        vertex_layout = {
            1: [-5,0,0],
            2: [-3,0,0],
            3: [-1,1,0],
            4: [-1,0,0],
            5: [-1,-1,0],
            6: [1,0,0],
            7: [3,0,0],
            8: [4,0,0],
            9: [5,1,0],
            10:[5,-1,0]
        }

        edge_weights = {
            (1,2): 2,
            (2,3): -2,
            (2,4): -5,
            (2,5): -9,
            (3,6): -6,
            (4,6): -8,
            (5,6): -4,
            (6,7): 2,
            (7,8): 20,
            (8,9): 3,
            (8,10): 5,
            (10,9): 1
        }

        return vertices,edges,U,negative_edges,vertex_layout,edge_weights


    
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


        vertices,edges,U,negative_edges,vertex_layout,edge_weights = self.create_r_removal_example()

        radius = 0.1
        edge_config = {
            # 'buff': radius/2,
            # 'max_tip_length_to_length_ratio': 0.15,
            # 'stroke_width': 3.5
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
        
        # for label in weight_labels:
        #     self.play(Write(label,run_time=0.5))
        # self.wait(2)


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
        U_indicator_label = Text("(x,U,y)",font_size=48,color=BLUE)
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


        new_text_idk = MathTex(
           # r"{\text{The set of edges}},
        )
        