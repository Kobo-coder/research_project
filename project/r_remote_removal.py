from manim import *
from collections import defaultdict
import random as rand
class r_remote_removal(Scene):

    def generate_vertex_layout(self,vertices, boundaries, min_distance=0.5):
        x_bound, y_bound = boundaries

        layout = {}

        x = np.linspace(x_bound, -x_bound, 10)
        y = np.linspace(y_bound, -y_bound, 5)

        xs, ys = np.meshgrid(x, y)
        points = [(x, y) for x, y in zip(xs.flatten(), ys.flatten())]

        for v in vertices:
            x,y = points[v-1]
            placed = False
            attempts = 0

            while not placed and attempts < 1000:
                x += rand.uniform(-0.3, 0.3)
                y += rand.uniform(-0.3, 0.3)

                if self.is_position_valid(x, y, [pos for pos in layout.values()], min_distance):
                    layout[v] = [x, y, 0]
                    placed = True

                attempts += 1

                if attempts % 100 == 0:
                    min_distance *= 0.95

        return layout
    
    def is_position_valid(self,x, y, existing_positions, min_dist):
        for pos in existing_positions:
            if np.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2) < min_dist:
                return False
        return True
    
    def create_graph(self,n,m):
        vertices = [i for i in range(n)]
        edges = []
        negative_vertices = set()
        negative_edges = set()
        max_degree = 4*(m/n)
        edge_count = 0
        degrees = defaultdict(int)
        weights = defaultdict(lambda: 0.0)
        u = 0; v = 0

        while edge_count < m:
            # may seem weird at first but we always try to make new edges otherwise we can get stuck in infinite loop
            # in case of negative
            u = rand.randint(0,n-1)
            v = rand.randint(0,n-1)
        
            #diabolical double while-loop - but we want to avoid self-loops
            while (u == v or edges.count((u,v))):
                u = rand.randint(0,n-1)
                v = rand.randint(0,n-1)

            negative = True if rand.random() < 0.3 else False

            if negative:
                if u in negative_vertices or degrees.get(u):
                    # skip because a negative vertex may only have one outgoing edge.
                    continue
                # add to negative vertex set and negative edges set
                negative_vertices.add(u)
                negative_edges.add((u,v))
                # increase degree of both vertices
                degrees[u] = degrees[u] + 1
                degrees[v] = degrees[v] + 1

            # check degree of vertices
            if (degrees[u] >= max_degree or degrees[v] >= max_degree):
                #skip because one may not have any more edges
                continue
            
            # randly choose a negative or positive weight depending on sign. 
            # We use a larger range for negative weights to increase the reach of vertices
            # TODO: Check if range must be decrease for negative weights if r-reach is too large
            weights[(u,v)] = rand.uniform(0,50.0) if not negative else rand.uniform(-100,-1)
            
            edges.append((u,v))
            edge_count += 1

        # TODO: Consider what is worth returning.
        return vertices,edges,weights,negative_vertices,negative_edges
    
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
            Write(title,run_time=3), 
            Create(h_line,run_time=3),
            
        )
        self.wait(3)
        self.play(
            Unwrite(title,run_time=3),
            Uncreate(h_line,run_time=3)
        )

        # radius = 1.0
        # arrow_length = radius + 0.1
        # g.add_edges(*edges,edge_type=Arrow,edge_config={'buff': arrow_length/2,'max_tip_length_to_length_ratio': 0.1,'stroke_width': 3.5})

        vertices,edges,weights,negative_vertices,negative_edges = self.create_graph(50,60)
        positions = self.generate_vertex_layout(vertices,(5, 2.5))
        print(vertices)
        print(edges)
        edge_config={}
    

        g = DiGraph(vertices, edges, edge_config = edge_config, vertex_config= {0: {"fill_opacity": 0}, 1: {"fill_color": WHITE}, 2: {"fill_color": WHITE}}, layout=positions)
        self.play(
            Create(g,run_time=4)
        )
        self.wait(4)
    