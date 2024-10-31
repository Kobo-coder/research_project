from manim import *
from collections import defaultdict

class SimplifyingAssumption2(Scene):
    def construct(self):
        num_vertices = 11
        vertices = [i for i in range(num_vertices)]
        edges = [(0,i) for i in range(1,num_vertices,1)]
        print(vertices)
        print(edges)

        edge_config={"tip_config": {"tip_length": 0.2, "tip_width": 0.2}}
        vertex_layout = defaultdict(int)
        vertex_layout[0]= [-2,0,0]
        offset = -2
        for i in range(1,num_vertices,1):
            vertex_layout[i] = [3,offset,0]
            offset = offset + 0.5
            
        #print(vertex_layout)
        
        g = DiGraph(vertices, edges, edge_config=edge_config,layout=vertex_layout)
        
        self.play(
            Create(g)
        )
        self.wait()

 
        num_vertices += 2
        vertices = [i for i in range(num_vertices)]
        edges = [(11,i) for i in range(1,num_vertices//2,1)]+[(12,i) for i in range(num_vertices//2,num_vertices-2,1)]
        edges += [(0,11),(0,12)]

        vertex_layout_2 = defaultdict(int)
        offset = -2.0
        vertex_layout_2[0] = [-3,0,0]
        vertex_layout_2[11] = [-2,1,0]
        vertex_layout_2[12] = [-2,-1,0]
        for i in range(1,num_vertices-2,1):
            vertex_layout_2[i] = [3,offset,0]
            offset = offset + 0.5

        for key,value in vertex_layout_2.items():
            print(str(key) + " : " + str(value))
        g_2 = DiGraph(vertices, edges, edge_config=edge_config,layout=vertex_layout_2)

        print(vertices)
        print(edges)

        self.play(
            g.animate.become(g_2)
        )
        
        
