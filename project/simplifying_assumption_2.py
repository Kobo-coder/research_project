from manim import *
from collections import defaultdict

from manim.utils.color.X11 import HOTPINK


class SimplifyingAssumption2(Scene):
    def construct(self):
        num_vertices = 10
        vertices = [i for i in range(num_vertices)]
        edges = [(0,i) for i in range(1,num_vertices,1)]
        print(vertices)
        print(edges)

        edge_config={"tip_config": {"tip_length": 0.2, "tip_width": 0.2},
                     (0, 10): {"stroke_color": HOTPINK}
                     }
        vertex_layout = defaultdict(int)
        vertex_layout[0]= [-2,0,0]
        offset = 2
        for i in range(1,num_vertices,1):
            vertex_layout[i] = [2, offset, 0]
            offset = offset - 0.5
            
        print(vertex_layout)
        
        g = DiGraph(vertices, edges, edge_config = edge_config, layout = vertex_layout, labels = False)

        self.play(
           Create(g)
        )
        self.wait(2)

        self.play(
            Flash(g.vertices[0], color=WHITE)
        )
 
        num_vertices += 1
        vertices = [i for i in range(num_vertices)]
        edges = [(0,10)] + [(10,i) for i in range(1,(num_vertices//2)+1)] + [(0, i) for i in range((num_vertices//2)+1, num_vertices-1)]

        vertex_layout_2 = defaultdict(int)
        offset = 2.0
        vertex_layout_2[0] = [-2,0,0]
        vertex_layout_2[10] = [-0.5,1,0]
        for i in range(1,num_vertices-1):
            vertex_layout_2[i] = [2, offset, 0]
            offset = offset - 0.5

        #for key,value in vertex_layout_2.items():
        #    print("old: "+str(key) + " : " + str(vertex_layout[key]))
        #    print("new : "+ str(key) + " : " + str(value))
        h = DiGraph(vertices, edges, edge_config = edge_config, layout = vertex_layout_2, labels = False)

        print(f"vertices: {vertices}")
        print(f"edges: {edges}")

        #self.play(
            #Uncreate(g),
            #Create(h)
            #ReplacementTransform(g, h)
            #FadeOut(g),
            #FadeIn(h)
            #TransformMatchingShapes(g,h)
        #)

        self.play(
            ReplacementTransform(g,h),
            *[ReplacementTransform(g.vertices[v], h.vertices[v]) for v in g.vertices if v in h.vertices],
            *[ReplacementTransform(g.edges[e], h.edges[e]) for e in g.edges if e in h.edges],
            Create(h.vertices[10])
        )
        self.wait(2)





class SimplifyingAssumption3(Scene):
    def construct(self):
        num_vertices = 10
        vertices = [i for i in range(num_vertices)]
        edges = [(0, i) for i in range(1, num_vertices, 1)]
        print(vertices)
        print(edges)

        edge_config = {"tip_config": {"tip_length": 0.2, "tip_width": 0.2},
                       (0, 10): {"stroke_color": HOTPINK}
                       }
        vertex_layout = defaultdict(int)
        vertex_layout[0] = [-2, 0, 0]
        offset = 2
        for i in range(1, num_vertices, 1):
            vertex_layout[i] = [2, offset, 0]
            offset = offset - 0.5

        print(vertex_layout)

        g = DiGraph(vertices, edges, edge_config=edge_config, layout=vertex_layout, labels=False)

        self.play(
            Create(g)
        )
        self.wait(2)

        self.play(
            Flash(g.vertices[0], color=WHITE)
        )

        num_vertices += 1
        vertices = [i for i in range(num_vertices)]
        edges = [(0, 10)] + [(10, i) for i in range(1, (num_vertices // 2) + 1)] + [(0, i) for i in
                                                                                    range((num_vertices // 2) + 1,
                                                                                          num_vertices - 1)]

        vertex_layout_2 = defaultdict(int)
        offset = 2.0
        vertex_layout_2[0] = [-2, 0, 0]
        vertex_layout_2[10] = [-0.5, 1, 0]
        for i in range(1, num_vertices - 1):
            vertex_layout_2[i] = [2, offset, 0]
            offset = offset - 0.5

        h = g.remove_edges(*[(0,2), (0,3), (0,4), (0,5)])
        #h = DiGraph(vertices, edges, edge_config=edge_config, layout=vertex_layout_2, labels=False)

        print(f"vertices: {vertices}")
        print(f"edges: {edges}")

        self.play(
            ReplacementTransform(g, h),
            #Uncreate(g),
            #Create(h)
        )
        self.wait(2)