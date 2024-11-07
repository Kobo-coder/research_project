from manim import *
from collections import defaultdict
from manim.utils.color.X11 import HOTPINK


class SimplifyingAssumption2(Scene):
    def construct(self):
        num_vertices = 13
        vertices = [i for i in range(num_vertices)]
        edges = [(0,1), (0,2)] + [(1,i) for i in range(3, 8)] + [(2,i) for i in range(8, len(vertices))]

        print(vertices)
        print(edges)

        edge_config={}#{"tip_config": {"tip_length": 0.2, "tip_width": 0.2}}
        vertex_layout = defaultdict(int)
        vertex_layout[0]= [-1.5,0,0]
        vertex_layout[1]= [-1.5,0,0]
        vertex_layout[2]= [-1.5,0,0]
        offset = 2.25
        for i in range(3,num_vertices):
            vertex_layout[i] = [1.5, offset, 0]
            offset = offset - 0.5
            
        print(vertex_layout)
        
        g = DiGraph(vertices, edges, edge_config = edge_config, vertex_config= {0: {"fill_opacity": 0}}, layout = vertex_layout, labels = False)

        self.play(
           Create(g)
        )
        self.wait(2)

        self.play(
            Flash(g.vertices[0], color=YELLOW)
        )


        upper = [3,4,5,6,7]
        lower = [8,9,10,11,12]

        vertex_layout[0] = [-2, 0, 0]
        vertex_layout[1] = [-0.5, 1.65, 0]
        vertex_layout[2] = [-0.5, -1.65, 0]
        offset = 2.25
        for i in range(3, num_vertices):
            if i in {3,4,5,6,7}:
                vertex_layout[i] = [1.5, offset+0.4, 0]
            else:
                vertex_layout[i] = [1.5, offset-0.4, 0]
            offset = offset - 0.5


        h = DiGraph(vertices, edges, edge_config = edge_config, layout = vertex_layout)

        self.play(
            AnimationGroup(
                g.vertices[0].animate.shift(LEFT * 0.5),
                g.vertices[1].animate.shift(UP * 1.65).shift(RIGHT),
                *[g.vertices[v].animate.shift(UP * 0.4) for v in upper],
                *[g.edges[1, v].animate.become(h.edges[1, v]) for v in upper],
                g.vertices[2].animate.shift(DOWN * 1.65).shift(RIGHT),
                *[g.vertices[e].animate.shift(DOWN * 0.4) for e in lower],
                *[g.edges[2, v].animate.become(h.edges[2, v]) for v in lower]
                )
        )
        self.wait()

        self.play(
            Create(g.vertices[0].set_opacity(1))
        )
        self.wait()

        self.play(
            Create(h.edges[(0, 1)]),
            Create(h.edges[(0, 2)])
        )
        self.wait()
