from manim import *

class CreateGraph(Scene):
    def construct(self):
        vertices = [1,2,3,4,5,6]
        edges = [(1,3), (2,3), (3,4), (3,5), (3,6)]
        lt = {1: [-1, 1, 0], 2: [-1, -1, 0], 3: [0, 0, 0], 4: [2, 1, 0], 5: [2, 0, 0], 6: [2, -1, 0]}
        g = Graph(vertices, edges, layout=lt,
                  vertex_config={3: {"fill_color": RED}},
                  edge_config={(3, 4): {"stroke_color": RED},
                               (3, 5): {"stroke_color": RED},
                               (3, 6): {"stroke_color": RED}})
        self.play(Create(g))

        self.wait()

        # Add new vertex 7 and position it
        new_vertex = 7
        g.add_vertices(new_vertex, positions={7: [0, 0, 0]})

        new_edges = [(1, 7), (2, 7), (7, 3)]


        self.play(
            Create(g.add_edges(*new_edges)),
            g.vertices[3].animate.shift(RIGHT * 1),
            g.vertices[4].animate.shift(RIGHT * 1),
            g.vertices[5].animate.shift(RIGHT * 1),
            g.vertices[6].animate.shift(RIGHT * 1),
            g.edges[(3,4)].animate.shift(RIGHT * 1),
            g.edges[(3,5)].animate.shift(RIGHT * 1),
            g.edges[(3,6)].animate.shift(RIGHT * 1),
            Create(g.add_edges(*new_edges))
        )

        neg_edge = [(7, 3)]
        self.play(
            FadeToColor(g.edges[(3,4)], WHITE),
            FadeToColor(g.edges[(3, 5)], WHITE),
            FadeToColor(g.edges[(3, 6)], WHITE),
            FadeToColor(g.vertices[3], WHITE),
            FadeToColor(g.vertices[7], RED),
            Create(g.add_edges(*neg_edge, edge_config={(7, 3): {"stroke_color": RED}}))
        )

        self.wait()