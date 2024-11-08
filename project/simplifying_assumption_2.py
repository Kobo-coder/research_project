from manim import *
from collections import defaultdict
import numpy as np
from scipy.fftpack import shift


class SimplifyingAssumption2(Scene):
    def construct(self):
        num_vertices = 13
        vertices_g = [i for i in range(num_vertices)]
        edges_g = [(0,1), (0,2)] + [(1,i) for i in range(3, 8)] + [(2,i) for i in range(8, len(vertices_g))]

        edge_config={}#{"tip_config": {"tip_length": 0.2, "tip_width": 0.2}}
        vertex_layout = defaultdict(int)
        vertex_layout[0] = [-1.5, 0, 0]
        vertex_layout[1] = [-1.5, 0, 0]
        vertex_layout[2] = [-1.5, 0, 0]
        offset = 2.25
        for upper in range(3,num_vertices):
            vertex_layout[upper] = [1.5, offset, 0]
            offset = offset - 0.5

        g = DiGraph(vertices_g, edges_g, edge_config = edge_config, vertex_config= {0: {"fill_opacity": 0}, 1: {"fill_color": WHITE}, 2: {"fill_color": WHITE}}, layout = vertex_layout)

        self.play(
           Create(g)
        )
        self.wait(2)

        text1 = Tex(r"(2) Every vertex must have a degree of at most $O(m/n)$", font_size=28).shift(UP * 2.5, LEFT*3)
        text2 = MathTex(f"n = {num_vertices - 3}, \ \  m = {len(edges_g)-1}", font_size=28).next_to(text1, DOWN, aligned_edge=LEFT)

        self.play(Write(text1))
        self.wait()
        self.play(Write(text2))
        self.wait()



        self.play(
            Flash(g.vertices[1], color=PURPLE),
            #FadeToColor(g.vertices[1], color=PURPLE),
            #FadeToColor(g.vertices[2], color=PURPLE)
        )

        upper = [3,4,5,6,7]
        lower = [8,9,10,11,12]

        vertices_h = (vertices_g + [13,14,15,16])
        edges_h = [(0, 1), (0, 2), (13,3), (13,4), (13,5), (14,6), (14,7), (15,8), (15,9), (15,10), (16,11), (16,12)]

        vertex_layout[0] = [-2, 0, 0]
        vertex_layout[1] = [-0.5, 1.25, 0]
        vertex_layout[13] = [-0.5, 1.25, 0]
        vertex_layout[14] = [-0.5, 1.25, 0]

        vertex_layout[2] = [-0.5, -1.25, 0]
        vertex_layout[15] = [-0.5, -1.25, 0]
        vertex_layout[16] = [-0.5, -1.25, 0]
        idx = 3
        for d in np.arange(1, -1.5, -0.5):
            vertex_layout[idx] = [1.5, d+1.25, 0]
            vertex_layout[idx+5] = [1.5, d-1.25, 0]
            idx += 1


        h = DiGraph(vertices_h, edges_h, edge_config = edge_config, layout = vertex_layout)

        self.play(
            AnimationGroup(
                g.vertices[0].animate.shift(LEFT * 0.5),
                g.vertices[1].animate.shift(UP * 1.25).shift(RIGHT),
                *[g.vertices[v].animate.become(h.vertices[v]) for v in upper],
                *[g.edges[(1, v)].animate.become(h.edges[(13, v)]) for v in upper[:3]],
                *[g.edges[(1, v)].animate.become(h.edges[(14, v)]) for v in upper[3:]],
                g.vertices[2].animate.shift(DOWN * 1.25).shift(RIGHT),
                *[g.vertices[e].animate.become(h.vertices[e]) for e in lower],
                *[g.edges[(2, v)].animate.become(h.edges[(15, v)]) for v in lower[:3]],
                *[g.edges[(2, v)].animate.become(h.edges[(16, v)]) for v in lower[3:]]
                )
        )
        self.wait()

        self.play(
            Create(g.vertices[0].set_opacity(1))
        )

        self.wait()
        self.play(FadeIn(h))
        self.wait()
        self.play(FadeOut(g))
        self.wait()


        upper_upper = [3,4,5]
        upper_lower = [6,7]
        lower_upper = [8,9,10]
        lower_lower = [11,12]
        self.play(h.animate.shift(LEFT))
        self.wait()

        h.vertices[1].set_opacity(0)
        h.vertices[2].set_opacity(0)

        vertex_layout_i = defaultdict(int)
        vertex_layout_i[0] = [-3, 0, 0]
        vertex_layout_i[1] = [-1.5, 1.25, 0]
        vertex_layout_i[2] = [-1.5, -1.25, 0]

        vertex_layout_i[13] = [-0.08, 1.75,0]
        vertex_layout_i[14] = [-0.08, 0.75, 0]
        vertex_layout_i[15] = [-0.08, -0.75, 0]
        vertex_layout_i[16] = [-0.08, -1.75, 0]

        # For upper_upper
        vertex_layout_i[upper_upper[0]] = [(-0.08 + 2), (1.75 + 0.5), 0]
        vertex_layout_i[upper_upper[1]] = [(-0.08 + 2), 1.75, 0]
        vertex_layout_i[upper_upper[2]] = [(-0.08 + 2), (1.75 - 0.5), 0]

        # For upper_lower
        vertex_layout_i[upper_lower[0]] = [(-0.08 + 2), (0.75 + 0.25), 0]
        vertex_layout_i[upper_lower[1]] = [(-0.08 + 2), (0.75 - 0.25), 0]

        # For lower_upper
        vertex_layout_i[lower_upper[0]] = [(-0.08 + 2), -(0.75 - 0.5), 0]
        vertex_layout_i[lower_upper[1]] = [(-0.08 + 2), -0.75, 0]
        vertex_layout_i[lower_upper[2]] = [(-0.08 + 2), -(0.75 + 0.5), 0]

        # For lower_lower
        vertex_layout_i[lower_lower[0]] = [(-0.08 + 2), -(1.75 - 0.25), 0]
        vertex_layout_i[lower_lower[1]] = [(-0.08 + 2), -(1.75 + 0.25), 0]

        i = DiGraph(vertices_h, edges_h+[(1,13), (1,14), (2,15), (2,16)], layout = vertex_layout_i)

        h.vertices[1].set_opacity(1)
        h.vertices[2].set_opacity(1)

        self.play(
            AnimationGroup(
                *[h.vertices[v].animate.move_to(vertex_layout_i[v]) for v in [13,14,15,16]],

                *[h.vertices[v].animate.become(i.vertices[v]) for v in upper_upper],
                *[h.edges[(13,e)].animate.become(i.edges[(13,e)]) for e in upper_upper],

                *[h.vertices[v].animate.become(i.vertices[v]) for v in upper_lower],
                *[h.edges[(14, e)].animate.become(i.edges[(14, e)]) for e in upper_lower],

                *[h.vertices[v].animate.become(i.vertices[v]) for v in lower_upper],
                *[h.edges[(15, e)].animate.become(i.edges[(15,e)]) for e in lower_upper],

                *[h.vertices[v].animate.become(i.vertices[v]) for v in lower_lower],
                *[h.edges[(16, e)].animate.become(i.edges[(16, e)]) for e in lower_lower],
            )
        )
        self.wait()

        self.play(
            AnimationGroup(
                # Create(i.vertices[1]),
                # Create(i.vertices[2]),
                # Create(i.vertices[13]),
                # Create(i.edges[(1, 13)]),
                # Create(i.edges[(1, 14)]),
                # Create(i.edges[(2, 15)]),
                # Create(i.edges[(2, 16)])
                FadeIn(i)
            )
        )



