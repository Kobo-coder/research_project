from manim import *
from collections import defaultdict

class SimplifyingAssumption2(Scene):
    def construct(self):
        num_vertices = 13
        vertices_g = [i for i in range(num_vertices)]
        edges_g = [(0,1), (0,2)] + [(1,i) for i in range(3, 8)] + [(2,i) for i in range(8, len(vertices_g))]

        print(vertices_g)
        print(edges_g)

        edge_config={}#{"tip_config": {"tip_length": 0.2, "tip_width": 0.2}}
        vertex_layout = defaultdict(int)
        vertex_layout[0]= [-1.5,0,0]
        vertex_layout[1]= [-1.5,0,0]
        vertex_layout[2]= [-1.5,0,0]
        offset = 2.25
        for i in range(3,num_vertices):
            vertex_layout[i] = [1.5, offset, 0]
            offset = offset - 0.5

        g = DiGraph(vertices_g, edges_g, edge_config = edge_config, vertex_config= {0: {"fill_opacity": 0}, 1: {"fill_color": WHITE}, 2: {"fill_color": WHITE}}, layout = vertex_layout, labels = False)

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
        print(vertices_h)
        print(edges_h)

        vertex_layout[0] = [-2, 0, 0]
        vertex_layout[1] = [-0.5, 1.65, 0]
        vertex_layout[13] = [-0.5, 1.65, 0]
        vertex_layout[14] = [-0.5, 1.65, 0]

        vertex_layout[2] = [-0.5, -1.65, 0]
        vertex_layout[15] = [-0.5, -1.65, 0]
        vertex_layout[16] = [-0.5, -1.65, 0]
        offset = 2.25
        for i in range(3, num_vertices):
            if i in {3,4,5,6,7}:
                vertex_layout[i] = [1.5, offset+0.4, 0]
            else:
                vertex_layout[i] = [1.5, offset-0.4, 0]
            offset = offset - 0.5


        h = DiGraph(vertices_h, edges_h, edge_config = edge_config, layout = vertex_layout)

        self.play(
            AnimationGroup(
                g.vertices[0].animate.shift(LEFT * 0.5),
                g.vertices[1].animate.shift(UP * 1.65).shift(RIGHT),
                *[g.vertices[v].animate.shift(UP * 0.4) for v in upper],
                *[g.edges[1, v].animate.become(h.edges[13, v]) for v in upper[:3]],
                *[g.edges[1, v].animate.become(h.edges[14, v]) for v in upper[3:]],
                g.vertices[2].animate.shift(DOWN * 1.65).shift(RIGHT),
                *[g.vertices[e].animate.shift(DOWN * 0.4) for e in lower],
                *[g.edges[2, v].animate.become(h.edges[15, v]) for v in lower[:3]],
                *[g.edges[2, v].animate.become(h.edges[16, v]) for v in lower[3:]]
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

        vertex_layout_i = vertex_layout
        vertex_layout_i[13] = [-0.58, 2.15,0]
        vertex_layout_i[14] = [-0.58, 1.15, 0]
        vertex_layout_i[15] = [-0.58, -1.15, 0]
        vertex_layout_i[16] = [-0.58, -2.15, 0]

        for (i,j) in [(upper_upper, upper_lower), (lower_upper, lower_lower)]:
            vertex_layout_i[i[0]] = [(-0.58 + 3), (2.15 + 0.5), 0]
            vertex_layout_i[i[1]] = [(-0.58 + 3), 2.15, 0]
            vertex_layout_i[i[2]] = [(-0.58 + 3), (2.15 - 0.5), 0]

            vertex_layout_i[j[0]] = [(-0.58+3),  (1.15+0.25), 0]
            vertex_layout_i[j[1]] = [(-0.58+3),  (1.15-0.25), 0]



        print("her")
        print(vertex_layout_i)

        i = DiGraph(vertices_h, edges_h+[(1,13), (1,14), (2,15), (2,16)], layout = vertex_layout_i)

        self.play(
            AnimationGroup(
                h.vertices[13].animate.shift(UP * 0.5).shift(RIGHT),
                h.vertices[14].animate.shift(DOWN * 0.5).shift(RIGHT),
                *[h.vertices[v].animate.move_to(vertex_layout_i[v]) for v in upper_upper],


                h.vertices[15].animate.shift(UP * 0.5).shift(RIGHT),
                h.vertices[16].animate.shift(DOWN * 0.5).shift(RIGHT),


                # h.vertices[1].animate.shift(UP * 1.65).shift(RIGHT),
                # #*[h.vertices[v].animate.shift(UP * 0.4) for v in upper],
                # *[g.edges[1, v].animate.become(h.edges[13, v]) for v in upper[:3]],
                # *[g.edges[1, v].animate.become(h.edges[14, v]) for v in upper[3:]],
                # g.vertices[2].animate.shift(DOWN * 1.65).shift(RIGHT),
                # *[g.vertices[e].animate.shift(DOWN * 0.4) for e in lower],
                # *[g.edges[2, v].animate.become(h.edges[15, v]) for v in lower[:3]],
                # *[g.edges[2, v].animate.become(h.edges[16, v]) for v in lower[3:]]
            )
        )
        print("13: ", h.vertices[13].get_midpoint())
        print("14: ", h.vertices[14].get_midpoint())
        print("15: ", h.vertices[15].get_midpoint())
        print("16: ", h.vertices[16].get_midpoint())
        self.wait()



