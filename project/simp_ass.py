from manim import *

class SimplifyingAssumption1(Scene):
    def construct(self):
        vertices = [1,2,3,4,5,6]
        pos_edges = [(1,3), (2,3)]
        neg_edges = [(3,4), (3,5), (3,6)]

        lt = {1: [-2, 1, 0], 2: [-2, -1, 0], 3: [0, 0, 0], 4: [2, 1, 0], 5: [2, 0, 0], 6: [2, -1, 0]}
        g = DiGraph(vertices, pos_edges+neg_edges, layout=lt,
                  vertex_config={3: {"fill_color": RED}},
                  edge_config={"tip_config": {"tip_length": 0.2, "tip_width": 0.2},
                                (3, 4): {"stroke_color": RED},
                                (3, 5): {"stroke_color": RED},
                                (3, 6): {"stroke_color": RED}
                                })
        self.play(Create(g))
        self.wait()

        edge_labels = {
            (3, 4): MathTex(r"w_1", "- w^*").set_color(RED).next_to(g.vertices[4].get_midpoint(), RIGHT*1.4),
            (3, 5): MathTex(r"w_2", "- w^*").set_color(RED).next_to(g.vertices[5].get_midpoint(), RIGHT*1.4),
            (3, 6): MathTex(r"w_3", "- w^*").set_color(RED).next_to(g.vertices[6].get_midpoint(), RIGHT*1.4)
        }


        for k in edge_labels:
             edge_labels[k][1].set_opacity(0)
             self.play(FadeToColor(g.vertices[3], RED),
                       #Uncreate(g.edges[k]),
                       #Create(g.edges[k].set_stroke(color=RED)),
                       Write(edge_labels[k][0]),
                       FadeToColor(g.edges[k], RED))
             ## Ide: gør dem rød, når der kommer vægt på

        self.wait()

        self.play(
            Flash(g.vertices[3], color=RED)
        )


        formula = MathTex(r"w^*", " = \min(", "w_1", ",", "w_2", ",", "w_3", ")")
        formula.shift(UP*2.5)

        formula[2].set_opacity(0)
        formula[4].set_opacity(0)
        formula[6].set_opacity(0)

        self.play(Write(formula[0]), Write(formula[1]), Write(formula[3]), Write(formula[5]), Write(formula[7]))
        self.wait()

        self.play(
            TransformFromCopy(edge_labels[(3, 4)][0], formula[2].copy().set_opacity(1)),
            TransformFromCopy(edge_labels[(3, 5)][0], formula[4].copy().set_opacity(1)),
            TransformFromCopy(edge_labels[(3, 6)][0], formula[6].copy().set_opacity(1)),
        )


        # Add new vertex 7 and position it
        new_vertex = 7
        g.add_vertices(new_vertex, positions={7: [0, 0, 0]})

        new_edges = [(1, 7), (2, 7), (7, 3)]
        neg_cluster = [3,4,5,6]
        pos_cluster = [1,2,7]

        self.play(
            [g.vertices[e].animate.shift(RIGHT * 1) for e in neg_cluster],
            [g.vertices[e].animate.shift(LEFT * 1) for e in pos_cluster],
            [g.edges[edge].animate.shift(RIGHT * 1) for edge in neg_edges],
            [g.edges[edge].animate.shift(LEFT * 1) for edge in pos_edges],
            [l.animate.shift(RIGHT * 1) for l in edge_labels.values()]
        )

        self.wait()

        new_neg_edge = (7, 3)
        g.add_edges(*new_edges)
        w_star_copy = MathTex(r"w^*").set_color(RED).next_to(g.edges[(7,3)].get_midpoint()).shift(DOWN*0.4).shift(LEFT*0.4)
        self.play(
            FadeToColor(g.vertices[7], RED),
            Create(g.add_edges(new_neg_edge, edge_config={(7, 3): {"stroke_color": RED}})),
            TransformFromCopy(formula[0], w_star_copy),
        )

        self.wait()
        self.play(
            FadeToColor(g.vertices[3], WHITE),
            [FadeToColor(l, WHITE) for l in edge_labels.values()],
            [FadeToColor(g.edges[edge], WHITE) for edge in neg_edges],
            [TransformFromCopy(formula[0], l[1].copy().set_opacity(1).set_color(WHITE)) for l in edge_labels.values()],
        )

        self.wait()
