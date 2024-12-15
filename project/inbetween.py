from manim import *

class Inbetween(Scene):
    def construct(self):
        text1 = Tex("In step (2) of the algorithm, we either find:", font_size=35)
        neg_sand = Tex("a negative sandwich", font_size=45)
        ort = Tex("or", font_size=35)
        ind_set = Tex("a $1$-hop independence set", font_size=45)


        text1.shift(UP*1.25)
        neg_sand.align_to(text1, LEFT).shift(UP*0.25).shift(RIGHT*0.5)
        ort.align_to(text1, LEFT).shift(DOWN*0.5)
        ind_set.align_to(text1, LEFT).shift(DOWN*1.25).shift(RIGHT*0.5)

        self.play(
            FadeIn(text1),
            FadeIn(neg_sand),
            FadeIn(ort),
            FadeIn(ind_set)
        )
        self.wait(10)
        self.play(
            FadeOut(text1),
            FadeOut(neg_sand),
            FadeOut(ort),
            FadeOut(ind_set)
        )
        self.wait(2)

