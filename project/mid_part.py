from manim import *

class MidPart(Scene):
    def construct(self):

        header = Tex("Fienman's elimination algorithm consists of 4 steps:", font_size=40)
        step1 = Tex("(1)  Betweenness Reduction", font_size=30)
        step2 = Tex("(2)  Finding a negative sandwich or $1$-hop independent sandwich", font_size=30)
        step3 = Tex("(3)  Making the negative sandwich $r$-remote", font_size=30)
        step4 = Tex("(4)  $r$-remote elimination by hop reduction", font_size=30)

        header.shift(UP*1.5)
        step1.align_to(header, LEFT).shift(UP*0.5).shift(RIGHT*0.25)
        step2.align_to(header, LEFT).shift(DOWN*0.25).shift(RIGHT*0.25)
        step3.align_to(header, LEFT).shift(DOWN*1).shift(RIGHT*0.25)
        step4.align_to(header, LEFT).shift(DOWN*1.75).shift(RIGHT*0.25)

        self.play(
            FadeIn(header),
        )
        self.wait()
        self.play(
            FadeIn(step1)
        )
        self.wait()
        self.play(
            FadeIn(step2)
        )
        self.wait()
        self.play(
            FadeIn(step3)
        )
        self.wait()
        self.play(
            FadeIn(step4)
        )
        self.wait(8)

        self.play(
            FadeToColor(step1, color=YELLOW),
            FadeToColor(step4, color=YELLOW)
        )
        self.wait()


        self.play(
            FadeOut(header),
            FadeOut(step1),
            FadeOut(step2),
            FadeOut(step3),
            FadeOut(step4)
        )
        self.wait(2)
