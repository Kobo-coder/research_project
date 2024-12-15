from manim import *

class ClosingRemarks(Scene):
    def construct(self):
        fs = 35
        text1 = Tex(r"Running steps (1), (2), (3), \& (4) eliminates $\Theta(k^{1/3})$ negative edges per round.", font_size=fs)
        text2 = Tex(r"These four steps steps are run several times\\until all negative edges in the graph are eliminated.", font_size=fs)
        text3 = Tex(r"After all edges are eliminated,\\the algorithm runs Dijkstra's algorithm to compute a shortest path\\from the source to all other vertices in the graph.", font_size=fs)

        text2.shift(UP)
        text3.shift(DOWN)

        thanks = Tex("Thank you for watching!", font_size=50)

        tag = Tex(r"This video is produced as part of a project at the IT-University of Copenhagen", font_size=25, color=GREY)
        mails = Tex(r"\{imag, jlhj\}@itu.dk", font_size=25, color=GREY)
        mails.shift(DOWN*3.5)
        tag.shift(DOWN*3)

        self.play(
            Write(text1)
        )
        self.wait(5)
        self.play(
            Unwrite(text1)
        )
        self.wait()

        self.play(Write(text2))
        self.play(Write(text3))
        self.wait(5)
        self.play(
            Unwrite(text2),
            Unwrite(text3)
        )
        self.wait(2)

        self.play(Write(thanks))
        self.play(FadeIn(tag),FadeIn(mails))
        self.wait(5)
        self.clear()

