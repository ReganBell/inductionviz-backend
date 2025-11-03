
from manim import *

# --------- Style constants (Distil.pub inspired) ---------
# Distil.pub color palette: off-white, warm gray, ink, accent green
INK = "#0E1111"
WARM_GRAY = "#EDEAE6"
ACCENT_GREEN = "#2ECF8B"
SOFT_BLUE = "#4A90E2"

ATTN_COLOR = SOFT_BLUE
MLP_COLOR = ACCENT_GREEN
RESID_COLOR = WARM_GRAY
EMBED_COLOR = "#7E57C2"  # Soft purple
UNEMBED_COLOR = "#FF9800"  # Soft orange

N_LAYERS = 48
HEADS_PER_LAYER = 12  # visual only; not drawn individually by default


class LayerBlock(VGroup):
    """
    Visual block for a single Transformer layer:
    [ Residual ] -> [ Multi-Head Attention ] -> [ Residual ] -> [ MLP ] -> [ Residual ]
    We render a compact representation: [Attn | MLP] inside a frame, with subtle residual bars.
    Horizontal layout: narrower blocks stacked vertically within each layer.
    """
    def __init__(self, idx, width=0.4, height=1.2, **kwargs):
        super().__init__(**kwargs)
        self.idx = idx

        frame = RoundedRectangle(corner_radius=0.06, width=width, height=height, color=INK, stroke_width=1.5, fill_opacity=0.05)

        # Attention and MLP segments stacked vertically inside the frame
        component_height = height * 0.40
        gap = height * 0.02

        attn = RoundedRectangle(corner_radius=0.04, width=width*0.75, height=component_height, color=ATTN_COLOR, fill_opacity=0.7, stroke_width=1, stroke_color=INK)
        mlp = RoundedRectangle(corner_radius=0.04, width=width*0.75, height=component_height, color=MLP_COLOR, fill_opacity=0.7, stroke_width=1, stroke_color=INK)

        attn.move_to(frame.get_top() + DOWN*(component_height/2 + gap))
        mlp.move_to(frame.get_bottom() + UP*(component_height/2 + gap))

        # residual bars (thin lines on left/right)
        resid_left = Line(frame.get_top()+LEFT*0.18, frame.get_bottom()+LEFT*0.18, color=RESID_COLOR, stroke_width=2)
        resid_right = Line(frame.get_top()+RIGHT*0.18, frame.get_bottom()+RIGHT*0.18, color=RESID_COLOR, stroke_width=2)

        label = Text(f"{idx+1}", font_size=14, color=INK, font=".AppleSystemUIFont").next_to(frame, DOWN, buff=0.15)

        self.frame = frame
        self.attn = attn
        self.mlp = mlp
        self.resid_left = resid_left
        self.resid_right = resid_right
        self.label = label

        self.add(frame, attn, mlp, resid_left, resid_right, label)


class GPT2Stack(VGroup):
    """
    Full stack: input -> embed -> [48 layers] -> unembed -> output
    Horizontal layout for layers.
    """
    def __init__(self, n_layers=N_LAYERS, **kwargs):
        super().__init__(**kwargs)

        # IO blocks (styled for Distil.pub aesthetic)
        self.input_token = RoundedRectangle(corner_radius=0.08, width=1.0, height=1.2, color=INK, stroke_width=1.5)\
            .set_fill(WARM_GRAY, opacity=0.3)
        input_text = Text("in", font_size=18, color=INK, font=".AppleSystemUIFont").move_to(self.input_token.get_center())

        self.embed = RoundedRectangle(corner_radius=0.08, width=1.2, height=1.2, color=EMBED_COLOR, stroke_width=1.5)\
            .set_fill(EMBED_COLOR, opacity=0.3)
        embed_text = Text("embed", font_size=16, color=INK, font=".AppleSystemUIFont").move_to(self.embed.get_center())

        self.unembed = RoundedRectangle(corner_radius=0.08, width=1.2, height=1.2, color=UNEMBED_COLOR, stroke_width=1.5)\
            .set_fill(UNEMBED_COLOR, opacity=0.3)
        unembed_text = Text("unembed", font_size=14, color=INK, font=".AppleSystemUIFont").move_to(self.unembed.get_center())

        self.output_token = RoundedRectangle(corner_radius=0.08, width=1.0, height=1.2, color=INK, stroke_width=1.5)\
            .set_fill(WARM_GRAY, opacity=0.3)
        output_text = Text("out", font_size=18, color=INK, font=".AppleSystemUIFont").move_to(self.output_token.get_center())

        # Horizontal row of layers
        self.layers = VGroup(*[LayerBlock(i) for i in range(n_layers)]).arrange(RIGHT, buff=0.08)

        # Layout horizontally: input -> embed -> [layers] -> unembed -> output
        row = VGroup(self.input_token, self.embed, self.layers, self.unembed, self.output_token)\
            .arrange(RIGHT, buff=0.3)
        row.move_to(ORIGIN)

        self.add(self.input_token, input_text, self.embed, embed_text, self.layers, self.unembed, unembed_text, self.output_token, output_text)


class GPT2ReductionScene(Scene):
    """
    Animation script:
      1) Show full pipeline.
      2) Emphasize 48 layers block.
      3) Remove all but one layer.
      4) Remove MLP in the surviving layer (attention-only).
      5) Hold, then fade, and reveal replay loop.
    """
    def construct(self):
        # Set white background (Distil.pub style)
        self.camera.background_color = "#FCFCFC"

        title = MarkupText("<b>From GPT‑2 (48 layers)</b> ➜ <b>Single Attention Layer</b>",
                          font_size=32, color=INK, font=".AppleSystemUIFont")
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        stack = GPT2Stack()
        self.play(FadeIn(stack, shift=UP, run_time=1.2))

        # 1) Emphasize layers
        brace = Brace(stack.layers, UP, color=INK)
        brace_label = Text("48 transformer layers", font_size=20, color=INK, font=".AppleSystemUIFont").next_to(brace, UP)
        self.play(GrowFromCenter(brace), FadeIn(brace_label, shift=DOWN))

        self.wait(0.6)

        # 2) Dim everything but the layers briefly to focus
        non_layers = VGroup(stack.input_token, stack.embed, stack.unembed, stack.output_token)
        self.play(non_layers.animate.set_opacity(0.3), run_time=0.6)
        self.wait(0.4)
        self.play(non_layers.animate.set_opacity(1.0), run_time=0.6)

        # 3) Remove all but one layer
        target_idx = 22  # center-ish layer to keep
        survivors = VGroup(stack.layers[target_idx])
        to_remove = VGroup(*[L for i, L in enumerate(stack.layers) if i != target_idx])

        self.play(*[L.animate.set_opacity(0.2) for L in to_remove], run_time=0.6)
        self.play(Indicate(survivors, scale_factor=1.05, color=ACCENT_GREEN), run_time=0.8)
        self.wait(0.3)

        self.play(FadeOut(to_remove, lag_ratio=0.01, run_time=1.0))
        self.play(survivors.animate.move_to(stack.layers.get_center()), run_time=0.8)

        # Update brace/label for single layer
        new_brace = Brace(survivors, UP, color=INK)
        new_brace_label = Text("Single transformer layer", font_size=20, color=INK, font=".AppleSystemUIFont").next_to(new_brace, UP)
        self.play(Transform(brace, new_brace), Transform(brace_label, new_brace_label))

        self.wait(0.4)

        # 4) Remove MLP from the surviving layer (attention-only)
        single = survivors[0]
        mlp_glow = single.mlp.copy().set_stroke(ACCENT_GREEN, width=6).set_fill(opacity=0)
        self.play(Create(mlp_glow), run_time=0.5)
        self.play(FadeOut(mlp_glow), run_time=0.4)

        mlp_cross = VGroup(
            Line(single.mlp.get_corner(UL), single.mlp.get_corner(DR), color="#E53935", stroke_width=4),
            Line(single.mlp.get_corner(UR), single.mlp.get_corner(DL), color="#E53935", stroke_width=4),
        )
        self.play(FadeIn(mlp_cross, scale=0.9), run_time=0.4)
        self.play(FadeOut(single.mlp, run_time=0.6))

        attn_only_label = Text("Attention only", font_size=20, color=ATTN_COLOR, font=".AppleSystemUIFont").next_to(survivors, RIGHT, buff=0.4)
        self.play(FadeIn(attn_only_label, shift=LEFT))

        self.wait(0.6)

        # 5) Hold, then fade to restart state (suggesting replay)
        footer = Text("…then we replay the story", font_size=20, color=INK, font=".AppleSystemUIFont").to_edge(DOWN, buff=0.3)
        self.play(FadeIn(footer))
        self.wait(0.6)

        self.play(
            *[FadeOut(mob) for mob in [survivors, brace, brace_label, attn_only_label, footer, stack]],
            FadeOut(title, shift=UP),
            run_time=1.2
        )

        # Optional: Bring everything back quickly for loopable feel
        end_card = MarkupText("<b>Replay</b>", font_size=32, color=INK, font=".AppleSystemUIFont")
        self.play(FadeIn(end_card, scale=0.9))
        self.wait(0.4)
        self.play(FadeOut(end_card))

# Notes:
# - Render locally with: manim -pqh gpt2_layers_animation.py GPT2ReductionScene
# - Horizontal layout now fits all 48 layers on screen
# - Distil.pub-inspired styling: white background (#FCFCFC), ink text (#0E1111), accent green (#2ECF8B)
# - Uses .AppleSystemUIFont (SF Pro on macOS)
#
# Web-native alternatives to explore:
# - Motion Canvas (TypeScript, similar to Manim but web-first): https://motioncanvas.io/
# - Remotion (React-based programmatic video): https://www.remotion.dev/
# - D3.js + Observable (interactive web visualizations): https://observablehq.com/
# - Framer Motion (React animations): https://www.framer.com/motion/
# - Three.js/React-Three-Fiber (3D web graphics)
# - GSAP (web animation library): https://greensock.com/gsap/
