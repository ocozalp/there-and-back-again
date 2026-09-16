"""Draw a diary scene with PIL.

Everything is drawn at ``SUPERSAMPLE`` times the target size and downsampled, so a 32x32 image still
has smooth edges. Coordinates inside the drawing functions are fractions of the canvas so the same
code produces 32x32 today and 64x64 in Phase 8.
"""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from .scenario import BILBO, GOLD, WOOD, Character, Scene

SUPERSAMPLE = 4
GROUND_Y = 0.86  # Bilbo's feet, as a fraction of the height

SKY_BY_TIME = {
    "dawn": (247, 183, 140),
    "day": (135, 206, 235),
    "dusk": (150, 90, 140),
    "night": (22, 26, 62),
}
GROUND_BY_LOCATION = {
    "shire": (98, 160, 60),
    "forest": (52, 92, 42),
    "mountains": (112, 112, 118),
    "river": (90, 150, 62),
    "cave": (60, 48, 44),
    "laketown": WOOD,
    "lonely_mountain": (96, 88, 82),
}


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(x * (1 - t) + y * t)) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def _dim(color: tuple[int, int, int], time: str) -> tuple[int, int, int]:
    """Darken scenery at night and dusk so lighting is consistent with the sky."""
    if time == "night":
        return _blend(color, (10, 10, 30), 0.55)
    if time == "dusk":
        return _blend(color, (60, 30, 60), 0.25)
    if time == "dawn":
        return _blend(color, (255, 200, 150), 0.15)
    return color


class _Canvas:
    """A tiny helper so drawing code can speak in fractions of the canvas."""

    def __init__(self, size: int):
        self.s = size
        self.img = Image.new("RGB", (size, size))
        self.d = ImageDraw.Draw(self.img, "RGBA")

    def p(self, x: float, y: float) -> tuple[float, float]:
        return (x * self.s, y * self.s)

    def rect(self, x0, y0, x1, y1, fill, outline=None):
        self.d.rectangle([self.p(x0, y0), self.p(x1, y1)], fill=fill, outline=outline)

    def ellipse(self, x0, y0, x1, y1, fill, outline=None, width=1):
        self.d.ellipse([self.p(x0, y0), self.p(x1, y1)], fill=fill, outline=outline, width=width)

    def poly(self, pts, fill):
        self.d.polygon([self.p(x, y) for x, y in pts], fill=fill)

    def line(self, pts, fill, width):
        self.d.line([self.p(x, y) for x, y in pts], fill=fill, width=max(1, int(width * self.s)))


def _draw_sky(c: _Canvas, scene: Scene) -> None:
    sky = SKY_BY_TIME[scene.time]
    if scene.weather == "rain":
        sky = _blend(sky, (105, 110, 125), 0.55)
    elif scene.weather == "snow":
        sky = _blend(sky, (190, 195, 205), 0.45)
    elif scene.weather == "fog":
        sky = _blend(sky, (200, 200, 205), 0.35)
    c.rect(0, 0, 1, 1, sky)


def _draw_celestial(c: _Canvas, scene: Scene, rng: random.Random) -> None:
    if scene.time == "night":
        for _ in range(14):
            x, y = rng.uniform(0.02, 0.98), rng.uniform(0.02, 0.5)
            c.ellipse(x, y, x + 0.012, y + 0.012, (235, 235, 220))
        c.ellipse(0.70, 0.08, 0.84, 0.22, (240, 240, 215))
        return
    if scene.weather in ("rain", "fog"):
        return
    pos = {"dawn": (0.14, 0.34), "day": (0.5, 0.14), "dusk": (0.84, 0.34)}[scene.time]
    color = (255, 220, 80) if scene.weather == "sun" else (240, 235, 210)
    r = 0.07
    c.ellipse(pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r, color)


def _draw_backdrop(c: _Canvas, scene: Scene, rng: random.Random) -> None:
    """Scenery behind Bilbo, chosen by location."""
    t = scene.time
    g = GROUND_Y
    if scene.location == "shire":
        hill = _dim((120, 180, 70), t)
        c.ellipse(-0.2, g - 0.22, 0.55, g + 0.2, hill)
        c.ellipse(0.45, g - 0.16, 1.2, g + 0.2, _dim((105, 165, 62), t))
        # a round green door in the near hill
        c.ellipse(0.10, g - 0.13, 0.22, g - 0.01, _dim((60, 110, 50), t))
        c.ellipse(0.155, g - 0.075, 0.175, g - 0.055, GOLD)
    elif scene.location == "forest":
        for i in range(5):
            x = 0.08 + i * 0.21 + rng.uniform(-0.03, 0.03)
            h = rng.uniform(0.35, 0.5)
            trunk = _dim((80, 55, 30), t)
            leaf = _dim((30, 75, 35), t)
            c.rect(x - 0.02, g - h * 0.4, x + 0.02, g, trunk)
            c.poly([(x - 0.11, g - h * 0.35), (x + 0.11, g - h * 0.35), (x, g - h)], leaf)
            c.poly([(x - 0.08, g - h * 0.6), (x + 0.08, g - h * 0.6), (x, g - h - 0.08)], leaf)
    elif scene.location in ("mountains", "lonely_mountain"):
        rock = _dim((88, 88, 96), t)
        snow = _dim((235, 235, 240), t)
        if scene.location == "mountains":
            peaks = [(0.0, 0.5, 0.45), (0.3, 0.6, 0.3), (0.65, 1.0, 0.4)]
            for x0, x1, top in peaks:
                xm = (x0 + x1) / 2
                c.poly([(x0 - 0.1, g), (x1 + 0.1, g), (xm, top)], rock)
                cap = 0.09
                c.poly([(xm - cap * 0.7, top + cap), (xm + cap * 0.7, top + cap), (xm, top)], snow)
        else:
            dark = _dim((62, 60, 66), t)
            c.poly([(0.05, g), (0.95, g), (0.5, 0.12)], dark)
            c.poly([(0.42, 0.24), (0.58, 0.24), (0.5, 0.12)], snow)
            if scene.item == "dragon":
                c.ellipse(0.44, 0.30, 0.56, 0.40, (240, 120, 40))
    elif scene.location == "river":
        water = _dim((60, 110, 190), t)
        c.rect(0, g - 0.26, 1, g - 0.10, water)
        for i in range(4):
            y = g - 0.24 + i * 0.04
            c.line([(0.05 + 0.1 * i, y), (0.25 + 0.1 * i, y)], _dim((150, 190, 235), t), 0.008)
        c.ellipse(-0.3, g - 0.42, 0.5, g - 0.2, _dim((105, 165, 62), t))
    elif scene.location == "laketown":
        water = _dim((55, 100, 170), t)
        c.rect(0, g - 0.34, 1, g - 0.04, water)
        for i in range(4):
            x = 0.08 + i * 0.24
            house = _dim((150, 105, 60), t)
            roof = _dim((90, 55, 40), t)
            c.line([(x + 0.03, g - 0.30), (x + 0.03, g - 0.05)], _dim(WOOD, t), 0.012)
            c.line([(x + 0.13, g - 0.30), (x + 0.13, g - 0.05)], _dim(WOOD, t), 0.012)
            c.rect(x, g - 0.44, x + 0.16, g - 0.30, house)
            c.poly([(x - 0.02, g - 0.44), (x + 0.18, g - 0.44), (x + 0.08, g - 0.54)], roof)
            if t in ("dusk", "night"):
                c.rect(x + 0.06, g - 0.41, x + 0.10, g - 0.36, (255, 210, 110))
    elif scene.location == "cave":
        rock = (48, 40, 46)
        # rock everywhere except an arched opening at the top center showing the sky
        c.rect(0, 0, 1, 1, (0, 0, 0, 0))
        c.d.rectangle([(0, 0), (c.s, c.s)], fill=rock)
        # reopen the arch by redrawing the sky region
        sky = SKY_BY_TIME[scene.time]
        c.d.pieslice([c.p(0.30, 0.0), c.p(0.70, 0.5)], 180, 360, fill=sky)
        c.rect(0.30, 0.25, 0.70, 0.36, sky)
        c.poly([(0.30, 0.36), (0.70, 0.36), (0.62, 0.55), (0.38, 0.55)], rock)
        for i in range(3):
            x = 0.1 + i * 0.35 + rng.uniform(-0.05, 0.05)
            c.poly([(x, g), (x + 0.1, g), (x + 0.05, g - rng.uniform(0.08, 0.16))], (70, 60, 66))


def _draw_ground(c: _Canvas, scene: Scene) -> None:
    ground = _dim(GROUND_BY_LOCATION[scene.location], scene.time)
    if scene.weather == "snow" and scene.location not in ("cave",):
        ground = _blend(ground, (240, 240, 245), 0.7)
    c.rect(0, GROUND_Y, 1, 1, ground)
    if scene.location == "laketown":
        for i in range(6):
            c.line(
                [(0, GROUND_Y + 0.02 + i * 0.025), (1, GROUND_Y + 0.02 + i * 0.025)],
                (90, 60, 30),
                0.004,
            )


def _draw_character(
    c: _Canvas, ch: Character, cx: float, facing: int, gy: float = GROUND_Y
) -> None:
    """Bilbo (or any Character): cloak, waistcoat, hood, head, curls, feet, stick, pipe."""
    h = ch.height
    f = facing
    head_r = h * 0.17
    head_cy = gy - h + head_r
    shoulder_y = gy - h * 0.62
    # walking stick on the facing side, behind the body
    c.line([(cx + f * 0.11, gy), (cx + f * 0.125, gy - h * 0.85)], WOOD, 0.014)
    # hood: a cloak-colored circle behind the head
    c.ellipse(
        cx - head_r * 1.35,
        head_cy - head_r * 1.3,
        cx + head_r * 1.35,
        head_cy + head_r * 1.2,
        ch.cloak,
    )
    # cloak body
    c.poly(
        [
            (cx - h * 0.28, gy - h * 0.03),
            (cx + h * 0.28, gy - h * 0.03),
            (cx + h * 0.16, shoulder_y),
            (cx - h * 0.16, shoulder_y),
        ],
        ch.cloak,
    )
    # waistcoat with brass buttons
    c.rect(cx - h * 0.10, shoulder_y + h * 0.02, cx + h * 0.10, gy - h * 0.14, ch.waistcoat)
    for i in range(3):
        y = shoulder_y + h * (0.08 + 0.1 * i)
        c.ellipse(cx - 0.006, y, cx + 0.006, y + 0.012, GOLD)
    # head
    c.ellipse(cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r, ch.skin)
    # curls: a cap of hair plus three curls
    c.d.pieslice(
        [
            c.p(cx - head_r * 1.05, head_cy - head_r * 1.05),
            c.p(cx + head_r * 1.05, head_cy + head_r * 0.95),
        ],
        200,
        340,
        fill=ch.hair,
    )
    for dx in (-0.8, 0.0, 0.8):
        x = cx + dx * head_r
        y = head_cy - head_r * 0.95
        c.ellipse(
            x - head_r * 0.35, y - head_r * 0.35, x + head_r * 0.35, y + head_r * 0.35, ch.hair
        )
    # eye on the facing side
    c.ellipse(
        cx + f * head_r * 0.35 - 0.006,
        head_cy - 0.006,
        cx + f * head_r * 0.35 + 0.006,
        head_cy + 0.006,
        (40, 30, 30),
    )
    # pipe
    mouth = (cx + f * head_r * 0.5, head_cy + head_r * 0.45)
    bowl = (cx + f * head_r * 1.5, head_cy + head_r * 0.25)
    c.line([mouth, bowl], WOOD, 0.010)
    c.ellipse(bowl[0] - 0.012, bowl[1] - 0.012, bowl[0] + 0.012, bowl[1] + 0.012, WOOD)
    c.ellipse(bowl[0] - 0.01, bowl[1] - 0.05, bowl[0] + 0.01, bowl[1] - 0.03, (200, 200, 200, 150))
    # big bare feet
    c.ellipse(cx - h * 0.30, gy - 0.025, cx - h * 0.04, gy + 0.012, ch.skin)
    c.ellipse(cx + h * 0.04, gy - 0.025, cx + h * 0.30, gy + 0.012, ch.skin)


def _draw_companion(c: _Canvas, scene: Scene, cx: float, f: int) -> None:
    t = scene.time
    g = GROUND_Y
    x = cx - f * 0.24  # opposite side from the walking stick
    if scene.companion == "wizard":
        robe = _dim((150, 150, 160), t)
        c.line([(x + f * 0.10, g), (x + f * 0.11, g - 0.6)], WOOD, 0.012)
        c.poly([(x - 0.11, g), (x + 0.11, g), (x + 0.04, g - 0.42), (x - 0.04, g - 0.42)], robe)
        c.ellipse(x - 0.045, g - 0.50, x + 0.045, g - 0.41, (235, 205, 175))
        c.poly(
            [(x - 0.05, g - 0.44), (x + 0.05, g - 0.44), (x, g - 0.30)], (230, 230, 230)
        )  # beard
        c.poly([(x - 0.09, g - 0.48), (x + 0.09, g - 0.48), (x, g - 0.68)], robe)  # hat
    elif scene.companion == "dwarf":
        hood = _dim((40, 70, 140), t)
        c.poly([(x - 0.09, g), (x + 0.09, g), (x + 0.05, g - 0.24), (x - 0.05, g - 0.24)], hood)
        c.ellipse(x - 0.045, g - 0.32, x + 0.045, g - 0.23, (235, 205, 175))
        c.poly([(x - 0.06, g - 0.26), (x + 0.06, g - 0.26), (x, g - 0.08)], (70, 40, 25))  # beard
        c.poly([(x - 0.06, g - 0.30), (x + 0.06, g - 0.30), (x, g - 0.40)], hood)  # pointed hood
        c.rect(x - 0.08, g - 0.03, x - 0.02, g, (40, 30, 25))
        c.rect(x + 0.02, g - 0.03, x + 0.08, g, (40, 30, 25))
    elif scene.companion == "pony":
        x = cx - f * 0.30  # the pony is wider than the others, so it stands further out
        body = _dim((140, 90, 50), t)
        c.ellipse(x - 0.14, g - 0.22, x + 0.10, g - 0.08, body)
        for dx in (-0.10, -0.05, 0.03, 0.08):
            c.line([(x + dx, g - 0.10), (x + dx, g)], body, 0.014)
        c.ellipse(x + 0.06, g - 0.30, x + 0.17, g - 0.20, body)
        c.line([(x - 0.14, g - 0.18), (x - 0.19, g - 0.06)], (70, 45, 25), 0.012)
    elif scene.companion == "eagle":
        wing = _dim((110, 75, 40), t)
        ex, ey = 0.22, 0.20
        c.poly(
            [
                (ex - 0.16, ey - 0.06),
                (ex, ey),
                (ex + 0.16, ey - 0.06),
                (ex + 0.12, ey),
                (ex, ey + 0.05),
                (ex - 0.12, ey),
            ],
            wing,
        )


def _draw_item(c: _Canvas, scene: Scene, cx: float, f: int) -> None:
    g = GROUND_Y
    h = BILBO.height
    hand = (cx - f * h * 0.24, g - h * 0.42)  # the free hand, opposite the stick
    if scene.item == "ring":
        c.ellipse(
            hand[0] - 0.02,
            hand[1] - 0.02,
            hand[0] + 0.02,
            hand[1] + 0.02,
            None,
            outline=GOLD,
            width=max(1, int(0.009 * c.s)),
        )
    elif scene.item == "map":
        x0 = hand[0] - f * 0.11
        c.rect(min(x0, hand[0]), hand[1] - 0.04, max(x0, hand[0]), hand[1] + 0.03, (225, 205, 160))
        mx = (x0 + hand[0]) / 2
        c.line([(mx - 0.015, hand[1] - 0.015), (mx + 0.015, hand[1] + 0.015)], (180, 40, 40), 0.006)
        c.line([(mx - 0.015, hand[1] + 0.015), (mx + 0.015, hand[1] - 0.015)], (180, 40, 40), 0.006)
    elif scene.item == "campfire":
        x = cx + f * 0.26
        c.line([(x - 0.07, g - 0.01), (x + 0.07, g - 0.04)], WOOD, 0.014)
        c.line([(x - 0.07, g - 0.04), (x + 0.07, g - 0.01)], WOOD, 0.014)
        c.poly([(x - 0.05, g - 0.03), (x + 0.05, g - 0.03), (x, g - 0.17)], (240, 120, 30))
        c.poly([(x - 0.025, g - 0.03), (x + 0.025, g - 0.03), (x, g - 0.11)], (255, 210, 80))
    elif scene.item == "dragon":
        red = (190, 40, 30)
        dx, dy = 0.78, 0.18
        c.poly(
            [
                (dx - 0.14, dy - 0.05),
                (dx - 0.03, dy),
                (dx + 0.03, dy),
                (dx + 0.14, dy - 0.06),
                (dx + 0.10, dy + 0.02),
                (dx - 0.10, dy + 0.02),
            ],
            red,
        )
        c.ellipse(dx - 0.03, dy - 0.02, dx + 0.05, dy + 0.03, red)
        c.line([(dx - 0.03, dy + 0.01), (dx - 0.12, dy + 0.07)], red, 0.010)


def _draw_weather(c: _Canvas, scene: Scene, rng: random.Random) -> None:
    if scene.weather == "rain":
        for _ in range(40):
            x, y = rng.uniform(0, 1), rng.uniform(0, 0.9)
            c.line([(x, y), (x - 0.02, y + 0.06)], (170, 190, 220, 190), 0.006)
    elif scene.weather == "snow":
        for _ in range(45):
            x, y = rng.uniform(0, 1), rng.uniform(0, 0.95)
            c.ellipse(x, y, x + 0.018, y + 0.018, (250, 250, 255))
    elif scene.weather == "fog":
        c.rect(0, 0, 1, 1, (215, 215, 220, 105))
        c.rect(0, 0.55, 1, 1, (215, 215, 220, 70))


def render_scene(scene: Scene, size: int = 32, character: Character = BILBO) -> Image.Image:
    """Render one scene to an RGB image of ``size`` x ``size`` pixels."""
    rng = random.Random(scene.seed)
    c = _Canvas(size * SUPERSAMPLE)
    _draw_sky(c, scene)
    _draw_celestial(c, scene, rng)
    _draw_backdrop(c, scene, rng)
    _draw_ground(c, scene)
    if scene.companion == "eagle" or scene.item == "dragon":
        pass  # drawn in the sky below, before Bilbo so he stays in front
    _draw_companion(c, scene, scene.bilbo_x, scene.facing)
    _draw_item(c, scene, scene.bilbo_x, scene.facing)
    _draw_character(c, character, scene.bilbo_x, scene.facing)
    _draw_weather(c, scene, rng)
    return c.img.resize((size, size), Image.LANCZOS)
