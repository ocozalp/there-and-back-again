"""The scenario: Bilbo's travel diary.

Everything story-specific lives here: who Bilbo is (as colors and proportions the renderer
understands), which scene attributes exist, and the stages of the journey. Change the story here
and nothing else needs to move.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Attribute vocabulary. Every attribute is categorical with a fixed, ordered set of values,
# so an integer label is always available for classification and conditioning.
# --------------------------------------------------------------------------------------

LOCATIONS = ["shire", "forest", "mountains", "river", "cave", "laketown", "lonely_mountain"]
WEATHERS = ["sun", "rain", "fog", "snow"]
TIMES = ["dawn", "day", "dusk", "night"]
COMPANIONS = ["none", "wizard", "dwarf", "pony", "eagle"]
ITEMS = ["none", "ring", "map", "campfire", "dragon"]

ATTRIBUTES: dict[str, list[str]] = {
    "location": LOCATIONS,
    "weather": WEATHERS,
    "time": TIMES,
    "companion": COMPANIONS,
    "item": ITEMS,
}

Color = tuple[int, int, int]


@dataclass(frozen=True)
class Character:
    """A character as the renderer sees it: a name and a palette.

    Bilbo is the only character until Phase 9, when Gollum arrives with a different palette and
    body shape.
    """

    name: str
    skin: Color
    hair: Color
    cloak: Color
    waistcoat: Color
    height: float  # fraction of the image height, feet to top of head


BILBO = Character(
    name="Bilbo",
    skin=(245, 205, 170),
    hair=(110, 70, 35),
    cloak=(56, 118, 60),
    waistcoat=(178, 44, 40),
    height=0.36,
)

WOOD = (120, 80, 40)
GOLD = (235, 190, 60)


@dataclass
class Scene:
    """One diary day: a fully specified picture and the parameters that made it.

    Attributes are strings from the vocabulary above. ``bilbo_x`` is Bilbo's horizontal position as
    a fraction of the width, ``facing`` is +1 (right) or -1 (left), and ``seed`` fixes every random
    detail (tree positions, raindrops, stars) so rendering is deterministic.
    """

    location: str = "shire"
    weather: str = "sun"
    time: str = "day"
    companion: str = "none"
    item: str = "none"
    bilbo_x: float = 0.5
    facing: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        for name, values in ATTRIBUTES.items():
            value = getattr(self, name)
            if value not in values:
                raise ValueError(f"{name}={value!r} not in {values}")
        if self.facing not in (1, -1):
            raise ValueError("facing must be +1 or -1")

    @classmethod
    def random(cls, rng: random.Random) -> Scene:
        return cls(
            location=rng.choice(LOCATIONS),
            weather=rng.choice(WEATHERS),
            time=rng.choice(TIMES),
            companion=rng.choice(COMPANIONS),
            item=rng.choice(ITEMS),
            bilbo_x=rng.uniform(0.3, 0.7),
            facing=rng.choice((1, -1)),
            seed=rng.randrange(2**31),
        )

    def labels(self) -> dict[str, int]:
        """Integer label per attribute, using the vocabulary order."""
        return {name: values.index(getattr(self, name)) for name, values in ATTRIBUTES.items()}

    @property
    def mood(self) -> str:
        """A crude mood used by the caption generator: cheerful, gloomy, or neutral."""
        if self.item == "dragon":
            return "gloomy"
        if self.weather == "sun" or self.item == "campfire":
            return "cheerful"
        if self.weather in ("rain", "fog") and self.time == "night":
            return "gloomy"
        return "neutral"


# --------------------------------------------------------------------------------------
# The journey. Phase 10 walks these stages in order to write the illustrated diary.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    name: str
    location: str
    days: int
    companions: tuple[str, ...] = field(default=("none",))
    items: tuple[str, ...] = field(default=("none",))


JOURNEY: list[Stage] = [
    Stage("Leaving Bag End", "shire", 3, ("none", "wizard", "dwarf"), ("none", "map")),
    Stage("Into the wild", "forest", 4, ("dwarf", "pony", "wizard"), ("none", "campfire", "map")),
    Stage("Over the mountains", "mountains", 4, ("dwarf", "wizard", "eagle"), ("none", "map")),
    Stage("Under the mountain", "cave", 3, ("none",), ("ring",)),
    Stage("The dark forest", "forest", 4, ("dwarf", "none"), ("ring", "none")),
    Stage("Down the river", "river", 3, ("dwarf", "none"), ("none", "ring")),
    Stage("The town on the lake", "laketown", 3, ("dwarf", "none"), ("map", "none")),
    Stage(
        "The Lonely Mountain", "lonely_mountain", 4, ("dwarf", "none"), ("dragon", "ring", "map")
    ),
    Stage("Back again", "shire", 2, ("wizard", "none"), ("ring", "none")),
]
