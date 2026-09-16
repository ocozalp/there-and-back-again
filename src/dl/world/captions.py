"""Turn a Scene into text: a short caption for conditioning and a diary entry in Bilbo's voice.

Both are template based. The point is not literary quality but a guaranteed, checkable link
between every word and a scene parameter, so later phases can ask "did the model bind 'rain' to
the rain attribute?" and get a ground-truth answer.
"""

from __future__ import annotations

import random

from .scenario import Scene

LOCATION_SHORT = {
    "shire": "in the Shire",
    "forest": "in the forest",
    "mountains": "in the mountains",
    "river": "by the river",
    "cave": "in a cave",
    "laketown": "in the lake town",
    "lonely_mountain": "at the Lonely Mountain",
}
WEATHER_SHORT = {
    "sun": "in the sun",
    "rain": "in the rain",
    "fog": "in the fog",
    "snow": "in the snow",
}
TIME_SHORT = {"dawn": "at dawn", "day": "at midday", "dusk": "at dusk", "night": "at night"}
COMPANION_SHORT = {
    "none": "",
    "wizard": "with a wizard",
    "dwarf": "with a dwarf",
    "pony": "with a pony",
    "eagle": "with an eagle overhead",
}
ITEM_SHORT = {
    "none": "",
    "ring": "holding a ring",
    "map": "reading a map",
    "campfire": "by a campfire",
    "dragon": "with a dragon in the distance",
}

LOCATION_PHRASES = {
    "shire": [
        "at home in the Shire",
        "among the green hills of the Shire",
        "on the road out of Hobbiton",
    ],
    "forest": ["deep in the forest", "under the dark trees", "on the forest path"],
    "mountains": ["high in the mountains", "on a narrow mountain path", "among the cold peaks"],
    "river": ["by the river", "on the riverbank", "beside the rushing water"],
    "cave": ["in a cave", "underground in the dark", "in a damp tunnel"],
    "laketown": [
        "in the town on the lake",
        "on the wooden piers of the lake town",
        "among the houses on stilts",
    ],
    "lonely_mountain": [
        "in the shadow of the Lonely Mountain",
        "at the foot of the great mountain",
        "on the slopes of the Lonely Mountain",
    ],
}
WEATHER_PHRASES = {
    "sun": ["The sun was out", "A bright, warm day", "Sunshine"],
    "rain": ["Rain again", "It rained without pause", "Wet through"],
    "fog": ["Fog so thick I could not see my own feet", "A grey fog hung over everything", "Mist"],
    "snow": ["Snow fell", "Cold, and snowing", "Snow on everything"],
}
TIME_PHRASES = {
    "dawn": ["at dawn", "as the sun came up", "early, before breakfast"],
    "day": ["at midday", "through the day", "in the afternoon"],
    "dusk": ["at dusk", "as the light faded", "toward evening"],
    "night": ["by night", "under the stars", "long after dark"],
}
COMPANION_PHRASES = {
    "none": [""],
    "wizard": [
        "The wizard walked with me and said little.",
        "Gandalf led the way, humming to himself.",
    ],
    "dwarf": [
        "One of the dwarves kept me company and complained about the food.",
        "A dwarf marched beside me, grumbling.",
    ],
    "pony": ["The pony carried the packs and ate my apple.", "My pony was patient with me."],
    "eagle": ["An eagle circled far overhead.", "A great eagle watched us from the sky."],
}
ITEM_PHRASES = {
    "none": [""],
    "ring": [
        "I turned the ring over in my pocket and told no one.",
        "The ring felt heavier today.",
    ],
    "map": [
        "We studied the map by the road and argued about the route.",
        "I unrolled the map again; the runes still made no sense to me.",
    ],
    "campfire": [
        "We lit a fire and I toasted bread on a stick.",
        "A small campfire, and stories until late.",
    ],
    "dragon": [
        "Far off, something large and red moved across the sky. I did not sleep well.",
        "We saw the dragon in the distance. Nobody spoke.",
    ],
}
CLOSERS = {
    "cheerful": [
        "I thought fondly of second breakfast.",
        "Not a bad day for a hobbit.",
        "My feet are sore but my spirits are good.",
    ],
    "gloomy": [
        "I miss my armchair.",
        "I wish I had never left Bag End.",
        "No handkerchief, no supper, no comfort.",
    ],
    "neutral": ["Walked a long way.", "Supper was thin.", "Slept badly, dreamt of eggs."],
}


def short_caption(scene: Scene) -> str:
    """A compact, fully deterministic description: one clause per attribute."""
    parts = [
        f"Bilbo {LOCATION_SHORT[scene.location]}",
        TIME_SHORT[scene.time],
        WEATHER_SHORT[scene.weather],
        COMPANION_SHORT[scene.companion],
        ITEM_SHORT[scene.item],
    ]
    return ", ".join(p for p in parts if p)


def diary_entry(scene: Scene, rng: random.Random | None = None) -> str:
    """A few sentences in Bilbo's voice. Randomness only picks among templates."""
    rng = rng or random.Random(scene.seed)
    weather = rng.choice(WEATHER_PHRASES[scene.weather])
    when = rng.choice(TIME_PHRASES[scene.time])
    where = rng.choice(LOCATION_PHRASES[scene.location])
    opener = f"{weather} {when}, {where}."
    sentences = [
        opener,
        rng.choice(COMPANION_PHRASES[scene.companion]),
        rng.choice(ITEM_PHRASES[scene.item]),
        rng.choice(CLOSERS[scene.mood]),
    ]
    return " ".join(s for s in sentences if s)
