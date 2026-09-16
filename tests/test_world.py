import random

import numpy as np
import pytest

from dl.world import ATTRIBUTES, Scene, diary_entry, generate_examples, render_scene, short_caption
from dl.world.captions import COMPANION_SHORT, ITEM_SHORT, LOCATION_SHORT, TIME_SHORT, WEATHER_SHORT
from dl.world.toy import make_mood_dataset


def test_render_shape_and_mode():
    img = render_scene(Scene(), size=32)
    assert img.size == (32, 32)
    assert img.mode == "RGB"
    img64 = render_scene(Scene(), size=64)
    assert img64.size == (64, 64)


def test_render_is_deterministic():
    rng = random.Random(3)
    scene = Scene.random(rng)
    a = np.asarray(render_scene(scene))
    b = np.asarray(render_scene(scene))
    assert np.array_equal(a, b)


def test_every_attribute_value_renders():
    for name, values in ATTRIBUTES.items():
        for value in values:
            scene = Scene(**{name: value})
            img = np.asarray(render_scene(scene))
            assert img.shape == (32, 32, 3)
            assert img.std() > 0, f"{name}={value} rendered a flat image"


def test_scene_rejects_unknown_values():
    with pytest.raises(ValueError):
        Scene(location="mordor")


def test_labels_match_vocabulary_order():
    scene = Scene(location="cave", weather="snow", time="night", companion="pony", item="map")
    labels = scene.labels()
    assert labels == {"location": 4, "weather": 3, "time": 3, "companion": 3, "item": 2}


def test_short_caption_names_every_attribute():
    rng = random.Random(0)
    for _ in range(50):
        scene = Scene.random(rng)
        caption = short_caption(scene)
        assert LOCATION_SHORT[scene.location] in caption
        assert WEATHER_SHORT[scene.weather] in caption
        assert TIME_SHORT[scene.time] in caption
        if scene.companion != "none":
            assert COMPANION_SHORT[scene.companion] in caption
        if scene.item != "none":
            assert ITEM_SHORT[scene.item] in caption


def test_diary_entry_is_deterministic_and_nonempty():
    scene = Scene(seed=42, weather="rain", companion="dwarf")
    assert diary_entry(scene) == diary_entry(scene)
    assert "dwar" in diary_entry(scene).lower()


def test_generate_examples():
    examples = generate_examples(5, seed=1)
    assert len(examples) == 5
    assert examples[0].image.shape == (32, 32, 3)
    assert examples[0].image.dtype == np.uint8
    again = generate_examples(5, seed=1)
    assert [e.scene for e in examples] == [e.scene for e in again]


def test_mood_dataset_is_balanced_enough():
    X, y = make_mood_dataset(1000, seed=0)
    assert X.shape == (1000, 2)
    assert set(np.unique(y)) == {-1, 1}
    frac = (y == 1).mean()
    assert 0.3 < frac < 0.7
