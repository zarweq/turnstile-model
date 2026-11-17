"""Разметка по одной картинке (без людей) и правка предложения мышкой."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from autosetup import find_lights, propose_static  # noqa: E402
from hud import Painter  # noqa: E402
from setup_ui import SetupUI  # noqa: E402
from turnstile_detector import GateLogic, Params  # noqa: E402


def row_scene(xs=(100, 260, 420, 580), skip=(), light_color=(0, 0, 230)):
    """Ряд стоек 40x160 с огоньком наверху каждой (BGR красный)."""
    img = np.full((480, 720, 3), 150, np.uint8)
    stands = []
    for i, x in enumerate(xs):
        cv2.rectangle(img, (x, 220), (x + 40, 380), (90, 90, 90), -1)
        cv2.circle(img, (x + 20, 240), 7, light_color, -1)
        if i not in skip:
            stands.append({"box": [x, 220, x + 40, 380], "conf": 0.5, "src": "nn"})
    return img, stands


def test_lights_found_on_stands():
    img, _ = row_scene()
    lights = find_lights(img)
    assert len(lights) == 4 and all(l_["color"] == "red" for l_ in lights)


def test_passages_between_neighbour_stands():
    img, stands = row_scene()
    gates, notes = propose_static(img, stands=stands, lights=find_lights(img))
    assert len(gates) == 3 and not any(g["auto"]["check"] for g in gates)
    (ax, ay), (bx, by) = gates[0]["line"]
    assert (ax, bx) == (140, 260) and ay == by == 380                 # по полу от стойки до стойки
    x1, y1, x2, y2 = gates[0]["indicator"]
    assert x1 <= 120 <= x2 and y1 <= 240 <= y2                        # огонёк соседней стойки
    for g in gates:
        GateLogic(g, Params())


def test_missed_stand_restored_from_its_light():
    img, stands = row_scene(skip=(2,))                                 # нейросеть пропустила 3-ю стойку
    gates, notes = propose_static(img, stands=stands, lights=find_lights(img))
    assert len(gates) == 3 and any("огонькам" in n for n in notes)


def test_too_wide_passage_is_flagged():
    img, stands = row_scene(xs=(100, 200, 300, 560))
    gates, _ = propose_static(img, stands=stands, lights=[])          # без огоньков: восстановить нечем
    assert [g["auto"]["check"] for g in gates][-1] is True


def test_nothing_found():
    img = np.full((480, 720, 3), 150, np.uint8)
    gates, notes = propose_static(img, stands=[], lights=[])
    assert gates == [] and notes


def test_editor_drag_and_delete_proposal():
    img, stands = row_scene()
    gates, _ = propose_static(img, stands=stands, lights=find_lights(img))
    gates[1]["auto"]["check"] = True
    ui = SetupUI(img, Painter(), gates)
    ui.press(140, 380)                                    # схватили левый конец линии A
    assert ui.drag == ("line", 0, 0)
    ui.move(150, 385)
    ui.release()
    assert ui.gates[0]["line"][0] == [150, 385] and ui.gates[0]["zone"][0] != gates[0]["zone"][0]
    cx = (gates[1]["indicator"][0] + gates[1]["indicator"][2]) // 2
    cy = (gates[1]["indicator"][1] + gates[1]["indicator"][3]) // 2
    ui.press(cx, cy)                                      # тащим индикатор B
    ui.move(cx + 30, cy)
    ui.release()
    assert ui.gates[1]["indicator"][0] == gates[1]["indicator"][0] + 30 and not ui.gates[1]["_check"]
    ui.move(520, 381)
    ui.key(ord("d"))                                      # удалить проём C под мышкой
    assert [g["name"] for g in ui.gates] == ["A", "B"]
    ui.press(600, 100)                                    # клик в пустоте — новый проём
    assert ui.pts == [(600, 100)]
    ui.undo()
    ui.key(13)
    assert ui.finished and len(ui.result()) == 2 and "auto" not in ui.result()[0]
    assert ui.render().shape == img.shape
