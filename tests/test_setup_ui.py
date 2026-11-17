"""Ручная разметка: клики и клавиши подаются напрямую, без окна."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from hud import Painter  # noqa: E402
from setup_ui import SetupUI, find_indicator_box  # noqa: E402
from turnstile_detector import GateLogic, Params  # noqa: E402


def frame():
    img = np.full((540, 960, 3), 120, np.uint8)
    cv2.circle(img, (500, 150), 12, (0, 0, 220), 4)          # красный индикатор
    return img


def test_three_clicks_make_a_gate():
    ui = SetupUI(frame(), Painter())
    ui.click(480, 200)
    ui.click(480, 300)
    ui.click(497, 148)                                        # чуть мимо центра — рамка найдётся сама
    x1, y1, x2, y2 = ui.indicator
    assert x1 < 490 < 510 < x2 and y1 < 140 < 160 < y2 and (x2 - x1) < 50
    ui.key(13)
    ui.key(13)                                                # второй Enter — закончить
    assert ui.finished
    g = ui.result()[0]
    assert g["name"] == "A" and g["kind"] == "turnstile" and len(g["zone"]) == 4
    GateLogic(g, Params())                                    # формат понятен основной программе


def test_undo_cargo_and_barrier():
    ui = SetupUI(frame(), Painter())
    ui.click(100, 100)
    ui.click(900, 100)                                        # ошиблись
    ui.undo()
    ui.click(300, 100)
    ui.click(500, 150)
    ui.key(ord("g"))                                          # это ворота
    ui.key(ord("t"))
    ui.click(300, 100)
    ui.click(200, 100)                                        # тумба
    ui.key(13)
    g = ui.result()[0]
    assert g["line"] == [[100, 100], [300, 100]] and g["kind"] == "cargo"
    assert g["barriers"] == [[[300, 100], [200, 100]]]


def test_undo_returns_to_previous_gate_and_esc_cancels():
    ui = SetupUI(frame(), Painter())
    for p in ((480, 200), (480, 300), (500, 150)):
        ui.click(*p)
    ui.key(13)
    ui.undo()                                                 # вернулись к проёму A
    assert len(ui.pts) == 3 and not ui.gates
    ui.key(27)
    assert ui.cancelled


def test_indicator_fallback_without_color():
    box = find_indicator_box(np.full((100, 100, 3), 120, np.uint8), (50, 50), 20)
    assert box[0] < 50 < box[2] and box[1] < 50 < box[3]


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_render_every_step(n):
    ui = SetupUI(frame(), Painter(ru=True))
    for p in ((480, 200), (480, 300), (500, 150))[:n]:
        ui.click(*p)
    ui.move(600, 400)
    assert ui.render().shape == (540, 960, 3)
