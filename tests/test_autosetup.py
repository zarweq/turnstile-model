"""Авторазметка на синтетике: мигающие индикаторы и потоки людей через проёмы разной ширины."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from autosetup import IndicatorFinder, TrackCollector, build_config, gate_from_tracks  # noqa: E402
from turnstile_detector import GateLogic, Params, dist  # noqa: E402

RED, GREEN = (0, 0, 220), (0, 220, 0)


def frames_with_indicators(n=60):
    """Два кольца-индикатора: A загорается зелёным, B всё время красный. Плюс крупный красный щит."""
    out = []
    for k in range(n):
        img = np.full((480, 720, 3), 110, np.uint8)
        cv2.circle(img, (200, 200), 9, GREEN if 20 <= k < 26 else RED, 3)     # A
        cv2.circle(img, (420, 200), 9, RED, 3)                                # B (никто не прикладывал)
        cv2.rectangle(img, (600, 20), (715, 470), RED, -1)                    # огнетушительный щит — не индикатор
        out.append(img)
    return out


def test_indicator_finder():
    f = IndicatorFinder((480, 720))
    for img in frames_with_indicators():
        f.add(img)
    found = f.find()
    toggled = [i for i in found if i["toggled"]]
    steady = [i for i in found if not i["toggled"]]
    assert len(toggled) == 1 and abs((toggled[0]["box"][0] + toggled[0]["box"][2]) / 2 - 200) < 8
    assert len(steady) == 1 and abs((steady[0]["box"][0] + steady[0]["box"][2]) / 2 - 420) < 8


def flow(cx, waist_y, spread_far, spread_waist, pw, n=8, seed=0):
    """n траекторий сверху вниз, которые сходятся к проёму (cx, waist_y) и расходятся после него."""
    rng = np.random.default_rng(seed)
    col = TrackCollector()
    for i in range(n):
        off_far, off_w = rng.uniform(-spread_far, spread_far), rng.uniform(-spread_waist, spread_waist)
        sign = 1 if i % 2 else -1                                  # ходят в обе стороны
        ys = np.linspace(waist_y - 150, waist_y + 150, 30)[::sign]
        for y in ys:
            k = min(abs(y - waist_y) / 150, 1)
            x = cx + off_w + (off_far - off_w) * k
            col.add([(x - pw / 2, y - 3 * pw, x + pw / 2, y)], [100 * (cx + 1) + i])
    return col.moving()


def test_small_and_big_gates():
    pw = 50
    small = flow(200, 260, 70, 4, pw)
    big = flow(600, 260, 120, 90, pw, seed=1)
    ind_s = {"box": [190, 190, 210, 210], "toggled": True}
    ind_b = {"box": [690, 190, 710, 210], "toggled": True}
    gs, gb = gate_from_tracks(small, ind_s), gate_from_tracks(big, ind_b)
    assert gs["kind"] == "turnstile" and gb["kind"] == "cargo"
    (ax, ay), (bx, by) = gs["line"]
    assert abs((ay + by) / 2 - 260) < 25 and abs((ax + bx) / 2 - 200) < 10    # линия на «талии» потока
    assert abs(ay - by) < 10                                                 # поперёк вертикального потока
    assert 1.0 * pw <= dist(*gs["line"]) <= 1.8 * pw
    assert dist(*gb["line"]) > 2.2 * pw and gb["scale"] < dist(*gb["line"])


def test_same_logic_far_from_camera():
    # тот же турникет вдали: всё вдвое мельче — тип тот же
    g = gate_from_tracks(flow(200, 260, 35, 2, 25), {"box": [195, 195, 205, 205], "toggled": True})
    assert g["kind"] == "turnstile"


def test_build_config_two_turnstiles_side_by_side():
    pw = 50
    tracks = flow(200, 260, 40, 4, pw) + flow(330, 260, 40, 4, pw, seed=2)
    inds = [{"box": [190, 220, 210, 240], "toggled": True}, {"box": [320, 220, 340, 240], "toggled": False}]
    cfg, notes = build_config("CAM-X", "src", (480, 720), inds, tracks)
    assert [g["name"] for g in cfg["gates"]] == ["A", "B"]
    xs = sorted((g["line"][0][0] + g["line"][1][0]) / 2 for g in cfg["gates"])
    assert abs(xs[0] - 200) < 12 and abs(xs[1] - 330) < 12
    assert any("B" in n for n in notes)                    # B без зелёного — просит проверить
    for g in cfg["gates"]:
        GateLogic(g, Params())                             # конфиг понятен основной программе


def test_no_traffic_gives_note():
    cfg, notes = build_config("C", "s", (480, 720), [{"box": [10, 10, 20, 20], "toggled": True}], [])
    assert cfg["gates"] == [] and notes
