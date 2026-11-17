"""Сценарии логики проёма (без камеры и нейросети): python -m pytest -q"""
import pytest

from turnstile_detector import (GateLogic, Params, Person, SELFTEST_CFG, selftest_cases, simulate, walker)


@pytest.mark.parametrize("name,kw,want", selftest_cases(), ids=[c[0] for c in selftest_cases()])
def test_selftest_scenarios(name, kw, want):
    assert simulate(**kw) == want


def test_owner_swipes_twice_for_two_people_is_ok():
    # «Если владелец приложил пропуск дважды для двоих — это норма»
    scn = lambda t: (([walker(1, 1.0, t)] if t >= 1 else []) + ([walker(2, 1.3, t)] if t >= 1.3 else []),
                     (0.5 <= t < 0.8) or (1.8 <= t < 2.1))
    assert simulate(scn, dur=6.0) == []


def test_owner_plus_three_ok_fourth_escort_is_limit():
    owner = Person(1, (60, 20, 90, 100), (95, 60))
    four = lambda t: ([owner], (t % 2.0) < 0.3 and t < 8.0)       # 4 прикладывания: владелец + 3
    assert simulate(four, dur=11.0) == []
    five = lambda t: ([owner], (t % 2.0) < 0.3 and t < 10.0)      # 5-е — нарушение
    assert simulate(five, dur=11.0) == ["LIMIT"]


def test_credit_expires():
    scn = lambda t: ([walker(1, 9.5, t)] if t >= 9.5 else [], 0.5 <= t < 1.0)   # пришёл через 10 с после зелёного
    assert simulate(scn, dur=14.0) == ["NO_GREEN"]


def test_cargo_gate_has_no_wide():
    cfg = dict(SELFTEST_CFG, kind="cargo")
    scn = lambda t: ([walker(1, 1.0, t, w=60)] if t >= 1 else [], 0.5 <= t < 1.0)
    assert simulate(scn, cfg=cfg) == []


def test_wide_with_two_greens_is_ok():
    scn = lambda t: ([walker(1, 1.5, t, w=60)] if t >= 1.5 else [], (0.2 <= t < 0.5) or (1.3 <= t < 1.6))
    assert simulate(scn) == []


def test_reverse_only_when_one_way():
    scn = lambda t: ([walker(1, 1.0, t, speed=-40, start=160)] if t >= 1 else [], 0.5 <= t < 1.0)
    assert simulate(scn) == []
    got = simulate(scn, params=Params(one_way=True), cfg=dict(SELFTEST_CFG, allowed_direction=-1))
    assert got == ["REVERSE"]


def test_jitter_on_line_counts_once():
    # человек топчется на линии туда-обратно — один проход
    xs = [90, 105, 95, 108, 96, 110, 130, 150]
    scn = lambda t: ([Person(1, (0, 60, 30, 140), (xs[min(int(t * 2), 7)], 100))], 0.2 <= t < 0.5)
    assert simulate(scn, dur=4.0) == []


def test_relink_does_not_join_same_side():
    # трек потерян и новый появился на той же стороне — прохода нет
    def scn(t):
        if 1.0 <= t < 2.0:
            return [walker(1, 1.0, t)], False        # 40 -> 76
        if 2.3 <= t < 3.0:
            return [Person(5, (75, 60, 105, 140), (90, 100))], False
        return [], False
    assert simulate(scn) == []


def test_incident_groups_alerts():
    g = GateLogic(SELFTEST_CFG, Params())
    evs = []
    for k in range(60):
        t = k / 10
        persons = ([walker(1, 1.0, t)] if t >= 1 else []) + ([walker(2, 1.5, t)] if t >= 1.5 else [])
        evs += [e for e in g.update(t, persons, False) if e["type"] == "violation"]
    assert [e["kind"] for e in evs].count("NO_GREEN") == 2
    assert len({e["incident"] for e in evs}) == 1


def test_memory_is_bounded():
    g = GateLogic(SELFTEST_CFG, Params())
    for k in range(3000):                      # 300 секунд потока, каждые 2 с новый человек
        t = k / 10
        tid = int(t // 2)
        g.update(t, [walker(tid, tid * 2.0, t)], (t % 2.0) < 0.3)
    assert len(g.counted) < 10 and len(g.last_pos) < 10 and len(g.sessions) < 40


def test_quick_start_updates_link(tmp_path, monkeypatch):
    import json
    import turnstile_detector as td
    cfg_path = tmp_path / "cam.json"
    cfg_path.write_text(json.dumps({"camera": "C", "source": "old", "gates": [SELFTEST_CFG],
                                    "params": {"green_frames": 3}}), encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _: "https://new/index.m3u8")
    seen = {}
    monkeypatch.setattr(td, "run", lambda cfg, params, show: seen.update(src=cfg["source"], gf=params.green_frames))
    td.quick_start(str(cfg_path))
    assert seen == {"src": "https://new/index.m3u8", "gf": 3}
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["source"] == "https://new/index.m3u8"


def test_id_reused_by_tracker_still_counts():
    # трекер отдал id ушедшего человека встречному: тот же tid снова пересекает линию через 3 с
    def scn(t):
        green = 0.5 <= t < 1.0
        if 1.0 <= t < 4.0:
            return [walker(1, 1.0, t, speed=70)], green                # 40 -> 250, ушёл от линии далеко
        if 4.0 <= t < 7.0:
            return [walker(1, 4.0, t, speed=-70, start=250)], green    # обратно через 4 с, без зелёного
        return [], green
    assert simulate(scn, dur=8.0) == ["NO_GREEN"]
    assert simulate(scn, dur=8.0, params=Params(recross_dist=99)) == []   # поведение v3: проход терялся
