from datetime import datetime, timedelta

from skud_check import check_rules, main, match_violations

T0 = datetime(2026, 11, 17, 9, 0, 0)


def ev(sec, card, color="белый", d=1, point="T3"):
    return {"t": T0 + timedelta(seconds=sec), "point": point, "card": card, "color": color, "dir": d}


def kinds(events, **kw):
    return [v["kind"] for v in check_rules(sorted(events, key=lambda e: e["t"]), **kw)]


def test_red_card_single_ok_double_escort():
    assert kinds([ev(0, "R1", "красный")]) == []
    assert kinds([ev(0, "R1", "красный"), ev(5, "R1", "красный")]) == ["RED_ESCORT"]


def test_white_limit():
    assert kinds([ev(i * 5, "W1") for i in range(4)]) == []
    assert kinds([ev(i * 5, "W1") for i in range(6)]) == ["LIMIT"]      # одна запись на серию


def test_passback():
    assert kinds([ev(0, "W1"), ev(3600, "W1", d=-1), ev(7200, "W1")]) == []
    assert kinds([ev(0, "W1"), ev(3600, "W1")]) == ["PASSBACK"]


def test_match_violation_to_card():
    events = [ev(0, "W1"), ev(100, "R1", "красный")]
    vio = [{"время": "2026-11-17 09:01:43", "камера": "CAM-01", "проём": "A", "тип": "NO_GREEN"}]
    rows = match_violations(vio, events, {"CAM-01/A": "T3"})
    assert rows[0]["пропуск"] == "R1" and rows[0]["красный"] == "да"


def test_cli(tmp_path):
    skud = tmp_path / "skud.csv"
    skud.write_text("время;точка;пропуск;цвет;направление\n"
                    "17.11.2026 09:00:00;T3;R1;красный;вход\n"
                    "17.11.2026 09:00:06;T3;R1;красный;вход\n", encoding="utf-8")
    main(["--skud", str(skud), "--out", str(tmp_path)])
    text = (tmp_path / "skud_report.csv").read_text(encoding="utf-8-sig")
    assert "RED_ESCORT" in text
