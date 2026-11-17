"""
skud_check.py — сверка журнала СКУД с тревогами детектора.

По видео нельзя узнать цвет пропуска и чей он. В журнале СКУД это есть, поэтому
часть правил объекта надёжнее проверять по нему:

  RED_ESCORT — красный пропуск приложен 2+ раза к одной точке в одну сторону за session_sec
               (красный проводит только владельца — второе прикладывание значит, что провели кого-то);
  LIMIT      — белый/жёлтый/синий пропуск приложен больше max_per_card раз за session_sec;
  PASSBACK   — пропуск дважды прошёл в одну сторону без выхода между проходами
               (второй проход позже session_sec) — признак передачи пропуска.

Кроме того, к каждой тревоге детектора из violations.csv подбирается ближайшее по времени
прикладывание на той же точке (пропуск и цвет) — оператору проще разбирать тревогу.

Журнал СКУД — CSV (разделитель ; или ,) со столбцами (имена задаются ключами):
    время; точка; пропуск; цвет; направление
Направление: "вход"/"выход" (или in/out, 1/-1). Если столбца нет, PASSBACK не проверяется.

Соответствие точек СКУД и проёмов детектора — JSON (необязательно):
    {"CAM-01/A": "Турникет 3 этаж левый", "CAM-01/B": "Турникет 3 этаж правый"}

Пример:
    python skud_check.py --skud skud.csv --violations out/CAM-01/violations.csv --map points.json
Результаты: skud_report.csv и violations_skud.csv рядом с журналом нарушений (или в --out).
"""

import argparse
import csv
import json
import sys
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M")
RED = {"красный", "red", "к"}
IN = {"вход", "in", "1", "+1", "entry"}
OUT = {"выход", "out", "-1", "exit"}


def parse_time(s):
    s = (s or "").strip()
    for f in TIME_FORMATS:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    return None


def parse_dir(s):
    s = (s or "").strip().lower()
    return 1 if s in IN else -1 if s in OUT else 0


def read_csv(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=";,\t")
    except (csv.Error, IndexError):
        class dialect(csv.excel):
            delimiter = ";"
    return list(csv.DictReader(text.splitlines(), dialect=dialect))


def load_skud(path, cols):
    events, bad = [], 0
    for row in read_csv(path):
        t = parse_time(row.get(cols["time"]))
        if t is None:
            bad += 1
            continue
        events.append({"t": t, "point": (row.get(cols["point"]) or "").strip(),
                       "card": (row.get(cols["card"]) or "").strip(),
                       "color": (row.get(cols["color"]) or "").strip().lower(),
                       "dir": parse_dir(row.get(cols["dir"]))})
    if bad:
        print(f"Пропущено строк без понятного времени: {bad}")
    events.sort(key=lambda e: e["t"])
    return events


def check_rules(events, session_sec=60.0, max_per_card=4, passback_hours=12.0):
    """Правила по одному журналу СКУД. Возвращает список нарушений (dict)."""
    out = []
    by_card = defaultdict(list)
    for e in events:
        if e["card"]:
            by_card[e["card"]].append(e)
    win = timedelta(seconds=session_sec)
    for card, evs in by_card.items():
        # серии прикладываний к одной точке в одну сторону
        groups = defaultdict(list)
        for e in evs:
            groups[(e["point"], e["dir"])].append(e)
        for (point, _), g in groups.items():
            red = g[0]["color"] in RED
            limit = 1 if red else max_per_card
            kind = "RED_ESCORT" if red else "LIMIT"
            j, flagged_until = 0, None
            for i, e in enumerate(g):
                while e["t"] - g[j]["t"] > win:
                    j += 1
                n = i - j + 1
                if n > limit and (flagged_until is None or e["t"] > flagged_until):
                    text = (f"Красный пропуск провёл ещё кого-то: прикладываний за {session_sec:.0f} с — {n}"
                            if red else
                            f"Превышен лимит сопровождения: прикладываний за {session_sec:.0f} с — {n} "
                            f"(можно {limit})")
                    out.append({"kind": kind, "t": e["t"], "point": point, "card": card,
                                "color": e["color"], "text": text})
                    flagged_until = e["t"] + win          # одна запись на серию
        # passback: два прохода в одну сторону без противоположного между ними
        last = None
        for e in evs:
            if e["dir"] == 0:
                continue
            if last is not None and e["dir"] == last["dir"]:
                gap = (e["t"] - last["t"]).total_seconds()
                if session_sec < gap <= passback_hours * 3600:
                    word = "вход" if e["dir"] == 1 else "выход"
                    out.append({"kind": "PASSBACK", "t": e["t"], "point": e["point"], "card": card,
                                "color": e["color"],
                                "text": f"Повторный {word} без обратного прохода (через {gap / 60:.0f} мин)"})
            last = e
    out.sort(key=lambda v: v["t"])
    return out


def match_violations(violations, events, point_map, window_sec=8.0):
    """К каждой тревоге детектора — ближайшее прикладывание на той же точке в пределах window_sec."""
    by_point = defaultdict(list)
    for e in events:
        by_point[e["point"]].append(e)
    times = {p: [e["t"] for e in evs] for p, evs in by_point.items()}
    rows = []
    for v in violations:
        key = f"{v.get('камера', '')}/{v.get('проём', '')}"
        point = point_map.get(key, key)
        t = parse_time(v.get("время"))
        best = None
        if t is not None and point in by_point:
            ts = times[point]
            i = bisect_left(ts, t)
            for k in (i - 1, i):
                if 0 <= k < len(ts):
                    d = abs((ts[k] - t).total_seconds())
                    if d <= window_sec and (best is None or d < best[0]):
                        best = (d, by_point[point][k])
        e = best[1] if best else {}
        rows.append({**v, "точка СКУД": point, "пропуск": e.get("card", ""), "цвет пропуска": e.get("color", ""),
                     "красный": "да" if e.get("color") in RED else ""})
    return rows


def write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter=";", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Сверка журнала СКУД с тревогами детектора")
    ap.add_argument("--skud", required=True, help="CSV-выгрузка журнала СКУД")
    ap.add_argument("--violations", help="violations.csv детектора (необязательно)")
    ap.add_argument("--map", help="JSON: {\"камера/проём\": \"точка СКУД\"}")
    ap.add_argument("--out", help="папка для результатов")
    ap.add_argument("--session-sec", type=float, default=60.0)
    ap.add_argument("--max-per-card", type=int, default=4)
    ap.add_argument("--passback-hours", type=float, default=12.0)
    ap.add_argument("--window-sec", type=float, default=8.0, help="окно сопоставления тревоги и прикладывания")
    for k, d in (("time", "время"), ("point", "точка"), ("card", "пропуск"), ("color", "цвет"),
                 ("dir", "направление")):
        ap.add_argument(f"--col-{k}", default=d, help=f"имя столбца ({d})")
    a = ap.parse_args(argv)

    cols = {k: getattr(a, f"col_{k}") for k in ("time", "point", "card", "color", "dir")}
    events = load_skud(a.skud, cols)
    if not events:
        sys.exit("В журнале СКУД нет ни одной строки с понятным временем — проверь имена столбцов (--col-...)")
    out_dir = Path(a.out or (Path(a.violations).parent if a.violations else Path(a.skud).parent))
    out_dir.mkdir(parents=True, exist_ok=True)

    found = check_rules(events, a.session_sec, a.max_per_card, a.passback_hours)
    report = [{"время": v["t"].strftime("%Y-%m-%d %H:%M:%S"), "тип": v["kind"], "точка": v["point"],
               "пропуск": v["card"], "цвет": v["color"], "описание": v["text"], "вердикт оператора": ""}
              for v in found]
    write_csv(out_dir / "skud_report.csv", report,
              ["время", "тип", "точка", "пропуск", "цвет", "описание", "вердикт оператора"])
    print(f"Событий СКУД: {len(events)}, нарушений по журналу: {len(found)} -> {out_dir / 'skud_report.csv'}")

    if a.violations:
        point_map = json.loads(Path(a.map).read_text(encoding="utf-8")) if a.map else {}
        vio = read_csv(a.violations)
        rows = match_violations(vio, events, point_map, a.window_sec)
        header = (list(vio[0].keys()) if vio else []) + ["точка СКУД", "пропуск", "цвет пропуска", "красный"]
        write_csv(out_dir / "violations_skud.csv", rows, header)
        n = sum(1 for r in rows if r["пропуск"])
        print(f"Тревог детектора: {len(rows)}, сопоставлено с СКУД: {n} -> {out_dir / 'violations_skud.csv'}")


if __name__ == "__main__":
    main()
