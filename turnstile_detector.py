"""
turnstile_detector.py (v4) — детекция нарушений прохода через турникет по видео.

Что ловит (на каждом проёме турникета отдельно):
  NO_GREEN  — человек прошёл линию проёма без зелёного сигнала (вдвоём по одному пропуску / без пропуска)
  ZONE      — в проёме одновременно больше людей, чем было зелёных сигналов
  WIDE      — подозрение: в одной рамке двое (рамка заметно шире обычной для этого проёма)
  HANDOVER  — возможная передача пропуска: руки двух людей по разные стороны турникета сошлись
  LIMIT     — один человек дал больше max_per_card зелёных подряд (лимит: владелец + 3)
  CLIMB     — перелаз через тумбу/ограждение (если в конфиге размечены "barriers")
  REVERSE   — проход в запрещённую сторону (только если в конфиге one_way = true)

Что нового в v4 по сравнению с v3:
  * проход не теряется, если трекер сменил id человека прямо на линии или потерял его
    на долю секунды (пролаз под триподом, перекрытие) — треки «сшиваются» через линию;
  * тревоги о проходах (NO_GREEN, WIDE, CLIMB, REVERSE) больше не глушат друг друга:
    у каждого типа своя пауза, а записи одного события объединяются номером инцидента;
  * зелёный засчитывается, только если горит green_frames кадров подряд (блики не дают разрешений);
  * перелаз через тумбы (CLIMB) — отрезки "barriers" в конфиге, размечаются в --setup;
  * для видеофайла можно обрабатывать не все кадры (process_fps) — анализ в разы быстрее;
  * исправлена утечка памяти (словарь учтённых проходов рос без ограничения).

Грузовые ворота размечаются как отдельный проём (тип "cargo"): логика та же, зелёный = один человек.
Красный пропуск по видео не отличить: для него — сверка с выгрузкой СКУД (skud_check.py).

Установка:
    pip install -r requirements.txt

Порядок работы:
    python turnstile_detector.py --selftest                       # проверка логики, камера не нужна
    python turnstile_detector.py --setup --config cam.json --source "ССЫЛКА"
    python turnstile_detector.py --calib --config cam.json        # проверить индикатор
    python turnstile_detector.py --config cam.json                # работа
  Ссылка на поток может меняться: её можно каждый раз передавать через --source.
  Вместо ссылки можно указать видеофайл — тогда время берётся из самого видео.

Результаты: папка out/<камера>/ — violations.csv, passes.csv и кадры нарушений.
"""

import argparse
import csv
import json
import statistics
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ============================ ПАРАМЕТРЫ ============================
@dataclass
class Params:
    # --- проход и зелёный сигнал ---
    credit_ttl: float = 8.0        # сколько секунд после зелёного можно пройти
    green_window: float = 10.0     # окно, в котором зелёные учитываются для правила ZONE
    green_min_gap: float = 1.0     # защита от двойного срабатывания одного сигнала
    green_frames: int = 2          # сколько кадров подряд должен гореть зелёный (защита от бликов)
    recross_sec: float = 5.0       # через сколько секунд тот же человек может быть посчитан снова
    recross_dist: float = 1.5      # ...или раньше, если он успел отойти от линии дальше (в ширинах проёма):
                                   # значит это уже другой проход (или трекер передал id другому человеку)
    one_way: bool = False          # True — проход против allowed_direction считается нарушением
    # --- сшивание треков через линию ---
    relink_sec: float = 1.5        # сколько секунд ждать «потерянного» человека на другой стороне
    relink_dist: float = 1.2       # насколько далеко (в ширинах проёма) от места потери он может появиться
    track_forget: float = 5.0      # через сколько секунд забыть невидимый трек
    # --- лимит сопровождения ---
    session_sec: float = 60.0
    max_per_card: int = 4          # владелец + 3
    swiper_max_dist: float = 3.0   # макс. расстояние человека до индикатора, в ширинах проёма
    # --- плотный проход ---
    zone_sec: float = 0.4          # сколько секунд подряд в зоне должно быть "лишних" людей
    wide_factor: float = 1.6       # рамка шире обычной во столько раз = подозрение на двоих
    min_width_samples: int = 10    # сколько обычных проходов нужно, чтобы знать "обычную" ширину
    # --- передача пропуска ---
    hand_dist: float = 0.35        # расстояние между кистями, в ширинах проёма
    hand_line_dist: float = 1.5    # кисти должны быть не дальше этого от линии проёма, в ширинах проёма
    hand_sec: float = 0.4
    # --- индикатор (серый купол, цветная окантовка) ---
    sat_min: int = 40
    val_min: int = 50
    min_colored: int = 4
    green_frac: float = 0.5
    # --- прочее ---
    cooldown: float = 4.0          # пауза между тревогами ОДНОГО ТИПА на одном проёме (ZONE, HANDOVER, LIMIT)
    incident_sec: float = 4.0      # тревоги одного проёма ближе этого — один инцидент
    model: str = "yolov8n-pose.pt"
    conf: float = 0.4
    kp_conf: float = 0.4           # минимальная уверенность точки кисти
    imgsz: int = 640
    process_fps: float = 0.0       # видеофайл: сколько кадров в секунду анализировать (0 — все)
    sound: bool = True             # звуковой сигнал при тревоге (Windows)
    ru_text: bool = False          # подписи на экране по-русски (через Pillow)
    device: str = ""               # "" — сам выберет видеокарту, "cpu" — процессор, "0" — первая видеокарта
    cam_fps: float = 8.0           # много камер: сколько кадров/с анализировать с каждой камеры


# Тревоги, которые относятся к конкретному проходу конкретного человека: каждая пишется,
# повторы того же человека отсекает recross_sec. Остальные — «состояния», для них действует cooldown.
PASS_KINDS = {"NO_GREEN", "WIDE", "CLIMB", "REVERSE"}


# ============================ ГЕОМЕТРИЯ ============================
def side(p, a, b):
    v = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return 1 if v > 0 else -1 if v < 0 else 0


def seg_intersect(p1, p2, a, b):
    """Пересекает ли отрезок p1-p2 отрезок a-b."""
    d1, d2 = side(p1, a, b), side(p2, a, b)
    d3, d4 = side(a, p1, p2), side(b, p1, p2)
    return d1 != d2 and d3 != d4 and d1 != 0 and d2 != 0


def dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def dist_point_seg(p, a, b):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return dist(p, (ax + t * dx, ay + t * dy))


def point_in_poly(p, poly):
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


# ============================ ЛОГИКА ПРОЁМА ============================
@dataclass
class Person:
    tid: int                 # id трека; -1 если трекер id не выдал
    box: tuple               # x1, y1, x2, y2
    anchor: tuple            # опорная точка (ноги / центр)
    wrists: list = field(default_factory=list)


class GateLogic:
    """Вся логика одного проёма. Не зависит от камеры и нейросети — проверяется в --selftest."""

    def __init__(self, cfg, params: Params):
        self.name = cfg.get("name", "A")
        self.zone = [tuple(p) for p in cfg["zone"]]
        self.a, self.b = tuple(cfg["line"][0]), tuple(cfg["line"][1])
        self.roi = tuple(int(v) for v in cfg["indicator"])
        self.allowed = cfg.get("allowed_direction", 1)
        self.barriers = [(tuple(s[0]), tuple(s[1])) for s in cfg.get("barriers", [])]
        self.p = params
        self.kind = cfg.get("kind", "turnstile")           # "turnstile" или "cargo" (грузовые ворота)
        # масштаб = ширина обычного проёма в пикселях; у широких ворот задаётся отдельно полем "scale"
        self.L = max(float(cfg.get("scale") or dist(self.a, self.b)), 1.0)
        self.ind_center = ((self.roi[0] + self.roi[2]) / 2, (self.roi[1] + self.roi[3]) / 2)

        self.credits = deque()
        self.recent_greens = deque()
        self.g_run = self.r_run = 0
        self.armed = True             # готов принять новый зелёный (после того как индикатор погас)
        self.last_green_t = -1e9
        self.last_pos = {}            # tid -> (anchor, t)
        self.counted = {}             # tid -> t последнего учтённого прохода
        self.climbed = {}             # tid -> t последнего перелаза
        self.sessions = {}            # tid -> deque времён зелёных
        self.widths = deque(maxlen=200)
        self.zone_since = None
        self.hand_since = None
        self.last_kind_t = {}         # тип тревоги -> t последней
        self.incident = 0
        self.last_violation_t = -1e9
        self.stats = Counter()

    # -- служебное --
    def _alert(self, events, t, kind, text, people=None, greens=None):
        if kind not in PASS_KINDS:
            if t - self.last_kind_t.get(kind, -1e9) < self.p.cooldown:
                return
        self.last_kind_t[kind] = t
        if t - self.last_violation_t > self.p.incident_sec:
            self.incident += 1
        self.last_violation_t = t
        self.stats[kind] += 1
        events.append({"type": "violation", "kind": kind, "text": text, "gate": self.name,
                       "people": people, "greens": greens, "incident": f"{self.name}-{self.incident}"})

    def typical_width(self):
        if len(self.widths) >= self.p.min_width_samples:
            return statistics.median(self.widths)
        return None

    def _near_line(self, pt, k=1.0):
        return dist_point_seg(pt, self.a, self.b) <= k * self.L

    def _relink(self, x, t, visible):
        """Новый трек у линии: ищем недавно потерянный трек на другой стороне — это тот же человек."""
        p = self.p
        if not (point_in_poly(x.anchor, self.zone) or self._near_line(x.anchor)):
            return None
        s_new = side(x.anchor, self.a, self.b)
        best = None
        for tid, (pos, ts) in self.last_pos.items():
            if tid in visible or tid == x.tid or t - ts > p.relink_sec or t - ts <= 0:
                continue
            if side(pos, self.a, self.b) in (0, s_new) or not self._near_line(pos, p.relink_dist):
                continue
            d = dist(pos, x.anchor)
            if d <= 2 * p.relink_dist * self.L and (best is None or d < best[0]):
                best = (d, tid, pos)
        if best is None:
            return None
        _, old, pos = best
        self.last_pos.pop(old, None)
        if old in self.counted:                       # уже учтённый проход переносится на новый id
            self.counted[x.tid] = self.counted.pop(old)
        return pos

    def _register_green(self, t, persons, ev):
        p = self.p
        self.last_green_t = t
        self.credits.append(t)
        self.recent_greens.append(t)
        near = [(dist(x.anchor, self.ind_center), x) for x in persons if x.tid >= 0]
        near = [(d, x) for d, x in near if d <= p.swiper_max_dist * self.L]
        if not near:
            return
        swiper = min(near, key=lambda z: z[0])[1].tid
        s = self.sessions.setdefault(swiper, deque())
        s.append(t)
        while s and t - s[0] > p.session_sec:
            s.popleft()
        if len(s) > p.max_per_card:
            self._alert(ev, t, "LIMIT", f"Больше {p.max_per_card} проходов подряд по одному пропуску",
                        greens=len(s))

    # -- основной шаг --
    def update(self, t, persons, is_green):
        p, ev = self.p, []

        # 1. Зелёный сигнал (с защитой от бликов): +1 разрешение
        if is_green:
            self.g_run, self.r_run = self.g_run + 1, 0
        else:
            self.g_run, self.r_run = 0, self.r_run + 1
        if self.armed and self.g_run >= p.green_frames:
            self.armed = False
            if t - self.last_green_t >= p.green_min_gap:
                self._register_green(t, persons, ev)
        if not self.armed and self.r_run >= p.green_frames:
            self.armed = True

        while self.credits and t - self.credits[0] > p.credit_ttl:
            self.credits.popleft()
        while self.recent_greens and t - self.recent_greens[0] > p.green_window:
            self.recent_greens.popleft()

        # 2. Пересечение линии проёма и ограждений
        visible = {x.tid for x in persons if x.tid >= 0}
        for x in persons:
            if x.tid < 0:
                continue
            prev = self.last_pos.get(x.tid)
            if prev is None:
                pos = self._relink(x, t, visible)
                if pos is not None:
                    prev = (pos, t)
            if side(x.anchor, self.a, self.b) == 0 and prev is not None:
                self.last_pos[x.tid] = (prev[0], t)     # точка ровно на линии: ждём следующего кадра
                continue
            self.last_pos[x.tid] = (x.anchor, t)
            if x.tid in self.counted and dist_point_seg(x.anchor, self.a, self.b) > p.recross_dist * self.L:
                self.counted.pop(x.tid)                    # ушёл от проёма — следующий проход считается заново
            if prev is None:
                continue

            # перелаз через тумбу: опорная точка пересекла ограждение, а не проём
            if any(seg_intersect(prev[0], x.anchor, a, b) for a, b in self.barriers) \
                    and not seg_intersect(prev[0], x.anchor, self.a, self.b):
                if t - self.climbed.get(x.tid, -1e9) >= p.recross_sec:
                    self.climbed[x.tid] = t
                    self._alert(ev, t, "CLIMB", "Перелаз через тумбу или ограждение", people=1)
                continue

            if not seg_intersect(prev[0], x.anchor, self.a, self.b):
                continue
            if t - self.counted.get(x.tid, -1e9) < p.recross_sec:
                continue
            self.counted[x.tid] = t
            direction = side(x.anchor, self.a, self.b)
            width = x.box[2] - x.box[0]
            typical = self.typical_width()
            wide = self.kind != "cargo" and typical is not None and width > p.wide_factor * typical

            if p.one_way and direction != self.allowed:
                ev.append({"type": "pass", "gate": self.name, "direction": direction, "green": False})
                self._alert(ev, t, "REVERSE", "Проход в запрещённом направлении", people=1)
                continue

            had_credit = bool(self.credits)
            if had_credit:
                self.credits.popleft()
            ev.append({"type": "pass", "gate": self.name, "direction": direction, "green": had_credit})
            if not had_credit:
                self._alert(ev, t, "NO_GREEN", "Проход без зелёного сигнала (вдвоём или без пропуска)",
                            people=1, greens=len(self.recent_greens))
            elif wide:
                if self.credits:
                    self.credits.popleft()          # второй зелёный был — считаем, что прошли двое законно
                else:
                    self._alert(ev, t, "WIDE", "Подозрение: двое в одной рамке на один зелёный",
                                people=2, greens=len(self.recent_greens))
            if not wide:
                self.widths.append(width)

        # забываем старое, чтобы память не росла
        for tid in [k for k, (_, ts) in self.last_pos.items() if t - ts > p.track_forget]:
            self.last_pos.pop(tid, None)
        for d in (self.counted, self.climbed):
            for tid in [k for k, ts in d.items() if t - ts > p.recross_sec]:
                d.pop(tid, None)
        for tid in [k for k, d in self.sessions.items() if not d or t - d[-1] > p.session_sec]:
            self.sessions.pop(tid, None)

        # 3. Людей в проёме больше, чем зелёных
        n_zone = sum(1 for x in persons if point_in_poly(x.anchor, self.zone))
        greens = len(self.recent_greens)
        if n_zone >= 2 and n_zone > greens:
            if self.zone_since is None:
                self.zone_since = t
            elif t - self.zone_since >= p.zone_sec:
                self._alert(ev, t, "ZONE", f"В проёме {n_zone} чел. на {greens} зелёных",
                            people=n_zone, greens=greens)
                self.zone_since = None
        else:
            self.zone_since = None

        # 4. Передача пропуска: кисти людей по разные стороны линии сошлись у турникета
        if self._hands_meet(persons):
            if self.hand_since is None:
                self.hand_since = t
            elif t - self.hand_since >= p.hand_sec:
                self._alert(ev, t, "HANDOVER", "Возможная передача пропуска через турникет", people=2)
                self.hand_since = None
        else:
            self.hand_since = None

        return ev

    def _hands_meet(self, persons):
        p = self.p
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                A, B = persons[i], persons[j]
                sa, sb = side(A.anchor, self.a, self.b), side(B.anchor, self.a, self.b)
                if sa == 0 or sb == 0 or sa == sb:
                    continue
                for wa in A.wrists:
                    for wb in B.wrists:
                        if dist(wa, wb) > p.hand_dist * self.L:
                            continue
                        mid = ((wa[0] + wb[0]) / 2, (wa[1] + wb[1]) / 2)
                        if dist_point_seg(mid, self.a, self.b) <= p.hand_line_dist * self.L:
                            return True
        return False


# ============================ САМОПРОВЕРКА ============================
SELFTEST_CFG = {"name": "T", "zone": [[80, 60], [120, 60], [120, 140], [80, 140]],
                "line": [[100, 60], [100, 140]], "indicator": [95, 40, 105, 50],
                "barriers": [[[100, 140], [100, 200]]]}       # тумба ниже проёма


def walker(tid, t0, t, y=100, speed=40, w=30, start=40):
    x = start + (t - t0) * speed
    return Person(tid, (x - w / 2, y - 40, x + w / 2, y + 40), (x, y))


def simulate(scn, dur=8.0, fps=10, warm=True, params=None, cfg=None):
    """Прогоняет сценарий scn(t) -> (persons, is_green) и возвращает список типов тревог."""
    g = GateLogic(cfg or SELFTEST_CFG, params or Params())
    if warm:
        g.widths.extend([30] * 20)
    out = []
    for k in range(int(dur * fps)):
        t = k / fps
        persons, green = scn(t)
        out += [e["kind"] for e in g.update(t, persons, green) if e["type"] == "violation"]
    return out


def selftest_cases():
    def green(*spans):
        return lambda t: any(a <= t < b for a, b in spans)

    def two_walkers(t1, t2, g):
        return lambda t: (([walker(1, t1, t)] if t >= t1 else []) + ([walker(2, t2, t)] if t >= t2 else []), g(t))

    def lost_on_line(g):
        # трекер теряет человека перед линией и выдаёт новый id уже за линией (пролаз, перекрытие)
        def scn(t):
            if 1.0 <= t < 2.3:
                return [walker(1, 1.0, t)], g(t)          # x: 40 -> 92
            if 2.8 <= t < 5.0:
                return [walker(7, 1.0, t)], g(t)          # x: 112 -> ...
            return [], g(t)
        return scn

    return [
        ("1 зелёный, 1 человек", dict(scn=lambda t: ([walker(1, 1.0, t)] if t >= 1 else [], 0.5 <= t < 1.0)), []),
        ("1 зелёный, 2 человека", dict(scn=two_walkers(1.0, 2.2, green((0.5, 1.0))), dur=6.0), ["NO_GREEN"]),
        ("2 зелёных, 2 человека", dict(scn=two_walkers(1.0, 3.0, green((0.5, 0.8), (2.5, 2.8))), dur=6.0), []),
        ("0 зелёных, 1 человек", dict(scn=lambda t: ([walker(1, 1.0, t)] if t >= 1 else [], False)), ["NO_GREEN"]),
        ("1 зелёный, широкая рамка",
         dict(scn=lambda t: ([walker(1, 1.0, t, w=60)] if t >= 1 else [], 0.5 <= t < 1.0)), ["WIDE"]),
        ("двое в зоне на 1 зелёный", dict(scn=lambda t: (
            [Person(-1, (75, 60, 105, 140), (90, 100)), Person(-1, (95, 60, 125, 140), (110, 100))]
            if 1.0 <= t < 2.5 else [], 0.5 <= t < 1.0), dur=4.0), ["ZONE"]),
        ("передача пропуска", dict(scn=lambda t: (
            [Person(1, (50, 60, 80, 140), (65, 100), [(98, 55)]),
             Person(2, (120, 60, 150, 140), (135, 100), [(104, 57)])] if 1.0 <= t < 2.5 else [], False),
            dur=4.0), ["HANDOVER"]),
        ("5 зелёных от одного", dict(scn=lambda t: ([Person(1, (60, 20, 90, 100), (95, 60))], (t % 2.0) < 0.3),
                                     dur=11.0), ["LIMIT"]),
        ("мимо проёма", dict(scn=lambda t: ([walker(1, 1.0, t, y=300)] if t >= 1 else [], False)), []),
        # --- новое в v4 ---
        ("потеря трека на линии, без зелёного", dict(scn=lost_on_line(green())), ["NO_GREEN"]),
        ("потеря трека на линии, с зелёным", dict(scn=lost_on_line(green((0.5, 1.0)))), []),
        ("блик индикатора в 1 кадр", dict(scn=lambda t: ([walker(1, 1.0, t)] if t >= 1 else [],
                                                         abs(t - 0.5) < 0.05)), ["NO_GREEN"]),
        ("двое без пропуска подряд", dict(scn=two_walkers(1.0, 2.0, green()), dur=6.0), ["NO_GREEN", "NO_GREEN"]),
        ("перелаз через тумбу", dict(scn=lambda t: ([walker(1, 1.0, t, y=170)] if t >= 1 else [], False)), ["CLIMB"]),
    ]


def selftest():
    ok = True
    for name, kw, want in selftest_cases():
        got = simulate(**kw)
        good = got == want
        ok &= good
        print(("OK  " if good else "FAIL"), name, "->", got, "" if good else f"(ожидалось {want})")
    print("\nЛогика в порядке." if ok else "\nЕсть ошибки в логике!")
    return ok


# ============================ РАБОТА С ВИДЕО ============================
def is_file_source(src):
    return isinstance(src, str) and not src.lower().startswith(("rtsp://", "http://", "https://")) and Path(src).exists()


class LatestFrame:
    """Читает живой поток в отдельном потоке и отдаёт только самый свежий кадр (без накопления задержки)."""

    def __init__(self, source):
        import cv2
        self.cv2, self.source = cv2, source
        self.frame, self.n, self.lock, self.stop = None, 0, threading.Lock(), False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        cap = None
        while not self.stop:
            if cap is None or not cap.isOpened():
                cap = self.cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    print("Поток недоступен, повтор через 5 с (ссылка могла протухнуть)...")
                    time.sleep(5)
                    continue
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                time.sleep(1)
                continue
            with self.lock:
                self.frame, self.n = frame, self.n + 1

    def get(self, last_n):
        with self.lock:
            if self.frame is None or self.n == last_n:
                return None, last_n
            return self.frame.copy(), self.n


def green_share(frame, roi, p: Params):
    """Доля зелёного среди цветных пикселей окантовки: ~0 красный, ~1 зелёный. Серый купол игнорируется."""
    import cv2
    x1, y1, x2, y2 = roi
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colored = (s >= p.sat_min) & (v >= p.val_min)
    if int(colored.sum()) < p.min_colored:
        return 0.0
    g = int((colored & (h >= 35) & (h <= 90)).sum())
    r = int((colored & ((h <= 12) | (h >= 165))).sum())
    return g / (g + r) if g + r else 0.0


def load_config(path, source_override=None):
    path = Path(path)
    if not path.exists():
        sys.exit(f"Нет файла {path}. Сначала запусти с --setup")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if source_override:
        cfg["source"] = source_override
    params = Params(**{k: v for k, v in cfg.get("params", {}).items() if k in Params.__dataclass_fields__})
    return cfg, params


def grab_frame(source):
    import cv2
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("Не удалось получить кадр — проверь ссылку (она могла протухнуть)")
    return frame


def click_segments(frame, n, gate_pts, title):
    """Разметка n отрезков (по 2 клика): верх тумбы / линия ограждения вдоль, по которой нельзя перелезать."""
    import cv2
    pts = []
    if n <= 0:
        return []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2 * n:
            pts.append((x, y))

    cv2.setMouseCallback("setup", on_click)
    while len(pts) < 2 * n:
        view = frame.copy()
        for k in range(4):
            cv2.line(view, gate_pts[k], gate_pts[(k + 1) % 4], (255, 200, 0), 1)
        cv2.line(view, gate_pts[4], gate_pts[5], (0, 0, 255), 2)
        for k in range(0, len(pts) - 1, 2):
            cv2.line(view, pts[k], pts[k + 1], (255, 0, 255), 2)
        for q in pts:
            cv2.circle(view, q, 4, (255, 0, 255), -1)
        cv2.putText(view, f"{title} {len(pts) // 2 + 1}/{n}: point {len(pts) % 2 + 1}/2", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("setup", view)
        if cv2.waitKey(30) == 27:
            sys.exit("Отменено")
    return [[list(pts[k]), list(pts[k + 1])] for k in range(0, 2 * n, 2)]


def setup_mode(config_path, source, camera=None):
    import cv2
    if not source:
        sys.exit("Для --setup укажи --source \"ссылка или файл\"")
    frame0 = grab_frame(source)
    camera = camera or input("Название камеры (например CAM-01): ").strip() or "camera"
    n_gates = int(input("Сколько проёмов турникета видно в кадре? [1]: ").strip() or 1)
    anchor = "center" if (input("Камера смотрит: 1 — сбоку, 2 — сверху [1]: ").strip() or "1") == "2" else "foot"
    steps = ["zone 1/4", "zone 2/4", "zone 3/4", "zone 4/4", "line 1/2", "line 2/2",
             "indicator top-left", "indicator bottom-right"]
    print("\nДля каждого проёма 8 кликов:\n"
          "  1-4: углы зоны проёма (площадка между створками, по кругу)\n"
          "  5-6: линия поперёк проёма (от стойки до стойки)\n"
          "  7-8: левый верхний и правый нижний угол индикатора (купол с окантовкой, с небольшим запасом)\n")
    gates = []
    for gi in range(n_gates):
        frame, pts = frame0.copy(), []

        def on_click(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 8:
                pts.append((x, y))

        cv2.namedWindow("setup")
        cv2.setMouseCallback("setup", on_click)
        while len(pts) < 8:
            view = frame.copy()
            for q in pts:
                cv2.circle(view, q, 4, (0, 0, 255), -1)
            if len(pts) >= 4:
                for k in range(4):
                    cv2.line(view, pts[k], pts[(k + 1) % 4], (255, 200, 0), 2)
            if len(pts) >= 6:
                cv2.line(view, pts[4], pts[5], (0, 0, 255), 2)
            cv2.putText(view, f"gate {gi + 1}/{n_gates}: click {steps[len(pts)]}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("setup", view)
            if cv2.waitKey(30) == 27:
                sys.exit("Отменено")
        (x1, y1), (x2, y2) = pts[6], pts[7]
        kind = "cargo" if (input(f"Проём {gi + 1}: 1 — обычный турникет, 2 — грузовые ворота [1]: ").strip()
                           or "1") == "2" else "turnstile"
        n_bar = int(input(f"Проём {gi + 1}: сколько тумб/ограждений разметить для CLIMB (по 2 клика) [0]: ")
                    .strip() or 0)
        barriers = click_segments(frame0, n_bar, pts, f"gate {gi + 1}/{n_gates} barrier")
        gates.append({"name": chr(ord("A") + gi), "kind": kind, "zone": pts[:4], "line": pts[4:6],
                      "indicator": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                      "barriers": barriers, "allowed_direction": 1})
    cv2.destroyAllWindows()
    # масштаб для грузовых ворот: ширина обычного турникета в этом же кадре, иначе половина ширины ворот
    normal = [dist(g["line"][0], g["line"][1]) for g in gates if g["kind"] == "turnstile"]
    for g in gates:
        if g["kind"] == "cargo":
            g["scale"] = round(statistics.median(normal) if normal else dist(g["line"][0], g["line"][1]) / 2, 1)
    cfg = {"camera": camera, "source": source, "anchor": anchor, "gates": gates, "params": {}}
    Path(config_path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено в {config_path}. Параметры можно менять в разделе \"params\" этого файла.")


def calib_mode(cfg, params):
    import cv2
    cap = cv2.VideoCapture(cfg["source"])
    print("Приложи пропуск: значение должно подняться с ~0 до ~1. Выход — q.\n"
          "Всегда 0 -> уменьши sat_min (до 25) или min_colored (до 2) в params.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        line = []
        for g in cfg["gates"]:
            s = green_share(frame, g["indicator"], params)
            x1, y1, x2, y2 = g["indicator"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if s > params.green_frac else (0, 0, 255), 2)
            line.append(f"{g['name']}: {s:.2f}{' GREEN' if s > params.green_frac else ''}")
        print("  ".join(line))
        cv2.imshow("calib", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


class Output:
    def __init__(self, camera):
        self.dir = Path("out") / camera
        self.dir.mkdir(parents=True, exist_ok=True)
        self.camera = camera

    def _row(self, name, header, row):
        path = self.dir / name
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8-sig") as f:   # utf-8-sig + ';' — открывается в Excel
            w = csv.writer(f, delimiter=";")
            if new:
                w.writerow(header)
            w.writerow(row)

    def passed(self, stamp, e):
        self._row("passes.csv", ["время", "камера", "проём", "направление", "был зелёный"],
                  [stamp, self.camera, e["gate"], e["direction"], "да" if e["green"] else "нет"])

    def violation(self, stamp, e, frame):
        import cv2
        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{e['gate']}_{e['kind']}.jpg"
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            (self.dir / fname).write_bytes(buf.tobytes())      # так работают и русские буквы в пути
        self._row("violations.csv",
                  ["время", "камера", "проём", "инцидент", "тип", "описание", "людей", "зелёных", "кадр",
                   "вердикт оператора"],
                  [stamp, self.camera, e["gate"], e["incident"], e["kind"], e["text"], e["people"], e["greens"],
                   fname, ""])
        print(f"[!] {stamp} проём {e['gate']}: {e['text']} -> {fname}")


def build_persons(res, anchor_mode, kp_conf):
    persons = []
    if res.boxes is None or len(res.boxes) == 0:
        return persons
    boxes = res.boxes.xyxy.tolist()
    ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [-1] * len(boxes)
    kxy = kcf = None
    if getattr(res, "keypoints", None) is not None and res.keypoints.xy is not None:
        kxy = res.keypoints.xy.tolist()
        kcf = res.keypoints.conf.tolist() if res.keypoints.conf is not None else None
    for i, (box, tid) in enumerate(zip(boxes, ids)):
        x1, y1, x2, y2 = box
        anchor = ((x1 + x2) / 2, (y1 + y2) / 2) if anchor_mode == "center" else ((x1 + x2) / 2, y2)
        wrists = []
        if kxy is not None and i < len(kxy) and len(kxy[i]) >= 11:
            for k in (9, 10):                                   # 9, 10 — левая и правая кисть
                c = kcf[i][k] if kcf is not None else 1.0
                wx, wy = kxy[i][k]
                if c >= kp_conf and (wx > 0 or wy > 0):
                    wrists.append((wx, wy))
        persons.append(Person(tid, (x1, y1, x2, y2), anchor, wrists))
    return persons


def draw_overlay(cv2, view, persons, gates, greens, painter):
    """Рамки людей, зоны, линии, тумбы и индикаторы всех проёмов на кадре."""
    for x in persons:
        x1, y1, x2, y2 = map(int, x.box)
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(view, tuple(map(int, x.anchor)), 4, (0, 255, 0), -1)
        cv2.putText(view, str(x.tid), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for w in x.wrists:
            cv2.circle(view, tuple(map(int, w)), 4, (255, 0, 255), -1)
    for g, is_green in zip(gates, greens):
        pts = [tuple(map(int, q)) for q in g.zone]
        for k in range(len(pts)):
            cv2.line(view, pts[k], pts[(k + 1) % len(pts)], (255, 200, 0), 1)
        cv2.line(view, tuple(map(int, g.a)), tuple(map(int, g.b)), (0, 0, 255), 2)
        for a, b in g.barriers:
            cv2.line(view, tuple(map(int, a)), tuple(map(int, b)), (255, 0, 255), 2)
        cv2.rectangle(view, g.roi[:2], g.roi[2:], (0, 255, 0) if is_green else (0, 0, 255), 2)
        painter.text(view, f"{g.name}: {painter.t('credits')} {len(g.credits)}", (int(g.a[0]), int(g.a[1]) - 8),
                     0.55, (255, 255, 255))


WINDOW = "Turnstile detector  (q - exit, f - fullscreen, space - pause)"


def need_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        sys.exit("Не установлены зависимости. Запусти start.bat (Windows) или start.command (Mac) "
                 "либо выполни: pip install -r requirements.txt")


def run(cfg, params, show=True):
    import cv2
    from hud import Painter, beep, draw_hud, waiting_frame
    YOLO = need_yolo()
    painter = Painter(params.ru_text)

    source = cfg["source"]
    file_mode = is_file_source(source)
    if not file_mode and not str(source).lower().startswith(("rtsp://", "http://", "https://")):
        sys.exit(f"Источник не найден: {source} (нужна ссылка rtsp/http или существующий видеофайл)")
    model = YOLO(params.model)
    gates = [GateLogic(g, params) for g in cfg["gates"]]
    out = Output(cfg.get("camera", "camera"))
    anchor_mode = cfg.get("anchor", "foot")

    if file_mode:
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(fps / params.process_fps)) if params.process_fps > 0 else 1
        reader, frame_i = None, 0
    else:
        reader, last_n = LatestFrame(source), 0
    alerts, totals = deque(maxlen=50), Counter()
    paused, fullscreen, last_frame_wall, size = False, False, time.time(), (960, 540)
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, *size)
    print("Запущено. Выход — q в окне или Ctrl+C.")

    try:
        while True:
            if file_mode:
                for _ in range(step - 1):                      # пропущенные кадры не отдаём в нейросеть
                    cap.grab()
                    frame_i += 1
                ok, frame = cap.read()
                if not ok:
                    break
                t = frame_i / fps
                frame_i += 1
                stamp = f"video+{t:.1f}s"
            else:
                frame, last_n = reader.get(last_n)
                if frame is None:
                    if show and time.time() - last_frame_wall > 3:
                        cv2.imshow(WINDOW, waiting_frame(painter, size))
                        if cv2.waitKey(200) & 0xFF == ord("q"):
                            break
                    else:
                        time.sleep(0.005)
                    continue
                last_frame_wall = time.time()
                t = time.time()
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            res = model.track(frame, persist=True, classes=[0], conf=params.conf, imgsz=params.imgsz,
                              tracker="bytetrack.yaml", verbose=False, device=params.device or None)[0]
            persons = build_persons(res, anchor_mode, params.kp_conf)

            view = frame.copy()
            events, greens = [], []
            for g in gates:
                is_green = green_share(frame, g.roi, params) > params.green_frac
                greens.append(is_green)
                events += g.update(t, persons, is_green)
            draw_overlay(cv2, view, persons, gates, greens, painter)

            for e in events:
                if e["type"] == "pass":
                    out.passed(stamp, e)
                else:
                    shot = view.copy()
                    cv2.putText(shot, f"{e['gate']}: {e['kind']}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    out.violation(stamp, e, shot)
                    alerts.append((time.time(), stamp[-8:] if not file_mode else stamp, None, e["gate"], e["kind"]))
                    totals[e["kind"]] += 1
                    if params.sound and show:
                        beep()

            if show:
                size = (view.shape[1], view.shape[0])
                draw_hud(painter, view, alerts, totals, time.time())
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and file_mode:              # пауза — только для записи
                    paused = True
                    draw_hud(painter, view, alerts, totals, time.time(), paused=True)
                    cv2.imshow(WINDOW, view)
                    while paused:
                        key = cv2.waitKey(50) & 0xFF
                        if key in (ord(" "), ord("q")):
                            paused = False
                if key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if key == ord("q"):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE) < 0:  # окно закрыли крестиком
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if reader is not None:
            reader.stop = True
        else:
            cap.release()
        if show:
            cv2.destroyAllWindows()
    total = sum((g.stats for g in gates), Counter())
    if total:
        print("Тревоги:", ", ".join(f"{k} — {v}" for k, v in total.most_common()))
    print("Остановлено. Результаты в папке", out.dir)


def ask(prompt, default=""):
    try:
        v = input(prompt).strip().strip('"')
    except EOFError:
        v = ""
    return v or default


def quick_start(config_path="cam.json"):
    """Запуск без параметров: всё спрашивается по шагам, дальше работает окно с видео."""
    print("=== Детектор нарушений прохода ===\n")
    path = Path(config_path)
    if not path.exists():
        print("Камера ещё не настроена. Нужна ссылка на поток (index.m3u8 из «Аргуса» или RTSP) "
              "или путь к видеофайлу.")
        source = ask("Ссылка или файл: ")
        if not source:
            sys.exit("Ссылка не указана")
        print("\nСейчас откроется кадр: отметь кликами зону, линию и индикатор каждого проёма.")
        setup_mode(config_path, source)
        cfg = json.loads(path.read_text(encoding="utf-8"))
    else:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        print(f"Камера: {cfg.get('camera', '?')}, проёмов: {len(cfg.get('gates', []))}")
        print(f"Текущий источник: {cfg.get('source')}")
        print("Ссылка из «Аргуса» живёт около суток — если она старая, вставь новую.")
        source = ask("Новая ссылка (Enter — оставить прежнюю, s — разметить заново): ")
        if source.lower() == "s":
            setup_mode(config_path, ask("Ссылка или файл для разметки: ", cfg.get("source", "")))
            cfg = json.loads(path.read_text(encoding="utf-8"))
        elif source:
            cfg["source"] = source
            path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    _, params = load_config(config_path)
    print("\nОткрываю окно. Клавиши: q — выход, f — во весь экран, пробел — пауза (для записи).")
    run(cfg, params, show=True)


def main():
    if len(sys.argv) == 1:
        if Path("cam.json").exists() and not Path("links.txt").exists():
            return quick_start()                         # старый режим: одна камера в cam.json
        import multicam
        return multicam.wizard()
    ap = argparse.ArgumentParser(description="Детекция нарушений прохода через турникет")
    ap.add_argument("--config", default="cam.json")
    ap.add_argument("--source", help="ссылка на поток (RTSP / m3u8) или видеофайл; заменяет source из конфига")
    ap.add_argument("--setup", action="store_true", help="разметить зоны, линии и индикаторы")
    ap.add_argument("--calib", action="store_true", help="проверить распознавание индикатора")
    ap.add_argument("--selftest", action="store_true", help="проверка логики без камеры")
    ap.add_argument("--no-show", action="store_true", help="не показывать окно с видео")
    ap.add_argument("--multi", action="store_true", help="все камеры из cameras/ и links.txt в одном окне")
    ap.add_argument("--auto", action="store_true",
                    help="авторазметка: всех камер из links.txt (или одной: --source и --camera)")
    ap.add_argument("--camera", default="CAM-01", help="имя камеры для --auto с --source")
    ap.add_argument("--observe", type=int, default=180, help="сколько секунд наблюдать при авторазметке")
    ap.add_argument("--ru", action="store_true", help="подписи на экране по-русски")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.auto or args.multi:
        import multicam
        gp = multicam.global_params()
        if args.ru:
            gp["ru_text"] = True
        if args.auto:
            sources = {args.camera: args.source} if args.source else multicam.read_links()
            if not sources:
                sys.exit("Нет ссылок: укажи --source или заполни links.txt")
            multicam.autosetup(sources, args.observe, multicam.make_params(gp), review=not args.no_show)
        if args.multi:
            multicam.run_many(multicam.load_cameras(), gp, show=not args.no_show)
        return
    if args.setup:
        return setup_mode(args.config, args.source)
    cfg, params = load_config(args.config, args.source)
    if args.ru:
        params.ru_text = True
    if args.calib:
        return calib_mode(cfg, params)
    run(cfg, params, show=not args.no_show)


if __name__ == "__main__":
    main()
