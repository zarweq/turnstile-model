"""
turnstile_detector_v3.py — детекция нарушений прохода через турникет по видео.

Что ловит (на каждом проёме турникета отдельно):
  NO_GREEN  — человек прошёл линию проёма без зелёного сигнала (вдвоём по одному пропуску / без пропуска)
  ZONE      — в проёме одновременно больше людей, чем было зелёных сигналов
  WIDE      — подозрение: в одной рамке двое (рамка заметно шире обычной для этого проёма)
  HANDOVER  — возможная передача пропуска: руки двух людей по разные стороны турникета сошлись
  LIMIT     — один человек дал больше MAX_PER_CARD зелёных подряд (лимит: владелец + 3)
  REVERSE   — проход в запрещённую сторону (только если в конфиге one_way = true)

Грузовые ворота размечаются как отдельный проём (тип "cargo"): логика та же, зелёный = один человек.
Пока НЕ ловит: красный пропуск (нужен цвет карты), перелаз через тумбу турникета.

Установка:
    pip install ultralytics opencv-python lap numpy

Порядок работы:
    python turnstile_detector_v3.py --selftest                       # проверка логики, камера не нужна
    python turnstile_detector_v3.py --setup --config cam.json --source "ССЫЛКА"
    python turnstile_detector_v3.py --calib --config cam.json        # проверить индикатор
    python turnstile_detector_v3.py --config cam.json                # работа
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
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


# ============================ ПАРАМЕТРЫ ============================
@dataclass
class Params:
    # --- проход и зелёный сигнал ---
    credit_ttl: float = 8.0        # сколько секунд после зелёного можно пройти
    green_window: float = 10.0     # окно, в котором зелёные учитываются для правила ZONE
    green_min_gap: float = 1.0     # защита от двойного срабатывания одного сигнала
    recross_sec: float = 5.0       # через сколько секунд тот же человек может быть посчитан снова
    one_way: bool = False          # True — проход против allowed_direction считается нарушением
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
    cooldown: float = 4.0          # пауза между тревогами одного проёма
    model: str = "yolov8n-pose.pt"
    conf: float = 0.4
    kp_conf: float = 0.4           # минимальная уверенность точки кисти
    imgsz: int = 640


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
        self.roi = tuple(cfg["indicator"])
        self.allowed = cfg.get("allowed_direction", 1)
        self.p = params
        self.kind = cfg.get("kind", "turnstile")           # "turnstile" или "cargo" (грузовые ворота)
        # масштаб = ширина обычного проёма в пикселях; у широких ворот задаётся отдельно полем "scale"
        self.L = max(float(cfg.get("scale") or dist(self.a, self.b)), 1.0)
        self.ind_center = ((self.roi[0] + self.roi[2]) / 2, (self.roi[1] + self.roi[3]) / 2)

        self.credits = deque()
        self.recent_greens = deque()
        self.was_green = False
        self.last_green_t = -1e9
        self.last_pos = {}        # tid -> (anchor, t)
        self.counted = {}         # tid -> t последнего учтённого прохода
        self.sessions = {}        # tid -> deque времён зелёных
        self.widths = deque(maxlen=200)
        self.zone_since = None
        self.hand_since = None
        self.last_alert_t = -1e9
        self.last_hand_t = -1e9

    # -- служебное --
    def _alert(self, events, t, kind, text, people=None, greens=None, own_cooldown=False):
        if own_cooldown:
            if t - self.last_hand_t < self.p.cooldown * 1.5:
                return
            self.last_hand_t = t
        else:
            if t - self.last_alert_t < self.p.cooldown:
                return
            self.last_alert_t = t
        events.append({"type": "violation", "kind": kind, "text": text, "gate": self.name,
                       "people": people, "greens": greens})

    def typical_width(self):
        if len(self.widths) >= self.p.min_width_samples:
            return statistics.median(self.widths)
        return None

    # -- основной шаг --
    def update(self, t, persons, is_green):
        p, ev = self.p, []

        # 1. Зелёный сигнал: +1 разрешение
        if is_green and not self.was_green and t - self.last_green_t >= p.green_min_gap:
            self.last_green_t = t
            self.credits.append(t)
            self.recent_greens.append(t)
            near = [(dist(x.anchor, self.ind_center), x) for x in persons if x.tid >= 0]
            near = [(d, x) for d, x in near if d <= p.swiper_max_dist * self.L]
            if near:
                swiper = min(near, key=lambda z: z[0])[1].tid
                s = self.sessions.setdefault(swiper, deque())
                s.append(t)
                while s and t - s[0] > p.session_sec:
                    s.popleft()
                if len(s) > p.max_per_card:
                    self._alert(ev, t, "LIMIT",
                                f"Больше {p.max_per_card} проходов подряд по одному пропуску",
                                greens=len(s), own_cooldown=True)
        self.was_green = is_green

        while self.credits and t - self.credits[0] > p.credit_ttl:
            self.credits.popleft()
        while self.recent_greens and t - self.recent_greens[0] > p.green_window:
            self.recent_greens.popleft()

        # 2. Пересечение линии проёма
        for x in persons:
            if x.tid < 0:
                continue
            prev = self.last_pos.get(x.tid)
            if side(x.anchor, self.a, self.b) == 0 and prev is not None:
                self.last_pos[x.tid] = (prev[0], t)     # точка ровно на линии: ждём следующего кадра
                continue
            self.last_pos[x.tid] = (x.anchor, t)
            if prev is None or not seg_intersect(prev[0], x.anchor, self.a, self.b):
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

        # забываем треки, которых давно не видно
        for tid in [k for k, (_, ts) in self.last_pos.items() if t - ts > 5.0]:
            self.last_pos.pop(tid, None)
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
        hand = False
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
                            hand = True
        if hand:
            if self.hand_since is None:
                self.hand_since = t
            elif t - self.hand_since >= p.hand_sec:
                self._alert(ev, t, "HANDOVER", "Возможная передача пропуска через турникет",
                            people=2, own_cooldown=True)
                self.hand_since = None
        else:
            self.hand_since = None

        return ev


# ============================ САМОПРОВЕРКА ============================
def selftest():
    cfg = {"name": "T", "zone": [[80, 60], [120, 60], [120, 140], [80, 140]],
           "line": [[100, 60], [100, 140]], "indicator": [95, 40, 105, 50]}

    def walker(tid, t0, t, y=100, speed=40, w=30, start=40):
        x = start + (t - t0) * speed
        return Person(tid, (x - w / 2, y - 40, x + w / 2, y + 40), (x, y))

    def run(scn, dur=8.0, fps=10, warm=True):
        g = GateLogic(cfg, Params())
        if warm:
            g.widths.extend([30] * 20)
        out, n = [], int(dur * fps)
        for k in range(n):
            t = k / fps
            persons, green = scn(t)
            out += [e["kind"] for e in g.update(t, persons, green) if e["type"] == "violation"]
        return out

    tests = []
    # 1. один зелёный, один человек -> чисто
    tests.append(("1 зелёный, 1 человек", run(lambda t: ([walker(1, 1.0, t)] if t >= 1 else [], 0.5 <= t < 1.0)), []))
    # 2. один зелёный, двое друг за другом -> NO_GREEN
    tests.append(("1 зелёный, 2 человека", run(
        lambda t: (([walker(1, 1.0, t)] if t >= 1 else []) + ([walker(2, 2.2, t)] if t >= 2.2 else []),
                   0.5 <= t < 1.0), dur=6.0), ["NO_GREEN"]))
    # 3. два зелёных, двое -> чисто
    tests.append(("2 зелёных, 2 человека", run(
        lambda t: (([walker(1, 1.0, t)] if t >= 1 else []) + ([walker(2, 3.0, t)] if t >= 3.0 else []),
                   (0.5 <= t < 0.8) or (2.5 <= t < 2.8)), dur=6.0), []))
    # 4. без зелёного -> NO_GREEN
    tests.append(("0 зелёных, 1 человек", run(lambda t: ([walker(1, 1.0, t)] if t >= 1 else [], False)), ["NO_GREEN"]))
    # 5. один зелёный, одна широкая рамка -> WIDE
    tests.append(("1 зелёный, широкая рамка", run(
        lambda t: ([walker(1, 1.0, t, w=60)] if t >= 1 else [], 0.5 <= t < 1.0)), ["WIDE"]))
    # 6. двое стоят в проёме, один зелёный -> ZONE
    tests.append(("двое в зоне на 1 зелёный", run(
        lambda t: ([Person(-1, (75, 60, 105, 140), (90, 100)), Person(-1, (95, 60, 125, 140), (110, 100))]
                   if 1.0 <= t < 2.5 else [], 0.5 <= t < 1.0), dur=4.0), ["ZONE"]))
    # 7. руки сошлись через турникет -> HANDOVER
    tests.append(("передача пропуска", run(
        lambda t: ([Person(1, (50, 60, 80, 140), (65, 100), [(98, 55)]),
                    Person(2, (120, 60, 150, 140), (135, 100), [(104, 57)])] if 1.0 <= t < 2.5 else [], False),
        dur=4.0), ["HANDOVER"]))
    # 8. пятый зелёный от одного человека -> LIMIT
    tests.append(("5 зелёных от одного", run(
        lambda t: ([Person(1, (60, 20, 90, 100), (95, 60))], (t % 2.0) < 0.3), dur=11.0), ["LIMIT"]))
    # 9. человек идёт мимо проёма (линию как бесконечную прямую пересёк, отрезок — нет) -> чисто
    tests.append(("мимо проёма", run(lambda t: ([walker(1, 1.0, t, y=300)] if t >= 1 else [], False)), []))

    ok = True
    for name, got, want in tests:
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


def setup_mode(config_path, source):
    import cv2
    if not source:
        sys.exit("Для --setup укажи --source \"ссылка или файл\"")
    frame0 = grab_frame(source)
    camera = input("Название камеры (например CAM-01): ").strip() or "camera"
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
        gates.append({"name": chr(ord("A") + gi), "kind": kind, "zone": pts[:4], "line": pts[4:6],
                      "indicator": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                      "allowed_direction": 1})
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
                  ["время", "камера", "проём", "тип", "описание", "людей", "зелёных", "кадр", "вердикт оператора"],
                  [stamp, self.camera, e["gate"], e["kind"], e["text"], e["people"], e["greens"], fname, ""])
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


def run(cfg, params, show=True):
    import cv2
    from ultralytics import YOLO

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
        reader, frame_i = None, 0
    else:
        reader, last_n = LatestFrame(source), 0
    print("Запущено. Выход — q в окне или Ctrl+C.")

    try:
        while True:
            if file_mode:
                ok, frame = cap.read()
                if not ok:
                    break
                t = frame_i / fps
                frame_i += 1
                stamp = f"video+{t:.1f}s"
            else:
                frame, last_n = reader.get(last_n)
                if frame is None:
                    time.sleep(0.005)
                    continue
                t = time.time()
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            res = model.track(frame, persist=True, classes=[0], conf=params.conf, imgsz=params.imgsz,
                              tracker="bytetrack.yaml", verbose=False)[0]
            persons = build_persons(res, anchor_mode, params.kp_conf)

            view = frame.copy()
            for x in persons:
                x1, y1, x2, y2 = map(int, x.box)
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(view, tuple(map(int, x.anchor)), 4, (0, 255, 0), -1)
                cv2.putText(view, str(x.tid), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                for w in x.wrists:
                    cv2.circle(view, tuple(map(int, w)), 4, (255, 0, 255), -1)

            events = []
            for g in gates:
                is_green = green_share(frame, g.roi, params) > params.green_frac
                events += g.update(t, persons, is_green)
                pts = [tuple(map(int, q)) for q in g.zone]
                for k in range(len(pts)):
                    cv2.line(view, pts[k], pts[(k + 1) % len(pts)], (255, 200, 0), 1)
                cv2.line(view, tuple(map(int, g.a)), tuple(map(int, g.b)), (0, 0, 255), 2)
                cv2.rectangle(view, g.roi[:2], g.roi[2:], (0, 255, 0) if is_green else (0, 0, 255), 2)
                cv2.putText(view, f"{g.name}: credits {len(g.credits)}", (int(g.a[0]), int(g.a[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            for e in events:
                if e["type"] == "pass":
                    out.passed(stamp, e)
                else:
                    shot = view.copy()
                    cv2.putText(shot, f"{e['gate']}: {e['kind']}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    out.violation(stamp, e, shot)

            if show:
                cv2.imshow("detector", view)
                if cv2.waitKey(1) == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if reader is not None:
            reader.stop = True
        if show:
            cv2.destroyAllWindows()
    print("Остановлено. Результаты в папке", out.dir)


def main():
    ap = argparse.ArgumentParser(description="Детекция нарушений прохода через турникет")
    ap.add_argument("--config", default="cam.json")
    ap.add_argument("--source", help="ссылка на поток (RTSP / m3u8) или видеофайл; заменяет source из конфига")
    ap.add_argument("--setup", action="store_true", help="разметить зоны, линии и индикаторы")
    ap.add_argument("--calib", action="store_true", help="проверить распознавание индикатора")
    ap.add_argument("--selftest", action="store_true", help="проверка логики без камеры")
    ap.add_argument("--no-show", action="store_true", help="не показывать окно с видео")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.setup:
        return setup_mode(args.config, args.source)
    cfg, params = load_config(args.config, args.source)
    if args.calib:
        return calib_mode(cfg, params)
    run(cfg, params, show=not args.no_show)


if __name__ == "__main__":
    main()
