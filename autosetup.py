"""
autosetup.py — автоматическая разметка проёмов по наблюдению за видео.

Вместо 8 кликов на проём программа несколько минут смотрит поток и находит сама:

1. Индикаторы — места, где пиксели бывают то насыщенно-красными, то зелёными
   (окантовка купола, ворота, галочка/крестик на считывателе пожарного выхода).
   Если за время наблюдения никто не прикладывал пропуск, берутся устойчиво красные
   небольшие пятна — такие проёмы помечаются «проверить».
2. Проёмы — по траекториям людей рядом с каждым индикатором: главное направление
   движения, «талия» потока (где люди идут теснее всего — это и есть проём),
   ширина проёма. Линия проёма — поперёк потока в самом узком месте.
3. Тип — маленький (турникет, трипод) или большой (грузовые ворота): ширина потока
   сравнивается с шириной человека в этом же месте кадра. Поэтому одинаково работает и
   для турникетов у камеры, и для дальних, и для широких ворот.

Результат — обычный конфиг камеры (тот же формат, что после --setup) и картинка
с разметкой для проверки. Разметку всегда стоит проверить глазами: в окне проверки
Enter — принять, m — разметить вручную, s — пропустить камеру.
"""

from pathlib import Path

import cv2
import numpy as np

from turnstile_detector import Params, dist

GREEN_H, RED_LO, RED_HI = (35, 90), 12, 165


# ============================ ИНДИКАТОРЫ ============================
class IndicatorFinder:
    """Копит по кадрам, какие пиксели бывали красными и какие зелёными."""

    def __init__(self, shape, p: Params = None, down=2):
        p = p or Params()
        self.p, self.down = p, down
        h, w = shape[0] // down, shape[1] // down
        self.red = np.zeros((h, w), np.uint16)
        self.green = np.zeros((h, w), np.uint16)
        self.n = 0

    def add(self, frame):
        small = cv2.resize(frame, (self.red.shape[1], self.red.shape[0]), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        colored = (s >= max(self.p.sat_min, 90)) & (v >= max(self.p.val_min, 80))
        self.green += (colored & (h >= GREEN_H[0]) & (h <= GREEN_H[1])).astype(np.uint16)
        self.red += (colored & ((h <= RED_LO) | (h >= RED_HI))).astype(np.uint16)
        self.n += 1

    def find(self, min_hits=2, max_area_frac=0.01, red_share=0.4):
        """Список {"box": [x1,y1,x2,y2], "toggled": bool} в координатах исходного кадра.

        Индикатор стоит на месте и почти всё время красный, иногда зелёный. Одежда идущих людей
        тоже бывает красной и зелёной, но в каждой точке недолго — поэтому нужна доля красного red_share.
        """
        if self.n == 0:
            return []
        toggled = (self.red >= red_share * self.n) & (self.green >= min_hits)
        steady_red = self.red >= 0.8 * self.n
        out = []
        for mask, is_toggle in ((toggled, True), (steady_red & ~toggled, False)):
            m = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
            n, _, stats, _ = cv2.connectedComponentsWithStats(m)
            max_area = max_area_frac * m.size
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area < 4 or area > max_area or w > 6 * h + 4 or h > 6 * w + 4:
                    continue
                pad = 1
                box = [int((x - pad) * self.down), int((y - pad) * self.down),
                       int((x + w + pad) * self.down), int((y + h + pad) * self.down)]
                if not is_toggle and any(_near(box, o["box"]) for o in out):
                    continue                                   # кусок уже найденного индикатора
                out.append({"box": [max(0, b) for b in box], "toggled": is_toggle})
        return out


def _near(a, b):
    """Рамки пересекаются или почти касаются (с запасом в свой размер)."""
    m = max(b[2] - b[0], b[3] - b[1], a[2] - a[0], a[3] - a[1])
    return not (a[2] + m < b[0] or b[2] + m < a[0] or a[3] + m < b[1] or b[3] + m < a[1])


def box_dist(p, box):
    """Расстояние от точки до рамки (0 — внутри)."""
    dx = max(box[0] - p[0], 0, p[0] - box[2])
    dy = max(box[1] - p[1], 0, p[1] - box[3])
    return (dx * dx + dy * dy) ** 0.5


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


# ============================ ПРОЁМЫ ПО ТРАЕКТОРИЯМ ============================
def split_tracks(tracks, indicators, max_dist):
    """Каждую траекторию отдаём ближайшему индикатору. Расстояние меряется до рамки человека,
    а не до ног: индикатор стоит на стойке на уровне пояса и при проходе оказывается «внутри» человека."""
    groups = [[] for _ in indicators]
    centers = [_center(i["box"]) for i in indicators]
    for tr in tracks:
        if len(tr["points"]) < 3:
            continue
        boxes = tr.get("boxes") or [(p[0], p[1], p[0], p[1]) for p in tr["points"]]
        best = None
        for gi, c in enumerate(centers):
            d = min(box_dist(c, b) for b in boxes)
            if d <= max_dist(gi) and (best is None or d < best[0]):
                best = (d, gi)
        if best:
            groups[best[1]].append(tr)
    return groups


def gate_from_tracks(tracks, indicator, cargo_factor=2.2):
    """Проём по траекториям, прошедшим у индикатора. None — если людей было слишком мало."""
    pts = np.array([p for tr in tracks for p in tr["points"]], float)
    widths = [w for tr in tracks for w in tr["widths"]]
    if len(tracks) < 3 or len(pts) < 10 or not widths:
        return None
    person_w = float(np.median(widths))
    # главное направление движения — по смещениям внутри траекторий
    steps = np.concatenate([np.diff(np.array(tr["points"], float), axis=0) for tr in tracks if len(tr["points"]) > 1])
    steps = steps[np.linalg.norm(steps, axis=1) > 1e-6]
    if len(steps) < 5:
        return None
    _, _, vt = np.linalg.svd(steps, full_matrices=False)  # ось без знака: люди ходят в обе стороны
    d = vt[0] / np.linalg.norm(vt[0])                     # вдоль потока
    n = np.array([-d[1], d[0]])                           # поперёк потока
    c0 = pts.mean(axis=0)
    along, across = (pts - c0) @ d, (pts - c0) @ n
    # «талия» потока: срез вдоль направления с наименьшим разбросом поперёк
    lo, hi = np.percentile(along, 10), np.percentile(along, 90)
    best = None
    for s in np.linspace(lo, hi, 15):
        sel = np.abs(along - s) <= max(person_w * 0.5, (hi - lo) / 15)
        if sel.sum() < 6:
            continue
        spread = np.percentile(across[sel], 90) - np.percentile(across[sel], 10)
        if best is None or spread < best[0]:
            best = (spread, s, float(np.median(across[sel])))
    if best is None:
        return None
    spread, s, mid = best
    width = max(spread + person_w, 1.1 * person_w)        # проём шире разброса центров на ширину человека
    center = c0 + d * s + n * mid
    kind = "cargo" if width > cargo_factor * person_w else "turnstile"
    a, b = center - n * width / 2, center + n * width / 2
    depth = max(0.6 * width, 1.2 * person_w) if kind == "turnstile" else 1.2 * person_w
    zone = [center - n * width / 2 - d * depth / 2, center + n * width / 2 - d * depth / 2,
            center + n * width / 2 + d * depth / 2, center - n * width / 2 + d * depth / 2]
    g = {"kind": kind, "line": [_ip(a), _ip(b)], "zone": [_ip(q) for q in zone],
            "indicator": list(indicator["box"]),
            "auto": {"passes": len(tracks), "person_w": round(person_w, 1), "width": round(width, 1),
                     "indicator_seen_green": indicator["toggled"], "check": not indicator["toggled"]}}
    if kind == "cargo":                                   # масштаб правил — ширина обычного турникета
        g["scale"] = round(1.3 * person_w, 1)
    return g


def _ip(p):
    return [int(round(p[0])), int(round(p[1]))]


def build_config(camera, source, frame_shape, indicators, tracks, anchor="foot"):
    """Собирает конфиг камеры. Возвращает (cfg, заметки для оператора)."""
    notes = []
    if not indicators:
        notes.append("индикаторы не найдены — нужна ручная разметка или наблюдение подольше")
        return {"camera": camera, "source": source, "anchor": anchor, "gates": [], "params": {}}, notes
    widths = [w for tr in tracks for w in tr["widths"]] or [frame_shape[1] / 20]
    pw = float(np.median(widths))
    groups = split_tracks(tracks, indicators, lambda gi: 1.5 * pw)
    gates = []
    for ind, grp in zip(indicators, groups):
        g = gate_from_tracks(grp, ind)
        if g is None:
            notes.append(f"индикатор {ind['box']}: мало проходов рядом ({len(grp)}), проём не построен")
            continue
        g["name"] = chr(ord("A") + len(gates))
        g["allowed_direction"] = 1
        g["barriers"] = []
        if g["auto"]["check"]:
            notes.append(f"проём {g['name']}: зелёный за время наблюдения не загорался — проверь индикатор")
        gates.append(g)
    cfg = {"camera": camera, "source": source, "anchor": anchor, "gates": gates, "params": {}}
    return cfg, notes


# ============================ НАБЛЮДЕНИЕ ============================
class TrackCollector:
    """Собирает траектории опорных точек людей (по id трекера)."""

    def __init__(self, anchor="foot"):
        self.anchor = anchor
        self.tracks = {}

    def add(self, boxes, ids):
        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            if tid < 0:
                continue
            pt = ((x1 + x2) / 2, (y1 + y2) / 2) if self.anchor == "center" else ((x1 + x2) / 2, y2)
            tr = self.tracks.setdefault(tid, {"points": [], "widths": [], "boxes": []})
            tr["points"].append(pt)
            tr["widths"].append(x2 - x1)
            tr["boxes"].append((x1, y1, x2, y2))

    def moving(self, min_path=None):
        out = []
        for tr in self.tracks.values():
            p = tr["points"]
            if len(p) < 3:
                continue
            path = dist(p[0], p[-1])
            if path >= (min_path if min_path is not None else 1.0 * float(np.median(tr["widths"]))):
                out.append(tr)
        return out


def draw_markup(frame, cfg, notes=()):
    img = frame.copy()
    for g in cfg["gates"]:
        color = (0, 200, 255) if g.get("auto", {}).get("check") else (255, 200, 0)
        pts = [tuple(q) for q in g["zone"]]
        for k in range(len(pts)):
            cv2.line(img, pts[k], pts[(k + 1) % len(pts)], color, 2)
        cv2.line(img, tuple(g["line"][0]), tuple(g["line"][1]), (0, 0, 255), 3)
        x1, y1, x2, y2 = g["indicator"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lbl = f"{g['name']} {'BIG' if g['kind'] == 'cargo' else 'small'} ({g['auto']['passes']} passes)"
        cv2.putText(img, lbl, (pts[0][0], max(15, pts[0][1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
    y = img.shape[0] - 10
    for i, _ in enumerate(notes):
        cv2.putText(img, f"note {i + 1}: see console", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        y -= 20
    return img



# ============================ РАЗМЕТКА ПО ОДНОЙ КАРТИНКЕ (БЕЗ ЛЮДЕЙ) ============================
# Стойки турникетов ищет нейросеть YOLOE по словам («turnstile», «speed gate», ...). Смысл слов посчитан
# заранее и лежит в models/turnstile_prompts.pt, поэтому на рабочем компьютере не нужен CLIP.
# Нейросеть находит не все стойки и иногда путает с ними киоски и стойки указателей, поэтому:
#   * стойка без огонька индикатора (красного/зелёного) отбрасывается, если огоньки в кадре вообще есть;
#   * огонёк в ряду стоек, которого не накрыла ни одна найденная стойка, — пропущенная стойка;
#   * проход — промежуток между соседними стойками, линия — по полу от одной стойки к другой.
# Результат — предложение: он открывается в редакторе, где точки можно перетащить.

PROMPTS_FILE = Path(__file__).with_name("models") / "turnstile_prompts.pt"
STAND_NAMES = {"turnstile", "speed gate", "tripod turnstile", "security gate", "turnstile gate",
               "access control gate", "barrier gate"}
_STAND_MODEL = None


def stand_model(weights="yoloe-11s-seg.pt"):
    """YOLOE с готовыми эмбеддингами слов. None — если ultralytics/торч нет или файл не скачался."""
    global _STAND_MODEL
    if _STAND_MODEL is None:
        try:
            import torch
            from ultralytics import YOLOE
            d = torch.load(PROMPTS_FILE, map_location="cpu")
            m = YOLOE(weights)
            m.set_classes(d["names"], d["pe"])
            _STAND_MODEL = m
        except Exception as e:                                   # нет сети, старая версия и т.п.
            print(f"  (поиск стоек нейросетью недоступен: {e}; ищу только по индикаторам)")
            _STAND_MODEL = False
    return _STAND_MODEL or None


def find_lights(frame, max_area_frac=0.004, s_min=90, v_min=100):
    """Огоньки индикаторов на одном кадре: маленькие насыщенные красные/зелёные пятна."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = (s >= s_min) & (v >= v_min)
    out = []
    for color, mask in (("red", bright & ((h <= 10) | (h >= 170))), ("green", bright & (h >= 40) & (h <= 90))):
        m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            x, y, w, hh, a = st[i]
            if 6 <= a <= max_area_frac * m.size and w <= 4 * hh + 3 and hh <= 4 * w + 3:
                out.append({"box": [int(x), int(y), int(x + w), int(y + hh)], "color": color})
    return out


def _inside(box, outer, pad=0.15):
    w, h = outer[2] - outer[0], outer[3] - outer[1]
    cx, cy = _center(box)
    return outer[0] - pad * w <= cx <= outer[2] + pad * w and outer[1] - pad * h <= cy <= outer[3] + pad * h


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def detect_stands(frame, model=None, conf=0.15):
    """Рамки стоек от нейросети (без людей), похожие рамки склеены."""
    model = model if model is not None else stand_model()
    if model is None:
        return []
    r = model.predict(frame, conf=conf, verbose=False)[0]
    boxes = []
    for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        if r.names[int(c)] in STAND_NAMES:
            boxes.append({"box": [int(v) for v in b], "conf": float(p), "src": "nn"})
    boxes.sort(key=lambda s: -s["conf"])
    keep = []
    for s in boxes:
        if all(_iou(s["box"], k["box"]) < 0.4 and not _inside(s["box"], k["box"], 0) for k in keep):
            keep.append(s)
    return keep


def propose_static(frame, stands=None, lights=None, model=None):
    """Предложение разметки по одной картинке: (список проёмов, заметки)."""
    notes = []
    lights = find_lights(frame) if lights is None else lights
    stands = detect_stands(frame, model) if stands is None else [dict(s) for s in stands]
    for s_ in stands:                                    # стойка без огонька — сомнительная, но не выбрасываем
        s_["lit"] = any(_inside(l_["box"], s_["box"]) for l_ in lights)
    if not stands:
        notes.append("стойки турникетов не найдены — нужна ручная разметка (m)")
        return [], notes
    widths = [s["box"][2] - s["box"][0] for s in stands]
    heights = [s["box"][3] - s["box"][1] for s in stands]
    sw, sh = float(np.median(widths)), float(np.median(heights))
    tops, bottoms = [s["box"][1] for s in stands], [s["box"][3] for s in stands]
    x_lo, x_hi = min(s["box"][0] for s in stands) - 2.5 * sw, max(s["box"][2] for s in stands) + 2.5 * sw
    # огоньки в полосе стоек, не попавшие ни в одну стойку, — пропущенные нейросетью стойки
    for l_ in lights:
        cx, cy = _center(l_["box"])
        if any(_inside(l_["box"], s["box"]) for s in stands) or not (x_lo <= cx <= x_hi):
            continue
        if not (min(tops) - 0.3 * sh <= cy <= min(bottoms) - 0.3 * sh):
            continue
        if any(s["src"] == "light" and abs(_center(s["box"])[0] - cx) < 0.6 * sw for s in stands):
            continue
        bottom = float(np.median(bottoms))
        stands.append({"box": [int(cx - sw / 2), int(min(cy - 0.1 * sh, bottom - sh)), int(cx + sw / 2), int(bottom)],
                       "conf": 0.0, "src": "light", "lit": True})
    added = sum(s["src"] == "light" for s in stands)
    if added:
        notes.append(f"добавлено стоек по огонькам индикаторов: {added}")
    stands.sort(key=lambda s: (s["box"][0] + s["box"][2]) / 2)
    pairs = [(L, R) for L, R in zip(stands, stands[1:])
             if -0.2 * sw <= R["box"][0] - L["box"][2] <= 4.0 * sw and abs(R["box"][3] - L["box"][3]) <= 0.6 * sh]
    gaps = [max(1.0, R["box"][0] - L["box"][2]) for L, R in pairs]
    med_gap = float(np.median(gaps)) if gaps else 0.0
    gates = []
    for L, R in pairs:
        gap = R["box"][0] - L["box"][2]
        a = (int(L["box"][2]), int(L["box"][3]))
        b = (int(R["box"][0]), int(R["box"][3]))
        if dist(a, b) < 0.3 * sw:
            continue
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        own = [l_ for l_ in lights if _inside(l_["box"], L["box"]) or _inside(l_["box"], R["box"])]
        if own:
            ind = min(own, key=lambda l_: dist(_center(l_["box"]), mid))["box"]
            pad = max(2, (ind[2] - ind[0]) // 2)
            ind = [ind[0] - pad, ind[1] - pad, ind[2] + pad, ind[3] + pad]
        else:
            t = L["box"]
            ind = [int(t[2] - 0.5 * sw), int(t[1]), int(t[2]), int(t[1] + 0.25 * sw)]
        from setup_ui import zone_around
        gates.append({"name": chr(ord("A") + len(gates)), "kind": "turnstile", "line": [list(a), list(b)],
                      "zone": zone_around(a, b), "indicator": [max(0, v) for v in ind], "barriers": [],
                      "allowed_direction": 1,
                      "auto": {"static": True, "stands": [L["src"], R["src"]],
                               "check": not own or not (L["lit"] or R["lit"]) or (len(gaps) > 1 and gap > 1.7 * med_gap),
                               "why": "нет огонька" if not own else
                                      ("слишком широкий — возможно, пропущена стойка"
                                       if len(gaps) > 1 and gap > 1.7 * med_gap else "")}})
    if not gates:
        notes.append(f"найдено стоек: {len(stands)}, но проходов между ними не получилось — разметьте вручную")
    for g in gates:
        if g["auto"]["check"]:
            notes.append(f"проём {g['name']}: проверьте ({g['auto']['why'] or 'стойки без огоньков'})")
    return gates, notes


# ============================ ОБУЧЕНИЕ НА ХОДУ ============================
class Learner:
    """
    Разметка ставится сама во время обычной работы. Программа копит:
      * какие места кадра мигают красным/зелёным (индикаторы),
      * траектории людей (по номерам трекера).
    Как только у индикатора, который хоть раз загорелся зелёным, набралось min_passes проходов,
    проём строится так же, как при авторазметке по людям, и сразу начинает работать.
    """

    def __init__(self, shape, params=None, anchor="foot", min_passes=3, every=10.0, max_tracks=3000):
        self.finder = IndicatorFinder(shape, params)
        self.tracks = TrackCollector(anchor)
        self.shape, self.min_passes, self.every, self.max_tracks = shape, min_passes, every, max_tracks
        self.last_try = None
        self.progress = 0            # больше всего проходов у одного индикатора
        self.done = False

    def add(self, frame, persons):
        self.finder.add(frame)
        if self.finder.n >= 30000:                               # счётчики не переполняются за сутки работы
            self.finder.red //= 2
            self.finder.green //= 2
            self.finder.n //= 2
        self.tracks.add([p.box for p in persons], [p.tid for p in persons])
        if len(self.tracks.tracks) > self.max_tracks:            # память не растёт: старые траектории забываем
            for k in list(self.tracks.tracks)[: len(self.tracks.tracks) - self.max_tracks]:
                del self.tracks.tracks[k]

    def try_build(self, t, camera="camera", source=""):
        """Раз в every секунд: новые проёмы (список) или None, если учиться ещё рано."""
        if self.last_try is not None and t - self.last_try < self.every:
            return None
        self.last_try = t
        inds = [i for i in self.finder.find() if i["toggled"]]
        if not inds:
            return None
        tracks = self.tracks.moving()
        widths = [w for tr in tracks for w in tr["widths"]] or [self.shape[1] / 20]
        groups = split_tracks(tracks, inds, lambda gi: 1.5 * float(np.median(widths)))
        self.progress = max((len(g) for g in groups), default=0)
        cfg, _ = build_config(camera, source, self.shape, inds, tracks)
        gates = [g for g in cfg["gates"] if g["auto"]["passes"] >= self.min_passes]
        if not gates:
            return None
        for k, g in enumerate(gates):
            g["name"] = chr(ord("A") + k)
            g["auto"]["learned"] = True
        self.done = len(gates) >= len(inds)
        return gates
