"""
setup_ui.py — ручная разметка камеры: 3 клика на проём, подсказки по-русски прямо на картинке.

На каждый проём:
  1. клик по одной стойке проёма (внизу, у пола — где стоят ноги проходящего);
  2. клик по второй стойке — между ними появится красная линия проёма;
  3. клик по индикатору (купол / лампа, которая загорается зелёным) — рамка подберётся сама.
Зона проёма строится автоматически вокруг линии.

Клавиши в окне:
  Enter     — принять проём (и потом — закончить разметку)
  Backspace — отменить последний клик (можно много раз)
  G         — переключить: турникет / грузовые ворота
  T         — добавить тумбу для правила «перелаз» (2 клика вдоль тумбы)
  Esc       — выйти без сохранения
"""

import numpy as np

from turnstile_detector import dist


def find_indicator_box(frame, pt, size_hint):
    """Рамка индикатора вокруг клика: ищем рядом цветное (красное/зелёное) пятно, иначе квадрат."""
    import cv2
    h, w = frame.shape[:2]
    r = int(max(12, size_hint))
    x0, y0, x1, y1 = max(0, pt[0] - r), max(0, pt[1] - r), min(w, pt[0] + r), min(h, pt[1] + r)
    half = int(max(6, size_hint / 3))
    fallback = [max(0, pt[0] - half), max(0, pt[1] - half), min(w, pt[0] + half), min(h, pt[1] + half)]
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return fallback
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    col = ((s >= 70) & (v >= 60) & ((hh <= 12) | (hh >= 165) | ((hh >= 35) & (hh <= 90)))).astype(np.uint8)
    if col.sum() < 4:
        return fallback
    col = cv2.dilate(col, np.ones((3, 3), np.uint8), iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(col)
    cx, cy = pt[0] - x0, pt[1] - y0
    best = None
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        d = float(np.min((xs - cx) ** 2 + (ys - cy) ** 2))
        if best is None or d < best[0]:
            best = (d, i)
    x, y, bw, bh, _ = stats[best[1]]
    pad = 3
    return [int(x0 + x - pad), int(y0 + y - pad), int(x0 + x + bw + pad), int(y0 + y + bh + pad)]


def zone_around(a, b, depth_factor=0.5):
    """Прямоугольник вокруг линии a-b: по depth_factor * длины линии в каждую сторону."""
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    L = float(np.hypot(*d)) or 1.0
    n = np.array([-d[1], d[0]]) / L * L * depth_factor
    return [[int(round(q[0])), int(round(q[1]))] for q in (a - n, b - n, b + n, a + n)]


STEPS = {
    0: "Шаг 1 из 3: кликните по ПЕРВОЙ стойке проёма — внизу, у пола",
    1: "Шаг 2 из 3: кликните по ВТОРОЙ стойке этого же проёма",
    2: "Шаг 3 из 3: кликните по ИНДИКАТОРУ (лампа, что горит красным/зелёным)",
    3: "Проём готов. Enter — принять,  G — турникет/ворота,  T — добавить тумбу",
}


class SetupUI:
    """Состояние разметки. Логика отделена от окна, поэтому проверяется тестами без экрана."""

    def __init__(self, frame, painter):
        self.frame, self.painter = frame, painter
        self.gates, self.pts, self.mouse = [], [], None
        self.kind = "turnstile"
        self.barriers, self.bar_pts = [], []
        self.adding_barrier = False
        self.indicator = None
        self.finished = self.cancelled = False

    # --- события ---
    def click(self, x, y):
        if self.adding_barrier:
            self.bar_pts.append((x, y))
            if len(self.bar_pts) == 2:
                self.barriers.append([list(self.bar_pts[0]), list(self.bar_pts[1])])
                self.bar_pts, self.adding_barrier = [], False
            return
        if len(self.pts) < 2:
            self.pts.append((x, y))
        elif len(self.pts) == 2:
            self.pts.append((x, y))
            size = dist(self.pts[0], self.pts[1]) / 4
            self.indicator = find_indicator_box(self.frame, (x, y), size)

    def move(self, x, y):
        self.mouse = (x, y)

    def key(self, k):
        """k — код клавиши (как из cv2.waitKeyEx). Русская раскладка тоже понимается: П = G, Е = T."""
        if k in (8, 127, 0x2E0000, 0x700000):              # Backspace / Delete
            self.undo()
        elif k in (13, 10):                                 # Enter
            if len(self.pts) == 3:
                self._accept()
            elif not self.pts and self.gates:
                self.finished = True
        elif k in (ord("g"), ord("G"), ord("п"), ord("П"), 0x47):
            self.kind = "cargo" if self.kind == "turnstile" else "turnstile"
        elif k in (ord("t"), ord("T"), ord("е"), ord("Е"), 0x54) and len(self.pts) == 3:
            self.adding_barrier, self.bar_pts = True, []
        elif k == 27:                                       # Esc
            if self.adding_barrier:
                self.adding_barrier, self.bar_pts = False, []
            else:
                self.cancelled = True

    def undo(self):
        if self.adding_barrier and self.bar_pts:
            self.bar_pts.pop()
        elif self.adding_barrier:
            self.adding_barrier = False
        elif self.barriers and len(self.pts) == 3:
            self.barriers.pop()
        elif self.pts:
            self.pts.pop()
            if len(self.pts) < 3:
                self.indicator = None
        elif self.gates:                                    # вернуться к предыдущему проёму
            g = self.gates.pop()
            self.pts = [tuple(g["line"][0]), tuple(g["line"][1]), tuple(g["_click"])]
            self.indicator, self.kind, self.barriers = g["indicator"], g["kind"], g["barriers"]

    def _accept(self):
        a, b = self.pts[0], self.pts[1]
        self.gates.append({"name": chr(ord("A") + len(self.gates)), "kind": self.kind,
                           "zone": zone_around(a, b), "line": [list(a), list(b)],
                           "indicator": list(self.indicator), "barriers": self.barriers,
                           "allowed_direction": 1, "_click": list(self.pts[2])})
        self.pts, self.indicator, self.kind, self.barriers = [], None, "turnstile", []

    def result(self):
        out = []
        for g in self.gates:
            g = {k: v for k, v in g.items() if not k.startswith("_")}
            out.append(g)
        return out

    # --- картинка ---
    def hint(self):
        if self.adding_barrier:
            return f"Тумба: кликните {'первый' if not self.bar_pts else 'второй'} конец тумбы (вдоль неё). Esc — отмена"
        if not self.pts and self.gates:
            return (f"Готово проёмов: {len(self.gates)}. Кликните по стойке следующего проёма "
                    "или нажмите Enter — закончить")
        return STEPS[len(self.pts)]

    def render(self):
        import cv2
        img = self.frame.copy()
        h, w = img.shape[:2]
        sc = max(0.6, min(1.4, w / 1100))
        for g in self.gates:
            self._draw_gate(cv2, img, g["line"], g["indicator"], g["kind"], g["barriers"], g["name"], done=True)
        if self.pts:
            line = [self.pts[0], self.pts[1] if len(self.pts) > 1 else (self.mouse or self.pts[0])]
            self._draw_gate(cv2, img, line, self.indicator, self.kind, self.barriers,
                            chr(ord("A") + len(self.gates)), done=False)
            if len(self.pts) == 2 and self.mouse:                     # куда сейчас тянемся к индикатору
                mid = ((self.pts[0][0] + self.pts[1][0]) // 2, (self.pts[0][1] + self.pts[1][1]) // 2)
                cv2.line(img, mid, self.mouse, (0, 255, 255), 1)
        for q in self.bar_pts:
            cv2.circle(img, q, 6, (255, 0, 255), -1)
        if self.mouse:
            x, y = self.mouse
            cv2.line(img, (x - 15, y), (x + 15, y), (0, 255, 255), 1)
            cv2.line(img, (x, y - 15), (x, y + 15), (0, 255, 255), 1)
        bh = int(46 * sc)
        cv2.rectangle(img, (0, 0), (w, bh), (40, 40, 40), -1)
        hint, hs = self.hint(), 0.75 * sc
        while hs > 0.35 and self.painter.size(hint, hs)[0] > w - 24:     # подсказка всегда влезает
            hs *= 0.92
        self.painter.text(img, hint, (12, int(bh * 0.72)), hs, (0, 255, 255))
        foot = "Backspace — отменить клик     Esc — выйти без сохранения"
        cv2.rectangle(img, (0, h - int(34 * sc)), (w, h), (40, 40, 40), -1)
        self.painter.text(img, foot, (12, h - int(10 * sc)), 0.6 * sc, (220, 220, 220))
        return img

    def _draw_gate(self, cv2, img, line, ind, kind, barriers, name, done):
        a, b = tuple(line[0]), tuple(line[1])
        if a != b:
            for q, r in zip(zone_around(a, b), zone_around(a, b)[1:] + zone_around(a, b)[:1]):
                cv2.line(img, tuple(q), tuple(r), (255, 200, 0), 1 if done else 2)
        cv2.line(img, a, b, (0, 0, 255), 3)
        for q in (a, b):
            cv2.circle(img, q, 6, (0, 0, 255), -1)
        if ind:
            cv2.rectangle(img, tuple(ind[:2]), tuple(ind[2:]), (0, 255, 0), 2)
        for s in barriers:
            cv2.line(img, tuple(s[0]), tuple(s[1]), (255, 0, 255), 3)
        label = f"{name}: {'ворота' if kind == 'cargo' else 'турникет'}"
        self.painter.text(img, label, (a[0] + 8, a[1] - 10), 0.6, (255, 255, 255), bg=(0, 0, 0))


def mark_gates(frame, painter, title="Setup"):
    """Открывает окно разметки. Возвращает список проёмов или None (отмена)."""
    import cv2
    ui = SetupUI(frame, painter)
    win = title
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    h, w = frame.shape[:2]
    k = min(1.0, 1500 / w, 850 / h)
    cv2.resizeWindow(win, int(w * k), int(h * k))

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            ui.click(x, y)
        elif ev == cv2.EVENT_RBUTTONDOWN:
            ui.undo()
        elif ev == cv2.EVENT_MOUSEMOVE:
            ui.move(x, y)

    cv2.setMouseCallback(win, on_mouse)
    while not (ui.finished or ui.cancelled):
        cv2.imshow(win, ui.render())
        key = cv2.waitKeyEx(30)
        if key == -1:
            if cv2.getWindowProperty(win, cv2.WND_PROP_AUTOSIZE) < 0:     # закрыли крестиком
                ui.finished = bool(ui.gates)
                ui.cancelled = not ui.gates
            continue
        ui.key(key if key > 255 else key & 0xFF)
    cv2.destroyWindow(win)
    return None if ui.cancelled else ui.result()
