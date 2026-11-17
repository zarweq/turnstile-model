"""
hud.py — всё, что рисуется на экране: подписи (латиница или кириллица), баннер тревоги,
список последних тревог, счётчики и мозаика для многих камер.

Кириллица включается параметром "ru_text": true. OpenCV сам кириллицу не рисует,
поэтому такой текст рисуется через Pillow (ставится вместе с ultralytics).
Если Pillow или шрифт не найдены, программа тихо возвращается к латинице.
"""

import math
from pathlib import Path

import cv2
import numpy as np

KIND_LABEL = {
    "en": {"NO_GREEN": "pass without green", "ZONE": "2+ people in gate", "WIDE": "two in one box",
           "HANDOVER": "card handover", "LIMIT": "escort limit", "CLIMB": "climb over",
           "REVERSE": "wrong direction"},
    "ru": {"NO_GREEN": "проход без зелёного", "ZONE": "двое в проёме", "WIDE": "двое в одной рамке",
           "HANDOVER": "передача пропуска", "LIMIT": "превышен лимит", "CLIMB": "перелаз",
           "REVERSE": "обратный проход"},
}
TEXT = {
    "en": {"violation": "VIOLATION", "gate": "gate", "none": "no violations", "paused": "PAUSED",
           "wait": "Waiting for video stream... (link may be expired)", "credits": "credits",
           "nosignal": "NO SIGNAL"},
    "ru": {"violation": "НАРУШЕНИЕ", "gate": "проём", "none": "нарушений нет", "paused": "ПАУЗА",
           "wait": "Жду видеопоток... (ссылка могла устареть)", "credits": "разрешений",
           "nosignal": "НЕТ СИГНАЛА"},
}

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


class Painter:
    """Рисует текст латиницей (OpenCV) или кириллицей (Pillow)."""

    def __init__(self, ru=False):
        self.lang, self._fonts, self._font_path = "en", {}, None
        if ru:
            import importlib.util
            if importlib.util.find_spec("PIL") is not None:
                self._font_path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
                if self._font_path:
                    self.lang = "ru"

    def t(self, key):
        return TEXT[self.lang][key]

    def label(self, kind):
        return KIND_LABEL[self.lang].get(kind, "")

    def _font(self, px):
        from PIL import ImageFont
        if px not in self._fonts:
            self._fonts[px] = ImageFont.truetype(self._font_path, px)
        return self._fonts[px]

    def size(self, text, scale):
        if self.lang == "ru":
            box = self._font(max(10, int(30 * scale))).getbbox(text)
            return box[2] - box[0], box[3] - box[1]
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, int(2 * scale)))
        return w, h

    def text(self, img, text, org, scale, color, bg=None, pad=4):
        """org — левый нижний угол строки, как в cv2.putText. color — BGR."""
        x, y = int(org[0]), int(org[1])
        w, h = self.size(text, scale)
        if bg is not None:
            cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + pad), bg, -1)
        if self.lang != "ru":
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, int(2 * scale)))
            return
        from PIL import Image, ImageDraw
        x0, y0 = max(0, x - pad), max(0, y - h - pad * 2)
        x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + pad * 2)
        if x1 <= x0 or y1 <= y0:
            return
        roi = Image.fromarray(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(roi).text((x - x0, y - y0), text, font=self._font(max(10, int(30 * scale))),
                                 fill=(color[2], color[1], color[0]), anchor="ls")
        img[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(roi), cv2.COLOR_RGB2BGR)


def draw_hud(painter, view, alerts, totals, now, paused=False, camera=None, side_list=True):
    """alerts: последовательность (wall_time, clock, camera, gate, kind)."""
    h, w = view.shape[:2]
    sc = max(0.5, min(1.2, w / 1280))
    if alerts and now - alerts[-1][0] < 4.0:
        _, clock, cam, gate, kind = alerts[-1]
        bh = int(48 * sc)
        cv2.rectangle(view, (0, 0), (w, bh), (0, 0, 200), -1)
        where = f"{cam} " if cam and cam != camera else ""
        painter.text(view, f"{painter.t('violation')}  {where}{painter.t('gate')} {gate}: {kind} — "
                           f"{painter.label(kind)}  [{clock}]", (10, int(bh * 0.72)), 0.8 * sc, (255, 255, 255))
    y = int(84 * sc)
    for _, clock, cam, gate, kind in (list(alerts)[-6:][::-1] if side_list else []):
        txt = f"{clock} {cam + ' ' if cam and cam != camera else ''}{gate} {kind}"
        tw, _ = painter.size(txt, 0.55 * sc)
        painter.text(view, txt, (w - tw - 10, y), 0.55 * sc, (80, 80, 255), bg=(0, 0, 0))
        y += int(30 * sc)
    txt = "  ".join(f"{k}:{v}" for k, v in totals.most_common()) or painter.t("none")
    if paused:
        txt = f"{painter.t('paused')}  {txt}"
    cv2.rectangle(view, (0, h - int(34 * sc)), (w, h), (0, 0, 0), -1)
    painter.text(view, txt, (10, h - int(10 * sc)), 0.6 * sc, (255, 255, 255))


def waiting_frame(painter, size, text=None):
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    painter.text(img, text or painter.t("wait"), (30, h // 2), 0.9, (255, 255, 255))
    return img


def grid_shape(n):
    cols = max(1, math.ceil(math.sqrt(n)))          # 20 камер -> 5 x 4
    rows = math.ceil(n / cols)
    return rows, cols


MOSAIC_TOP, MOSAIC_BOTTOM = 52, 36          # полосы под баннер тревоги и счётчики — не закрывают камеры


def mosaic(painter, tiles, names, alert_until, now, tile_w=384, tile_h=216):
    """Склеивает кадры всех камер в сетку. Камера с недавней тревогой — в красной рамке."""
    n = len(tiles)
    rows, cols = grid_shape(n)
    canvas = np.zeros((MOSAIC_TOP + rows * tile_h + MOSAIC_BOTTOM, cols * tile_w, 3), np.uint8)
    for i, (img, name) in enumerate(zip(tiles, names)):
        r, c = divmod(i, cols)
        y, x = MOSAIC_TOP + r * tile_h, c * tile_w
        if img is None:
            canvas[y:y + tile_h, x:x + tile_w] = 30
            painter.text(canvas, painter.t("nosignal"), (x + 12, y + tile_h // 2), 0.6, (200, 200, 200))
        else:
            canvas[y:y + tile_h, x:x + tile_w] = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        hot = alert_until.get(name, 0) > now
        cv2.rectangle(canvas, (x, y), (x + tile_w - 1, y + tile_h - 1), (0, 0, 255) if hot else (60, 60, 60),
                      4 if hot else 1)
        painter.text(canvas, f"{i + 1}. {name}", (x + 8, y + 24), 0.55, (255, 255, 255), bg=(0, 0, 0))
    return canvas


def tile_at(n, canvas_size, point):
    """Номер плитки мозаики под точкой (для клика мышью) или None."""
    rows, cols = grid_shape(n)
    w, h = canvas_size
    h -= MOSAIC_TOP + MOSAIC_BOTTOM
    y = point[1] - MOSAIC_TOP
    if y < 0 or y >= h:
        return None
    c, r = int(point[0] * cols / w), int(y * rows / h)
    i = r * cols + c
    return i if 0 <= i < n and 0 <= c < cols else None


def beep():
    """Короткий системный звук на Windows (асинхронно, не тормозит видео). На других ОС — ничего."""
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONHAND)
    except Exception:
        pass
