"""
camid.py — какая это камера? Узнаём по картинке, а не по имени в файле.

Ссылки из «Аргуса» безымянные, меняются каждый день и идут в случайном порядке. Поэтому:
  * ссылки можно вставлять списком в любом порядке, без имён (links.txt, буфер обмена, HAR-файл);
  * при разметке камеры запоминается её вид: cameras/<камера>_ref.jpg;
  * при запуске с каждой ссылки берётся кадр и сравнивается со всеми запомненными видами
    (особые точки ORB по неподвижному фону: стены, стойки, двери — люди и свет мешают мало);
  * если адрес ссылки без токена совпадает с прошлым — это та же камера, без сравнения;
  * похожие камеры (одинаковые этажи) помечаются «проверить»: решает оператор по картинке.
"""

import json
import re
import threading
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

URL_RE = re.compile(r"""(?:https?|rtsp)://[^\s"'<>]+""", re.I)


# ============================ ССЫЛКИ ============================
def extract_urls(text):
    """Все ссылки из любого текста: links.txt, вставка из буфера, кусок HAR/логов. Порядок сохраняется."""
    seen, out = set(), []
    for u in URL_RE.findall(text or ""):
        u = u.rstrip(".,;)]}")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def is_video_url(u):
    low = u.lower()
    return low.startswith("rtsp://") or ".m3u8" in low or ".mp4" in low or "/live" in low


def urls_from_har(path):
    """Ссылки на видеопотоки из HAR-файла (вкладка Network -> «Save all as HAR»)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig", errors="ignore"))
        urls = [e["request"]["url"] for e in data.get("log", {}).get("entries", [])]
    except (ValueError, KeyError, AttributeError):
        urls = extract_urls(Path(path).read_text(encoding="utf-8-sig", errors="ignore"))
    # из HLS берём только плейлисты index/playlist, не отдельные сегменты .ts
    out = [u for u in urls if ".m3u8" in u.lower() or u.lower().startswith("rtsp://")]
    return extract_urls("\n".join(out))


def url_key(u):
    """Адрес без токена/параметров: если он совпал с прошлым — это та же камера."""
    s = urlsplit(u)
    return f"{s.scheme}://{s.hostname}{s.path}".lower()


# ============================ КАДР С ТАЙМАУТОМ ============================
def background(frames):
    """Медиана нескольких кадров: проходящие люди исчезают, остаётся сама сцена."""
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    if len(frames) < 3:
        return frames[-1]
    shape = frames[0].shape
    frames = [f for f in frames if f.shape == shape]
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def grab_frame(source, timeout=20.0, n=7, every=5):
    """Вид сцены из ссылки или файла (медиана n кадров через every); None, если не открылось за timeout."""
    box = {"frames": []}

    def work():
        cap = cv2.VideoCapture(source)
        k, last = 0, None
        while len(box["frames"]) < n and k < n * every + 10:
            ok, f = cap.read()
            if not ok:
                break
            last = f
            if k >= 2 and k % every == 0:        # первые кадры потока бывают серыми/битыми
                box["frames"].append(f)
            k += 1
        cap.release()
        box["frame"] = background(box["frames"] or [last])

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(timeout)
    return box.get("frame")


def grab_many(sources, timeout=20.0):
    """Кадры со всех ссылок параллельно: {ссылка: кадр или None}."""
    out = {}
    threads = []
    for s in sources:
        th = threading.Thread(target=lambda s=s: out.__setitem__(s, grab_frame(s, timeout)), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join(timeout + 5)
    return {s: out.get(s) for s in sources}


# ============================ ОТПЕЧАТОК КАМЕРЫ ============================
_ORB = None


class Fingerprint:
    W = 640

    def __init__(self, frame):
        global _ORB
        if _ORB is None:
            _ORB = cv2.ORB_create(nfeatures=1500)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        k = self.W / g.shape[1]
        g = cv2.resize(g, (self.W, int(g.shape[0] * k)), interpolation=cv2.INTER_AREA)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)   # выравниваем свет
        self.kp, self.des = _ORB.detectAndCompute(g, None)
        self.small = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        e = cv2.resize(g, (320, int(320 * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA)
        self.edges = cv2.Canny(cv2.GaussianBlur(e, (3, 3), 0), 40, 120) > 0
        self.near = cv2.distanceTransform((~self.edges).astype(np.uint8), cv2.DIST_L2, 3) <= 3


def similarity(a: Fingerprint, b: Fingerprint):
    """0..1. Берётся лучшее из двух: особые точки (для сцен с деталями) и контуры (для голых стен)."""
    return max(_orb_score(a, b), edge_score(a, b))


def edge_score(a: Fingerprint, b: Fingerprint):
    """Насколько контуры одной картинки лежат на контурах другой — сверх случайного совпадения."""
    if a.edges.shape != b.edges.shape:
        return 0.0
    out = []
    for x, y in ((a, b), (b, a)):
        n = int(x.edges.sum())
        if n < 50:
            return 0.0
        hit = float((x.edges & y.near).sum()) / n
        chance = float(y.near.mean())
        out.append(max(0.0, (hit - chance) / max(1e-6, 1 - chance)))
    return min(out)


def _orb_score(a: Fingerprint, b: Fingerprint):
    if a.des is None or b.des is None or len(a.kp) < 8 or len(b.kp) < 8:
        return 0.0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a.des, b.des, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return len(good) / 100
    src = np.float32([a.kp[x.queryIdx].pt for x in good])
    dst = np.float32([b.kp[x.trainIdx].pt for x in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    inl = int(mask.sum()) if mask is not None else 0
    return min(1.0, inl / max(30, 0.25 * min(len(a.kp), len(b.kp))))


def _ncc(x, y):
    x, y = x - x.mean(), y - y.mean()
    d = float(np.sqrt((x * x).sum() * (y * y).sum())) or 1.0
    return max(0.0, float((x * y).sum()) / d)


# ============================ СОПОСТАВЛЕНИЕ ============================
def match_cameras(frames, refs, known_keys=None, min_sim=0.25, margin=0.1):
    """
    frames: {ссылка: кадр}; refs: {камера: кадр-образец}; known_keys: {камера: url_key прошлой ссылки}.
    Возвращает список {"url", "camera" (или None — новая), "score", "check", "why"}.
    """
    known_keys = known_keys or {}
    urls = [u for u, f in frames.items() if f is not None]
    names = list(refs)
    by_key = {k: n for n, k in known_keys.items() if k}
    result, used = {}, set()
    for u in urls:                                            # 1. тот же адрес без токена
        n = by_key.get(url_key(u))
        if n and n not in used:
            result[u] = {"url": u, "camera": n, "score": 1.0, "check": False, "why": "та же ссылка"}
            used.add(n)
    fps = {u: Fingerprint(frames[u]) for u in urls if u not in result}
    rfp = {n: Fingerprint(refs[n]) for n in names if n not in used}
    S = np.zeros((len(fps), len(rfp)))
    fu, rn = list(fps), list(rfp)
    for i, u in enumerate(fu):
        for j, n in enumerate(rn):
            S[i, j] = similarity(fps[u], rfp[n])
    pairs = []
    if S.size:                                                # 2. лучшее взаимное соответствие
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(-S)
            pairs = list(zip(rows, cols))
        except ImportError:
            taken_r, taken_c = set(), set()
            for i, j in sorted(np.ndindex(S.shape), key=lambda ij: -S[ij]):
                if i not in taken_r and j not in taken_c:
                    pairs.append((i, j))
                    taken_r.add(i)
                    taken_c.add(j)
    for i, j in pairs:
        if S[i, j] < min_sim:
            continue
        row = np.delete(S[i], j)
        second = float(row.max()) if row.size else 0.0
        result[fu[i]] = {"url": fu[i], "camera": rn[j], "score": round(float(S[i, j]), 2),
                         "check": S[i, j] - second < margin,
                         "why": "похожа и на другую камеру — проверьте" if S[i, j] - second < margin else "по виду"}
    out = []
    matched_scores = {r["camera"]: r["score"] for r in result.values()}
    for u in frames:
        if u in result:
            out.append(result[u])
        elif frames[u] is None:
            out.append({"url": u, "camera": None, "score": 0, "check": False, "why": "не открылась"})
        else:
            i = fu.index(u) if u in fu else None
            dup = None
            if i is not None and len(rn):
                j = int(np.argmax(S[i]))
                if S[i, j] >= min_sim and rn[j] in matched_scores:
                    dup = rn[j]
            out.append({"url": u, "camera": None, "score": 0, "check": False,
                        "why": f"повтор камеры {dup}" if dup else "новая камера", "dup": dup})
    return out
