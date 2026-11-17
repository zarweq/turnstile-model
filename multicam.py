"""
multicam.py — одна программа на все камеры объекта (рассчитано на 20 и больше).

Как устроено:
  * у каждой камеры свой поток чтения (берётся только свежий кадр, задержка не копится);
  * кадры всех камер, подошедшие по времени (cam_fps на камеру), идут в нейросеть ОДНОЙ пачкой —
    на видеокарте это в разы быстрее, чем гонять камеры по очереди;
  * у каждой камеры свой трекер людей и своя логика проёмов (та же GateLogic, что и для одной камеры);
  * на экране мозаика всех камер; камера с тревогой загорается красной рамкой,
    клик по камере (или цифра) — открыть её крупно, Esc — назад к мозаике;
  * журналы: out/<камера>/... как раньше и общий out/violations_all.csv по всем камерам.

Файлы:
  cameras/<камера>.json — разметка камеры (создаётся авторазметкой или --setup);
  cameras/params.json   — общие параметры для всех камер (необязательно);
  links.txt             — ссылки на потоки, по строке на камеру: CAM-01 https://.../index.m3u8
                          (ссылки из «Аргуса» живут около суток — обновляется только этот файл).
"""

import csv
import json
import re
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np

from turnstile_detector import (GateLogic, LatestFrame, Output, Params, WINDOW, build_persons, draw_overlay,
                                green_share, is_file_source, need_yolo, setup_mode)

CAM_DIR = Path("cameras")
LINKS = Path("links.txt")


# ============================ КОНФИГИ И ССЫЛКИ ============================
NAME_RE = re.compile(r"^[\w.\-]{1,32}$")          # CAM-07, Этаж3_лево; слова с «:» и т.п. — не имя


def _is_name(s):
    return bool(NAME_RE.match(s)) and any(ch.isdigit() for ch in s)


def parse_links(text):
    """Строки со ссылками в любом виде: «ссылка», «ИМЯ ссылка», путь к видеофайлу, мусор вокруг.
    Возвращает [(подсказка-имя или None, ссылка)]. Имя нужно только старым камерам без образца вида."""
    from camid import extract_urls
    out, seen = [], set()
    for line in (text or "").splitlines():
        line = line.strip().strip('"')
        if not line or line.startswith("#"):
            continue
        urls = extract_urls(line)
        if not urls:                                    # видеофайлы: любое слово строки, которое есть на диске
            whole = line.split(None, 1)[-1].strip().strip('"')
            urls = [whole] if Path(whole).is_file() else \
                [t.strip('"') for t in line.split() if Path(t.strip('"')).is_file()]
        first = line.split(None, 1)[0]
        hint = first if len(line.split(None, 1)) == 2 and _is_name(first) and not Path(first).is_file() \
            else None
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append((hint if len(urls) == 1 else None, u))
    return out


def collect_links(links_path=LINKS, har_dir="."):
    """Ссылки из links.txt и из всех HAR-файлов в папке программы."""
    from camid import urls_from_har
    pairs = parse_links(Path(links_path).read_text(encoding="utf-8-sig")) if Path(links_path).exists() else []
    seen = {u for _, u in pairs}
    for har in sorted(Path(har_dir).glob("*.har")):
        for u in urls_from_har(har):
            if u not in seen:
                seen.add(u)
                pairs.append((None, u))
    return pairs


def read_links(path=LINKS):
    """Старый формат «ИМЯ ссылка»: {имя: ссылка}. Строки без имени пропускаются."""
    return {h: u for h, u in parse_links(Path(path).read_text(encoding="utf-8-sig")) if h} \
        if Path(path).exists() else {}


def write_links_template(path=LINKS):
    Path(path).write_text(
        "# Вставьте сюда ссылки на камеры — по одной на строку, В ЛЮБОМ ПОРЯДКЕ, имена не нужны.\n"
        "# Программа сама узнаёт камеру по картинке и подставляет её разметку.\n"
        "# Ссылка — index.m3u8 из вкладки Network в «Аргусе», RTSP камеры или путь к видеофайлу.\n"
        "# Можно не копировать по одной: в Network нажмите правой кнопкой -> «Save all as HAR»\n"
        "# и положите .har-файл в папку программы — ссылки возьмутся из него.\n", encoding="utf-8")


def global_params(cam_dir=CAM_DIR):
    p = Path(cam_dir) / "params.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def make_params(*dicts):
    merged = {}
    for d in dicts:
        merged.update(d or {})
    return Params(**{k: v for k, v in merged.items() if k in Params.__dataclass_fields__})


def all_cameras(cam_dir=CAM_DIR):
    """{камера: конфиг} всех размеченных камер."""
    out = {}
    for f in sorted(Path(cam_dir).glob("*.json")):
        if f.name == "params.json":
            continue
        cfg = json.loads(f.read_text(encoding="utf-8"))
        out[cfg.get("camera", f.stem)] = cfg
    return out


def load_cameras(cam_dir=CAM_DIR, links_path=LINKS):
    """Конфиги размеченных камер (с проёмами). Старый формат links.txt «ИМЯ ссылка» заменяет ссылку."""
    links = read_links(links_path)
    cams = []
    for name, cfg in all_cameras(cam_dir).items():
        if name in links:
            cfg["source"] = links[name]
        if cfg.get("gates") or needs_learning(cfg):
            cams.append(cfg)
    return cams


def ref_image(name, cam_dir=CAM_DIR):
    """Образец вида камеры: снимок при разметке (или картинка авторазметки у старых камер)."""
    import cv2
    for suffix in ("_ref.jpg", "_auto.jpg"):
        p = Path(cam_dir) / f"{name}{suffix}"
        if p.exists():
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
    return None


def save_camera(cfg, cam_dir=CAM_DIR, frame=None):
    import cv2
    Path(cam_dir).mkdir(exist_ok=True)
    path = Path(cam_dir) / f"{cfg['camera']}.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    if frame is not None:                                # образец вида — по нему камера узнаётся завтра
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            (Path(cam_dir) / f"{cfg['camera']}_ref.jpg").write_bytes(buf.tobytes())
    return path


def next_name(taken):
    i = 1
    while f"CAM-{i:02d}" in taken:
        i += 1
    return f"CAM-{i:02d}"


def resolve_links(pairs, cam_dir=CAM_DIR, show=False, timeout=20.0):
    """
    Какая ссылка какой камере. pairs: [(подсказка-имя или None, ссылка)].
    Возвращает (конфиги размеченных камер с сегодняшними ссылками, {новое имя: ссылка}, отчёт).
    """
    from camid import grab_many, match_cameras, url_key
    cams = all_cameras(cam_dir)
    urls = [u for _, u in pairs]
    print(f"Проверяю {len(urls)} ссылок (беру по кадру с каждой)...")
    frames = grab_many(urls, timeout)
    refs = {n: img for n in cams if (img := ref_image(n, cam_dir)) is not None}
    keys = {n: c.get("source_key") for n, c in cams.items()}
    report = match_cameras(frames, refs, keys)
    hints = {u: h for h, u in pairs if h}
    taken = {r["camera"] for r in report if r["camera"]}
    for r in report:                                     # старые камеры без образца — по имени из links.txt
        h = hints.get(r["url"])
        if r["camera"] is None and r["why"] == "новая камера" and h in cams and h not in refs and h not in taken:
            r.update(camera=h, score=0, check=True, why="по имени (нет образца вида)")
            taken.add(h)
    today, new = [], {}
    for r in report:
        if r["camera"]:
            cfg = cams[r["camera"]]
            cfg["source"], cfg["source_key"] = r["url"], url_key(r["url"])
            save_camera(cfg, cam_dir, frames[r["url"]] if r["camera"] not in refs else None)
            if cfg.get("gates") or needs_learning(cfg):
                today.append(cfg)
        elif r["why"] == "новая камера":
            name = hints.get(r["url"]) if hints.get(r["url"]) not in cams else None
            name = name if name and name not in new else next_name(set(cams) | set(new))
            new[name] = r["url"]
            r["camera_new"] = name
    _print_report(report, cams)
    if show:
        _show_report(report, frames, refs, cam_dir)
    return today, new, report


def _print_report(report, cams):
    print()
    for i, r in enumerate(report, 1):
        short = r["url"] if len(r["url"]) < 60 else r["url"][:40] + "..." + r["url"][-15:]
        if r["camera"]:
            mark = "  ПРОВЕРЬТЕ" if r["check"] else ""
            print(f"  {i:2}. {short} -> {r['camera']} ({r['why']}, сходство {r['score']}){mark}")
        elif r.get("camera_new"):
            print(f"  {i:2}. {short} -> новая камера, будет {r['camera_new']}")
        else:
            print(f"  {i:2}. {short} -> {r['why']}")
    missing = sorted(set(cams) - {r["camera"] for r in report if r["camera"]})
    if missing:
        print("  Сегодня нет ссылки для:", ", ".join(missing))


def _show_report(report, frames, refs, cam_dir):
    """Картинка «образец | сейчас» по каждой узнанной камере — глазами проверить, что всё сошлось."""
    import cv2
    from hud import Painter
    p = Painter(ru=True)
    rows = []
    for r in report:
        if not r["camera"] or frames.get(r["url"]) is None:
            continue
        ref = refs.get(r["camera"])
        cur = cv2.resize(frames[r["url"]], (320, 180))
        left = cv2.resize(ref, (320, 180)) if ref is not None else np.zeros_like(cur)
        row = np.hstack([left, cur])
        color = (0, 165, 255) if r["check"] else (0, 200, 0)
        cv2.rectangle(row, (0, 0), (639, 179), color, 3)
        p.text(row, f"{r['camera']}: образец | сейчас  {'ПРОВЕРЬТЕ' if r['check'] else 'ок'}", (8, 22), 0.5,
               (255, 255, 255), bg=(0, 0, 0))
        rows.append(row)
    if not rows:
        return
    cols = 2 if len(rows) > 4 else 1
    while len(rows) % cols:
        rows.append(np.zeros_like(rows[0]))
    sheet = np.vstack([np.hstack(rows[i:i + cols]) for i in range(0, len(rows), cols)])
    cv2.imwrite(str(Path(cam_dir) / "_links_check.jpg"), sheet)
    win = "Links check"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, sheet)
    print("\nПроверьте картинку: слева — как камера выглядела при разметке, справа — что пришло по ссылке.\n"
          "Нажмите любую клавишу в окне, чтобы продолжить.")
    cv2.waitKey(0)
    cv2.destroyWindow(win)


# ============================ ИСТОЧНИКИ КАДРОВ ============================
class FileReader:
    """Видеофайл: кадры по порядку, время — из самого видео, лишние кадры пропускаются."""

    def __init__(self, path, fps_limit):
        import cv2
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.step = max(1, round(self.fps / fps_limit)) if fps_limit > 0 else 1
        self.i, self.done = 0, False

    def next(self):
        for _ in range(self.step - 1):
            self.cap.grab()
            self.i += 1
        ok, frame = self.cap.read()
        if not ok:
            self.done = True
            return None, None
        t = self.i / self.fps
        self.i += 1
        return frame, t

    def close(self):
        self.cap.release()


class StreamReader:
    def __init__(self, source):
        self.reader, self.last_n, self.done = LatestFrame(source), 0, False

    def next(self):
        frame, self.last_n = self.reader.get(self.last_n)
        return (frame, time.time()) if frame is not None else (None, None)

    def close(self):
        self.reader.stop = True


def open_source(source, fps_limit):
    if is_file_source(source):
        return FileReader(source, fps_limit)
    if not str(source).lower().startswith(("rtsp://", "http://", "https://")):
        return None
    return StreamReader(source)


# ============================ ОДНА КАМЕРА В ОБЩЕМ ЦИКЛЕ ============================
def needs_learning(cfg):
    return bool(cfg.get("auto_learn")) or not cfg.get("gates")


class Camera:
    def __init__(self, cfg, params, cam_dir=CAM_DIR):
        from tracker import SimpleTracker
        self.cfg, self.p, self.cam_dir = cfg, params, cam_dir
        self.learner = None
        self.learn = needs_learning(cfg)
        self.name = cfg.get("camera", "camera")
        self.anchor = cfg.get("anchor", "foot")
        self.gates = [GateLogic(g, params) for g in cfg["gates"]]
        self.tracker = SimpleTracker()
        self.reader = open_source(cfg.get("source", ""), params.cam_fps)
        self.out = Output(self.name)
        self.file = isinstance(self.reader, FileReader)
        self.last_proc = 0.0
        self.last_frame_wall = time.time()
        self.view = None
        self.stats = Counter()

    def due(self, now):
        if self.reader is None or self.reader.done:
            return False
        return self.file or now - self.last_proc >= 1.0 / max(self.p.cam_fps, 0.1)

    def stamp(self, t):
        return f"video+{t:.1f}s" if self.file else datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")

    def status(self, ru=True):
        if not self.learn:
            return ""
        n = self.learner.progress if self.learner else 0
        need = self.learner.min_passes if self.learner else 3
        if self.cfg.get("gates") and not self.cfg.get("learned"):
            return f"по картинке, уточняю {n}/{need}" if ru else f"guess, learning {n}/{need}"
        return f"обучение {n}/{need}" if ru else f"learning {n}/{need}"

    def learn_step(self, frame, persons, t):
        """Обучение на ходу: копим индикаторы и траектории, при успехе — новая разметка сразу в работу."""
        from autosetup import Learner, draw_markup
        if not self.learn:
            return False
        if self.learner is None:
            self.learner = Learner(frame.shape, self.p, self.anchor)
        self.learner.add(frame, persons)
        gates = self.learner.try_build(t, self.name, self.cfg.get("source", ""))
        if not gates or (self.cfg.get("learned") and len(gates) <= len(self.cfg.get("gates", []))):
            if self.learner.done and self.cfg.get("learned"):
                self.learn = False
            return False                                   # нового ничего — текущую разметку не трогаем
        self.cfg["gates"] = gates
        self.cfg["learned"] = True
        self.cfg["auto_learn"] = not self.learner.done
        self.gates = [GateLogic(g, self.p) for g in gates]
        save_camera(self.cfg, self.cam_dir)
        try:
            import cv2
            cv2.imwrite(str(Path(self.cam_dir) / f"{self.name}_auto.jpg"), draw_markup(frame, self.cfg))
        except Exception:
            pass
        if self.learner.done:
            self.learn = False
        print(f"[+] {self.name}: разметка выучена по проходам людей — проёмов {len(gates)}"
              f"{'' if self.learner.done else ' (учусь дальше: есть индикаторы без проёма)'}")
        return True


def persons_from(res, cam, t):
    persons = build_persons(res, cam.anchor, cam.p.kp_conf)
    ids = cam.tracker.update([x.box for x in persons], t)
    for x, tid in zip(persons, ids):
        x.tid = tid
    return persons


class AllOutput:
    """Общий журнал нарушений всех камер."""

    def __init__(self, path=Path("out") / "violations_all.csv"):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, stamp, camera, e):
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            if new:
                w.writerow(["время", "камера", "проём", "инцидент", "тип", "описание", "вердикт оператора"])
            w.writerow([stamp, camera, e["gate"], e["incident"], e["kind"], e["text"], ""])


# ============================ ГЛАВНЫЙ ЦИКЛ ============================
def run_many(cfgs, base_params=None, show=True, max_seconds=None, model=None, cam_dir=CAM_DIR):
    import cv2
    from hud import Painter, beep, draw_hud, mosaic, tile_at
    base = base_params or {}
    cams = [Camera(c, make_params(base, c.get("params")), cam_dir) for c in cfgs]
    bad = [c.name for c in cams if c.reader is None]
    if bad:
        print("Нет ссылки или файла для камер:", ", ".join(bad), "— проверь links.txt")
    cams = [c for c in cams if c.reader is not None]
    if not cams:
        sys.exit("Нет ни одной камеры с рабочей ссылкой")
    p0 = cams[0].p
    model = model or need_yolo()(p0.model)
    painter = Painter(p0.ru_text)
    allout = AllOutput()
    alerts, totals, alert_until = deque(maxlen=100), Counter(), {}
    zoom, fullscreen, start = None, False, time.time()
    click = {}

    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 720)
        cv2.setMouseCallback(WINDOW, lambda ev, x, y, *_: click.update(pt=(x, y)) if ev == cv2.EVENT_LBUTTONDOWN
                             else None)
    print(f"Запущено камер: {len(cams)}. Выход — q в окне или Ctrl+C.")
    last_draw = 0.0
    try:
        while True:
            now = time.time()
            if max_seconds and now - start > max_seconds:
                break
            batch = []
            for cam in cams:
                if cam.due(now):
                    frame, t = cam.reader.next()
                    if frame is not None:
                        cam.last_proc, cam.last_frame_wall = now, now
                        batch.append((cam, frame, t))
            if all(c.reader.done for c in cams):
                break

            if batch:
                results = model.predict([b[1] for b in batch], classes=[0], conf=p0.conf, imgsz=p0.imgsz,
                                        device=p0.device or None, verbose=False)
                for (cam, frame, t), res in zip(batch, results):
                    persons = persons_from(res, cam, t)
                    cam.learn_step(frame, persons, t)
                    events, greens = [], []
                    for g in cam.gates:
                        is_green = green_share(frame, g.roi, cam.p) > cam.p.green_frac
                        greens.append(is_green)
                        events += g.update(t, persons, is_green)
                    view = frame.copy()
                    draw_overlay(cv2, view, persons, cam.gates, greens, painter)
                    cam.view = view
                    stamp = cam.stamp(t)
                    for e in events:
                        if e["type"] == "pass":
                            cam.out.passed(stamp, e)
                            continue
                        if cam.cfg.get("gates") and not cam.cfg.get("learned") and cam.learn:
                            e["text"] += " [разметка по картинке, ещё не уточнена]"
                        shot = view.copy()
                        painter.text(shot, f"{cam.name} {e['gate']}: {e['kind']}", (10, 30), 0.9, (0, 0, 255))
                        cam.out.violation(stamp, e, shot)
                        allout.add(stamp, cam.name, e)
                        cam.stats[e["kind"]] += 1
                        totals[e["kind"]] += 1
                        alerts.append((time.time(), stamp[-8:], cam.name, e["gate"], e["kind"]))
                        alert_until[cam.name] = time.time() + 6
                        if cam.p.sound and show:
                            beep()
            else:
                time.sleep(0.005)

            if not show or now - last_draw < 0.08:                     # экран — не чаще ~12 раз/с
                continue
            last_draw = now
            names = [c.name for c in cams]
            labels = [f"{c.name}  {c.status(painter.lang == 'ru')}".strip() for c in cams]
            if zoom is None:
                tiles = [c.view if now - c.last_frame_wall < 10 or c.file else None for c in cams]
                canvas = mosaic(painter, tiles, names, alert_until, now, labels=labels)
            else:
                c = cams[zoom]
                canvas = c.view.copy() if c.view is not None else mosaic(painter, [None], [c.name], {}, now)
            draw_hud(painter, canvas, alerts, totals, now, camera=None if zoom is None else cams[zoom].name,
                     side_list=zoom is not None)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if "pt" in click:
                pt = click.pop("pt")
                zoom = tile_at(len(cams), (canvas.shape[1], canvas.shape[0]), pt) if zoom is None else None
            if ord("1") <= key <= ord("9") and key - ord("1") < len(cams):
                zoom = key - ord("1")
            if key in (27, 8, ord("0")):                                # Esc / Backspace / 0 — к мозаике
                zoom = None
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE) < 0:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for c in cams:
            c.reader.close()
        if show:
            cv2.destroyAllWindows()
    for c in cams:
        if c.stats:
            print(f"  {c.name}: " + ", ".join(f"{k} — {v}" for k, v in c.stats.most_common()))
    print("Всего тревог:", sum(totals.values()), "| общий журнал:", allout.path)
    return totals


# ============================ АВТОРАЗМЕТКА ВСЕХ КАМЕР ============================
def observe(sources, seconds, params, model, show=False):
    """Смотрит все камеры одновременно. Возвращает {камера: (кадр, индикаторы, траектории)}."""
    import cv2
    from autosetup import IndicatorFinder, TrackCollector
    from tracker import SimpleTracker
    st = {}
    for name, src in sources.items():
        r = open_source(src, params.cam_fps)
        if r is None:
            print(f"{name}: нет ссылки или файла — пропускаю")
            continue
        st[name] = {"reader": r, "finder": None, "tracks": TrackCollector(), "tracker": SimpleTracker(),
                    "frame": None, "last": 0.0, "bg": [], "n": 0}
    start = last_print = time.time()
    while st and time.time() - start < seconds:
        batch = []
        now = time.time()
        for name, s in st.items():
            if s["reader"].done:
                continue
            if not isinstance(s["reader"], FileReader) and now - s["last"] < 1.0 / params.cam_fps:
                continue
            frame, t = s["reader"].next()
            if frame is None:
                continue
            s["last"] = now
            if s["finder"] is None:
                s["finder"] = IndicatorFinder(frame.shape, params)
            s["finder"].add(frame)
            s["frame"] = frame
            s["n"] += 1
            if s["n"] % 15 == 1 and len(s["bg"]) < 15:      # кадры для образца вида без людей
                s["bg"].append(frame)
            batch.append((name, frame, t))
        if all(s["reader"].done for s in st.values()):
            break
        if not batch:
            time.sleep(0.01)
            continue
        results = model.predict([b[1] for b in batch], classes=[0], conf=params.conf, imgsz=params.imgsz,
                                device=params.device or None, verbose=False)
        for (name, frame, t), res in zip(batch, results):
            boxes = res.boxes.xyxy.tolist() if res.boxes is not None and len(res.boxes) else []
            s = st[name]
            s["tracks"].add(boxes, s["tracker"].update(boxes, t))
        if time.time() - last_print > 10:
            last_print = time.time()
            left = int(seconds - (last_print - start))
            print(f"  наблюдаю... осталось ~{left} с, людей с траекториями: " +
                  ", ".join(f"{n}: {len(s['tracks'].moving())}" for n, s in st.items()))
        if show and batch:
            cv2.imshow("autosetup", batch[-1][1])
            cv2.waitKey(1)
    for s in st.values():
        s["reader"].close()
    from camid import background
    return {n: (background(s["bg"] + [s["frame"]]), s["finder"].find() if s["finder"] else [], s["tracks"].moving())
            for n, s in st.items() if s["frame"] is not None}


def autosetup(sources, seconds=180, params=None, model=None, review=True, cam_dir=CAM_DIR, anchor="foot"):
    """Авторазметка списка камер {имя: ссылка}. Сохраняет cameras/<имя>.json и картинку с разметкой."""
    import cv2
    from autosetup import build_config, draw_markup
    params = params or Params()
    model = model or need_yolo()(params.model)
    print(f"Авторазметка {len(sources)} камер: смотрю видео {seconds} с. "
          "Чем больше людей пройдёт и приложит пропуск за это время, тем точнее.")
    seen = observe(sources, seconds, params, model)
    saved = []
    Path(cam_dir).mkdir(exist_ok=True)
    for name, (frame, inds, tracks) in seen.items():
        cfg, notes = build_config(name, sources[name], frame.shape, inds, tracks, anchor)
        img = draw_markup(frame, cfg, notes)
        cv2.imwrite(str(Path(cam_dir) / f"{name}_auto.jpg"), img)
        small = sum(g["kind"] == "turnstile" for g in cfg["gates"])
        print(f"\n{name}: проёмов {len(cfg['gates'])} (турникетов {small}, ворот {len(cfg['gates']) - small})")
        for n in notes:
            print("   -", n)
        choice = "y" if cfg["gates"] else "m"
        if review:
            cv2.imshow("autosetup", img)
            print("   Enter — принять, m — разметить вручную, s — пропустить камеру")
            k = cv2.waitKey(0) & 0xFF
            choice = {13: "y", 10: "y", ord("m"): "m", ord("s"): "s"}.get(k, "y")
        if choice == "y" and cfg["gates"]:
            saved.append(save_camera(cfg, cam_dir, frame))
        elif choice == "m" and review:
            cv2.destroyWindow("autosetup")
            if setup_mode(str(Path(cam_dir) / f"{name}.json"), sources[name], camera=name):
                saved.append(Path(cam_dir) / f"{name}.json")
    if review:
        cv2.destroyAllWindows()
    missing = [n for n in sources if n not in seen]
    if missing:
        print("Не удалось получить видео:", ", ".join(missing))
    return saved


# ============================ МАСТЕР ЗАПУСКА ============================
def _ask(prompt, default=""):
    try:
        v = input(prompt).strip().strip('"')
    except EOFError:
        v = ""
    return v or default


def _open_file(path):
    import os
    import subprocess
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))                                  # noqa: S606 — открыть в Блокноте
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
    except Exception:
        pass


def _read_pasted():
    """Многострочная вставка в консоль: до пустой строки."""
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


def wizard(observe_sec=180, cam_dir=CAM_DIR, links_path=LINKS):
    """Запуск без параметров: ссылки (в любом порядке) -> узнать камеры -> разметить новые -> мозаика."""
    from datetime import date
    print("=== Детектор нарушений прохода (все камеры) ===\n")
    links_path, cam_dir = Path(links_path), Path(cam_dir)
    cam_dir.mkdir(exist_ok=True)
    have = collect_links(links_path)
    print("Вставьте ссылки на камеры — можно все сразу, в любом порядке, хоть с лишним текстом.\n"
          "Закончите пустой строкой (нажмите Enter ещё раз).")
    if have:
        print(f"Или сразу Enter — взять сохранённые ({len(have)} шт. из links.txt / HAR-файлов).")
    pasted = _read_pasted()
    if pasted.strip():
        pairs = parse_links(pasted)
        if not pairs:
            sys.exit("Во вставленном тексте не нашлось ни одной ссылки")
        links_path.write_text(f"# ссылки на {date.today():%d.%m.%Y}\n" + "\n".join(u for _, u in pairs) + "\n",
                              encoding="utf-8")
    else:
        pairs = have
    if not pairs:
        write_links_template(links_path)
        _open_file(links_path)
        sys.exit(f"Ссылок нет. Вставьте их в окно при запуске или в {links_path.resolve()}")

    gp_path = cam_dir / "params.json"
    if not gp_path.exists():
        ru = _ask("Подписи на экране по-русски? (y — да, Enter — латиница): ").lower() in ("y", "д", "да", "yes")
        gp_path.write_text(json.dumps({"ru_text": ru, "cam_fps": 8}, ensure_ascii=False, indent=2), encoding="utf-8")

    today, new, _ = resolve_links(pairs, cam_dir, show=_gui_available())
    print(f"\nУзнано размеченных камер: {len(today)}, новых: {len(new)}")
    choice = _ask("Enter — старт, a — разметить ВСЕ сегодняшние камеры заново: ").lower()
    if choice == "a":
        new.update({c["camera"]: c["source"] for c in today})
    if new:
        print(f"\nНовых камер: {len(new)}. Как размечать?\n"
              "  Enter — всё само: точки ставятся по картинке сразу, а потом программа сама уточняет их\n"
              "          по проходящим людям прямо во время работы (кликать ничего не нужно)\n"
              "  e     — то же, но показать предложение по картинке для проверки мышкой\n"
              "  m     — вручную с нуля (3 клика на проём)")
        how = _ask("Выбор: ").lower()
        if how in ("e", "m"):
            for n, src in new.items():
                print(f"\n--- {n} ---")
                setup_mode(str(cam_dir / f"{n}.json"), src, camera=n, auto=how == "e", anchor="foot")
        else:
            auto_new_cameras(new, cam_dir)
        marked = all_cameras(cam_dir)
        today = [c for c in today if c["camera"] not in new]
        today += [marked[n] for n in new if n in marked and (marked[n].get("gates") or needs_learning(marked[n]))]
    if not today:
        sys.exit("Нет ни одной размеченной камеры")
    print("\nОткрываю окно. Клик по камере — крупно, Esc — назад к мозаике, f — весь экран, q — выход.")
    run_many(today, global_params(cam_dir), show=True)


def auto_new_cameras(sources, cam_dir=CAM_DIR):
    """Новые камеры без единого клика: предложение по картинке + обучение на ходу."""
    from autosetup import propose_static
    from camid import grab_frame
    for name, src in sources.items():
        frame = grab_frame(src)
        if frame is None:
            print(f"  {name}: ссылка не открылась — пропускаю")
            continue
        try:
            gates, _ = propose_static(frame)
        except Exception as e:                              # нет модели, нет сети — просто учимся на ходу
            print(f"  {name}: поиск по картинке не удался ({e})")
            gates = []
        cfg = {"camera": name, "source": src, "anchor": "foot", "gates": gates, "params": {},
               "auto_learn": True, "learned": False}
        save_camera(cfg, cam_dir, frame)
        what = f"по картинке найдено проёмов: {len(gates)}" if gates else "по картинке турникеты не найдены"
        print(f"  {name}: {what}; дальше разметка уточнится сама по проходящим людям")


def _gui_available():
    """Можно ли открывать окна (в серверной сборке OpenCV без экрана — нельзя)."""
    import cv2
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return "NONE" not in line.upper()
    return True
