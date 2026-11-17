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
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

from turnstile_detector import (GateLogic, LatestFrame, Output, Params, WINDOW, build_persons, draw_overlay,
                                green_share, is_file_source, need_yolo, setup_mode)

CAM_DIR = Path("cameras")
LINKS = Path("links.txt")


# ============================ КОНФИГИ И ССЫЛКИ ============================
def read_links(path=LINKS):
    links = {}
    if not Path(path).exists():
        return links
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            links[parts[0]] = parts[1].strip().strip('"')
    return links


def write_links_template(path=LINKS):
    Path(path).write_text(
        "# Одна строка — одна камера: НАЗВАНИЕ  ССЫЛКА\n"
        "# Ссылка — index.m3u8 из вкладки Network в «Аргусе», RTSP камеры или путь к видеофайлу.\n"
        "# Ссылки из «Аргуса» живут около суток: каждый день обновляй только этот файл.\n"
        "CAM-01 https://example/CAM-01/index.m3u8\n"
        "CAM-02 rtsp://user:pass@10.0.0.12/stream1\n", encoding="utf-8")


def global_params(cam_dir=CAM_DIR):
    p = Path(cam_dir) / "params.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def make_params(*dicts):
    merged = {}
    for d in dicts:
        merged.update(d or {})
    return Params(**{k: v for k, v in merged.items() if k in Params.__dataclass_fields__})


def load_cameras(cam_dir=CAM_DIR, links_path=LINKS):
    """Конфиги всех размеченных камер; ссылки из links.txt заменяют старые."""
    links = read_links(links_path)
    cams = []
    for f in sorted(Path(cam_dir).glob("*.json")):
        if f.name == "params.json":
            continue
        cfg = json.loads(f.read_text(encoding="utf-8"))
        name = cfg.get("camera", f.stem)
        if name in links:
            cfg["source"] = links[name]
        if cfg.get("gates"):
            cams.append(cfg)
    return cams


def save_camera(cfg, cam_dir=CAM_DIR):
    Path(cam_dir).mkdir(exist_ok=True)
    path = Path(cam_dir) / f"{cfg['camera']}.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


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
class Camera:
    def __init__(self, cfg, params):
        from tracker import SimpleTracker
        self.cfg, self.p = cfg, params
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
def run_many(cfgs, base_params=None, show=True, max_seconds=None, model=None):
    import cv2
    from hud import Painter, beep, draw_hud, mosaic, tile_at
    base = base_params or {}
    cams = [Camera(c, make_params(base, c.get("params"))) for c in cfgs]
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
            if zoom is None:
                tiles = [c.view if now - c.last_frame_wall < 10 or c.file else None for c in cams]
                canvas = mosaic(painter, tiles, names, alert_until, now)
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
                    "frame": None, "last": 0.0}
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
    return {n: (s["frame"], s["finder"].find() if s["finder"] else [], s["tracks"].moving())
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
            saved.append(save_camera(cfg, cam_dir))
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


def wizard(observe_sec=180, cam_dir=CAM_DIR, links_path=LINKS):
    """Запуск без параметров: ссылки -> авторазметка новых камер -> мозаика всех камер."""
    print("=== Детектор нарушений прохода (все камеры) ===\n")
    links_path, cam_dir = Path(links_path), Path(cam_dir)
    if not links_path.exists():
        print("Файла links.txt со ссылками на камеры ещё нет.")
        link = _ask("Вставь ссылку на одну камеру (или Enter — создать links.txt для многих камер): ")
        if link:
            links_path.write_text(f"CAM-01 {link}\n", encoding="utf-8")
        else:
            write_links_template(links_path)
            _open_file(links_path)
            print(f"\nСоздан {links_path.resolve()}. Впиши туда камеры (по строке: НАЗВАНИЕ ССЫЛКА), "
                  "сохрани и запусти программу снова.")
            return
    links = read_links(links_path)
    if not links:
        _open_file(links_path)
        sys.exit(f"В {links_path} нет ни одной камеры. Формат строки: CAM-01 https://.../index.m3u8")

    gp_path = cam_dir / "params.json"
    cam_dir.mkdir(exist_ok=True)
    if not gp_path.exists():
        ru = _ask("Подписи на экране по-русски? (y — да, Enter — латиница): ").lower() in ("y", "д", "да", "yes")
        gp_path.write_text(json.dumps({"ru_text": ru, "cam_fps": 8}, ensure_ascii=False, indent=2), encoding="utf-8")
    params = make_params(global_params(cam_dir))

    done = {c.get("camera") for c in load_cameras(cam_dir, links_path)}
    new = {n: s for n, s in links.items() if n not in done}
    print(f"Камер в links.txt: {len(links)}, размечено: {len(done & set(links))}, новых: {len(new)}")
    choice = _ask("Enter — старт, a — переразметить ВСЕ камеры автоматически: ").lower()
    if choice == "a":
        new = dict(links)
    if new:
        how = _ask(f"Разметка {len(new)} камер: Enter — автоматически (~{observe_sec // 60} мин), "
                   "m — вручную кликами: ").lower()
        if how == "m":
            for n, s in new.items():
                print(f"\n--- {n} ---")
                setup_mode(str(cam_dir / f"{n}.json"), s, camera=n)
        else:
            mins = _ask(f"Сколько минут наблюдать? Нужно хотя бы 3 прохода через каждый проём "
                        f"(Enter — {observe_sec // 60}): ")
            if mins.replace(",", ".").replace(".", "", 1).isdigit():
                observe_sec = int(float(mins.replace(",", ".")) * 60)
            autosetup(new, observe_sec, params, review=True, cam_dir=cam_dir)
    cfgs = [c for c in load_cameras(cam_dir, links_path) if c.get("camera") in links]
    if not cfgs:
        sys.exit("Нет ни одной размеченной камеры")
    print("\nОткрываю окно. Клик по камере — крупно, Esc — назад к мозаике, f — весь экран, q — выход.")
    run_many(cfgs, global_params(cam_dir), show=True)
