"""Камера узнаётся по виду: ссылки в любом порядке, без имён."""
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import multicam  # noqa: E402
from camid import Fingerprint, extract_urls, match_cameras, similarity, url_key, urls_from_har  # noqa: E402
import turnstile_detector as td  # noqa: E402


def scene(seed, w=960, h=540):
    """Процедурная «комната»: прямоугольники, круги, надписи — много особых точек, у каждой сцены свои."""
    r = np.random.default_rng(seed)
    img = np.full((h, w, 3), r.integers(60, 180, 3), np.uint8)
    for _ in range(60):
        x, y = int(r.integers(0, w)), int(r.integers(0, h))
        c = tuple(int(v) for v in r.integers(0, 255, 3))
        if r.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(r.integers(10, 120)), y + int(r.integers(10, 120))), c, -1)
        else:
            cv2.circle(img, (x, y), int(r.integers(5, 40)), c, 2)
    for i in range(6):
        cv2.putText(img, f"ROOM {seed}-{i}", (int(r.integers(0, w - 200)), int(r.integers(30, h))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


def other_day(img, seed):
    """Та же камера в другой день: другой свет, шум, человек в кадре."""
    r = np.random.default_rng(seed)
    v = np.clip(img.astype(np.float32) * r.uniform(0.6, 1.3) + r.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    x = int(r.integers(0, img.shape[1] - 120))
    cv2.rectangle(v, (x, 150), (x + 110, 520), (40, 40, 60), -1)
    return v


def video(path, img, n=5):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (img.shape[1], img.shape[0]))
    for _ in range(n):
        w.write(img)
    w.release()
    return str(path)


def test_extract_urls_from_messy_text():
    text = ('copy: "https://argus/hls/7f3a/index.m3u8?token=abc", ещё rtsp://10.0.0.5/s1; '
            'и снова https://argus/hls/7f3a/index.m3u8?token=abc')
    assert extract_urls(text) == ["https://argus/hls/7f3a/index.m3u8?token=abc", "rtsp://10.0.0.5/s1"]
    assert url_key("https://Argus/hls/7f3a/index.m3u8?token=abc") == url_key("https://argus/hls/7f3a/index.m3u8?t=2")


def test_urls_from_har(tmp_path):
    har = {"log": {"entries": [{"request": {"url": u}} for u in (
        "https://argus/app.js", "https://argus/hls/a/index.m3u8?t=1", "https://argus/hls/a/seg001.ts",
        "https://argus/hls/b/index.m3u8?t=2", "https://argus/hls/a/index.m3u8?t=1")]}}
    (tmp_path / "x.har").write_text(json.dumps(har), encoding="utf-8")
    assert urls_from_har(tmp_path / "x.har") == ["https://argus/hls/a/index.m3u8?t=1",
                                                 "https://argus/hls/b/index.m3u8?t=2"]


def test_parse_links_formats(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    text = f"# комментарий\nCAM-07 rtsp://a/1\nhttps://b/2.m3u8\n{f}\n\nмусор без ссылки\n"
    assert multicam.parse_links(text) == [("CAM-07", "rtsp://a/1"), (None, "https://b/2.m3u8"), (None, str(f))]


def test_same_camera_other_day_vs_other_rooms():
    rooms = [scene(i) for i in range(5)]
    for i, img in enumerate(rooms):
        fp = Fingerprint(other_day(img, 100 + i))
        scores = [similarity(fp, Fingerprint(r)) for r in rooms]
        assert int(np.argmax(scores)) == i and scores[i] > 0.5
        assert max(s for j, s in enumerate(scores) if j != i) < 0.2


def test_match_shuffled_links():
    rooms = {f"CAM-{i:02d}": scene(i) for i in range(1, 6)}
    order = ["CAM-03", "CAM-05", "CAM-01", "CAM-04", "CAM-02"]
    frames = {f"https://argus/random{k}/index.m3u8": other_day(rooms[n], k) for k, n in enumerate(order)}
    frames["https://argus/new/index.m3u8"] = scene(99)                       # новой камеры нет в разметке
    frames["https://argus/dead/index.m3u8"] = None                           # не открылась
    res = match_cameras(frames, rooms)
    assert [r["camera"] for r in res[:5]] == order
    assert res[5]["camera"] is None and res[5]["why"] == "новая камера"
    assert res[6]["why"] == "не открылась"


def test_resolve_links_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cams = tmp_path / "cameras"
    for i in (1, 2):                                    # вчера размечены две камеры
        multicam.save_camera({"camera": f"CAM-0{i}", "source": "old", "gates": [td.SELFTEST_CFG]}, cams, scene(i))
    links = [video(tmp_path / "a.avi", other_day(scene(2), 7)),      # сегодня: в другом порядке
             video(tmp_path / "b.avi", scene(42)),                  # + новая камера
             video(tmp_path / "c.avi", other_day(scene(1), 8))]
    today, new, report = multicam.resolve_links([(None, u) for u in links], cams)
    assert {c["camera"]: c["source"] for c in today} == {"CAM-02": links[0], "CAM-01": links[2]}
    assert new == {"CAM-03": links[1]}
    saved = json.loads((cams / "CAM-02.json").read_text(encoding="utf-8"))
    assert saved["source"] == links[0]


def test_wizard_pasted_links(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cams = tmp_path / "cameras"
    multicam.save_camera({"camera": "CAM-01", "source": "old", "gates": [td.SELFTEST_CFG]}, cams, scene(1))
    a = video(tmp_path / "a.avi", scene(5))                     # новая
    b = video(tmp_path / "b.avi", other_day(scene(1), 3))       # это CAM-01
    answers = iter([f"смотри: {a}", b, "", "y", "", ""])        # вставка, пустая строка, ru, старт, «всё само»
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    monkeypatch.setattr("autosetup.propose_static", lambda frame: ([], ["не найдено"]))
    calls = {}
    monkeypatch.setattr(multicam, "run_many", lambda cfgs, gp, show: calls.update(
        run=sorted((c["camera"], c["source"], bool(c.get("auto_learn"))) for c in cfgs), ru=gp["ru_text"]))
    multicam.wizard()
    assert calls == {"ru": True, "run": [("CAM-01", b, False), ("CAM-02", a, True)]}
    assert (cams / "CAM-02_ref.jpg").exists()                  # образец вида новой камеры сохранён
    assert a in (tmp_path / "links.txt").read_text(encoding="utf-8")


def test_wizard_without_links_creates_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    monkeypatch.setattr(multicam, "_open_file", lambda p: None)
    with pytest.raises(SystemExit):
        multicam.wizard()
    assert "В ЛЮБОМ ПОРЯДКЕ" in (tmp_path / "links.txt").read_text(encoding="utf-8")
