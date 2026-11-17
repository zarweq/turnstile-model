"""Сквозной тест: синтетическое видео -> индикатор -> логика -> CSV. Нейросеть заменена заглушкой."""
import csv
import sys
import types

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

import turnstile_detector as td  # noqa: E402

FPS, N = 10, 80
GREEN_FRAMES = range(5, 9)          # зелёный 0.5–0.8 с


def person_x(tid, k):
    """Человек 1 идёт с 1.0 с, человек 2 — с 2.0 с (зелёный был один -> второй нарушитель)."""
    t0 = {1: 1.0, 2: 2.0}[tid]
    t = k / FPS
    return None if t < t0 else 40 + (t - t0) * 40


class FakeYOLO:
    def __init__(self, *_):
        self.k = 0

    def track(self, frame, **_):
        boxes, ids = [], []
        for tid in (1, 2):
            x = person_x(tid, self.k)
            if x is not None:
                boxes.append([x - 15, 60, x + 15, 140])
                ids.append(tid)
        self.k += 1

        res = types.SimpleNamespace(boxes=_Boxes(_T(boxes), _T(ids) if ids else None), keypoints=None)
        return [res]


class _T(list):
    """Имитация тензора ultralytics: .tolist() и .int()."""
    def tolist(self):
        return list(self)

    def int(self):
        return self


class _Boxes:
    def __init__(self, xyxy, ids):
        self.xyxy, self.id = xyxy, ids

    def __len__(self):
        return len(self.xyxy)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (240, 200))
    for k in range(N):
        img = np.full((200, 240, 3), 128, np.uint8)
        color = (0, 200, 0) if k in GREEN_FRAMES else (0, 0, 200)      # BGR: зелёный / красный
        cv2.rectangle(img, (93, 38), (107, 52), color, 3)
        w.write(img)
    w.release()
    return path


def test_pipeline(tmp_path, video, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.chdir(tmp_path)
    cfg = {"camera": "TEST", "source": str(video), "anchor": "center",
           "gates": [dict(td.SELFTEST_CFG, name="A")]}
    td.run(cfg, td.Params(), show=False)

    with open(tmp_path / "out/TEST/passes.csv", encoding="utf-8-sig") as f:
        passes = list(csv.DictReader(f, delimiter=";"))
    with open(tmp_path / "out/TEST/violations.csv", encoding="utf-8-sig") as f:
        vio = list(csv.DictReader(f, delimiter=";"))
    assert [p["был зелёный"] for p in passes] == ["да", "нет"]
    assert [v["тип"] for v in vio] == ["NO_GREEN"]
    assert (tmp_path / "out/TEST" / vio[0]["кадр"]).exists()
    assert "NO_GREEN — 1" in capsys.readouterr().out


def test_green_share_detects_colors():
    img = np.full((20, 20, 3), 128, np.uint8)
    cv2.rectangle(img, (2, 2), (17, 17), (0, 200, 0), 2)
    assert td.green_share(img, (0, 0, 20, 20), td.Params()) > 0.9
    cv2.rectangle(img, (2, 2), (17, 17), (0, 0, 200), 2)
    assert td.green_share(img, (0, 0, 20, 20), td.Params()) < 0.1


class FakeBatchYOLO(FakeYOLO):
    """predict() для пачки кадров: все камеры читают одно видео синхронно."""

    def predict(self, frames, **_):
        res = self.track(frames[0])[0]
        return [res] * len(frames)


def test_multicam_20_cameras(tmp_path, video, monkeypatch):
    import multicam
    monkeypatch.chdir(tmp_path)
    cfgs = [{"camera": f"CAM-{i:02d}", "source": str(video), "anchor": "center",
             "gates": [dict(td.SELFTEST_CFG, name="A")]} for i in range(1, 21)]
    totals = multicam.run_many(cfgs, {"cam_fps": 10}, show=False, model=FakeBatchYOLO())
    assert totals == {"NO_GREEN": 20}
    with open(tmp_path / "out/violations_all.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    assert sorted(r["камера"] for r in rows) == [f"CAM-{i:02d}" for i in range(1, 21)]


def test_links_and_cameras(tmp_path):
    import json
    import multicam
    (tmp_path / "links.txt").write_text("# комментарий\nCAM-01 rtsp://a/1\nCAM-02  \"https://b/2.m3u8\"\n\n",
                                        encoding="utf-8")
    cams = tmp_path / "cameras"
    cams.mkdir()
    (cams / "params.json").write_text('{"ru_text": true}', encoding="utf-8")
    (cams / "CAM-01.json").write_text(json.dumps({"camera": "CAM-01", "source": "old",
                                                  "gates": [td.SELFTEST_CFG]}), encoding="utf-8")
    links = multicam.read_links(tmp_path / "links.txt")
    assert links == {"CAM-01": "rtsp://a/1", "CAM-02": "https://b/2.m3u8"}
    loaded = multicam.load_cameras(cams, tmp_path / "links.txt")
    assert [c["source"] for c in loaded] == ["rtsp://a/1"]
    assert multicam.make_params(multicam.global_params(cams)).ru_text is True


def test_wizard_new_cameras_go_to_autosetup(tmp_path, monkeypatch):
    import json
    import multicam
    monkeypatch.chdir(tmp_path)
    (tmp_path / "links.txt").write_text("CAM-01 a.mp4\nCAM-02 b.mp4\n", encoding="utf-8")
    answers = iter(["y", "", "", "10"])               # русские подписи, старт, авторазметка, 10 мин
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    calls = {}

    def fake_auto(sources, sec, params, review, cam_dir):
        calls["auto"] = sorted(sources)
        calls["sec"] = sec
        for n, s in sources.items():
            multicam.save_camera({"camera": n, "source": s, "gates": [td.SELFTEST_CFG]}, cam_dir)

    monkeypatch.setattr(multicam, "autosetup", fake_auto)
    monkeypatch.setattr(multicam, "run_many", lambda cfgs, gp, show: calls.update(run=[c["camera"] for c in cfgs],
                                                                                   ru=gp["ru_text"]))
    multicam.wizard()
    assert calls == {"auto": ["CAM-01", "CAM-02"], "sec": 600, "run": ["CAM-01", "CAM-02"], "ru": True}
    assert json.loads((tmp_path / "cameras/params.json").read_text(encoding="utf-8"))["ru_text"] is True


def test_wizard_creates_links_template(tmp_path, monkeypatch):
    import multicam
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(multicam, "_open_file", lambda p: None)
    multicam.wizard()
    assert "CAM-01" in (tmp_path / "links.txt").read_text(encoding="utf-8")
