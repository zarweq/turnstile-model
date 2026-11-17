import time
from collections import Counter, deque

import numpy as np
import pytest

pytest.importorskip("cv2")
from hud import MOSAIC_TOP, Painter, draw_hud, mosaic, tile_at  # noqa: E402


@pytest.mark.parametrize("ru", [False, True])
def test_mosaic_20_cameras_and_click(ru):
    p = Painter(ru)
    tiles = [np.full((540, 960, 3), i * 10, np.uint8) for i in range(20)]
    tiles[3] = None                                           # камера без сигнала
    names = [f"CAM-{i + 1:02d}" for i in range(20)]
    now = time.time()
    canvas = mosaic(p, tiles, names, {"CAM-07": now + 5}, now)
    assert canvas.shape == (MOSAIC_TOP + 4 * 216 + 36, 5 * 384, 3)       # 5 x 4
    draw_hud(p, canvas, deque([(now, "09:00:00", "CAM-07", "A", "NO_GREEN")]), Counter(NO_GREEN=1), now,
             side_list=False)
    size = (canvas.shape[1], canvas.shape[0])
    assert tile_at(20, size, (10, MOSAIC_TOP + 5)) == 0
    assert tile_at(20, size, (384 * 1 + 5, MOSAIC_TOP + 216 * 1 + 5)) == 6        # CAM-07
    assert tile_at(20, size, (5, 5)) is None                                     # клик по баннеру
