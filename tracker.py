"""
tracker.py — лёгкий трекер людей для режима многих камер.

Почему свой: для 20 камер нейросеть обрабатывает кадры всех камер одной пачкой (batch),
а встроенный трекер ultralytics рассчитан на один поток. Здесь у каждой камеры свой
экземпляр трекера: сопоставление по IoU рамок + расстоянию центров (венгерский алгоритм
через lap, без него — жадно), потерянные треки живут max_lost секунд.
"""

import numpy as np


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def assign(cost, max_cost):
    """Пары (строка, столбец) с минимальной суммарной стоимостью, только cost <= max_cost."""
    if cost.size == 0:
        return []
    try:
        import lap
        _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=max_cost)
        return [(i, j) for i, j in enumerate(x) if j >= 0]
    except ImportError:
        pairs, used_r, used_c = [], set(), set()
        for i, j in sorted(np.ndindex(cost.shape), key=lambda ij: cost[ij]):
            if cost[i, j] > max_cost:
                break
            if i not in used_r and j not in used_c:
                pairs.append((i, j))
                used_r.add(i)
                used_c.add(j)
        return pairs


class SimpleTracker:
    def __init__(self, max_lost=1.0, min_iou=0.1, max_jump=1.5):
        self.max_lost = max_lost          # секунд держать потерянный трек
        self.min_iou = min_iou
        self.max_jump = max_jump          # макс. смещение центра за кадр, в высотах рамки
        self.tracks = {}                  # id -> {"box": array, "t": last_seen, "v": скорость центра}
        self.next_id = 1

    def update(self, boxes, t):
        """boxes: список [x1, y1, x2, y2]. Возвращает список id той же длины."""
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        ids = [-1] * len(boxes)
        for tid in [k for k, tr in self.tracks.items() if t - tr["t"] > self.max_lost]:
            del self.tracks[tid]
        tids = list(self.tracks)
        if tids and len(boxes):
            pred = []
            for k in tids:                                  # предсказываем положение по скорости
                tr = self.tracks[k]
                pred.append(tr["box"] + np.r_[tr["v"], tr["v"]] * (t - tr["t"]))
            pred = np.array(pred)
            iou = iou_matrix(pred, boxes)
            pc = (pred[:, :2] + pred[:, 2:]) / 2
            bc = (boxes[:, :2] + boxes[:, 2:]) / 2
            hgt = np.maximum(pred[:, 3] - pred[:, 1], 1.0)
            jump = np.linalg.norm(pc[:, None] - bc[None], axis=2) / hgt[:, None]
            cost = (1 - iou) + 0.5 * jump
            cost[(iou < self.min_iou) & (jump > 0.5)] = 1e6
            cost[jump > self.max_jump] = 1e6
            for i, j in assign(cost, 1e5):
                k, tr = tids[i], self.tracks[tids[i]]
                dt = max(t - tr["t"], 1e-3)
                c_old = (tr["box"][:2] + tr["box"][2:]) / 2
                v = (bc[j] - c_old) / dt
                tr["v"] = 0.6 * tr["v"] + 0.4 * v
                tr["box"], tr["t"] = boxes[j], t
                ids[j] = k
        for j, b in enumerate(boxes):
            if ids[j] == -1:
                self.tracks[self.next_id] = {"box": b, "t": t, "v": np.zeros(2)}
                ids[j] = self.next_id
                self.next_id += 1
        return ids
