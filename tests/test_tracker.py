from tracker import SimpleTracker


def box(x, y, w=40, h=120):
    return [x - w / 2, y - h, x + w / 2, y]


def test_ids_stable_when_people_pass_each_other():
    tr = SimpleTracker()
    seen = {}
    for k in range(40):
        t = k / 10
        a, b = 100 + 8 * k, 420 - 8 * k                     # идут навстречу, расходятся по высоте
        ids = tr.update([box(a, 300), box(b, 360)], t)
        seen.setdefault("a", set()).add(ids[0])
        seen.setdefault("b", set()).add(ids[1])
    assert len(seen["a"]) == 1 and len(seen["b"]) == 1 and seen["a"] != seen["b"]


def test_short_occlusion_keeps_id():
    tr = SimpleTracker(max_lost=1.0)
    first = tr.update([box(100, 300)], 0.0)[0]
    for k in range(1, 5):
        tr.update([box(100 + 10 * k, 300)], k / 10)
    again = tr.update([box(100 + 10 * 9, 300)], 0.9)[0]    # 0.5 с не видели
    assert again == first


def test_long_absence_new_id():
    tr = SimpleTracker(max_lost=1.0)
    a = tr.update([box(100, 300)], 0.0)[0]
    b = tr.update([box(100, 300)], 3.0)[0]
    assert a != b
