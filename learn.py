"""Обучение ранкера на твоих метках: просмотры (хорошо) + удаления (брак).

Порог включения: >=10 клипов с просмотрами и >=3 удалённых.
Иначе весов нет и скоринг идёт как раньше.
"""
import json
import os
import re
import sqlite3

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "data.db")
WPATH = os.path.join(BASE, "learn.json")
FEATS = ("viral", "hook", "whole", "peak")


def _probs(verdict: str):
    out = {}
    pats = {"viral": r"виральность (\d\.\d+)", "hook": r"хук (\d\.\d+)",
            "whole": r"цельность (\d\.\d+)", "peak": r"пик (\d\.\d+)"}
    for f in FEATS:
        m = re.search(pats[f], verdict or "")
        if not m:
            m = re.search(rf"{f} (\d\.\d+)", verdict or "")
        out[f] = float(m.group(1)) if m else 0.5
    m = re.search(r"Drex (\d+)", verdict or "")
    out["score"] = (int(m.group(1)) / 100) if m else 0.5
    return out


def dataset():
    """[(features, y)] — y=1 хит (просмотры выше медианы), y=0 брак/низ."""
    c = sqlite3.connect(DB, timeout=10)
    try:
        rows = c.execute(
            "SELECT verdict, views FROM clips WHERE views != 0").fetchall()
        try:
            refs = c.execute(
                "SELECT viral, hookp, whole, peak, score, views FROM refs"
                " WHERE views > 0").fetchall()
        except Exception:
            refs = []
    finally:
        c.close()
    pos = [(r[0], r[1]) for r in rows if r[1] and r[1] > 0]
    neg = [r[0] for r in rows if r[1] == -1]
    vs = sorted(v for _, v in pos)
    med = vs[len(vs) // 2] if vs else 0
    data = [(_probs(v), 1 if w >= med else 0) for v, w in pos]
    data += [(_probs(v), 0) for v in neg]
    # залётные референсы — эталонные хиты (не зависят от меток клипов)
    for vr, hk, wh, pk, sc, vw in refs:
        data.append(({"viral": vr or 0.5, "hook": hk or 0.5,
                      "whole": wh or 0.5, "peak": pk or 0.5,
                      "score": (sc or 70) / 100}, 1))
    return data


def train():
    """Считает веса признаков. Возвращает веса или None (мало данных)."""
    data = dataset()
    pos = [x for x, y in data if y == 1]
    neg = [x for x, y in data if y == 0]
    if not neg and len(pos) >= 8:
        # негативов нет (одни хиты) — сравниваем с нейтралью 0.5
        neg = [{f: 0.5 for f in FEATS + ("score",)}] * max(3, len(pos) // 3)
    if len(pos) < 8 or len(neg) < 2 or len(data) < 10:
        return None
    w = {}
    for f in FEATS + ("score",):
        mp = sum(x[f] for x in pos) / len(pos)
        mn = sum(x[f] for x in neg) / len(neg)
        w[f] = round(mp - mn, 4)
    norm = sum(abs(v) for v in w.values()) or 1.0
    w = {k: round(v / norm, 4) for k, v in w.items()}
    payload = {"weights": w, "n_pos": len(pos), "n_neg": len(neg)}
    with open(WPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return payload


def load_weights():
    try:
        return json.load(open(WPATH, encoding="utf-8")).get("weights")
    except Exception:
        return None


def adjust(score: int, verdict: str):
    """Поправка -10..+10 к скору по выученным весам. Без весов — 0."""
    w = load_weights()
    if not w:
        return 0
    x = _probs(verdict)
    dot = sum(w.get(f, 0) * x.get(f, 0.5) for f in FEATS + ("score",))
    return max(-10, min(10, int(round(dot * 20))))


def status():
    data = dataset()
    pos = sum(1 for _, y in data if y == 1)
    neg = sum(1 for _, y in data if y == 0)
    w = load_weights()
    try:
        c = sqlite3.connect(DB, timeout=10)
        try:
            nrefs = c.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
        finally:
            c.close()
    except Exception:
        nrefs = 0
    return {"labeled": len(data), "pos": pos, "neg": neg, "refs": nrefs,
            "trained": bool(w), "weights": w}
