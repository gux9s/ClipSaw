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
    for f in FEATS:
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
    finally:
        c.close()
    pos = [(r[0], r[1]) for r in rows if r[1] and r[1] > 0]
    neg = [r[0] for r in rows if r[1] == -1]
    if not pos:
        return []
    vs = sorted(v for _, v in pos)
    med = vs[len(vs) // 2]
    data = [(_probs(v), 1 if w >= med else 0) for v, w in pos]
    data += [(_probs(v), 0) for v in neg]
    return data


def train():
    """Считает веса признаков. Возвращает веса или None (мало данных)."""
    data = dataset()
    pos = [x for x, y in data if y == 1]
    neg = [x for x, y in data if y == 0]
    if len(pos) < 10 or len(neg) < 3 or len(data) < 15:
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
    return {"labeled": len(data), "pos": pos, "neg": neg,
            "trained": bool(w), "weights": w}
