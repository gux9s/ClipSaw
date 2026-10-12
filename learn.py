"""Обучение ранкера на твоих метках: просмотры (хорошо) + удаления (брак).

v2: богатые признаки (DREX-оценки + текст хука + длительность + темп речи),
L2 логистическая регрессия со стандартизацией, балансировка классов,
взвешивание референсов по просмотрам. Обратно совместим с learn.json v1.

Порог обучения: >=6 клипов с просмотрами и >=2 удалённых (было 10+3 —
слишком долго ждать). Меньше данных — работает PRIOR (ручной приор
из виральных практик), поэтому adjust() полезен даже с нуля.
"""
import json
import math
import os
import re
import sqlite3

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "data.db")
WPATH = os.path.join(BASE, "learn.json")
FEATS = ("viral", "hook", "whole", "peak")  # совместимость с v1
FEATS_V2 = ("viral", "hook", "whole", "peak", "score01",
            "dur_fit", "wps_fit", "q_hook", "excl", "trigger",
            "dialog", "complete", "payoff", "filler_pen", "lore_hit",
            "punch", "clarity", "comment")

# --- лексиконы (корни, lowercase, substring-match) ---
TRIGGER_ROOTS = (
    "секрет", "тайн", "измен", "любовн", "развод", "свадьб", "беремен",
    "денег", "деньг", "долг", "полиц", "драк", "скандал", "вран", "врешь",
    "врёшь", "обман", "уходи", "уйди", "люблю", "ненавижу", "дурак",
    "идиот", "дебил", "шок", "правд", "призна", "сюрприз", "наслед",
    "увол", "босс", "начальник", "поцелу", "ночь", "никогда", "всегда",
    "убью", "боюсь", "стыд", "позор", "хватит", "замолчи", "тест",
    "врач", "больниц", "смерт", "умер", "вернул", "уехал", "прости",
    "прощай", "навсегда", "стоп", "смотри", "подожди", "слушай",
    "кто отец", "ты кто", "что ты", "как ты мог", "я всё знаю", "все знаю",
)
FILLER_WORDS = (
    "ну", "короче", "типа", "вот", "значит", "как бы", "привет",
    "здравствуй", "добрый", "алло", "ладно", "понятно", "так сказать",
)
PAYOFF_ROOTS = (
    "продолжени", "часть 2", "части 2", "дальше", "узнаешь", "подожди",
    "стоп", "вдруг", "оказалось", "наконец", "а потом", "и тут",
    "что будет", "а если", "смотри", "а ты", "а он", "а она",
)
BROKEN_END = {"и", "а", "но", "что", "как", "не", "на", "в", "с", "к",
              "от", "для", "или", "если", "когда", "потому", "чтобы",
              "это", "же", "ли", "да", "нет", "уже"}

# Ручной приор: разумное ранжирование с нуля (стандартизованная шкала).
PRIOR_W = {
    "viral": 1.2, "hook": 1.0, "whole": 0.5, "peak": 0.8, "score01": 1.0,
    "dur_fit": 0.4, "wps_fit": 0.3, "q_hook": 0.5, "excl": 0.4,
    "trigger": 0.7, "dialog": 0.4, "complete": 0.5, "payoff": 0.6,
    "filler_pen": -0.8, "lore_hit": 0.4,
    "punch": 0.6, "clarity": 0.5, "comment": 0.5,
}
PRIOR_BIAS = -0.2


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
    for f, pat in (("punch", r"панч (\d\.\d+)"), ("clarity", r"ясность (\d\.\d+)"),
                   ("comment", r"дискуссия (\d\.\d+)")):
        mm = re.search(pat, verdict or "")
        out[f] = float(mm.group(1)) if mm else 0.5
    return out


def text_features(title: str = "", hook: str = "", dur: float = 0.0) -> dict:
    """Текстовые признаки из заголовка/хука. Всё 0..1, filler_pen — штраф 0..1."""
    text = f"{title or ''} {hook or ''}".strip()
    low = text.lower()
    words = re.findall(r"[а-яёa-z0-9]+", low)
    n_words = len(words)
    dur_s = float(dur or 0) or 0.0

    # sweet spot длительности 35с (сериальные клипы 25-45с)
    if dur_s > 0:
        dur_fit = max(0.0, 1.0 - abs(dur_s - 35.0) / 35.0)
    else:
        dur_fit = 0.5
    # темп речи: идеал 2-3.5 слова/с
    if dur_s > 0 and n_words:
        wps = n_words / dur_s
        wps_fit = max(0.0, 1.0 - abs(wps - 2.6) / 2.6)
    else:
        wps_fit = 0.5

    head = low[:140]
    q_hook = 1.0 if "?" in head else 0.0
    excl = min(low.count("!") / 2.0, 1.0)
    trigger = sum(1 for r in TRIGGER_ROOTS if r in low)
    trigger = min(trigger / 2.0, 1.0)
    n_sent = len(re.findall(r"[.!?…]+", text))
    dialog = min(n_sent / 6.0, 1.0)
    tail = low[-160:]
    payoff = 1.0 if any(r in tail for r in PAYOFF_ROOTS) else 0.0
    # вопрос без ответа в конце — тоже клиффхэнгер
    if tail.rstrip().endswith("?"):
        payoff = 1.0
    stripped = text.strip()
    if stripped and stripped[-1] in ".!?…":
        last_w = (re.findall(r"[а-яёa-z0-9]+", low) or [""])[-1]
        complete = 0.0 if last_w in BROKEN_END else 1.0
    else:
        last_w = (re.findall(r"[а-яёa-z0-9]+", low) or [""])[-1] if words else ""
        complete = 0.0 if (not stripped or last_w in BROKEN_END) else 0.3
    filler_pen = sum(1 for fw in FILLER_WORDS if fw in low.split() or low.startswith(fw + " "))
    # приветствие в самом начале — сильный штраф хуку
    if re.match(r"\s*(привет|здравствуй|добрый (день|вечер|утро)|алло)\b", low):
        filler_pen += 1.0
    filler_pen = min(filler_pen / 2.0, 1.0)
    return {
        "dur_fit": round(dur_fit, 3), "wps_fit": round(wps_fit, 3),
        "q_hook": q_hook, "excl": round(excl, 3),
        "trigger": round(trigger, 3), "dialog": round(dialog, 3),
        "complete": complete, "payoff": payoff,
        "filler_pen": round(filler_pen, 3),
    }


def _lore_hit(title: str, hook: str) -> float:
    try:
        d = os.path.join(BASE, "lore")
        names = set()
        for f in os.listdir(d):
            if f.endswith(".json"):
                data = json.load(open(os.path.join(d, f), encoding="utf-8"))
                for c in data.get("characters", []):
                    for a in [c.get("name", "")] + c.get("aliases", []):
                        a = (a or "").strip().lower()
                        if len(a) > 2:
                            names.add(a)
        if names:
            low = f"{title or ''} {hook or ''}".lower()
            for nm in names:
                if nm in low:
                    return 1.0
    except Exception:
        pass
    return 0.0


def full_features(verdict: str = "", title: str = "", hook: str = "",
                   dur: float = 0.0) -> dict:
    """Полный вектор v2: DREX-оценки + текст + длительность."""
    p = _probs(verdict)
    tf = text_features(title, hook, dur)
    return {
        "viral": p["viral"], "hook": p["hook"], "whole": p["whole"],
        "peak": p["peak"], "score01": p["score"],
        "punch": p["punch"], "clarity": p["clarity"], "comment": p["comment"],
        **tf, "lore_hit": _lore_hit(title, hook),
    }


def dataset():
    """[(features(v2), y, weight)] — y=1 хит, y=0 брак/низ. weight — важность."""
    try:
        c = sqlite3.connect(DB, timeout=10)
    except Exception:
        return []
    try:
        try:
            rows = c.execute(
                "SELECT verdict, title, hook, dur, score, views FROM clips"
                " WHERE views != 0").fetchall()
        except Exception:
            try:
                rows = c.execute(
                    "SELECT verdict, views FROM clips WHERE views != 0").fetchall()
                rows = [(r[0], "", "", 0, 0, r[1]) for r in rows]
            except Exception:
                rows = []
        try:
            refs = c.execute(
                "SELECT viral, hookp, whole, peak, score, views, hook, dur"
                " FROM refs WHERE views > 0").fetchall()
        except Exception:
            refs = []
    finally:
        try:
            c.close()
        except Exception:
            pass
    pos = [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows if r[5] and r[5] > 0]
    neg = [(r[0], r[1], r[2], r[3], r[4]) for r in rows if len(r) > 5 and r[5] == -1]
    # если старая схема без колонок — neg из verdict
    if not neg and rows and len(rows[0]) == 2:
        neg = [(r[0], "", "", 0, 0) for r in rows if r[1] == -1]
    vs = sorted(v for _, _, _, _, _, v in pos)
    med = vs[len(vs) // 2] if vs else 0
    data = []
    for v, t, h, d, sc, w in pos:
        data.append((full_features(v, t or "", h or "", d or 0),
                     1 if w >= med else 0, 1.0))
    for v, t, h, d, sc in neg:
        data.append((full_features(v, t or "", h or "", d or 0), 0, 1.5))
    # залётные референсы — эталонные хиты, вес по логарифму просмотров
    for vr, hk, wh, pk, sc, vw, hkt, dr in refs:
        feat = {"viral": vr or 0.5, "hook": hk or 0.5,
                "whole": wh or 0.5, "peak": pk or 0.5,
                "score01": (sc or 70) / 100}
        feat.update(text_features(hkt or "", hkt or "", dr or 0))
        feat["lore_hit"] = _lore_hit(hkt or "", "")
        w = 0.5 + min(math.log10(max(int(vw or 0), 10)) - 1.0, 3.0) / 3.0
        data.append((feat, 1, round(w, 3)))
    return data


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _train_logreg(X, y, sw, l2: float = 1.0, iters: int = 3000,
                  lr: float = 0.5):
    """Взвешенная L2 лог-регрессия на стандартизованных признаках (numpy)."""
    try:
        import numpy as np
    except ImportError:
        return None, None
    Xa = np.array(X, dtype=float)
    ya = np.array(y, dtype=float)
    wa = np.array(sw, dtype=float)
    n, d = Xa.shape
    mu = Xa.mean(axis=0)
    sd = Xa.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Xs = (Xa - mu) / sd
    # стартуем с приора, чтобы малая выборка не уходила в шум
    w = np.array([PRIOR_W.get(f, 0.0) for f in FEATS_V2], dtype=float) * 0.5
    b = float(PRIOR_BIAS) * 0.5
    pos_w = wa[ya == 1].sum() or 1.0
    neg_w = wa[ya == 0].sum() or 1.0
    cw = np.where(ya == 1, len(y) / 2.0 / max(pos_w, 1e-9) * wa,
                  len(y) / 2.0 / max(neg_w, 1e-9) * wa)
    for it in range(iters):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = (p - ya) * cw
        gw = (Xs.T @ err) / len(y) + l2 * w / max(n, 1)
        gb = float(err.mean())
        step = lr / (1.0 + it / 500.0)
        w -= step * gw
        b -= step * gb
    return w, b, mu, sd


def _accuracy(X, y, w, mu, sd, b) -> float:
    try:
        import numpy as np
    except ImportError:
        return 0.0
    Xs = (np.array(X, dtype=float) - mu) / sd
    p = 1.0 / (1.0 + np.exp(-np.clip(Xs @ w + b, -30, 30)))
    pred = (p >= 0.5).astype(int)
    return float((pred == np.array(y)).mean())


def train():
    """Считает веса. Возвращает payload или None (мало данных)."""
    data = dataset()
    pos = [(x, wt) for x, y, wt in data if y == 1]
    neg = [(x, wt) for x, y, wt in data if y == 0]
    if not neg and len(pos) >= 4:
        # негативов нет (одни хиты) — сравниваем с нейтралью 0.5
        neut = {f: 0.5 for f in FEATS_V2}
        neut.update({"dur_fit": 0.5, "wps_fit": 0.5, "q_hook": 0.0,
                     "excl": 0.0, "trigger": 0.0, "dialog": 0.3,
                     "complete": 0.5, "payoff": 0.0, "filler_pen": 0.3,
                     "lore_hit": 0.0})
        neg = [(dict(neut), 1.0)] * max(2, len(pos) // 3)
        data = data + [(dict(neut), 0, 1.0)] * len(neg)
    if len(pos) < 6 or len(neg) < 2 or len(data) < 8:
        return None
    X = [[float(x.get(f, 0.5)) for f in FEATS_V2] for x, _, _ in data]
    y = [yy for _, yy, _ in data]
    sw = [wt for _, _, wt in data]
    res = _train_logreg(X, y, sw)
    if res[0] is None:
        # без numpy — старый метод разности средних (фолбэк)
        w = {}
        for f in FEATS_V2:
            mp = sum(x[f] for x, yy, _ in data if yy == 1) / max(len(pos), 1)
            mn = sum(x[f] for x, yy, _ in data if yy == 0) / max(len(neg), 1)
            w[f] = round(mp - mn, 4)
        norm = sum(abs(v) for v in w.values()) or 1.0
        w = {k: round(v / norm, 4) for k, v in w.items()}
        payload = {"version": 2, "weights": w, "bias": 0.0,
                   "features": list(FEATS_V2),
                   "n_pos": len(pos), "n_neg": len(neg), "acc": 0.0,
                   "method": "mean-diff"}
        with open(WPATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        return payload
    w, b, mu, sd = res
    acc = round(_accuracy(X, y, w, mu, sd, b), 3)
    payload = {
        "version": 2,
        "weights": {f: round(float(w[i]), 4) for i, f in enumerate(FEATS_V2)},
        "bias": round(float(b), 4),
        "means": {f: round(float(mu[i]), 4) for i, f in enumerate(FEATS_V2)},
        "stds": {f: round(float(sd[i]), 4) for i, f in enumerate(FEATS_V2)},
        "features": list(FEATS_V2),
        "n_pos": len(pos), "n_neg": len(neg), "acc": acc,
        "method": "logreg-l2",
    }
    with open(WPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return payload


def bootstrap_prior():
    """Холодный старт: пишет приорные веса, чтобы adjust() работал без меток."""
    payload = {
        "version": 2,
        "weights": dict(PRIOR_W),
        "bias": PRIOR_BIAS,
        "means": {f: 0.5 for f in FEATS_V2},
        "stds": {f: 0.25 for f in FEATS_V2},
        "features": list(FEATS_V2),
        "n_pos": 0, "n_neg": 0, "acc": 0.0,
        "method": "prior", "prior": True,
    }
    with open(WPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return payload


def load_weights():
    try:
        return json.load(open(WPATH, encoding="utf-8")).get("weights")
    except Exception:
        return None


def _load_payload():
    try:
        return json.load(open(WPATH, encoding="utf-8"))
    except Exception:
        return None


def _score_v2(feat: dict, payload: dict) -> float:
    """Линейный скор z по v2-весам (или приору)."""
    w = payload.get("weights", {})
    feats = payload.get("features", list(FEATS_V2))
    mu = payload.get("means", {})
    sd = payload.get("stds", {})
    b = float(payload.get("bias", 0.0))
    z = b
    for f in feats:
        x = float(feat.get(f, 0.5))
        m = float(mu.get(f, 0.5))
        s = float(sd.get(f, 0.25)) or 0.25
        z += float(w.get(f, 0.0)) * ((x - m) / s)
    return z


def adjust(score: int, verdict: str, title: str = "", hook: str = "",
           dur: float = 0.0, use_prior: bool = True):
    """Поправка -10..+10 к скору. v2 — по всем признакам, v1 — как раньше.
    use_prior=False — без обученной модели возвращает 0 (чистый DREX-скор);
    так вызывает отбор клипов. Приор холодного старта применяется только
    при use_prior=True (анализ, тесты)."""
    payload = _load_payload()
    if payload and payload.get("version") == 2 and payload.get("weights"):
        if payload.get("prior") and not use_prior:
            return 0
        # v1-файл без means (старый mean-diff формат) — нормированная шкала
        if not payload.get("means") and payload.get("method") == "mean-diff":
            w = payload["weights"]
            x = full_features(verdict, title, hook, dur)
            dot = sum(w.get(f, 0) * x.get(f, 0.5) for f in FEATS_V2)
            return max(-10, min(10, int(round(dot * 20))))
        feat = full_features(verdict, title, hook, dur)
        z = _score_v2(feat, payload)
        # калибровка: z~=0 -> 0, z=±2 -> ±10
        return max(-10, min(10, int(round(_sigmoid(z) * 20 - 10))))
    if payload and payload.get("weights") and "viral" in payload["weights"]:
        # v1 формат: только 5 признаков из вердикта
        w = payload["weights"]
        x = _probs(verdict)
        dot = sum(w.get(f, 0) * x.get(f, 0.5) for f in FEATS + ("score",))
        return max(-10, min(10, int(round(dot * 20))))
    # холодный старт — приор даёт поправку, только если явно разрешён
    if not use_prior:
        return 0
    prior_payload = {"weights": PRIOR_W, "bias": PRIOR_BIAS,
                     "means": {f: 0.5 for f in FEATS_V2},
                     "stds": {f: 0.25 for f in FEATS_V2},
                     "features": list(FEATS_V2)}
    feat = full_features(verdict, title, hook, dur)
    z = _score_v2(feat, prior_payload)
    return max(-10, min(10, int(round(_sigmoid(z) * 20 - 10))))


def status():
    data = dataset()
    pos = sum(1 for _, y, _ in data if y == 1)
    neg = sum(1 for _, y, _ in data if y == 0)
    payload = _load_payload()
    w = (payload or {}).get("weights")
    try:
        c = sqlite3.connect(DB, timeout=10)
        try:
            nrefs = c.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
        finally:
            c.close()
    except Exception:
        nrefs = 0
    out = {"labeled": len(data), "pos": pos, "neg": neg, "refs": nrefs,
           "trained": bool(w and not (payload or {}).get("prior")),
           "weights": w}
    if payload:
        out.update({k: payload.get(k) for k in
                    ("version", "method", "acc", "bias", "prior") if k in payload})
        out["n_pos"] = payload.get("n_pos", pos)
        out["n_neg"] = payload.get("n_neg", neg)
    return out


if __name__ == "__main__":
    import sys

    cmd = (sys.argv[1] if len(sys.argv) > 1 else "status").lower()
    if cmd == "train":
        res = train()
        if res:
            print(json.dumps(res, ensure_ascii=False, indent=1))
        else:
            print("Мало меток — пишу приор холодного старта.")
            print(json.dumps(bootstrap_prior(), ensure_ascii=False, indent=1))
    elif cmd == "prior":
        print(json.dumps(bootstrap_prior(), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=1))
