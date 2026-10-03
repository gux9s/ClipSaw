"""AI-нарезка как Opus Clip: транскрибация -> скоринг моментов -> вертикаль 9:16 -> субтитры."""
import json
import os
import re
import subprocess
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
ASR_MODELS = {"base": "⚡ База — быстро", "small": "🎯 Small — точнее, медленнее",
              "giga": "GigaAM v3 — топ для русского"}
_MODEL_CACHE = {}


def _whisper_model(name: str):
    from faster_whisper import WhisperModel
    name = name if name in ("base", "small") else "base"
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = WhisperModel(
            name, device="cpu", compute_type="int8", download_root=MODELS)
    return _MODEL_CACHE[name]


def _transcribe_giga(path: str, limit_min: int = 0, progress=None):
    """GigaAM v3 e2e RNNT: топ для русского, с пословкой.
    Длинное аудио — окнами по 25с (лимит transcribe), шов 2с против дублей."""
    import gigaam
    import tempfile
    if "giga" not in _MODEL_CACHE:
        _MODEL_CACHE["giga"] = gigaam.load_model("e2e_rnnt")
    model = _MODEL_CACHE["giga"]
    import cv2
    probe = cv2.VideoCapture(path)
    n = int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = probe.get(cv2.CAP_PROP_FPS) or 0
    probe.release()
    total = (n / fps) if n and fps else 0
    if limit_min and limit_min > 0:
        total = min(total or limit_min * 60, limit_min * 60)
    if not total:
        return [], "ru"
    WIN, OV = 20.0, 2.0
    words = []
    starts = []
    s = 0.0
    while s < total:
        starts.append(s)
        s += WIN
    for k, st in enumerate(starts):
        cs = max(0.0, st - (OV if st > 0 else 0))
        chunk = os.path.join(tempfile.gettempdir(), f"giga_{os.getpid()}_{k}.wav")
        try:
            subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", str(round(cs, 2)),
                            "-t", str(WIN + OV), "-i", path,
                            "-ar", "16000", "-ac", "1", chunk],
                           check=True, timeout=300)
            r = model.transcribe(chunk, word_timestamps=True)
            for w in (r.words or []):
                a = float(w.start) + cs
                if st > 0 and a < st + 1.0:
                    continue  # дубль из нахлёста
                if a >= total:
                    break
                words.append({"w": str(w.text).strip(),
                              "s": round(a, 2), "e": round(a + max(float(w.end - w.start), 0.05), 2)})
        except Exception:
            pass
        finally:
            try:
                if os.path.exists(chunk):
                    os.remove(chunk)
            except OSError:
                pass
        if progress:
            progress(int(34 * (k + 1) / max(len(starts), 1)), "транскрибация GigaAM…")
    words.sort(key=lambda x: x["s"])
    # слова -> сегменты: пауза 0.8с или длина 12с
    segs, cur = [], []
    for w in words:
        if cur and ((w["s"] - cur[-1]["e"] > 0.8 and cur[-1]["e"] - cur[0]["s"] >= 1.5)
                    or (w["e"] - cur[0]["s"] > 12)):
            segs.append({"start": cur[0]["s"], "end": cur[-1]["e"],
                         "text": " ".join(x["w"] for x in cur), "words": cur})
            cur = []
        cur.append(w)
    if cur:
        segs.append({"start": cur[0]["s"], "end": cur[-1]["e"],
                     "text": " ".join(x["w"] for x in cur), "words": cur})
    return segs, "ru"
GAMEPLAY_DIR = os.path.join(BASE, "gameplay")
GAMEPLAY_FILE = os.path.join(GAMEPLAY_DIR, "subway.mp4")

# Сплит-раскладка 1080x1920: верх сериал 50% / низ геймплей 50%
OUT_W, OUT_H = 1080, 1920
TOP_H = int(OUT_H * 0.5)          # 960
BOT_H = OUT_H - TOP_H             # 960

# Первые секунды видео — заставки студий/логотипы: моменты раньше не начинаем
MIN_START = 15.0

# Баннер: вставка примерно в середину клипа, ускорен, по центру (≥25% экрана)
BANNER_FILE = os.path.join(BASE, "assets", "banner.mp4")
BANNER_SPEED = 1.3
BANNER_W = 1080  # во всю ширину: 1080×607 ≈ 32% площади 1080×1920
BANNER_ZOOM = 1.22  # наезд: контент баннера впритык к краям, текст цел (замерено)
BANNER_GAP = 8  # зазор под фильмом: баннер висит прямо под верхом, не задевая
FFMPEG = os.path.expanduser("~/.local/bin/ffmpeg")
os.environ["PATH"] = (os.path.expanduser("~/.deno/bin") + ":"
                      + os.path.expanduser("~/.local/bin") + ":"
                      + os.environ.get("PATH", ""))


def _has_videotoolbox() -> bool:
    try:
        r = subprocess.run([FFMPEG, "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=30)
        return "h264_videotoolbox" in r.stdout
    except Exception:
        return False


_VT = _has_videotoolbox()


def _video_codec_args() -> list:
    """x264 CRF — софтверный, но заметно качественнее железного CBR:
    психовизуалка, адаптивный битрейт. Скорость та же (бутылка — Python-цикл)."""
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-preset", "veryfast"]
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
            "-preset", "veryfast"]


# ---------- 1. Транскрибация ----------

def transcribe(path: str, limit_min: int = 0, progress=None, model_name: str = "base"):
    """faster-whisper / GigaAM -> segments[{start,end,text,words[]}]."""
    if model_name == "giga":
        return _transcribe_giga(path, limit_min, progress)
    src = path
    tmp_cut = None
    if limit_min and limit_min > 0:
        tmp_cut = path + ".cut.mp4"
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", path,
                        "-t", str(limit_min * 60), "-c", "copy", tmp_cut],
                       check=True, timeout=300)
        src = tmp_cut
    model = _whisper_model(model_name)
    if progress:
        progress(2, "транскрибация…")

    def _collect(segs_iter, info):
        out = []
        for s in segs_iter:
            words = [{"w": w.word.strip(), "s": float(round(w.start, 2)), "e": float(round(w.end, 2))}
                     for w in (s.words or []) if w.word.strip()]
            # чистка таймингов: сортировка, без наложений и нулевых длительностей
            words.sort(key=lambda x: x["s"])
            clean = []
            for w in words:
                if w["e"] <= w["s"]:
                    w["e"] = w["s"] + 0.08
                if clean and w["s"] < clean[-1]["e"]:
                    w["s"] = clean[-1]["e"]
                    if w["e"] <= w["s"]:
                        w["e"] = w["s"] + 0.08
                clean.append(w)
            out.append({"start": float(round(s.start, 2)), "end": float(round(s.end, 2)),
                        "text": s.text.strip(), "words": clean})
            if progress and out:
                progress(min(34, int(34 * s.end / max(info.duration, 1))), "транскрибация…")
        return out

    segs, info = model.transcribe(src, beam_size=5, word_timestamps=True,
                                  vad_filter=True,
                                  vad_parameters=dict(min_silence_duration_ms=500))
    out = _collect(segs, info)
    if not out:  # VAD иногда выедает всё аудио — второй проход без него
        if progress:
            progress(2, "транскрибация без VAD…")
        segs, info = model.transcribe(src, beam_size=5, word_timestamps=True)
        out = _collect(segs, info)
    if tmp_cut and os.path.exists(tmp_cut):
        os.remove(tmp_cut)
    return out, info.language


# ---------- 2. Скоринг моментов ----------

def _ssl_ctx():
    """Системный cert-бандл Python протух — ходим в Google с certifi."""
    try:
        import certifi
        import ssl
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


_CTX = _ssl_ctx()


DREX_URL = "https://drex.nace.ai/v1/systemone"
DREX_MODEL = "drex-v1.5"


def _drex_call(key: str, state: str, questions: dict):
    """Один батч в DREX: до 512 вопросов за миллисекунды."""
    import socket
    import time
    import urllib.error
    body = json.dumps({"model": DREX_MODEL, "state": state,
                       "questions": questions}).encode()
    last = None
    for attempt in range(4):
        req = urllib.request.Request(
            DREX_URL, data=body, headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=55, context=_CTX))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 529):
                time.sleep(4 * (attempt + 1))
                continue
            raise
        except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
            last = e  # зависший батч — повторяем
            time.sleep(3 * (attempt + 1))
            continue
    raise last


def _drex_score(segs, key: str, want: int, min_len: int, max_len: int):
    """Кандидаты-окна -> DREX (viral/hook/whole) -> скор 0-100. Без текстовой генерации."""
    # чанки транскрипта, чтобы влезть в лимит токенов модели
    chunks, cur, cur_chars = [], [], 0
    for s in segs:
        t = f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}"
        cur.append(s)
        cur_chars += len(t)
        if cur_chars > 18000:
            chunks.append(cur)
            cur, cur_chars = [], 0
    if cur:
        chunks.append(cur)
    out = []
    for csegs in chunks:
        state = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in csegs)
        wins, questions = [], {}
        step = max(1, len(csegs) // 50)  # не больше ~50 стартов на чанк
        for i in range(0, len(csegs), step):
            j = i
            while j < len(csegs) and csegs[j]["end"] - csegs[i]["start"] <= max_len:
                dur = csegs[j]["end"] - csegs[i]["start"]
                if dur >= min_len * 0.7:
                    s0, e0 = csegs[i]["start"], csegs[j]["end"]
                    k = len(wins)
                    wins.append((s0, e0, " ".join(s["text"] for s in csegs[i:j + 1])))
                    tag = f"[{s0:.1f}-{e0:.1f}]"
                    questions[f"w{k}_viral"] = {
                        "type": "noul",
                        "instructions": f"Moment {tag} will go viral on TikTok: hook in first seconds, emotion, quotable, story payoff."}
                    questions[f"w{k}_hook"] = {
                        "type": "noul",
                        "instructions": f"Moment {tag} opens with a strong hook that stops scrolling."}
                    questions[f"w{k}_whole"] = {
                        "type": "noul",
                        "instructions": f"Moment {tag} is a complete thought, not cut off mid-sentence."}
                    questions[f"w{k}_peak"] = {
                        "type": "noul",
                        "instructions": f"Moment {tag} has an emotional peak, plot twist or cliffhanger ending."}
                j += 1
        if not wins:
            continue
        # батчи по 100 окон (400 вопросов): быстрее отвечают, лимит API — 512
        ans = {}
        wkeys = [f"w{k}_{sfx}" for k in range(len(wins))
                 for sfx in ("viral", "hook", "whole", "peak")]
        for b in range(0, len(wkeys), 400):
            sub = {k: questions[k] for k in wkeys[b:b + 400]}
            part = _drex_call(key, state, sub).get("answers", {})
            ans.update(part)
        for k, (s0, e0, txt) in enumerate(wins):
            def _p(suffix):
                a = (ans.get(f"w{k}_{suffix}") or {})
                return float(a.get("noul", 0.5))
            pv, ph, pw, pk = _p("viral"), _p("hook"), _p("whole"), _p("peak")
            score = int(round(100 * (0.45 * pv + 0.20 * ph + 0.15 * pw + 0.20 * pk)))
            title = " ".join(txt.split()[:9]) + "…"
            out.append({
                "start": s0, "end": e0, "score": score, "title": title[:120],
                "verdict": (f"Drex {score}/100: виральность {pv:.2f}, "
                            f"хук {ph:.2f}, цельность {pw:.2f}, пик {pk:.2f}."),
                "hook": "",
            })
    return sorted(out, key=lambda m: -m["score"])[:want * 3 + 10]


def detect_scenes(src: str, min_len: float = 15.0, max_len: float = 120.0):
    """PySceneDetect -> сцены [(start,end)], короткие склеены, длинные режутся пополам."""
    from scenedetect import detect, ContentDetector
    raw = [(s.get_seconds(), e.get_seconds()) for s, e in
           detect(src, ContentDetector(threshold=27.0))]
    if not raw:
        return []
    merged, cs, ce = [], raw[0][0], raw[0][1]
    for s, e in raw[1:]:
        if ce - cs < min_len:
            ce = e  # короткая сцена (<15с) — клеим к следующей
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    out = []
    for s, e in merged:  # длинные пилим на куски ≤120с
        while e - s > max_len:
            out.append((s, s + max_len))
            s += max_len
        out.append((s, e))
    return [(round(s, 1), round(e, 1)) for s, e in out]


def snap_to_scenes(s: float, e: float, scenes, tol: float = 4.0):
    """Притягивает границы момента к границам сцен (±tol сек)."""
    if scenes:
        starts = [x[0] for x in scenes]
        ends = [x[1] for x in scenes]
        ns = min(starts, key=lambda x: abs(x - s))
        ne = min(ends, key=lambda x: abs(x - e))
        if abs(ns - s) <= tol:
            s = ns
        if abs(ne - e) <= tol:
            e = ne
    return s, e


def _word_bounds(segs):
    b = []
    for s in segs:
        for w in (s.get("words") or []):
            try:
                b.append((float(w["s"]), float(w["e"])))
            except (TypeError, ValueError, KeyError):
                continue
    return sorted(b)


def snap_to_speech(s: float, e: float, bounds, tol: float = 1.2):
    """Границы клипа — ровно по краям слов: не режем слово и фразу пополам."""
    import bisect
    if not bounds:
        return s, e
    pts = sorted({p for w in bounds for p in w})
    i = bisect.bisect_left(pts, s)
    near = [p for p in pts[max(0, i - 2):i + 2] if abs(p - s) <= tol]
    if near:
        s = min(near, key=lambda p: abs(p - s))
    j = bisect.bisect_left(pts, e)
    near = [p for p in pts[max(0, j - 2):j + 2] if abs(p - e) <= tol]
    if near:
        e = min(near, key=lambda p: abs(p - e))
    return s, e


def parse_stamps(text: str):
    """Ручные таймкоды: строки 'мм:сс-мм:сс' / 'ч:мм:сс-...' / 'сек-сек'."""
    def ts(t):
        t = t.strip().replace(",", ".")
        if re.fullmatch(r"\d+(\.\d+)?", t):
            return float(t)
        p = t.split(":")
        try:
            nums = [float(x) for x in p]
        except ValueError:
            return None
        mul = 1
        total = 0.0
        for n in reversed(nums):
            total += n * mul
            mul *= 60
        return total
    out = []
    for line in (text or "").splitlines():
        line = line.strip().strip("-–— ")
        if not line or "-" not in line.replace("–", "-"):
            continue
        a, b = re.split(r"\s*[-–—]\s*", line, maxsplit=1)
        s, e = ts(a), ts(b)
        if s is not None and e is not None and e - s >= 3:
            out.append({"start": round(s, 1), "end": round(e, 1),
                        "score": 100, "title": "Вручную"})
    return out


def pick_moments(segs, lang: str, key: str, want: int, min_len: int, max_len: int,
                 scenes=None, min_gap: float = 30.0):
    if not key:
        raise RuntimeError("Нет DREX-ключа — вставь его в настройках")
    try:
        cands = _drex_score(segs, key, want, min_len, max_len)
    except Exception as e:
        raise RuntimeError(f"DREX недоступен: {e}") from e
    scorer = "drex"
    if not cands:
        raise RuntimeError("DREX вернул пусто — попробуй ещё раз")
    # Весь пул в отбор: сначала разводка по времени, качество — планкой внутри.
    # Два прохода: (зазор, пол) = (норм, 45) затем (узко, 35) — количество добивается.
    pool = sorted(cands, key=lambda m: -m.get("score", 0))
    if not pool:
        raise RuntimeError("DREX вернул пусто — попробуй ещё раз")
    # адаптивный зазор: на часовом видео под 50 клипов зазор жмём, иначе не влезут
    total_dur = (segs[-1]["end"] - segs[0]["start"]) if segs else 0
    if total_dur > 0:
        fit_gap = (total_dur - want * max_len) / max(want, 1)
        min_gap = max(5.0, min(min_gap, fit_gap)) if fit_gap > 0 else 5.0
    # жадный отбор без пересечений + привязка к сценам + подгон длины
    bounds = _word_bounds(segs)
    picked, used = [], set()
    for gap, floor in ((min_gap, 45), (10.0, 35)):
        if len(picked) >= want:
            break
        for mi, m in enumerate(pool):
            if mi in used:
                continue
            if m.get("score", 0) < floor:
                continue
            s, e = float(m["start"]), float(m["end"])
            if s < MIN_START - 4.0 and e < MIN_START + 4.0:
                continue  # заставка/логотип в начале — пропускаем целиком
            if any(not (e + gap < p["start"] or s - gap > p["end"]) for p in picked):
                continue
            dur = e - s
            if dur > max_len:
                c = (s + e) / 2
                s, e = max(0, segs[0]["start"] + 0, c - max_len / 2), c + max_len / 2
            elif dur < min_len:
                c = (s + e) / 2
                s, e = max(0, c - min_len / 2), c + min_len / 2
            s, e = snap_to_scenes(s, e, scenes)
            s = max(s, MIN_START)  # привязка к сцене могла утащить на заставку
            s, e = snap_to_speech(s, e, bounds)  # режем строго между словами
            s = max(s, MIN_START)  # и спич-снап не лезет на заставку
            if e - s < 5:
                continue
            used.add(mi)
            picked.append({"start": float(round(s, 1)), "end": float(round(e, 1)),
                           "score": int(m.get("score", 70)), "title": m.get("title", ""),
                           "verdict": str(m.get("verdict", ""))[:240],
                           "hook": str(m.get("hook", "") or m.get("title", ""))[:40]})
            if len(picked) >= want:
                break
    return picked, scorer


# ---------- 3. Трек лиц ----------

def _yunet_detector(w: int, h: int):
    import cv2
    path = os.path.join(MODELS, "yunet.onnx")
    if not os.path.exists(path):
        return None
    det = cv2.FaceDetectorYN.create(path, "", (w, h), 0.6, 0.3, 5000)
    return det


def track_faces(src: str, start: float, end: float, every: float = 0.7):
    """Возвращает [(t, cx_norm)] либо []."""
    import cv2
    cap = cv2.VideoCapture(src)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    det = _yunet_detector(W, H)
    haar = None
    if det is None:
        haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    pts, t = [], start
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            t += every
            continue
        cx = None
        if det is not None:
            try:
                _, faces = det.detect(fr)
                if faces is not None and len(faces):
                    f = max(faces, key=lambda b: b[2] * b[3])
                    cx = (f[0] + f[2] / 2) / W
            except Exception:
                det = None
        if cx is None and haar is not None:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(g, (640, int(640 * H / W)))
            fs = haar.detectMultiScale(small, 1.2, 4, minSize=(60, 60))
            if len(fs):
                f = max(fs, key=lambda b: b[2] * b[3])
                cx = (f[0] + f[2] / 2) / 640
        if cx is not None:
            pts.append((t, float(min(max(cx, 0), 1))))
        t += every
    cap.release()
    if len(pts) < 2:
        return []
    # сглаживание скользящим средним
    sm = []
    for i, (t, x) in enumerate(pts):
        win = pts[max(0, i - 2):i + 3]
        sm.append((t, sum(p[1] for p in win) / len(win)))
    return sm


def face_x_at(track, t: float) -> float:
    if not track:
        return 0.5
    if t <= track[0][0]:
        return track[0][1]
    for (t0, x0), (t1, x1) in zip(track, track[1:]):
        if t0 <= t <= t1:
            k = (t - t0) / max(t1 - t0, 1e-6)
            return x0 + (x1 - x0) * k
    return track[-1][1]


# ---------- 3.5. Сплит: верх контент / низ геймплей ----------

def ensure_gameplay() -> str | None:
    """Возвращает путь к закэшированному геймплею; если нет — качает с YouTube."""
    if os.path.exists(GAMEPLAY_FILE) and os.path.getsize(GAMEPLAY_FILE) > 10 << 20:
        return GAMEPLAY_FILE
    os.makedirs(GAMEPLAY_DIR, exist_ok=True)
    try:
        subprocess.run(
            ["yt-dlp", "--no-playlist",
             "-f", "bv*[height<=720]+ba/b[height<=720]/b",
             "--merge-output-format", "mp4",
             "-o", os.path.join(GAMEPLAY_DIR, "subway.%(ext)s"),
             "ytsearch1:subway surfers gameplay 1 hour no commentary"],
            timeout=1200, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    if os.path.exists(GAMEPLAY_FILE):
        return GAMEPLAY_FILE
    return None


def _warm_start(cap, max_skip: int = 10):
    """Прогрев декодера после seek: выкидываем битые кадры и чёрный fade-in,
    чтобы первый кадр клипа (обложка!) не был чёрным. Сдвиг ≤0.4с."""
    import cv2
    import numpy as np
    for _ in range(2):  # мусор декодера после seek
        if not cap.read()[0]:
            return
    for _ in range(max_skip):
        ok, fr = cap.read()
        if not ok:
            return
        if float(np.mean(fr)) >= 10:
            # не чёрный — откатить нельзя, но следующий read даст почти тот же
            # кадр; кладём обратно через grab-пропуск? проще: стоим здесь,
            # сдвиг на пару кадров незаметен
            # возвращаем кадр в поток, прокрутив назад на 1
            try:
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if pos and pos > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos - 1)
            except Exception:
                pass
            return


def make_thumb(video_path: str, thumb_path: str, w: int = 360) -> bool:
    """Постер для сетки: первый не-чёрный кадр из 25/40/55/70/15% длительности."""
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if n <= 0:
            cap.release()
            return False
        import numpy as np
        for frac in (0.25, 0.4, 0.55, 0.7, 0.15, 0.85):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
            ok, fr = cap.read()
            if ok and fr is not None and float(np.mean(fr)) >= 14:
                h, ww = fr.shape[:2]
                small = cv2.resize(fr, (w, int(round(h * w / max(ww, 1)))),
                                   interpolation=cv2.INTER_AREA)
                ok = cv2.imwrite(thumb_path, small,
                                 [cv2.IMWRITE_JPEG_QUALITY, 70])
                cap.release()
                return bool(ok)
        cap.release()
    except Exception:
        pass
    return False


def media_duration(path: str) -> float | None:
    """Фактическая длительность готового файла (с учётом фриза под баннер)."""
    import cv2
    try:
        c = cv2.VideoCapture(path)
        n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = c.get(cv2.CAP_PROP_FPS) or 0
        c.release()
        if n > 0 and fps > 0:
            return round(n / fps, 1)
    except Exception:
        pass
    return None


def _cover_resize(fr, tw: int, th: int, cx: float = 0.5, cy: float = 0.5,
                  drift_y: float = 0.0, zoom: float = 1.0):
    """Crop-fill: масштабирует кадр чтобы закрыть tw×th, вырезает окно.
    cx/cy — центр окна 0..1, drift_y — сдвиг окна по Y в пикселях,
    zoom — доп. наезд (>1 прячет края и ватермарки)."""
    import cv2
    h, w = fr.shape[:2]
    sc = max(tw / max(w, 1), th / max(h, 1)) * zoom
    sw, sh = max(int(round(w * sc)), tw), max(int(round(h * sc)), th)
    if (sw, sh) != (w, h):
        # вверх — Lanczos (края), вниз — Area (без муара)
        inter = cv2.INTER_LANCZOS4 if sc > 1.0 else cv2.INTER_AREA
        fr = cv2.resize(fr, (sw, sh), interpolation=inter)
    x0 = int(min(max(cx * sw - tw / 2, 0), max(sw - tw, 0)))
    y0 = int(min(max(cy * sh - th / 2 + drift_y, 0), max(sh - th, 0)))
    return fr[y0:y0 + th, x0:x0 + tw]


def _quality_vf() -> str:
    """Базовая цепочка качества для TikTok: давим шумы пережатого исходника,
    слегка точим после апскейла. Всегда, не только с антидетектом."""
    return "hqdn3d=5:4:3:3,unsharp=5:5:0.5:5:5:0.0"


def _antidetect_filters():
    """Усиленный антидетект: сдвиг цвета + лёгкое зерно (видео),
    подъём питча на 2% без смены длительности (аудио)."""
    vf = "eq=contrast=1.07:saturation=1.12:brightness=0.015,noise=alls=3:allf=t"
    af = "aresample=44100,asetrate=44100*1.02,aresample=44100,atempo=0.9804"
    return vf, af


_BANNER_AUDIO = None


def _banner_has_audio(path: str) -> bool:
    global _BANNER_AUDIO
    if _BANNER_AUDIO is None:
        try:
            r = subprocess.run([FFMPEG, "-i", path],
                               capture_output=True, text=True, timeout=30)
            _BANNER_AUDIO = " Audio:" in r.stderr
        except Exception:
            _BANNER_AUDIO = False
    return _BANNER_AUDIO


def _chroma_overlay(bg, fg, box_w: int = BANNER_W, cy: int | None = None,
                    zoom: float = 1.0, y_top: int | None = None):
    """Баннер: зелёный фон выбит хромакеем, лёгкий despill против ореола.
    cy — центр по вертикали (по умолчанию центр кадра), y_top — верх бокса,
    zoom — наезд с центр-кропом (убирает поля дизайна до краёв)."""
    import cv2
    import numpy as np
    h0, w0 = fg.shape[:2]
    bw = min(box_w, bg.shape[1])
    bh = max(int(round(h0 * bw / max(w0, 1))), 1)
    if bh > bg.shape[0]:
        bh = bg.shape[0]
        bw = max(int(round(w0 * bh / max(h0, 1))), 1)
    fg = cv2.resize(fg, (bw, bh), interpolation=cv2.INTER_LANCZOS4)
    if zoom > 1.0:  # наезд: режем поля, центр-кроп обратно в бокс
        zw, zh = int(round(bw * zoom)), int(round(bh * zoom))
        big = cv2.resize(fg, (zw, zh), interpolation=cv2.INTER_LANCZOS4)
        dx, dy = (zw - bw) // 2, (zh - bh) // 2
        fg = big[dy:dy + bh, dx:dx + bw]
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(fg, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([85, 255, 255]))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    fg = fg.astype(np.float32)
    mx = np.maximum(fg[..., 2], fg[..., 0])
    fg[..., 1] -= np.clip(fg[..., 1] - mx, 0, None) * 0.85
    a = (mask.astype(np.float32) / 255.0)[..., None]  # 1 = зелень = прозрачно
    Hh, Ww = bg.shape[:2]
    x0 = (Ww - bw) // 2
    if y_top is not None:
        y0 = min(max(y_top, 0), max(Hh - bh, 0))
    else:
        y0 = (cy - bh // 2) if cy is not None else (Hh - bh) // 2
        y0 = min(max(y0, 0), max(Hh - bh, 0))
    roi = bg[y0:y0 + bh, x0:x0 + bw].astype(np.float32)
    bg[y0:y0 + bh, x0:x0 + bw] = np.clip(roi * a + fg * (1 - a), 0, 255).astype(np.uint8)
    return bg


def _extract_pcm(path: str, ss=None, dur=None, af=None,
                 sr: int = 44100, ch: int = 2):
    """Декодирует кусок аудио в numpy int16 (samples × ch)."""
    import numpy as np
    cmd = [FFMPEG, "-y", "-v", "error"]
    if ss is not None:
        cmd += ["-ss", str(ss)]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-i", path]
    if af:
        cmd += ["-af", af]
    cmd += ["-f", "s16le", "-ac", str(ch), "-ar", str(sr), "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("звук не извлёкся")
    return np.frombuffer(r.stdout, dtype=np.int16).reshape(-1, ch)


def _seek_live(gcap, gdur: float, need: float, tries: int = 3):
    """Ставит геймплей на живое место: пробует случайные точки, меряет
    движение между двумя кадрами (0.4с) и берёт самое активное.
    Отсекает меню, профиль, сундуки и темноту."""
    import cv2
    import numpy as np
    import random
    span = max(gdur - need - 0.5, 0)
    if span <= 0:
        gcap.set(cv2.CAP_PROP_POS_MSEC, 0)
        return 0.0
    best, best_score = 0.0, -1.0
    for _ in range(tries):
        s = random.uniform(0, span)
        gcap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
        ok, fa = gcap.read()
        if not ok:
            continue
        for _ in range(24):  # ~0.4с вперёд без декодирования
            if not gcap.grab():
                break
        ok, fb = gcap.read()
        if not ok:
            continue
        ga = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY).astype(int)
        gb = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY).astype(int)
        motion = float(np.abs(gb - ga).mean())
        bright = float(gb.mean())
        score = motion if bright > 18 else -1.0  # тёмные заставки мимо
        if score > best_score:
            best, best_score = s, score
    gcap.set(cv2.CAP_PROP_POS_MSEC, best * 1000)
    return best


_BANNER_EFF = None


def _banner_eff() -> float:
    global _BANNER_EFF
    if _BANNER_EFF is None:
        try:
            import cv2
            c = cv2.VideoCapture(BANNER_FILE)
            n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            f = c.get(cv2.CAP_PROP_FPS) or 60
            c.release()
            _BANNER_EFF = (n / max(f, 1)) / BANNER_SPEED if n else 4.22
        except Exception:
            _BANNER_EFF = 4.22
    return _BANNER_EFF


def render_split_clip(src: str, moment: dict, segs, out: str,
                      facetrack: bool = True, captions: bool = True,
                      progress=None, antidetect: bool = False,
                      mirror: bool = False, gameplay: str | None = None,
                      banner: bool = True, banner_path: str | None = None,
                      max_total: float | None = None):
    """Вертикаль 1080x1920: верх 40% — сериал (crop-fill), низ 60% — геймплей
    (закольцован, немой). Сабы — у шва. Звук — только из сериала."""
    import cv2
    import random
    from PIL import Image, ImageDraw
    gp = gameplay or ensure_gameplay()
    if not gp or not os.path.exists(gp):
        raise RuntimeError("нет геймплея — положи gameplay/subway.mp4")
    start, end = float(moment["start"]), float(moment["end"])
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fps = min(float(fps), 30)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if nfr and fps > 0:
        end = min(end, nfr / fps)
    if end - start < 1:
        cap.release()
        raise RuntimeError("момент за пределами видео")
    dur = end - start
    if banner and max_total:
        # итог вместе с баннерами влезает в линзу: ужимаем контент
        be = _banner_eff()
        for _ in range(3):
            n = max(1, min(6, int(round(dur / 50.0))))
            cap_dur = max_total - n * be
            if dur <= cap_dur or cap_dur < 5:
                break
            dur = cap_dur
        end = start + dur

    gcap = cv2.VideoCapture(gp)
    gn = int(gcap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    gfps = gcap.get(cv2.CAP_PROP_FPS) or 30
    gdur = gn / max(gfps, 1) if gn else 0
    # живая стартовая точка геймплея для каждого клипа (мимо меню и сундуков)
    if gdur > dur + 1:
        _seek_live(gcap, gdur, dur)
    else:
        gcap.set(cv2.CAP_PROP_POS_MSEC, 0)
    # шаг кадров геймплея: геймплей 60fps, выход 25-30fps — пропускаем кадры,
    # иначе низ пойдёт в слоу-мо
    g_step = max(gfps, 1) / max(fps, 1)
    g_frac = 0.0

    track = track_faces(src, start, end) if facetrack else []
    font = _font(62)
    seam = TOP_H

    # --- баннеры: 1 на ~50с контента (150-179с → 3 штуки), места равномерно ---
    wins = []  # (output_start, output_end, content_start)
    bcap, b_step, b_frac = None, 0.0, 0.0
    bpath = banner_path or BANNER_FILE
    use_banner = bool(banner) and os.path.exists(bpath)
    b_eff = 0.0
    if use_banner:
        bcap = cv2.VideoCapture(bpath)
        bn = int(bcap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        bfps = bcap.get(cv2.CAP_PROP_FPS) or 60
        b_eff = (bn / max(bfps, 1)) / BANNER_SPEED  # 5.5с → ~4.2с
        n = max(1, min(6, int(round(dur / 50.0))))
        if dur < b_eff + 3:
            use_banner = False  # клип короче баннера — пропускаем
        else:
            for k in range(n):
                if n == 1:
                    lo, hi = dur * 0.33, dur * 0.67 - b_eff
                    cA = random.uniform(lo, hi) if hi > lo else (dur - b_eff) / 2
                else:
                    cA = dur * (k + 1) / (n + 1) - b_eff / 2
                    cA = min(max(cA, 0.5), max(dur - b_eff - 0.5, 0.5))
                oA = cA + k * b_eff
                wins.append((oA, oA + b_eff, cA))
            b_step = max(bfps, 1) * BANNER_SPEED / max(fps, 1)

    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}",
           "-r", str(int(fps)),
           "-i", "pipe:0"]
    mix_tmp = None
    if use_banner:
        # фриз: контент останавливается на время баннера и продолжается
        # с того же места — клип длиннее на суммарный баннер
        froze = b_eff * len(wins)
        out_dur = dur + froze
        # --- звук собираем посемпльно в numpy: оригинал + баннеры + оригинал ---
        import numpy as np
        sr, ch = 44100, 2
        af_src = ("aresample=44100,asetrate=44100*1.02,"
                  "aresample=44100,atempo=0.9804") if antidetect else None
        orig = _extract_pcm(src, ss=start, dur=dur, af=af_src, sr=sr, ch=ch)
        n_total = int(round(out_dur * sr))
        need = int(round(b_eff * sr))
        if _banner_has_audio(bpath):
            ban = _extract_pcm(
                bpath, af=f"atempo={BANNER_SPEED},aresample={sr},volume=0.5",
                sr=sr, ch=ch)
            ban = ban[:need]
            if len(ban) < need:
                ban = np.vstack([ban, np.zeros((need - len(ban), ch),
                                               dtype=np.int16)])
        else:
            ban = np.zeros((need, ch), dtype=np.int16)
        parts, prev_s = [], 0
        for (_, _, cA) in wins:
            n_cA = min(int(round(cA * sr)), len(orig))
            parts.append(orig[prev_s:n_cA])
            parts.append(ban)
            prev_s = n_cA
        parts.append(orig[prev_s:])
        tl = np.vstack(parts)
        if len(tl) < n_total:
            tl = np.vstack([tl, np.zeros((n_total - len(tl), ch), dtype=np.int16)])
        else:
            tl = tl[:n_total]
        mix_tmp = out + ".mix.s16"
        tl.tofile(mix_tmp)
        cmd += ["-f", "s16le", "-ar", str(sr), "-ac", str(ch), "-i", mix_tmp]
        qvf = _quality_vf()
        if antidetect:
            vf, _ = _antidetect_filters()
            qvf += "," + vf
        cmd += ["-vf", qvf]
        cmd += ["-map", "0:v", "-map", "1:a"]
    else:
        out_dur = dur
        cmd += ["-ss", str(start), "-t", str(round(dur, 2)), "-i", src,
                "-map", "0:v", "-map", "1:a?"]
        qvf = _quality_vf()
        if antidetect:
            vf, af = _antidetect_filters()
            qvf += "," + vf
            cmd += ["-vf", qvf, "-af", af]
        else:
            cmd += ["-vf", qvf]
    cmd += _video_codec_args() + ["-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
            "-t", str(round(out_dur, 2)), out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    _warm_start(cap)  # первый кадр не чёрный
    total = max(int(round(out_dur * fps)), 1)
    last_top, last_bot, last_win = None, None, -1
    try:
        for k in range(total):
            t_rel = k / fps  # время на ВЫХОДЕ
            if use_banner:
                acc, ct_rel, frozen = 0.0, t_rel, False
                for (oA, oB, cA) in wins:
                    if t_rel >= oB:
                        acc += b_eff
                    elif t_rel >= oA:
                        ct_rel, frozen = cA, True
                        break
                else:
                    ct_rel = t_rel - acc
            else:
                ct_rel, frozen = t_rel, False
            t = start + ct_rel  # время в КОНТЕНТЕ (сабы, трек лиц)
            # цикл в RGB: одна конверсия вместо трёх на кадр
            if frozen and last_top is not None:
                fr = last_top
            else:
                ok, fr = cap.read()
                if not ok:
                    if last_top is None:
                        break
                    fr = last_top
                else:
                    if mirror:
                        fr = cv2.flip(fr, 1)
                    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                last_top = fr
            if track:
                cx = face_x_at(track, t)
                if mirror:
                    cx = 1.0 - cx
            else:
                cx = 0.5
            top = _cover_resize(fr, OUT_W, TOP_H, cx=cx, cy=0.5,
                                zoom=(1.04 if antidetect else 1.0))

            # низ НЕ фризим — геймплей едет дальше всегда
            g_frac += g_step
            n_adv, g_frac = int(g_frac), g_frac - int(g_frac)
            gok, gfr = False, None
            for _ in range(max(n_adv, 1)):  # кольцо при конце файла
                gok, gfr = gcap.read()
                if not gok:
                    gcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    gok, gfr = gcap.read()
                    if not gok:
                        break
            if not gok:
                if last_bot is None:
                    break
                gfr = last_bot
            else:
                gfr = cv2.cvtColor(gfr, cv2.COLOR_BGR2RGB)
            last_bot = gfr
            # медленный дрейф окна по ВРЕМЕНИ КОНТЕНТА — во фризе стоит вместе с картинкой
            drift = (ct_rel * 12) % 400
            bot = _cover_resize(gfr, OUT_W, BOT_H, cx=0.5, cy=0.4, drift_y=drift)

            frame = cv2.vconcat([top, bot])
            banner_on = False
            if frozen:
                if last_win < 0 or t_rel < wins[last_win][0] or t_rel > wins[last_win][1]:
                    for wi, (oA, oB, _cA) in enumerate(wins):
                        if oA <= t_rel <= oB:
                            last_win = wi
                            break
                    bcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    b_frac = 0.0
                b_frac += b_step
                n_adv, b_frac = int(b_frac), b_frac - int(b_frac)
                bok, bfr = False, None
                for _ in range(max(n_adv, 1)):
                    bok, bfr = bcap.read()
                    if not bok:
                        break
                if bok and bfr is not None:
                    # баннер висит прямо под фильмом (зазор 8px), от края до края
                    frame = _chroma_overlay(frame, bfr, y_top=TOP_H + BANNER_GAP,
                                            zoom=BANNER_ZOOM)
                    banner_on = True
            if captions and not banner_on:
                words = _active_words(segs, t)
                if words:
                    ch, act = _caption_states(words, t)
                    if ch:
                        img = Image.fromarray(frame)
                        dr = ImageDraw.Draw(img)
                        lines = _split_lines(dr, font, ch, OUT_W - 140)
                        hs = []
                        for ln in lines:
                            txt = " ".join(w["w"] for w in ln)
                            bbox = dr.textbbox((0, 0), txt, font=font, stroke_width=3)
                            hs.append(bbox[3] - bbox[1])
                        block = sum(hs) + 8 * (len(lines) - 1)
                        y = min(seam - 36 - block, 1560)
                        for li, ln in enumerate(lines):
                            txt = " ".join(w["w"] for w in ln)
                            bbox = dr.textbbox((0, 0), txt, font=font, stroke_width=3)
                            tw = bbox[2] - bbox[0]
                            x = (OUT_W - tw) / 2
                            for wi, w in enumerate(ln):
                                pre = " ".join(v["w"] for v in ln[:wi])
                                pre = (pre + " ") if pre else ""
                                bx = dr.textbbox((0, 0), pre, font=font, stroke_width=3)
                                wx = x + (bx[2] - bx[0])
                                col = "#FFD60A" if wi == act else "white"
                                dr.text((wx, y), w["w"], font=font, fill=col,
                                        stroke_width=3, stroke_fill="black")
                            y += hs[li] + 8
                        frame = __import__("numpy").asarray(img)
            proc.stdin.write(frame.tobytes())
            if progress and k % 15 == 0:
                progress(k / total)
    finally:
        cap.release()
        gcap.release()
        if bcap is not None:
            bcap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=600)
        if mix_tmp:
            try:
                if os.path.exists(mix_tmp):
                    os.remove(mix_tmp)
            except OSError:
                pass
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("рендер не удался")
    return out


# ---------- 4. Рендер вертикали + субтитры ----------

def _font(size: int):
    from PIL import ImageFont
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _active_words(segs, t: float):
    for s in segs:
        if s["start"] - 0.15 <= t <= s["end"] + 0.4 and s.get("words"):
            return s["words"]
    return None


def _split_lines(dr, font, words, max_w: int):
    """Строки по реальной ширине: жадно добиваем до max_w, максимум 2 строки."""
    def _w(t):
        bb = dr.textbbox((0, 0), t, font=font, stroke_width=3)
        return bb[2] - bb[0]
    lines, line, wdt = [], [], 0
    for w in words:
        ww = _w(w["w"])
        add = ww + (_w(" ") if line else 0)
        if line and wdt + add > max_w:
            lines.append(line)
            if len(lines) >= 2:
                break
            line, wdt = [w], ww
        else:
            line.append(w)
            wdt += add
    if line and len(lines) < 2:
        lines.append(line)
    return lines[:2]


def _caption_states(words, t: float, per_line: int = 4):
    """Чанки по per_line слов; возвращает (показанные слова, индекс активного)."""
    if not words:
        return [], -1
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= per_line or re.search(r"[.!?…]$", w["w"]):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    for ch in chunks:
        if ch[0]["s"] - 0.15 <= t <= ch[-1]["e"] + 0.4:
            act = -1
            for k, w in enumerate(ch):
                if w["s"] - 0.1 <= t <= w["e"] + 0.15:
                    act = k
            return ch, act
    return [], -1


def fast_cut(src: str, moment: dict, out: str):
    """Мгновенная нарезка без перекодирования: исходное качество и кодек, как есть 16:9."""
    start, dur = float(moment["start"]), float(moment["end"]) - float(moment["start"])
    if dur < 1:
        raise RuntimeError("момент короче секунды")
    subprocess.run([FFMPEG, "-y", "-v", "error",
                    "-ss", str(round(start, 2)), "-t", str(round(dur, 2)), "-i", src,
                    "-c", "copy", "-movflags", "+faststart", out],
                   timeout=600, check=True)
    if not os.path.exists(out):
        raise RuntimeError("нарезка не удалась")
    return out


def render_clip(src: str, moment: dict, segs, out: str, facetrack: bool = True,
                captions: bool = True, progress=None, vmode: str = "face",
                antidetect: bool = False, mirror: bool = False,
                banner: bool = True,
                max_total: float | None = None):
    if vmode == "split":
        return render_split_clip(src, moment, segs, out, facetrack=facetrack,
                                 captions=captions, progress=progress,
                                 antidetect=antidetect, mirror=mirror,
                                 banner=banner,
                                 max_total=max_total)
    import cv2
    from PIL import Image, ImageDraw
    start, end = float(moment["start"]), float(moment["end"])
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fps = min(float(fps), 30)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if nfr and fps > 0:
        end = min(end, nfr / fps)  # момент не может вылезти за конец видео
    if end - start < 1:
        cap.release()
        raise RuntimeError("момент за пределами видео")
    dur = end - start
    crop_w = int(H * 9 / 16)
    track = track_faces(src, start, end) if facetrack and vmode == "face" and W > crop_w else []
    font = _font(62)

    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1920", "-r", str(int(fps)),
           "-i", "pipe:0",
           "-ss", str(start), "-t", str(round(dur, 2)), "-i", src,
           "-map", "0:v", "-map", "1:a?"]
    qvf = _quality_vf()
    if antidetect:
        vf, af = _antidetect_filters()
        qvf += "," + vf
        cmd += ["-vf", qvf, "-af", af]
    else:
        cmd += ["-vf", qvf]
    cmd += _video_codec_args() + ["-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
            "-t", str(round(dur, 2)), out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    _warm_start(cap)  # первый кадр не чёрный
    total = max(int(round(dur * fps)), 1)
    last = None
    try:
        for k in range(total):
            ok, fr = cap.read()
            if not ok:
                if last is None:
                    break
                fr = last  # дырка в декодере — повторяем кадр, длина точная
            else:
                if mirror:
                    fr = cv2.flip(fr, 1)
                if antidetect:  # наезд 4%: прячет края, лого и ватермарки
                    h0, w0 = fr.shape[:2]
                    fr = cv2.resize(fr, (int(w0 * 1.04), int(h0 * 1.04)),
                                    interpolation=cv2.INTER_LANCZOS4)
                    dx, dy = (fr.shape[1] - w0) // 2, (fr.shape[0] - h0) // 2
                    fr = fr[dy:dy + h0, dx:dx + w0]
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            last = fr
            t = start + k / fps
            if vmode == "blur":
                # размытый фон + резкий центр 16:9 (сериал: в кадре часто несколько человек)
                small = cv2.resize(fr, (270, 480), interpolation=cv2.INTER_AREA)
                small = cv2.GaussianBlur(small, (0, 0), 12)
                bg = cv2.resize(small, (1080, 1920), interpolation=cv2.INTER_LINEAR)
                bg = cv2.convertScaleAbs(bg, alpha=0.72)
                fh = int(1080 * H / max(W, 1))
                fg = cv2.resize(fr, (1080, fh), interpolation=cv2.INTER_LANCZOS4)
                y0 = (1920 - fh) // 2
                frame = bg
                frame[y0:y0 + fh] = fg
                cap_y = y0 + fh + 48
            else:
                cx = face_x_at(track, t)
                if mirror:
                    cx = 1.0 - cx
                x0 = int(min(max(cx * W - crop_w / 2, 0), max(W - crop_w, 0)))
                crop = fr[:, x0:x0 + crop_w] if W > crop_w else fr
                frame = cv2.resize(crop, (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
                cap_y = 1230
            if captions:
                words = _active_words(segs, t)
                if words:
                    ch, act = _caption_states(words, t)
                    if ch:
                        img = Image.fromarray(frame)
                        dr = ImageDraw.Draw(img)
                        lines = _split_lines(dr, font, ch, OUT_W - 140)
                        y = min(cap_y, 1560)
                        for ln in lines:
                            txt = " ".join(w["w"] for w in ln)
                            bbox = dr.textbbox((0, 0), txt, font=font, stroke_width=3)
                            tw = bbox[2] - bbox[0]
                            x = (1080 - tw) / 2
                            # сначала обычные, потом активное поверх
                            for wi, w in enumerate(ln):
                                pre = " ".join(v["w"] for v in ln[:wi])
                                pre = (pre + " ") if pre else ""
                                bx = dr.textbbox((0, 0), pre, font=font, stroke_width=3)
                                wx = x + (bx[2] - bx[0])
                                col = "#FFD60A" if wi == act else "white"
                                dr.text((wx, y), w["w"], font=font, fill=col,
                                        stroke_width=3, stroke_fill="black")
                            y += (bbox[3] - bbox[1]) + 8
                        frame = __import__("numpy").asarray(img)
            proc.stdin.write(frame.tobytes())
            if progress and k % 15 == 0:
                progress(k / total)
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=600)
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("рендер не удался")
    return out
