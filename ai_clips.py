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
GAMEPLAYS = {
    "subway": {"file": "subway.mp4",
               "query": "ytsearch1:subway surfers gameplay 1 hour no commentary"},
    "minecraft": {"file": "minecraft.mp4",
                  "query": "ytsearch1:minecraft parkour gameplay no commentary"},
}
GAMEPLAY_FILE = os.path.join(GAMEPLAY_DIR, "subway.mp4")

# Сплит-раскладка 1080x1920: верх сериал 40% / низ геймплей 60%
OUT_W, OUT_H = 1080, 1920
TOP_H = int(OUT_H * 0.4)          # 768
BOT_H = OUT_H - TOP_H             # 1152

# Первые секунды видео — заставки студий/логотипы: моменты раньше не начинаем
MIN_START = 15.0

# Баннер: вставка строго по середине итога, ЦЕЛИКОМ без кропа, ускорен x1.4,
# по центру, фон выбит хромакеем (≥25% экрана — норма рекламодателя)
BANNER_FILE = os.path.join(BASE, "assets", "banner.mp4")
BANNER_SPEED = 1.4
BANNER_W = 1022  # бокс 1022×537 = 26.5% экрана (норма рекламодателя 25%+)
BANNER_ZOOM = 1.0  # без наезда: весь баннер виден целиком, ничего не режем
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
    """Железо (VideoToolbox) даёт скорость: клип 50с кодируется ~1 мин.
    Битрейт держим TikTok-дружелюбным 6-8 Мбит VBR: файл 50с весит
    ~35-50 МБ вместо 90-110, быстрее грузится и не застревает
    в транскодинге TikTok (одна из причин вечных 0 просмотров).
    Джиттер битрейта + случайный GOP — плюс к уникальности отпечатков."""
    import random as _r
    if _VT:
        br = _r.randint(6, 8)
        return ["-c:v", "h264_videotoolbox", "-pix_fmt", "yuv420p",
                "-b:v", f"{br}M", "-maxrate", f"{br + 3}M",
                "-bufsize", f"{2 * br}M", "-allow_sw", "1",
                "-g", str(_r.randint(48, 72))]
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(_r.randint(20, 22)),
            "-preset", "fast", "-tune", "film",
            "-g", str(_r.randint(48, 72))]


# Единая громкость под ленту: тихие сериалы подтягиваем к уровню TikTok.
_LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"
# ВАЖНО: loudnorm без финального ресемпла поднимает выход до 96/192 кГц —
# такие файлы TikTok плохо перекодирует (вечные 0 просмотров без нарушений).
# Поэтому везде после loudnorm добавляем возврат на 48 кГц стерео.
_TIKTOK_AF_TAIL = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"


def _tiktok_af(*parts: str) -> str:
    """Собирает аудиоцепочку, гарантируя 48кГц на выходе.
    Принимает кусочки (antidetect, loudnorm...), в конец дописывает ресемпл."""
    chain = ",".join(p for p in parts if p)
    if "loudnorm" in chain and "aresample=48000" not in chain:
        chain = chain + "," + _TIKTOK_AF_TAIL
    return chain


# Финальные аудио-аргументы кодирования: TikTok ждёт 48кГц AAC.
_AUDIO_OUT = ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]


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


_DREX_BATCH = 160  # вопросов на вызов: короткие батчи отвечают точнее


def _base_questions(tag: str, domain: str = "series") -> dict:
    """Тур 1: 4 базовых измерения момента. domain=streams — критерии под
    стримы (фраги, клатчи, фейлы), series — под сериалы. Движок один."""
    if domain == "streams":
        return {
            "viral": {
                "type": "noul",
                "instructions": f"Moment {tag} will go viral on TikTok.",
                "criteria": {
                    "true": ("Opens with a hook in the first 2 seconds: insane flick, "
                             "impossible clutch, hilarious fail, scream reaction, or trash-talk. "
                             "Visible high emotion (hype, shock, rage, laughter) or jaw-dropping "
                             "skill. A quotable moment viewers will clip and share. "
                             "Enjoyable with zero context."),
                    "false": ("Lobby waiting, load screens, settings menus, buying phase, "
                              "quiet looting with nothing happening, stream starting soon screens, "
                              "or a moment only fans of this streamer get.")}},
            "hook": {
                "type": "noul",
                "instructions": f"Moment {tag} opens with a strong hook that stops scrolling.",
                "criteria": {
                    "true": ("First seconds explode: kill feed pops, clutch begins, fail unfolds, "
                             "streamer screams or laughs — impossible to scroll past."),
                    "false": ("Starts with menu navigation, load screens, quiet looting, "
                              "streamer silent and nothing happening, or mid-round calm.")}},
            "whole": {
                "type": "noul",
                "instructions": f"Moment {tag} is a complete mini-story, not cut off mid-sentence.",
                "criteria": {
                    "true": ("Self-contained: round or fight starts and resolves, punchline lands, "
                             "reaction completes, or a deliberate cliffhanger before the outcome."),
                    "false": ("Breaks off mid-fight, mid-sentence, round continues past the cut, "
                              "or starts mid-action with no setup.")}},
            "peak": {
                "type": "noul",
                "instructions": f"Moment {tag} has an emotional peak, plot twist or cliffhanger ending.",
                "criteria": {
                    "true": ("Peak moment: multi-kill, 1vX clutch, insane flick, epic fail, "
                             "or a cut right before the outcome — viewer must watch part 2."),
                    "false": ("Flat round with no kills, no fails, nothing happens, "
                              "everything routine.")}},
        }
    return {
        "viral": {
            "type": "noul",
            "instructions": f"Moment {tag} will go viral on TikTok.",
            "criteria": {
                "true": ("Opens with a hook in the first 2 seconds: shocking accusation, "
                         "exposed secret, threat, ultimatum, love confession, or a burning "
                         "question. Visible strong emotion (anger, tears, jealousy, triumph) "
                         "or open conflict/scandal. A quotable punchline viewers will repeat "
                         "in comments. A clear story turn or payoff happens inside the moment — "
                         "enjoyable with zero prior context."),
                "false": ("Opens with greetings, filler, everyday small talk, scene-setting, "
                          "recap of previous events, credits talk, background chatter, "
                          "or a moment that only makes sense knowing prior episodes.")}},
        "hook": {
            "type": "noul",
            "instructions": f"Moment {tag} opens with a strong hook that stops scrolling.",
            "criteria": {
                "true": ("The very first phrase is a shock, secret, threat, accusation, "
                         "ultimatum, confession, insult, or a question that demands an answer — "
                         "a scroller cannot pass it by."),
                "false": ("Starts with greetings, filler openers, everyday chatter, "
                          "scene-setting, or a sentence visibly continued from before "
                          "(starts mid-thought).")}},
        "whole": {
            "type": "noul",
            "instructions": f"Moment {tag} is a complete mini-story, not cut off mid-sentence.",
            "criteria": {
                "true": ("Self-contained: ends on a finished sentence, punchline, decision, "
                         "reaction line, or a deliberate cliffhanger pause."),
                "false": ("Breaks off mid-word, mid-sentence, in the middle of an explanation, "
                          "or right as a new thought begins.")}},
        "peak": {
            "type": "noul",
            "instructions": f"Moment {tag} has an emotional peak, plot twist or cliffhanger ending.",
            "criteria": {
                "true": ("Betrayal reveal, fight, slap, confession, unexpected entrance, "
                         "plot twist, or a cut right before the answer — "
                         "viewer must watch part 2."),
                "false": ("Flat emotional line from start to finish, everything resolved "
                          "and explained, nothing left hanging.")}},
    }


def _extra_questions(tag: str) -> dict:
    """Тур 2 (финал, только шорт-лист): 3 точных измерения для разводки близких."""
    return {
        "punch": {
            "type": "noul",
            "instructions": f"Moment {tag} has a quotable punchline.",
            "criteria": {
                "true": ("A single striking phrase viewers will quote in comments and captions: "
                         "a brutal truth, insult, confession, threat, or ultimatum."),
                "false": ("Flat everyday wording, nothing memorable, nothing worth quoting.")}},
        "clarity": {
            "type": "noul",
            "instructions": f"Moment {tag} is fully clear to a first-time viewer.",
            "criteria": {
                "true": ("Zero context needed: anyone instantly gets WHAT is happening "
                         "and WHY it matters."),
                "false": ("Requires knowing the characters, prior episodes, "
                          "or off-screen events to understand.")}},
        "comment": {
            "type": "noul",
            "instructions": f"Moment {tag} will spark arguments in comments.",
            "criteria": {
                "true": ("Moral dilemma, betrayal, shocking behavior, hot take — "
                         "viewers will pick sides and fight in comments."),
                "false": ("Nothing to argue about; neutral everyday scene.")}},
    }


def _ask_drex(key: str, states: list, per_chunk: list, workers: int = 8) -> dict:
    """Параллельные батчи вопросов по чанкам. Упавший батч = ответы 0.5
    (берёт дефолт в скоринге), а не смерть всего прогона."""
    batches = []
    for ci, qs in enumerate(per_chunk):
        if not qs:
            continue
        keys = list(qs)
        for b in range(0, len(keys), _DREX_BATCH):
            batches.append((ci, {k: qs[k] for k in keys[b:b + _DREX_BATCH]}))

    def _one(job):
        ci, sub = job
        try:
            return _drex_call(key, states[ci], sub).get("answers", {})
        except Exception:
            return {}

    ans = {}
    if not batches:
        return ans
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as ex:
        for part in ex.map(_one, batches):
            ans.update(part)
    return ans


def _drex_score(segs, key: str, want: int, min_len: int, max_len: int,
                scenes=None, domain: str = "series"):
    """Кандидаты-окна -> DREX тур 1 (4 меры) -> шорт-лист -> DREX тур 2
    (панч/ясность/дискуссия) -> скор 0-100. Без текстовой генерации.
    Отбор — только DREX: качество из плотной сетки, перекрывающихся чанков
    (моменты через границу не теряются), меток сцен и пауз в контексте."""
    if not segs:
        return []
    # --- 1. глобальная сетка окон: каждый сегмент — старт, окна через
    # границы чанков разрешены (привязка к чанкам — только для контекста) ---
    per_start = 10 if len(segs) <= 300 else (6 if len(segs) <= 600 else 4)
    wins, seen = [], set()
    for i in range(len(segs)):
        base = float(segs[i]["start"])
        cands_j = []
        j = i
        while j < len(segs) and float(segs[j]["end"]) - base <= max_len:
            if float(segs[j]["end"]) - base >= min_len * 0.7:
                cands_j.append(j)
            j += 1
        if len(cands_j) > per_start:
            idx = sorted(set(round(k * (len(cands_j) - 1) / (per_start - 1))
                             for k in range(per_start)))
            cands_j = [cands_j[k] for k in idx]
        for j in cands_j:
            s0, e0 = base, float(segs[j]["end"])
            kk = (round(s0, 1), round(e0, 1))
            if kk in seen:
                continue
            seen.add(kk)
            wins.append((s0, e0, i, j))
    if not wins:
        return []
    # --- 2. перекрывающиеся чанки контекста (18к символов, нахлёст 5к) ---
    def _slen(s):
        return len(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}")
    chunks = []
    cur, cur_chars = [], 0
    for s in segs:
        cur.append(s)
        cur_chars += _slen(s)
        if cur_chars >= 18000:
            chunks.append(cur)
            tail, tc = [], 0
            for x in reversed(cur):
                tail.append(x)
                tc += _slen(x)
                if tc >= 5000:
                    break
            tail.reverse()
            cur, cur_chars = tail, tc
    if cur and (not chunks or cur[0] is not chunks[-1][0]):
        chunks.append(cur)
    if not chunks:
        return []
    spans = [(float(c[0]["start"]), float(c[-1]["end"])) for c in chunks]
    # --- 3. состояния: метки новых сцен + длинные паузы ---
    starts = sorted(float(s[0]) for s in (scenes or [])) if scenes else []
    pos_of = {id(s): n for n, s in enumerate(segs)}
    states = []
    for c in chunks:
        lines = []
        for s in c:
            pre = ""
            st = float(s["start"])
            if starts and min((abs(st - x) for x in starts), default=99.0) <= 2.0:
                pre = "▶ НОВАЯ СЦЕНА "
            p = pos_of.get(id(s), 0)
            if p > 0:
                gap = st - float(segs[p - 1]["end"])
                if gap > 1.5:
                    lines.append(f"…пауза {gap:.0f}с…")
            lines.append(f"{pre}[{s['start']:.1f}-{s['end']:.1f}] {s['text']}")
        states.append("\n".join(lines))

    def _assign(s0: float, e0: float) -> int:
        best, best_span = -1, None
        for n, (a, b) in enumerate(spans):
            if a <= s0 + 0.01 and b >= e0 - 0.01:
                if best_span is None or (b - a) < best_span:
                    best, best_span = n, (b - a)
        if best >= 0:
            return best
        ov, bn = -1.0, 0
        for n, (a, b) in enumerate(spans):
            o = min(b, e0) - max(a, s0)
            if o > ov:
                ov, bn = o, n
        return bn

    # --- 4. тур 1: базовые 4 меры по всем окнам ---
    per_chunk = [dict() for _ in chunks]
    wchunk = []
    for k, (s0, e0, _i, _j) in enumerate(wins):
        n = _assign(s0, e0)
        wchunk.append(n)
        tag = f"[{s0:.1f}-{e0:.1f}]"
        for sk, q in _base_questions(tag, domain).items():
            per_chunk[n][f"w{k}_{sk}"] = q
    ans1 = _ask_drex(key, states, per_chunk)
    base = []
    for k, (s0, e0, _i, _j) in enumerate(wins):
        def _p(suffix, _k=k):
            a = (ans1.get(f"w{_k}_{suffix}") or {})
            return float(a.get("noul", 0.5))
        pv, ph, pw, pk = _p("viral"), _p("hook"), _p("whole"), _p("peak")
        base.append((k, 0.45 * pv + 0.20 * ph + 0.15 * pw + 0.20 * pk,
                     (pv, ph, pw, pk)))
    # --- 5. тур 2: шорт-лист + 3 точные меры, финал 70/30 ---
    short = sorted(base, key=lambda r: -r[1])[:max(want * 6, 30)]
    per_chunk2 = [dict() for _ in chunks]
    for rank, (k, _bv, _bp) in enumerate(short):
        s0, e0, _i, _j = wins[k]
        tag = f"[{s0:.1f}-{e0:.1f}]"
        for sk, q in _extra_questions(tag).items():
            per_chunk2[wchunk[k]][f"f{rank}_{sk}"] = q
    ans2 = _ask_drex(key, states, per_chunk2)
    out = []
    for rank, (k, bv, (pv, ph, pw, pk)) in enumerate(short):
        def _q(suffix, _r=rank):
            a = (ans2.get(f"f{_r}_{suffix}") or {})
            return float(a.get("noul", 0.5))
        pp, pc, pm = _q("punch"), _q("clarity"), _q("comment")
        extra = (pp + pc + pm) / 3.0
        score = int(round(100 * (0.70 * bv + 0.30 * extra)))
        s0, e0, i, j = wins[k]
        txt = " ".join(s["text"] for s in segs[i:j + 1])
        title = " ".join(txt.split()[:9]) + "…"
        out.append({
            "start": s0, "end": e0, "score": score, "title": title[:120],
            "verdict": (f"Drex {score}/100: виральность {pv:.2f}, "
                        f"хук {ph:.2f}, цельность {pw:.2f}, пик {pk:.2f}, "
                        f"панч {pp:.2f}, ясность {pc:.2f}, дискуссия {pm:.2f}."),
            "hook": "",
        })
    return sorted(out, key=lambda m: -m["score"])[:want * 6 + 20]


def _drex_direct(segs, key: str, want: int, min_len: int, max_len: int,
                 scenes=None):
    """Прямой режим: DREX читает пронумерованные строки транскрипта и сам
    метит границы — open (здесь начинается виральное) и close (здесь резать).
    Код не предлагает окна, а только режет там, где указал DREX."""
    if not segs:
        return []

    def _slen(s):
        return len(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}")

    chunks = []
    cur, cur_chars = [], 0
    for s in segs:
        cur.append(s)
        cur_chars += _slen(s)
        if cur_chars >= 18000:
            chunks.append(cur)
            tail, tc = [], 0
            for x in reversed(cur):
                tail.append(x)
                tc += _slen(x)
                if tc >= 5000:
                    break
            tail.reverse()
            cur, cur_chars = tail, tc
    if cur and (not chunks or cur[0] is not chunks[-1][0]):
        chunks.append(cur)
    if not chunks:
        return []
    starts = sorted(float(s[0]) for s in (scenes or [])) if scenes else []
    pos_of = {id(s): n for n, s in enumerate(segs)}
    states = []
    for c in chunks:
        lines = []
        for s in c:
            n = pos_of.get(id(s), 0)
            pre = ""
            st = float(s["start"])
            if starts and min((abs(st - x) for x in starts), default=99.0) <= 2.0:
                pre = "▶ НОВАЯ СЦЕНА "
            if n > 0:
                gap = st - float(segs[n - 1]["end"])
                if gap > 1.5:
                    lines.append(f"…пауза {gap:.0f}с…")
            lines.append(f"L{n} {pre}[{s['start']:.1f}-{s['end']:.1f}] {s['text']}")
        states.append("\n".join(lines))
    per_chunk = [dict() for _ in chunks]
    for ci, c in enumerate(chunks):
        for s in c:
            n = pos_of.get(id(s), 0)
            lid = f"L{n}"
            per_chunk[ci][f"{lid}_open"] = {
                "type": "noul",
                "instructions": f"Line {lid} OPENS a viral TikTok moment.",
                "criteria": {
                    "true": ("Right here starts something viral: a shocking accusation, exposed "
                             "secret, threat, ultimatum, love confession, insult, or a burning "
                             "question — the exact phrase that stops scrolling. Emotion or "
                             "conflict ignites within the next 1-2 lines."),
                    "false": ("Greetings, filler, small talk, scene-setting, recap, or a sentence "
                              "continued from before. Nothing viral starts here.")}}
            per_chunk[ci][f"{lid}_open2"] = {
                "type": "noul",
                "instructions": f"Line {lid} is a scroll-stopper opening.",
                "criteria": {
                    "true": ("The first phrase shocks, intrigues, or provokes — the viewer must "
                             "hear what comes next. Instant curiosity spike."),
                    "false": ("Mundane opening: no curiosity spike, easy to swipe past.")}}
            per_chunk[ci][f"{lid}_close"] = {
                "type": "noul",
                "instructions": f"Line {lid} ENDS a viral TikTok moment.",
                "criteria": {
                    "true": ("The payoff lands here: punchline, confession, reveal, decision, "
                             "plot twist, or a question left hanging for part 2. "
                             "A perfect cut point right after this line."),
                    "false": ("The thought continues past this line, the explanation is unfinished, "
                              "or it is a flat filler ending. Cutting here would break the moment.")}}
            per_chunk[ci][f"{lid}_close2"] = {
                "type": "noul",
                "instructions": f"Line {lid} is a good cut point.",
                "criteria": {
                    "true": ("Ends on punch, twist, decision, or an open question — cutting here "
                             "keeps the tension at maximum."),
                    "false": ("Cuts mid-thought or lands on flat filler.")}}
    ans = _ask_drex(key, states, per_chunk)
    def _qq(n, *sfx):
        vs = [float((ans.get(f"L{n}_{s}") or {}).get("noul", 0.0)) for s in sfx]
        return sum(vs) / len(vs)
    if ENSEMBLE_BOUNDS:
        po = [_qq(n, "open", "open2") for n in range(len(segs))]
        pc = [_qq(n, "close", "close2") for n in range(len(segs))]
    else:
        po = [_qq(n, "open") for n in range(len(segs))]
        pc = [_qq(n, "close") for n in range(len(segs))]
    opens = sorted([i for i in range(len(segs)) if po[i] >= DIRECT_OPEN_T],
                   key=lambda i: -po[i])
    taken, out = [], []
    for i in opens:
        s0 = float(segs[i]["start"])
        if any(a - 0.01 <= s0 <= b + 0.01 for a, b in taken):
            continue
        best, bestp = -1, -1.0
        for j in range(i, len(segs)):
            e0 = float(segs[j]["end"])
            dur = e0 - s0
            if dur > max_len:
                break
            if dur >= min_len * 0.7 and pc[j] >= DIRECT_CLOSE_T and pc[j] > bestp:
                best, bestp = j, pc[j]
        if best < 0:
            continue
        e0 = float(segs[best]["end"])
        if any(not (e0 <= a or s0 >= b) for a, b in taken):
            continue
        taken.append((s0, e0))
        score = int(round(100 * (0.55 * po[i] + 0.45 * bestp)))
        txt = " ".join(s["text"] for s in segs[i:best + 1])
        title = " ".join(txt.split()[:9]) + "…"
        out.append({
            "start": s0, "end": e0, "score": score, "title": title[:120],
            "verdict": (f"Drex {score}/100: открытие {po[i]:.2f}, "
                        f"концовка {bestp:.2f}."),
            "hook": "",
        })
    return sorted(out, key=lambda m: -m["score"])[:want * 6 + 20]


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


_LORE = None


def _lore_names():
    """Имена персонажей (lore/*.json): моменты с героями получают бонус —
    зритель идёт на знакомые лица."""
    global _LORE
    if _LORE is None:
        _LORE = set()
        try:
            d = os.path.join(BASE, "lore")
            for f in os.listdir(d):
                if f.endswith(".json"):
                    data = json.load(open(os.path.join(d, f), encoding="utf-8"))
                    for c in data.get("characters", []):
                        for a in [c.get("name", "")] + c.get("aliases", []):
                            a = (a or "").strip().lower()
                            if len(a) > 2:
                                _LORE.add(a)
        except Exception:
            pass
    return _LORE


def _segs_text(segs, s: float, e: float) -> str:
    return " ".join(x.get("text", "") for x in segs
                     if x.get("end", 0) > s - 0.5 and x.get("start", 0) < e + 0.5)


_REFS_CACHE = (0, None)


def _ref_boosts():
    """Профиль залётных референсов: медианная длина + топ-слова хуков.
    Кэш на 60с. Возвращает (ref_dur|None, [слова])."""
    import re
    import sqlite3
    import time
    global _REFS_CACHE
    now = time.time()
    if now - _REFS_CACHE[0] < 60 and _REFS_CACHE[1] is not None:
        return _REFS_CACHE[1]
    out = (None, [])
    try:
        c = sqlite3.connect(os.path.join(BASE, "data.db"), timeout=10)
        try:
            rows = c.execute("SELECT dur, hook, top FROM refs").fetchall()
        finally:
            c.close()
        if rows:
            ds = sorted(r[0] for r in rows if r[0] and r[0] > 0)
            ref_dur = ds[len(ds) // 2] if ds else None
            stop = {"это", "как", "что", "для", "так", "уже", "если", "когда",
                    "или", "его", "ее", "её", "меня", "тебя", "себя", "просто",
                    "очень", "даже", "только", "потом", "зачем", "почему"}
            freq = {}
            for r in rows:
                for t in re.findall(r"[а-яА-ЯёЁa-zA-Z]{4,}", f"{r[1] or ''} {r[2] or ''}".lower()):
                    if t not in stop:
                        freq[t] = freq.get(t, 0) + 1
            words = sorted(freq, key=lambda w: -freq[w])[:25]
            out = (ref_dur, words)
    except Exception:
        pass
    _REFS_CACHE = (now, out)
    return out


def _tight_span(s: float, e: float, bounds):
    """Границы впритык к речи: ни секунды тишины по краям.
    Возвращает (ns, ne, coverage 0..1)."""
    inside = [(ws, we) for ws, we in bounds if we > s + 0.1 and ws < e - 0.1]
    if not inside:
        return s, e, 1.0
    ns = max(s, inside[0][0] - 0.15)
    ne = min(e, inside[-1][1] + 0.4)
    if ne - ns < 3:
        return s, e, 1.0
    speech = sum(min(we, ne) - max(ws, ns) for ws, we in inside)
    cov = max(0.0, min(1.0, speech / max(ne - ns, 0.01)))
    return ns, ne, cov


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


# ---------- 2.5. Эвристика виральности (текст + структура) ----------
# DREX смотрит смысл, но слеп к тикток-механике: первые 2 секунды решают,
# филлер в начале убивает досмотры, дубли одной шутки каннибализируют друг
# друга. Эвристика считает 0..100 и подмешивается к DREX (68/32), а MMR-отбор
# давит near-дубли по Жаккару.

_HEUR_TRIGGER = (
    "секрет", "тайн", "измен", "любовн", "развод", "свадьб", "беремен",
    "денег", "деньг", "долг", "полиц", "драк", "скандал", "вран", "врешь",
    "врёшь", "обман", "уходи", "уйди", "люблю", "ненавижу", "дурак",
    "идиот", "дебил", "шок", "правд", "призна", "сюрприз", "наслед",
    "увол", "босс", "начальник", "поцелу", "ночь", "никогда", "всегда",
    "убью", "боюсь", "стыд", "позор", "хватит", "замолчи", "тест",
    "врач", "больниц", "смерт", "умер", "вернул", "уехал", "прости",
    "прощай", "навсегда", "стоп", "смотри", "подожди",
    "кто отец", "как ты мог", "я всё знаю", "все знаю",
)
_HEUR_FILLER_START = re.compile(
    r"^\s*(ну|короче|типа|вот|значит|как бы|привет|здравствуй|"
    r"добрый (день|вечер|утро)|алло|ладно|слушай|давай|так)\b")
_HEUR_PAYOFF = (
    "продолжени", "часть 2", "части 2", "дальше", "узнаешь", "подожди",
    "стоп", "вдруг", "оказалось", "наконец", "а потом", "и тут",
    "что будет", "а если", "смотри", "а ты", "а он", "а она",
)
_HEUR_BROKEN_END = {"и", "а", "но", "что", "как", "не", "на", "в", "с",
                    "к", "от", "для", "или", "если", "когда", "потому",
                    "чтобы", "это", "же", "ли", "уже"}

# Переключатели отбора. По умолчанию — чистый DREX: эвристика и MMR-дедуп
# НЕ влияют на скор и отбор. Функции heuristic_moment_score /
# moment_similarity / fuse_scores остаются для анализа и тестов.
ENABLE_HEURISTIC = False  # True — подмешивать эвристику к DREX (68/32)
ENABLE_MMR = False        # True — давить near-дубли по Жаккару
# DREX сам метит границы моментов на строках транскрипта (open/close),
# код только режет там, где DREX указал. Сетка окон — лишь страховка,
# если прямых меток мало. DREX не умеет генерировать текст с таймкодами
# (decides instead of writing: только noul/choice/score), поэтому метки
# на пронумерованных строках — ближайший к «сам находит и режет» режим.
DIRECT_DREX = True
DIRECT_OPEN_T = 0.55   # порог «здесь начинается виральный момент»
DIRECT_CLOSE_T = 0.50  # порог «здесь он заканчивается»
ENSEMBLE_BOUNDS = True  # границы двумя формулировками, среднее (точнее)
TOURNAMENT = True       # pairwise-финал топа: DREX-choice «кто виральнее»
TITLE_PICK = False      # заголовок выбирает DREX-choice (кликбейт)
CTA_PART2 = False       # клиффхэнгер -> плашка «ЖДИ ЧАСТЬ 2» + метка ч.2


def heuristic_moment_score(text: str, dur: float = 30.0,
                           ref_dur: float | None = None) -> tuple:
    """Эвристический скор 0..100 + разбор по компонентам.
    text — полный текст момента, dur — длительность сек."""
    t = (text or "").strip()
    low = t.lower()
    if not t:
        return 0, {}
    head = low[:150]
    tail = low[-170:]
    words = re.findall(r"[а-яёa-z0-9]+", low)
    n_words = len(words) or 1

    # 1. хук первых ~2 сек (0-28)
    hook = 12.0  # база: нейтральное начало
    if "?" in head:
        hook += 10
    if "!" in head:
        hook += 6
    th = sum(1 for r in _HEUR_TRIGGER if r in head)
    hook += min(th * 7, 14)
    if _HEUR_FILLER_START.match(low):
        hook -= 12  # филлер/приветствие в начале — свайп
    if t and t[0].islower():
        hook -= 4  # начали с середины фразы
    hook = max(0.0, min(28.0, hook))

    # 2. эмоциональный пик (0-20)
    emo = min(low.count("!") * 4, 12)
    emo += min(low.count("?") * 2, 6)
    caps = sum(1 for w in re.findall(r"[А-ЯЁ]{4,}", t) if len(w) >= 4)
    emo += min(caps * 2, 6)
    emo_hits = sum(1 for r in ("люблю", "ненавиж", "урод", "шок",
                               "убью", "позор", "хватит") if r in low)
    emo += min(emo_hits * 3, 8)
    emo = max(0.0, min(20.0, emo + 4.0))  # +4 база за живую речь

    # 3. диалоговая динамика (0-14)
    n_sent = len(re.findall(r"[.!?…]+", t))
    if n_sent >= 8:
        dlg = 10.0
    elif n_sent >= 4:
        dlg = 8.0
    elif n_sent >= 2:
        dlg = 5.0
    else:
        dlg = 2.0  # монолог одной фразой — скучно
    wps = n_words / max(float(dur or 30.0), 1.0)
    if 1.8 <= wps <= 3.8:
        dlg += 4.0
    elif 1.2 <= wps <= 4.5:
        dlg += 2.0
    else:
        dlg -= 2.0  # тянучка или скороговорка
    dlg = max(0.0, min(14.0, dlg))

    # 4. цельность концовки (0-14)
    comp = 0.0
    stripped = t.strip()
    if stripped and stripped[-1] in ".!?…":
        comp += 8.0
        if stripped[-1] == "?":
            comp += 3.0  # вопрос в конце — ждём ответ
    last_w = (re.findall(r"[а-яёa-z0-9]+", low) or [""])[-1]
    if last_w in _HEUR_BROKEN_END or (stripped and stripped[-1] == ","):
        comp = 0.0  # обрыв на полуслове
    if t and t[0].isupper():
        comp += 3.0
    comp = max(0.0, min(14.0, comp))

    # 5. payoff / клиффхэнгер (0-14)
    pay = 0.0
    if any(r in tail for r in _HEUR_PAYOFF):
        pay += 7.0
    if any(r in tail for r in ("правд", "оказалось", "измен", "развод",
                               "беремен", "умер", "вернул", "прости")):
        pay += 7.0
    if stripped.endswith("…") or stripped.endswith("..."):
        pay += 3.0
    pay = max(0.0, min(14.0, pay + 2.0))  # +2 база

    # 6. длина (0-10): sweet spot = медиана залётных или 35с
    sweet = ref_dur if (ref_dur and ref_dur > 0) else 35.0
    dd = abs(float(dur or 30.0) - sweet)
    if dd <= 8:
        ln = 10.0
    elif dd <= 15:
        ln = 7.0
    elif dd <= 25:
        ln = 4.0
    else:
        ln = 1.0
    if (dur or 0) < 12:
        ln = 0.0
    total = hook + emo + dlg + comp + pay + ln
    parts = {"hook": round(hook, 1), "emo": round(emo, 1),
             "dialog": round(dlg, 1), "complete": round(comp, 1),
             "payoff": round(pay, 1), "length": round(ln, 1)}
    return int(round(max(0.0, min(100.0, total)))), parts


def moment_similarity(a: str, b: str) -> float:
    """Жаккар по словам 4+ букв (0..1). Дубли одной шутки — высокое значение."""
    ta = {w for w in re.findall(r"[а-яёa-z]{4,}", (a or "").lower())}
    tb = {w for w in re.findall(r"[а-яёa-z]{4,}", (b or "").lower())}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def fuse_scores(drex: float, heur: float) -> int:
    """68% DREX + 32% эвристика. При сильном расхождении (>30) тянем к
    среднему — один из скоров явно ошибся."""
    d, h = float(drex or 0), float(heur or 0)
    fused = 0.68 * d + 0.32 * h
    if abs(d - h) > 30:
        fused = 0.5 * fused + 0.5 * (d + h) / 2.0
    return int(round(max(0.0, min(100.0, fused))))


def _tournament_order(pool, segs, key, top_n: int = 12):
    """Пары топа: DREX-choice «какой момент наберёт больше просмотров».
    Порядок — по числу побед, скоры не трогаем (чистый DREX)."""
    import random
    top = list(pool[:max(min(top_n, len(pool)), 0)])
    if len(top) < 3:
        return pool
    texts = [_segs_text(segs, float(m["start"]), float(m["end"])) for m in top]
    states, per_chunk, pairs = [], [], []
    n = 0
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            A, B = (a, b) if random.random() < 0.5 else (b, a)
            states.append(
                f"Момент A [{float(top[A]['start']):.0f}с]: {texts[A][:600]}\n"
                f"Момент B [{float(top[B]['start']):.0f}с]: {texts[B][:600]}")
            per_chunk.append({f"m{n}_win": {
                "type": "choice",
                "instructions": "Which moment will get more views on TikTok?",
                "criteria": {"A": str(top[A].get("title", ""))[:80],
                             "B": str(top[B].get("title", ""))[:80]}}})
            pairs.append((A, B))
            n += 1
    ans = _ask_drex(key, states, per_chunk)
    wins = [0] * len(top)
    for k, (A, B) in enumerate(pairs):
        w = (ans.get(f"m{k}_win") or {}).get("choice", "A")
        wins[B if w == "B" else A] += 1
    scored = sorted(zip(top, wins),
                    key=lambda t: (t[1], float(t[0].get("score", 0))),
                    reverse=True)
    for m, w in zip(top, wins):
        m["_twins"] = w
    return [m for m, _ in scored] + list(pool[len(top):])


def _polish_picks(picked, segs, key):
    """На каждый взятый клип: DREX-choice заголовка (кликбейт) + noul
    клиффхэнгера (часть 2). Молчание DREX = оставляем как было."""
    if not picked:
        return
    states, per_chunk, metas = [], [], []
    for p in picked:
        txt = _segs_text(segs, float(p["start"]), float(p["end"])) or p.get("title", "")
        words = txt.split()
        cands = []
        t0 = (p.get("title", "") or "").rstrip("… ").strip()
        if t0:
            cands.append(t0[:80])
        if len(words) > 6:
            cands.append(" ".join(words[:14])[:80])
        if len(words) > 12:
            cands.append(" ".join(words[-10:])[:80])
        cands = [c for c in dict.fromkeys(c for c in cands if c.strip())]
        states.append(txt[:1500])
        qs = {}
        if TITLE_PICK and len(cands) >= 2:
            qs["ttl"] = {"type": "choice",
                         "instructions": "Which title gets the most clicks on TikTok for this moment?",
                         "criteria": {chr(65 + i): c for i, c in enumerate(cands[:4])}}
        if CTA_PART2:
            qs["cliff"] = {
                "type": "noul",
                "instructions": "This moment ends on an unresolved cliffhanger.",
                "criteria": {
                    "true": ("The answer comes in part 2: viewers must follow the profile for "
                             "the continuation. Ends on a question, shock freeze, or "
                             "'to be continued' tension."),
                    "false": ("Self-contained ending: resolved and explained, no part 2 needed.")}}
        per_chunk.append(qs)
        metas.append(cands[:4])
    namespaced = [{f"p{i}_{k}": v for k, v in qs.items()}
                  for i, qs in enumerate(per_chunk)]
    ans = _ask_drex(key, states, namespaced)
    for i, p in enumerate(picked):
        cands = metas[i]
        if TITLE_PICK and len(cands) >= 2:
            ch = (ans.get(f"p{i}_ttl") or {}).get("choice", "A")
            idx = max(0, min(ord(str(ch)[:1].upper()) - 65, len(cands) - 1))
            p["title"] = cands[idx][:120]
            p["hook"] = cands[idx][:40]
        if CTA_PART2:
            cp = float((ans.get(f"p{i}_cliff") or {}).get("noul", 0.0))
            cta = cp >= 0.65
            p["cta"] = cta
            p["of"] = 2 if cta else 1
            p["part"] = 1 if cta else 0
            if cta:
                p["verdict"] = (str(p.get("verdict", "")) + " Клиффхэнгер: жди часть 2.")[:300]


def pick_moments(segs, lang: str, key: str, want: int, min_len: int, max_len: int,
                 scenes=None, min_gap: float = 30.0, domain: str = "series"):
    if not key:
        raise RuntimeError("Нет DREX-ключа — вставь его в настройках")
    cands = []
    if DIRECT_DREX:
        # прямой режим: границы ставит сам DREX на строках транскрипта
        try:
            cands = _drex_direct(segs, key, want, min_len, max_len,
                                 scenes=scenes)
        except Exception:
            cands = []
    try:
        if len(cands) < max(want, 5):
            # страховка сеткой: добираем окнами, которых ещё нет
            have = {(round(float(m["start"]), 1), round(float(m["end"]), 1))
                    for m in cands}
            grid = _drex_score(segs, key, want, min_len, max_len,
                               scenes=scenes, domain=domain)
            for m in grid:
                kk = (round(float(m["start"]), 1), round(float(m["end"]), 1))
                if kk not in have:
                    have.add(kk)
                    cands.append(m)
            cands = sorted(cands, key=lambda m: -m.get("score", 0))
    except Exception as e:
        if not cands:
            raise RuntimeError(f"DREX недоступен: {e}") from e
    scorer = "drex"
    if not cands:
        raise RuntimeError("DREX вернул пусто — попробуй ещё раз")
    # Весь пул в отбор: сначала разводка по времени, качество — планкой внутри.
    # Два прохода: (зазор, пол) = (норм, 45) затем (узко, 35) — количество добивается.
    # v2: к DREX подмешана эвристика (68/32), отбор — MMR: качество минус
    # штраф за текстовую похожесть на уже взятые (дубли одной шутки).
    if ENABLE_HEURISTIC:
        try:
            _rd, _rw = _ref_boosts()
        except Exception:
            _rd, _rw = None, []
        for m in cands:
            try:
                _txt = _segs_text(segs, float(m["start"]), float(m["end"]))
            except Exception:
                _txt = m.get("title", "")
            _dur = max(float(m["end"]) - float(m["start"]), 1.0)
            try:
                _h, _ = heuristic_moment_score(
                    f"{m.get('title', '')} {_txt}", _dur, _rd)
            except Exception:
                _h = 50
            m["_heur"] = _h
            m["_full"] = _txt
            m["_fused"] = fuse_scores(m.get("score", 0), _h)
        pool = sorted(cands, key=lambda m: -m.get("_fused", m.get("score", 0)))
    else:
        # чистый DREX: исходный порядок по скору модели, без подмесов
        for m in cands:
            m["_heur"] = None
            m["_full"] = ""
            m["_fused"] = int(m.get("score", 0))
        pool = sorted(cands, key=lambda m: -m.get("score", 0))
    if not pool:
        raise RuntimeError("DREX вернул пусто — попробуй ещё раз")
    if TOURNAMENT and len(pool) >= 3:
        try:
            pool = _tournament_order(pool, segs, key)
        except Exception:
            pass
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
            fused0 = float(m.get("_fused", m.get("score", 0)))
            if fused0 < floor:
                continue
            s, e = float(m["start"]), float(m["end"])
            if s < MIN_START - 4.0 and e < MIN_START + 4.0:
                continue  # заставка/логотип в начале — пропускаем целиком
            if any(not (e + gap < p["start"] or s - gap > p["end"]) for p in picked):
                continue
            if ENABLE_MMR:
                # MMR: near-дубль уже взятого — пропускаем (Жаккар > 0.55)
                try:
                    _mtxt = f"{m.get('title', '')} {m.get('_full', '')}"
                    _maxsim = 0.0
                    for p in picked:
                        _s = moment_similarity(_mtxt, p.get("_full", ""))
                        if _s > _maxsim:
                            _maxsim = _s
                    if _maxsim > 0.55:
                        continue
                except Exception:
                    _maxsim = 0.0
            else:
                _maxsim = 0.0
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
            s, e, cov = _tight_span(s, e, bounds)  # тишина по краям — вон
            if e - s < 5:
                continue
            if cov < 0.35:
                continue  # почти одно молчание — не сцена, а пауза
            if e - s < min_len:
                # дотягиваем до низа линзы по целым словам (середину слова не режем)
                need = min_len - (e - s)
                left = [ws for ws, we in bounds
                        if ws < s and (s - ws) <= need / 2 + 3.0]
                if left:
                    s = min(left)
                s = max(s, MIN_START)
                need2 = min_len - (e - s)
                right = [we for ws, we in bounds
                         if we > e and (we - e) <= need2 + 3.0]
                if right:
                    e = max(right)
                if segs:
                    e = min(e, float(segs[-1]["end"]))
                if e - s < 5:
                    continue
            if e - s > max_len:
                # верх линзы — урезаем по концу слова (итог с баннером влезет в 59)
                e = s + max_len
                cut = s + 5
                for ws, we in bounds:
                    if ws < s - 0.1:
                        continue
                    if we > e + 0.01:
                        break
                    cut = we
                e = max(cut, s + 5)
            sc = int(m.get("score", 70))
            if cov < 0.5:  # половина молчания — минус за рыхлость
                sc = max(0, sc - 8)
            # MMR-штраф за похожесть на взятое (только при ENABLE_MMR)
            if ENABLE_MMR:
                try:
                    _ms = float(_maxsim)
                    if _ms > 0.25:
                        sc = max(0, sc - int(round((_ms - 0.25) * 30)))
                except Exception:
                    pass
            names = _lore_names()
            hit = ""
            low = ""
            if names:
                low = (m.get("title", "") + " " + _segs_text(segs, s, e)).lower()
                for nm in names:
                    if nm in low:
                        hit = nm
                        break
            if hit:
                sc = min(97, sc + 6)  # знакомый герой в кадре = смотрят охотнее
            rd, rw = _ref_boosts()
            if rd and abs((e - s) - rd) <= 8:
                sc = min(97, sc + 4)  # длина как у залётных референсов
            if rw:
                if not low:
                    low = (m.get("title", "") + " " + _segs_text(segs, s, e)).lower()
                low_words = set(low.split())
                hits = sum(1 for w in rw if w in low_words)
                if hits:
                    sc = min(97, sc + min(6, hits * 2))  # слова из залётных хуков
            try:  # поправка выученного ранкера v2 (-10..+10), только если обучен:
                # use_prior=False — без обучения добавка 0, скор чисто DREX
                import learn as learn_mod
                sc = max(0, min(97, sc + learn_mod.adjust(
                    sc, m.get("verdict", ""),
                    m.get("title", ""), m.get("hook", "") or m.get("title", ""),
                    float(e) - float(s), use_prior=False)))
            except Exception:
                pass
            used.add(mi)
            picked.append({"start": float(round(s, 1)), "end": float(round(e, 1)),
                           "score": sc, "title": m.get("title", ""),
                           "verdict": str(m.get("verdict", ""))[:240],
                           "_full": str(m.get("_full", ""))[:500],
                           "hook": str(m.get("hook", "") or m.get("title", ""))[:40]})
            if len(picked) >= want:
                break
    if picked and key and (TITLE_PICK or CTA_PART2):
        try:
            _polish_picks(picked, segs, key)
        except Exception:
            pass
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
    """Возвращает [(t, cx_norm)] либо []. Трек — украшение: любая ошибка
    (нет детекторов в сборке cv2 и т.п.) даёт [] и центр-кадр, но не роняет рендер."""
    import cv2
    try:
        cap = cv2.VideoCapture(src)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        det = _yunet_detector(W, H)
        haar = None
        if det is None and hasattr(cv2, "CascadeClassifier"):
            try:
                haar = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                if haar is not None and haar.empty():
                    haar = None
            except Exception:
                haar = None
        if det is None and haar is None:
            cap.release()
            return []
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
                try:
                    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                    small = cv2.resize(g, (640, int(640 * H / W)))
                    fs = haar.detectMultiScale(small, 1.2, 4, minSize=(60, 60))
                except Exception:
                    fs = []
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
    except Exception:
        try:
            cap.release()
        except Exception:
            pass
        return []


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


_CAM_CACHE = {}


def detect_webcam(src: str, samples: int = 12):
    """Находит вебку стримера: статичная рамка + живое лицо внутри.
    Возвращает (x, y, w, h) в долях 0..1 либо None. Результат кешируется."""
    import cv2
    import numpy as np
    if src in _CAM_CACHE:
        return _CAM_CACHE[src]
    try:
        cap = cv2.VideoCapture(src)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 0
        if W <= 0 or H <= 0 or n <= 0 or fps <= 0:
            cap.release()
            return None
        dur = n / fps
        SW = 640
        SH = max(int(round(H * SW / max(W, 1))), 1)
        found = []
        for i in range(samples):
            t = dur * (0.05 + 0.9 * i / max(samples - 1, 1))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f1 = cap.read()
            if not ok or f1 is None:
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, (t + 0.5) * 1000)
            ok, f2 = cap.read()
            if not ok or f2 is None:
                f2 = f1
            small1 = cv2.resize(f1, (SW, SH), interpolation=cv2.INTER_AREA)
            small2 = cv2.resize(f2, (SW, SH), interpolation=cv2.INTER_AREA)
            g1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
            # рамка вебки — прямые горизонтали/вертикали (Хаф): рвётся реже контуров
            edges = cv2.Canny(g1, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70,
                                    minLineLength=int(SW * 0.10), maxLineGap=10)
            if lines is None:
                continue
            import math
            HH, VV = [], []
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                dx, dy = x2 - x1, y2 - y1
                if dx == 0 and dy == 0:
                    continue
                ang = abs(math.degrees(math.atan2(dy, dx))) % 180.0
                if ang < 6 or ang > 174:
                    HH.append((min(x1, x2), max(x1, x2), (y1 + y2) / 2))
                elif 84 < ang < 96:
                    VV.append((min(y1, y2), max(y1, y2), (x1 + x2) / 2))
            for i in range(len(HH)):
                for j in range(i + 1, len(HH)):
                    yA, yB = sorted([HH[i][2], HH[j][2]])
                    if not (0.06 * SH <= yB - yA <= 0.65 * SH):
                        continue
                    for k in range(len(VV)):
                        for l in range(k + 1, len(VV)):
                            xA, xB = sorted([VV[k][2], VV[l][2]])
                            if not (0.06 * SW <= xB - xA <= 0.65 * SW):
                                continue
                            if not (0.5 <= (xB - xA) / max(yB - yA, 1) <= 2.2):
                                continue
                            ok = True
                            for xa, xb in ((HH[i][0], HH[i][1]),
                                           (HH[j][0], HH[j][1])):
                                if min(xb, xB) - max(xa, xA) < (xB - xA) * 0.6:
                                    ok = False
                            for ya, yb in ((VV[k][0], VV[k][1]),
                                           (VV[l][0], VV[l][1])):
                                if min(yb, yB) - max(ya, yA) < (yB - yA) * 0.6:
                                    ok = False
                            if not ok:
                                continue
                            x, y, w, h = xA, yA, xB - xA, yB - yA
                            # живое лицо: движение внутри умеренное (не 0 и не каша)
                            xi, yi = max(int(x), 0), max(int(y), 0)
                            xe, ye = min(int(x + w), SW), min(int(y + h), SH)
                            if xe - xi < 8 or ye - yi < 8:
                                continue
                            roi1 = g1[yi:ye, xi:xe].astype(int)
                            roi2 = g2[yi:ye, xi:xe].astype(int)
                            mot = float(np.abs(roi1 - roi2).mean()) if roi1.size else 0.0
                            mot_s = max(0.0, 1.0 - abs(mot - 12.0) / 12.0) if mot < 40 else 0.0
                            border = 0.8  # рамка уже доказана линиями Хафа
                            found.append((x / SW, y / SH, w / SW, h / SH,
                                          border, mot_s))
        cap.release()
        # кластеры по положению/размеру: рамка стоит на месте весь стрим
        clusters = []
        for x, y, w, h, border, mot_s in found:
            hit = None
            for c in clusters:
                mx, my, mw, mh, n_c, b_c, m_c = c
                if (abs(x + w / 2 - (mx + mw / 2)) < 0.05
                        and abs(y + h / 2 - (my + mh / 2)) < 0.05
                        and abs(w - mw) / max(mw, 1e-6) < 0.3
                        and abs(h - mh) / max(mh, 1e-6) < 0.3):
                    hit = c
                    break
            if hit is None:
                clusters.append([x, y, w, h, 1, border, mot_s])
            else:
                k = hit[4]
                hit[0] = (hit[0] * k + x) / (k + 1)
                hit[1] = (hit[1] * k + y) / (k + 1)
                hit[2] = (hit[2] * k + w) / (k + 1)
                hit[3] = (hit[3] * k + h) / (k + 1)
                hit[4] = k + 1
                hit[5] = max(hit[5], border)
                hit[6] = max(hit[6], mot_s)
        best, best_sc = None, 0.0
        for x, y, w, h, cnt_n, border, mot_s in clusters:
            cx = x + w / 2
            corner = 1.0 if (cx < 0.32 or cx > 0.68) else 0.3
            sc = (cnt_n / max(samples, 1)) * 0.5 + min(border, 1.0) * 0.25 \
                + mot_s * 0.15 + corner * 0.1
            if sc > best_sc:
                best, best_sc = (x, y, w, h), sc
        rect = best if best is not None and best_sc >= 0.45 else None
        _CAM_CACHE[src] = rect
        return rect
    except Exception:
        return None


def render_twitch_clip(src: str, moment: dict, segs, out: str,
                       captions: bool = True, progress=None,
                       antidetect: bool = False, mirror: bool = False,
                       banner: bool = True, banner_path: str | None = None,
                       max_total: float | None = None):
    """Вертикаль 1080x1920 для стримов: верх 30% — вебка стримера
    (авто-поиск рамкой), низ 70% — его катка (вебка там затемнена).
    Режимы как в клипах: антидетект, зеркало, баннер Мусордроп по середине
    (целиком, 95%, 1.4х). Сабы у шва, звук оригинала."""
    import cv2
    from PIL import Image, ImageDraw
    OUT_W2, OUT_H2 = 1080, 1920
    TOP_H2, BOT_H2 = int(OUT_H2 * 0.30), OUT_H2 - int(OUT_H2 * 0.30)
    start, end = float(moment["start"]), float(moment["end"])
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fps = round(min(float(fps or 30), 30), 2)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if nfr and fps > 0:
        end = min(end, nfr / fps)
    if end - start < 1:
        cap.release()
        raise RuntimeError("момент за пределами видео")
    dur = end - start
    cam = detect_webcam(src) or (0.02, 0.02, 0.30, 0.17)
    cx0, cy0, cw0, ch0 = (cam[0] * W, cam[1] * H, cam[2] * W, cam[3] * H)
    if mirror:
        cx0 = W - cx0 - cw0
    font = _font(62)
    seam = TOP_H2
    # хук-надпись на первые 3 секунды: крупный текст сверху
    font_hook = _font(84)
    hook_text = (moment.get("hook") or moment.get("title") or "").strip().upper()
    hook_words = [{"w": w} for w in hook_text.split()[:12]]
    import random
    # --- баннер: как в сплите (фриз + хромакей целиком, 95%, 1.4х) ---
    od = dur
    if banner and max_total:
        be = _banner_eff(banner_path)
        for _ in range(3):
            n = max(1, min(6, int(round(od / 50.0))))
            if od + n * be <= max_total or od < 5:
                break
            od = max_total - n * be
        dur = od
        end = start + dur
    wins = []
    bcap, b_step, b_frac = None, 0.0, 0.0
    bpath = banner_path or BANNER_FILE
    use_banner = bool(banner) and os.path.exists(bpath)
    tw_key = _banner_green(bpath) if use_banner else False
    b_eff = 0.0
    if use_banner:
        bcap = cv2.VideoCapture(bpath)
        bn = int(bcap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        bfps = bcap.get(cv2.CAP_PROP_FPS) or 60
        b_eff = (bn / max(bfps, 1)) / BANNER_SPEED
        n = max(1, min(6, int(round(od / 50.0))))
        if od < b_eff + 3:
            use_banner = False
        else:
            for k in range(n):
                if n == 1:
                    oA = (od + b_eff) / 2  # старт строго по середине итога
                    cA = oA
                else:
                    cA = od * (k + 1) / (n + 1) - b_eff / 2
                    cA = min(max(cA, 0.5), max(od - b_eff - 0.5, 0.5))
                    oA = cA + k * b_eff
                wins.append((oA, oA + b_eff, cA))
            b_step = max(bfps, 1) * BANNER_SPEED / max(fps, 1)
        bj = {"dy": random.randint(-6, 6),
              "gain": random.uniform(0.97, 1.03),
              "ns": random.uniform(0.8, 1.6),
              "asp": BANNER_SPEED}
    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1920",
           "-r", str(fps),
           "-i", "pipe:0"]
    mix_tmp = None
    if use_banner:
        froze = b_eff * len(wins)
        out_dur = od + froze
        import numpy as np
        sr, chn = 48000, 2
        af_parts = []
        if antidetect:
            af_parts.append(_antidetect_filters()[1])
        else:
            _pk = random.uniform(1.005, 1.012)
            af_parts.append(
                f"aresample=44100,asetrate=44100*{_pk:.4f},"
                f"aresample=44100,atempo={1 / _pk:.4f}")
        af_parts.append(_LOUDNORM)
        af_parts.append(f"volume={random.uniform(0.95, 1.05):.3f}")
        orig = _extract_pcm(src, ss=start, dur=dur, af=",".join(af_parts),
                            sr=sr, ch=chn)
        n_total = int(round(out_dur * sr))
        need = int(round(b_eff * sr))
        if _banner_has_audio(bpath):
            ban = _extract_pcm(
                bpath, af=f"atempo={bj['asp']:.3f},aresample={sr},volume=0.5",
                sr=sr, ch=chn)
            ban = ban[:need]
            if len(ban) < need:
                ban = np.vstack([ban, np.zeros((need - len(ban), chn),
                                               dtype=np.int16)])
        else:
            ban = np.zeros((need, chn), dtype=np.int16)
        parts, prev_s = [], 0
        for (_, _, cA) in wins:
            n_cA = min(int(round(cA * sr)), len(orig))
            parts.append(orig[prev_s:n_cA])
            parts.append(ban)
            prev_s = n_cA
        parts.append(orig[prev_s:])
        tl = np.vstack(parts)
        if len(tl) < n_total:
            tl = np.vstack([tl, np.zeros((n_total - len(tl), chn), dtype=np.int16)])
        else:
            tl = tl[:n_total]
        mix_tmp = out + ".mix.s16"
        tl.tofile(mix_tmp)
        cmd += ["-f", "s16le", "-ar", str(sr), "-ac", str(chn), "-i", mix_tmp]
        qvf = _quality_vf()
        if antidetect:
            vf, _ = _antidetect_filters()
            qvf += "," + vf
        cmd += ["-vf", qvf]
        cmd += ["-map", "0:v", "-map", "1:a"] + _rand_meta()
    else:
        out_dur = od
        cmd += ["-ss", str(start), "-t", str(round(dur, 2)), "-i", src,
                "-map", "0:v", "-map", "1:a?"] + _rand_meta()
        qvf = _quality_vf()
        spd_af = [_LOUDNORM]
        if antidetect:
            vf, af = _antidetect_filters()
            qvf += "," + vf
            spd_af.append(af)
        else:
            _tk = random.uniform(1.005, 1.012)
            spd_af.append(
                f"aresample=44100,asetrate=44100*{_tk:.4f},"
                f"aresample=44100,atempo={1 / _tk:.4f}")
        spd_af.append(f"volume={random.uniform(0.95, 1.05):.3f}")
        cmd += ["-vf", qvf, "-af", _tiktok_af(*spd_af)]
    cmd += _video_codec_args() + ["-pix_fmt", "yuv420p",
            *_AUDIO_OUT, "-movflags", "+faststart",
            "-t", str(round(out_dur, 2)), out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    _warm_start(cap)
    total = max(int(round(out_dur * fps)), 1)
    # проекция вебки на низ: cover-кроп всего кадра в 1080xBOT_H2
    sc = max(OUT_W2 / max(W, 1), BOT_H2 / max(H, 1))
    sw, sh = W * sc, H * sc
    bx0, by0 = (sw - OUT_W2) / 2, (sh - BOT_H2) / 2
    last_top, last_win = None, -1
    if antidetect:
        zm = random.uniform(1.03, 1.06)
        jx, jy = random.uniform(-0.04, 0.04), random.uniform(-0.04, 0.04)
    else:
        zm = random.uniform(1.01, 1.025)
        jx, jy = random.uniform(-0.015, 0.015), random.uniform(-0.015, 0.015)
    try:
        for k in range(total):
            t_rel = k / fps
            if use_banner:
                acc, ct_base, frozen = 0.0, t_rel, False
                for (oA, oB, cA) in wins:
                    if t_rel >= oB:
                        acc += b_eff
                    elif t_rel >= oA:
                        ct_base, frozen = cA, True
                        break
                else:
                    ct_base = t_rel - acc
            else:
                ct_base, frozen = t_rel, False
            t = start + ct_base
            if frozen and last_top is not None:
                fr = last_top
            else:
                ok, fr = cap.read()
                if not ok or fr is None:
                    if last_top is None:
                        break
                    fr = last_top
                else:
                    if mirror:
                        fr = cv2.flip(fr, 1)
                    if antidetect:
                        h0, w0 = fr.shape[:2]
                        fr = cv2.resize(fr, (int(w0 * zm), int(h0 * zm)),
                                        interpolation=cv2.INTER_LANCZOS4)
                        dx, dy = (fr.shape[1] - w0) // 2, (fr.shape[0] - h0) // 2
                        fr = fr[dy:dy + h0, dx:dx + w0]
                    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                last_top = fr
            crop = fr[max(int(cy0), 0):max(int(cy0 + ch0), 1),
                       max(int(cx0), 0):max(int(cx0 + cw0), 1)]
            if crop.size == 0:
                crop = fr
            top = _cover_resize(crop, OUT_W2, TOP_H2, cx=0.5, cy=0.5)
            bot = _cover_resize(fr, OUT_W2, BOT_H2, cx=0.5 + jx, cy=0.5 + jy)
            # затемняем вебку внизу, чтобы не двоилась
            ox0, oy0 = int(cx0 * sc - bx0), int(cy0 * sc - by0)
            ox1, oy1 = int((cx0 + cw0) * sc - bx0), int((cy0 + ch0) * sc - by0)
            ox0, oy0 = max(ox0, 0), max(oy0, 0)
            ox1, oy1 = min(ox1, OUT_W2), min(oy1, BOT_H2)
            if ox1 > ox0 and oy1 > oy0:
                roi = bot[oy0:oy1, ox0:ox1]
                dark = cv2.GaussianBlur(roi, (0, 0), 12)
                bot[oy0:oy1, ox0:ox1] = (dark * 0.35).astype("uint8")
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
                    _bh0, _bw0 = bfr.shape[:2]
                    _bh = int(round(_bh0 * min(1022, OUT_W2) / max(_bw0, 1)))
                    _bh = min(_bh, OUT_H2)
                    _by = (OUT_H2 - _bh) // 2 + random.randint(-15, 15)
                    frame = _banner_overlay(
                        frame, bfr, y_top=_by,
                        zoom=1.0, bj=bj, opacity=0.95, key_green=tw_key)
                    banner_on = True
            if captions and not banner_on:
                words = _active_words(segs, t)
                if words:
                    ch, act = _caption_states(words, t)
                    if ch:
                        img = Image.fromarray(frame)
                        dr = ImageDraw.Draw(img)
                        lines = _split_lines(dr, font, ch, OUT_W2 - 140)
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
                            x = (OUT_W2 - tw) / 2
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
            if hook_words and t_rel < 3.0:
                imgh = Image.fromarray(frame)
                drh = ImageDraw.Draw(imgh)
                hl = _split_lines(drh, font_hook, hook_words, OUT_W2 - 100)[:2]
                hy = seam + 24
                for ln in hl:
                    txt = " ".join(w["w"] for w in ln)
                    bbox = drh.textbbox((0, 0), txt, font=font_hook, stroke_width=4)
                    tw = bbox[2] - bbox[0]
                    drh.text(((OUT_W2 - tw) / 2, hy), txt, font=font_hook,
                             fill="#FFD60A", stroke_width=4, stroke_fill="black")
                    hy += (bbox[3] - bbox[1]) + 10
                frame = __import__("numpy").asarray(imgh)
            proc.stdin.write(frame.tobytes())
            if progress and k % 15 == 0:
                progress(k / total)
    finally:
        cap.release()
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
    if proc.returncode != 0 or not _has_video(out):
        raise RuntimeError("рендер не удался")
    return out


# ---------- 3.5. Сплит: верх контент / низ геймплей ----------

def _readable_video(path: str) -> bool:
    """Файл открывается и отдаёт хотя бы 1 видеокадр (cv2)."""
    import cv2
    try:
        c = cv2.VideoCapture(path)
        ok, fr = c.read()
        c.release()
        return bool(ok) and fr is not None and getattr(fr, "size", 0) > 0
    except Exception:
        return False


def _has_video(path: str) -> bool:
    """В готовом файле есть видеодорожка с кадрами, а не аудио-пустышка."""
    import cv2
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 50000:
            return False
        c = cv2.VideoCapture(path)
        n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        ok, fr = c.read()
        c.release()
        return bool(ok) and (n > 0 or (fr is not None and fr.size > 0))
    except Exception:
        return False


def tiktok_safety_check(path: str) -> dict:
    """Предзаливная проверка файла под TikTok. Возвращает
    {ok, problems[], info{}} — файл с ok=False почти наверняка получит 0
    просмотров (битый звук 96к, гигантский вес, не та частота кадров и т.п.)."""
    problems, info = [], {}
    try:
        size = os.path.getsize(path)
    except OSError:
        return {"ok": False, "problems": ["файл не найден"], "info": {}}
    info["size_mb"] = round(size / 1e6, 1)
    if size > 70 * 1024 * 1024:
        problems.append(f"тяжёлый файл ({info['size_mb']} МБ): будет долго грузиться и может зависнуть в обработке — норма до ~50 МБ")
    if size < 50000:
        problems.append("файл подозрительно маленький — возможно, битый")
    try:
        r = subprocess.run([FFMPEG, "-i", path],
                           capture_output=True, text=True, timeout=30)
        err = r.stderr or ""
    except Exception:
        err = ""
    m = re.search(r"Video:.*?(\d{3,5})x(\d{3,5}).*?(\d+(?:\.\d+)?)\s*fps", err)
    if m:
        info["wh"] = f"{m.group(1)}x{m.group(2)}"
        try:
            info["fps"] = float(m.group(3))
        except ValueError:
            pass
        if f"{m.group(1)}x{m.group(2)}" != "1080x1920":
            problems.append(f"разрешение {m.group(1)}x{m.group(2)} вместо 1080x1920")
    ma = re.search(r"Audio:.*?(\d{4,6})\s*Hz", err)
    if ma:
        info["audio_hz"] = int(ma.group(1))
        if int(ma.group(1)) not in (44100, 48000):
            problems.append(f"звук {ma.group(1)} Гц вместо 48к — TikTok плохо такое перекодирует (было главной причиной нулей)")
    elif "Audio:" not in err:
        problems.append("нет звуковой дорожки — такие видео TikTok почти не раздаёт")
    md = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if md:
        dur = int(md.group(1)) * 3600 + int(md.group(2)) * 60 + float(md.group(3))
        info["dur"] = round(dur, 1)
        if dur < 5:
            problems.append("короче 5с — неINFOрмативно для ленты")
    # первый кадр не чёрный (обложка решает CTR)
    try:
        import cv2
        import numpy as np
        c = cv2.VideoCapture(path)
        ok, fr = c.read()
        c.release()
        if ok and fr is not None and float(np.mean(fr)) < 10:
            problems.append("первый кадр чёрный — сменится обложка, упадёт CTR")
    except Exception:
        pass
    return {"ok": not problems, "problems": problems, "info": info}


def ensure_gameplay(name: str = "subway") -> str | None:
    """Возвращает путь к закэшированному геймплею; если нет — качает с YouTube."""
    meta = GAMEPLAYS.get(name) or GAMEPLAYS["subway"]
    dest = os.path.join(GAMEPLAY_DIR, meta["file"])
    if os.path.exists(dest):
        # битый/недокачанный геймплей — не сносим сразу (мимолетный сбой чтения
        # убивал бы гигабайты): переименовываем в .bad, докачаем заново
        if os.path.getsize(dest) > 10 << 20 and _readable_video(dest):
            return dest
        try:
            bad = dest + ".bad"
            if os.path.exists(bad):
                os.remove(bad)
            os.rename(dest, bad)
        except OSError:
            pass
    os.makedirs(GAMEPLAY_DIR, exist_ok=True)
    try:
        subprocess.run(
            ["yt-dlp", "--no-playlist",
              "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
             "--merge-output-format", "mp4",
             "-o", os.path.join(GAMEPLAY_DIR, meta["file"].rsplit(".", 1)[0] + ".%(ext)s"),
             meta["query"]],
            timeout=1200, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    if os.path.exists(dest) and _readable_video(dest):
        return dest
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
                                 [cv2.IMWRITE_JPEG_QUALITY, 82])
                cap.release()
                return bool(ok)
        cap.release()
    except Exception:
        pass
    # cv2 слеп к файлу — пробуем достать кадр напрямую ffmpeg
    try:
        return _ffmpeg_thumb(video_path, thumb_path, w)
    except Exception:
        return False


def _ffmpeg_thumb(video_path: str, thumb_path: str, w: int = 360) -> bool:
    """Запасной путь обложки: кадр через ffmpeg (когда cv2 не читает)."""
    try:
        r = subprocess.run([FFMPEG, "-i", video_path],
                           capture_output=True, text=True, timeout=30)
        m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
        if not m:
            return False
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        if dur <= 0:
            return False
        for frac in (0.25, 0.4, 0.55, 0.7, 0.15):
            try:
                if os.path.exists(thumb_path):
                    os.remove(thumb_path)
            except OSError:
                pass
            subprocess.run(
                [FFMPEG, "-y", "-v", "error", "-ss", str(round(dur * frac, 2)),
                 "-i", video_path, "-frames:v", "1",
                 "-vf", f"scale={w}:-2", "-q:v", "4", thumb_path],
                timeout=60, check=False)
            try:
                if os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0:
                    return True
            except OSError:
                pass
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


def _rand_meta() -> list:
    """Уникальные метаданные файла: рандомные время создания, кодер, соль.
    Content ID и хэши видят каждый рендер новым файлом."""
    import random
    import time
    ts = time.strftime("%Y-%m-%dT%H:%M:%S.000Z",
                       time.gmtime(time.time() - random.randint(0, 900 * 86400)))
    enc = f"Lavf{random.choice(['58.29', '59.27', '60.16', '61.7'])}.{random.randint(100, 104)}"
    salt = "".join(random.choice("abcdef0123456789") for _ in range(8))
    return ["-map_metadata", "-1",
            "-metadata", f"creation_time={ts}",
            "-metadata", f"encoder={enc}",
            "-metadata", f"comment={salt}"]


def _antidetect_filters():
    """Усиленный антидетект: ВСЕ параметры случайные в диапазоне —
    каждый рендер уникален: сдвиг цвета + лёгкое зерно (видео),
    подъём питча на 1-4% без смены длительности (аудио)."""
    import random
    vf = (f"eq=contrast={random.uniform(1.05, 1.09):.3f}:"
          f"saturation={random.uniform(1.10, 1.14):.3f}:"
          f"brightness={random.uniform(0.010, 0.020):.3f},"
          f"hue=h={random.uniform(-4, 4):.1f},"
          f"noise=alls={random.randint(2, 5)}:allf=t")
    k = random.uniform(1.01, 1.04)
    af = (f"aresample=44100,asetrate=44100*{k:.4f},"
          f"aresample=44100,atempo={1 / k:.4f}")
    return vf, af


_BANNER_AUDIO = {}


def _banner_has_audio(path: str) -> bool:
    global _BANNER_AUDIO
    if path not in _BANNER_AUDIO:
        try:
            r = subprocess.run([FFMPEG, "-i", path],
                               capture_output=True, text=True, timeout=30)
            _BANNER_AUDIO[path] = " Audio:" in r.stderr
        except Exception:
            _BANNER_AUDIO[path] = False
    return _BANNER_AUDIO[path]


_GREEN_CACHE = {}
# Параметры ключа подобраны на RubUP green-screen (0.20/0.10: зелень ушла
# полностью, без красного ореола; 0.30 уже красит контент в красный).
_KEY_SIM, _KEY_BLEND = 0.20, 0.10


def _banner_green(path: str | None = None) -> bool:
    """У баннера зелёный фон под хромакей? Смотрим УГЛЫ кадров с отступом
    0.5%: у хромакей-файлов (RubUP green-screen) зелень идёт по самой
    кромке, у обычных креативов (banner.mp4) углы чёрные.
    Важно смотреть именно кромку, а не долю зелени: у обычных креативов
    зелёного контента внутри может быть даже больше (зелёный бак)."""
    key = path or BANNER_FILE
    if key in _GREEN_CACHE:
        return _GREEN_CACHE[key]
    res = False
    try:
        import cv2
        import tempfile
        _w, _h, _n, _f = _probe_wh_fps(key)
        dur = _n / max(_f, 1) if _n and _f else 0
        if dur > 0:
            hits, total = 0, 0
            for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
                tmp = os.path.join(tempfile.gettempdir(),
                                   f"bngrn_{os.getpid()}.png")
                try:
                    subprocess.run(
                        [FFMPEG, "-y", "-v", "error",
                         "-ss", f"{dur * frac:.2f}", "-i", key,
                         "-frames:v", "1", tmp],
                        timeout=60, check=False)
                    fr = cv2.imread(tmp)
                    if fr is None:
                        continue
                    h, w = fr.shape[:2]
                    ix = max(int(w * 0.005), 2)
                    iy = max(int(h * 0.005), 2)
                    for yy, xx in ((iy, ix), (iy, w - 1 - ix),
                                   (h - 1 - iy, ix), (h - 1 - iy, w - 1 - ix)):
                        b, g, r = (int(v) for v in fr[yy, xx])
                        total += 1
                        if g > 150 and (g - r) > 60 and (g - b) > 60:
                            hits += 1
                except Exception:
                    continue
                finally:
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except OSError:
                        pass
            res = total >= 8 and hits / max(total, 1) >= 0.6
    except Exception:
        res = False
    _GREEN_CACHE[key] = res
    return res


def _banner_key_vf(path: str | None = None) -> str:
    """Кусок ffmpeg-цепочки для вычитания зелени (или '' для обычных
    креативов). Вставлять ПЕРЕД format=yuva420p."""
    if _banner_green(path):
        return (f"chromakey=0x00FF00:{_KEY_SIM}:{_KEY_BLEND},"
                f"despill=type=green,")
    return ""


def _green_alpha_bgr(fg_bgr):
    """Покадровый хромакей для numpy-пути (twitch): маска 0..1, 1=оставить.
    Зелень в прозрачность + растушёвка краёв от ореола."""
    import cv2
    import numpy as np
    b, g, r = cv2.split(fg_bgr)
    gi, ri, bi = g.astype(int), r.astype(int), b.astype(int)
    gm = (((gi > 150) & ((gi - ri) > 60) & ((gi - bi) > 60))
          .astype(np.uint8) * 255)
    soft = cv2.GaussianBlur(gm, (0, 0), 1.5) / 255.0
    return 1.0 - soft


def _despill_bgr(fg_bgr):
    """Давит зелёный отблеск на краях контента после кеинга."""
    import cv2
    import numpy as np
    b, g, r = cv2.split(fg_bgr)
    mx = np.maximum(r.astype(np.int16), b.astype(np.int16))
    ex = np.clip(g.astype(np.int16) - mx - 25, 0, 255).astype(np.float32)
    g2 = (g.astype(np.float32) - ex * 0.75).clip(0, 255).astype(np.uint8)
    return cv2.merge([b, g2, r])


def _banner_overlay(bg, fg, box_w: int = BANNER_W, cy: int | None = None,
                    zoom: float = 1.0, y_top: int | None = None,
                    opacity: float = 0.95,
                    bj: dict | None = None, key_green: bool = False):
    """Баннер поверх кадра. key_green=True — вычитает зелёный фон
    (хромакей-файлы типа RubUP green-screen): зелень в прозрачность
    с растушёвкой + despill, сквозь неё видно видео.
    cy — центр по вертикали (по умолчанию центр кадра), y_top — верх бокса,
    zoom — наезд с центр-кропом,
    opacity — непрозрачность баннера (0.95 = фон чуть просвечивает)."""
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
    keep_mask = None
    if key_green:
        keep_mask = _green_alpha_bgr(fg)
        fg = _despill_bgr(fg)
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB).astype(np.float32)
    if bj is not None:  # сублиминальный джиттер: глаз не видит, хэш другой
        fg *= bj.get("gain", 1.0)
        ns = bj.get("ns", 0.0)
        if ns > 0:
            fg += np.random.normal(0, ns, fg.shape).astype(np.float32)
    fg = fg.astype(np.float32)
    if keep_mask is not None:
        alpha = (keep_mask * opacity)[..., None]  # попиксельная прозрачность
    else:
        alpha = opacity  # обычная плашка целиком, фон чуть просвечивает
    Hh, Ww = bg.shape[:2]
    x0 = (Ww - bw) // 2
    if y_top is not None:
        y0 = min(max(y_top, 0), max(Hh - bh, 0))
    else:
        y0 = (cy - bh // 2) if cy is not None else (Hh - bh) // 2
        y0 = min(max(y0, 0), max(Hh - bh, 0))
    roi = bg[y0:y0 + bh, x0:x0 + bw].astype(np.float32)
    bg[y0:y0 + bh, x0:x0 + bw] = np.clip(
        roi * (1 - alpha) + fg * alpha, 0, 255).astype(np.uint8)
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


_BANNER_EFF = {}


def _banner_eff(path: str | None = None) -> float:
    global _BANNER_EFF
    key = path or BANNER_FILE
    if key not in _BANNER_EFF:
        try:
            import cv2
            c = cv2.VideoCapture(key)
            n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            f = c.get(cv2.CAP_PROP_FPS) or 60
            c.release()
            _BANNER_EFF[key] = (n / max(f, 1)) / BANNER_SPEED if n else 4.22
        except Exception:
            _BANNER_EFF[key] = 4.22
    return _BANNER_EFF[key]


def _ass_captions(segs, start: float, spd: float, wins, b_eff: float,
                  path: str, part_title: str = "") -> bool:
    """ASS-субтитры с пословным караоке (белый -> жёлтый, как было в PIL).
    Время — уже в координатах итога (фризы баннеров учтены).
    Возвращает True если есть хоть одна реплика."""
    def _ts(t: float) -> str:
        t = max(t, 0.0)
        return f"{int(t // 3600)}:{int((t % 3600) // 60):02d}:{t % 60:05.2f}"

    def _c2o(t: float) -> float:
        t = t / max(spd, 1e-6)
        acc = 0.0
        for (_oA, _oB, cA) in wins:
            if cA <= t:
                acc += b_eff
        return t + acc

    head = ["[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "ScaledBorderAndShadow: yes",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Hook,Arial,64,&H00FFFFFF,&H000AD6FF,&H80000000,&H80000000,"
            "-1,0,0,0,100,100,0,0,1,3,1,5,40,40,40,1",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
            "Effect, Text"]
    body = []
    if part_title:
        body.append("Dialogue: 0,0:00:00.00,9:59:59.00,Hook,,0,0,0,,"
                    "{\\an8\\pos(540,100)}" + part_title.replace("{", "(").replace("}", ")"))
    n_dlgs = 0
    dlgs = []
    for s in segs:
        ws = [w for w in (s.get("words") or [])
              if float(w.get("e", 0)) > start - 0.15]
        if not ws:
            continue
        chunks, cur = [], []
        for w in ws:
            if float(w.get("s", 0)) < start - 0.15:
                continue
            cur.append(w)
            if len(cur) >= 4 or re.search(r"[.!?…]$", str(w.get("w", ""))):
                chunks.append(cur)
                cur = []
        if cur:
            chunks.append(cur)
        for ch in chunks:
            cs0 = _c2o(float(ch[0]["s"]) - start - 0.15)
            ce0 = _c2o(float(ch[-1]["e"]) - start + 0.4)
            if ce0 <= 0.1:
                continue
            if any(oA - 0.05 <= (cs0 + ce0) / 2 <= oB + 0.05 for oA, oB, _c in wins):
                continue  # под баннером сабы не показываем
            txt = "".join("{\\k%d}%s " % (
                max(1, int(round((float(w["e"]) - float(w["s"])) * 100))),
                str(w["w"]).replace("{", "(").replace("}", ")")) for w in ch)
            dlgs.append([cs0, ce0,
                         f"{{\\an5\\pos(540,660)}}{txt.strip()}"])
    # концы — ровно до начала следующей: иначе реплики налезают друг на друга
    dlgs.sort(key=lambda d: d[0])
    for i, (s0, e0, tx) in enumerate(dlgs):
        if i + 1 < len(dlgs):
            e0 = min(e0, dlgs[i + 1][0] - 0.01)
        if e0 - s0 < 0.2:
            continue
        body.append(f"Dialogue: 0,{_ts(s0)},{_ts(e0)},Hook,,0,0,0,,{tx}")
        n_dlgs += 1
    if part_title:
        n_dlgs += 1
    if n_dlgs == 0:
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(head) + "\n" + "\n".join(body) + "\n")
    return True


def _run_ffmpeg_progress(cmd, total_dur: float, progress=None) -> int:
    """Запуск ffmpeg с живым прогрессом по out_time_ms. Возвращает returncode."""
    import threading
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    total_ms = max(total_dur * 1000.0, 1.0)

    def _watch():
        try:
            for raw in proc.stdout:
                try:
                    line = raw.decode("utf-8", "ignore").strip()
                except Exception:
                    continue
                if line.startswith("out_time_ms="):
                    try:
                        ms = int(line.split("=", 1)[1])
                    except ValueError:
                        ms = 0
                    if progress:
                        try:
                            progress(min(0.99, ms / total_ms))
                        except Exception:
                            pass
                elif line.strip() == "progress=end":
                    if progress:
                        try:
                            progress(1.0)
                        except Exception:
                            pass
                    break
        except Exception:
            pass

    th = threading.Thread(target=_watch, daemon=True)
    th.start()
    try:
        rc = proc.wait(timeout=1800)
    finally:
        try:
            th.join(timeout=5)
        except Exception:
            pass
    return rc


def _fesc(p: str) -> str:
    return p.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _prev_keyframe(src: str, t: float) -> float:
    """Время ближайшего ключевого кадра <= t (для точного быстрого сика)."""
    try:
        r = subprocess.run(
            [FFMPEG, "-v", "info", "-skip_frame", "nokey", "-i", src,
             "-vf", "select='eq(pict_type\\,I)',showinfo", "-f", "null", "-"],
            capture_output=True, text=True, timeout=180)
        best = 0.0
        for m in re.finditer(r"pts_time:([\d.]+)", r.stderr or ""):
            v = float(m.group(1))
            if v <= t + 0.05:
                best = v
        return best
    except Exception:
        return 0.0


def render_split_clip(src: str, moment: dict, segs, out: str,
                      facetrack: bool = True, captions: bool = True,
                      progress=None, antidetect: bool = False,
                      mirror: bool = False, gameplay: str | None = None,
                      banner: bool = True, banner_path: str | None = None,
                      max_total: float | None = None):
    """Вертикаль 1080x1920: верх 40% — сериал (кроп за лицом по 2с / центр),
    низ 60% — геймплей (закольцован, немой). Сабы ASS с караоке у шва.
    Звук — только из сериала. Сборка одним графом ffmpeg (без покадрового цикла)."""
    import cv2
    import random
    gp = (gameplay if gameplay and os.path.exists(gameplay) else
          ensure_gameplay(gameplay if gameplay in GAMEPLAYS else "subway"))
    if not gp or not os.path.exists(gp):
        raise RuntimeError("нет геймплея — положи gameplay/subway.mp4")
    start, end = float(moment["start"]), float(moment["end"])
    cap = cv2.VideoCapture(src)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30) or 30
    nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if nfr and src_fps > 0:
        end = min(end, nfr / src_fps)
    if end - start < 1:
        cap.release()
        raise RuntimeError("момент за пределами видео")
    dur = end - start
    cap.release()
    # скорость верха ломает пиксельный и звуковой отпечатки разом.
    # Сабы/трек лиц идут по контентному времени — синхрон цел.
    # Always-on: даже без чекбокса лёгкий дрейф ±1.5%, иначе все клипы
    # пачки идут 1:1 с оригиналом и TikTok режет их в 0 как дубли.
    if antidetect:
        spd = random.uniform(0.97, 1.03)
    else:
        spd = random.uniform(0.985, 1.015)
    od = dur / spd  # длина контента на выходе
    if banner and max_total:
        # итог вместе с баннерами влезает в линзу: ужимаем контент
        be = _banner_eff(banner_path)
        for _ in range(3):
            n = max(1, min(6, int(round(od / 50.0))))
            if od + n * be <= max_total or od < 5:
                break
            od = max_total - n * be
        dur = od * spd
        end = start + dur
    # точный сик: мотаем на ключевой кадр, дальше режем фильтром по секундам
    kf = _prev_keyframe(src, start)
    off = max(start - kf, 0.0)

    import cv2
    gcap = cv2.VideoCapture(gp)
    gn = int(gcap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    gfps = float(gcap.get(cv2.CAP_PROP_FPS) or 30) or 30
    gW = int(gcap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    gH = int(gcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    gdur = gn / max(gfps, 1) if gn else 0
    # живая стартовая точка геймплея для каждого клипа (мимо меню и сундуков)
    goff = 0.0
    if gdur > od + 1:
        try:
            goff = float(_seek_live(gcap, gdur, od) or 0.0)
        except Exception:
            goff = 0.0
    gcap.release()
    cap0 = cv2.VideoCapture(src)
    W = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    H = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap0.release()

    track = track_faces(src, start, end) if facetrack else []
    # --- баннеры: те же позиции (строго середина итога), сборка — графом ниже ---
    bpath = banner_path or BANNER_FILE
    use_banner = bool(banner) and os.path.exists(bpath)
    b_eff = 0.0
    wins = []
    if use_banner:
        b_eff = _banner_eff(bpath)
        n = max(1, min(6, int(round(od / 50.0))))
        if od < b_eff + 3:
            use_banner = False  # клип короче баннера — пропускаем
        else:
            for k in range(n):
                if n == 1:
                    oA = (od + b_eff) / 2
                    cA = oA
                else:
                    cA = od * (k + 1) / (n + 1) - b_eff / 2
                    cA = min(max(cA, 0.5), max(od - b_eff - 0.5, 0.5))
                    oA = cA + k * b_eff
                wins.append((oA, oA + b_eff, cA))
    out_dur = od + b_eff * len(wins)
    # --- верх: куски по 2с со статичным кропом за лицом (или центр) ---
    if antidetect:
        Z = random.uniform(1.03, 1.06)
        jx, jy = random.uniform(-0.04, 0.04), random.uniform(-0.04, 0.04)
    else:
        Z = random.uniform(1.01, 1.025)
        jx, jy = random.uniform(-0.015, 0.015), random.uniform(-0.015, 0.015)
    sc0 = max(1080.0 / max(W, 1), 768.0 / max(H, 1)) * Z
    sw0, sh0 = int(round(W * sc0)), int(round(H * sc0))
    SEG = 2.0
    segs2 = []
    t = 0.0
    while t < od - 1e-6:
        segs2.append((t, min(t + SEG, od)))
        t += SEG
    if not segs2:
        segs2 = [(0.0, od)]
    vf_ad = ""
    af_ad = ""
    if antidetect:
        _vf0, _af0 = _antidetect_filters()
        vf_ad = "," + _vf0
        af_ad = "," + _af0
    else:
        # always-on: лёгкий питч + цвет, глаз не видит, Content ID — да
        _k = random.uniform(1.005, 1.012)
        af_ad = (f",aresample=44100,asetrate=44100*{_k:.4f},"
                 f"aresample=44100,atempo={1 / _k:.4f}")
        vf_ad = (f",eq=contrast={random.uniform(1.02, 1.04):.3f}:"
                 f"saturation={random.uniform(1.04, 1.08):.3f}:"
                 f"brightness={random.uniform(0.004, 0.010):.3f}")
    fc = [f"[0:v]trim=start={off:.3f}:end={off + dur:.3f},setpts=(PTS-STARTPTS)/{spd:.4f}[cbase]",
          f"[cbase]split={len(segs2)}" + "".join(f"[cs{i}]" for i in range(len(segs2)))]
    # подменяем split-источник на именованные чанки
    vparts = []
    for pi, (a, b) in enumerate(segs2):
        if track:
            cx = face_x_at(track, start + (a + b) / 2 * spd)
        else:
            cx = 0.5
        fx = (cx + jx) * sw0
        fy = (0.5 + jy) * sh0
        x = int(round(min(max(fx - 540.0, 0.0), max(sw0 - 1080, 0.0))))
        y = int(round(min(max(fy - 384.0, 0.0), max(sh0 - 768, 0.0))))
        vparts.append(
            f"[cs{pi}]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS,"
            f"fps=30,scale={sw0}:{sh0}:flags=lanczos,"
            f"crop=1080:768:x={x}:y={y},setsar=1[fp{pi}]")
    fc.extend(vparts)
    fc.append("".join(f"[fp{pi}]" for pi in range(len(segs2)))
              + f"concat=n={len(segs2)}:v=1:a=0[vtop0]")
    if mirror:
        fc.append("[vtop0]hflip[vtop]")
    else:
        fc.append("[vtop0]null[vtop]")
    # --- низ: геймплей кольцом. Случайный флип низа у каждого клипа:
    # одинаковый subway сверху разных моментов даёт разный низ-хеш ---
    _gflip = ",hflip" if random.random() < 0.5 else ""
    fc.append(f"[1:v]fps=30{_gflip},scale=1080:1152:force_original_aspect_ratio=increase,"
              f"crop=1080:1152:x='(in_w-out_w)/2+({jx:.4f}*in_w)':"
              f"y='(in_h-out_h)/2+({jy:.4f}*in_h)',setsar=1[vbot]")
    fc.append("[vtop][vbot]vstack=inputs=2[vbase]")
    # --- баннеры: фриз + оверлей (как было покадрово, только графом) ---
    vlink = "[vbase]"
    if use_banner:
        n_br = 1 + 2 * len(wins)
        fc.append(f"[vbase]split={n_br}" + "".join(f"[vb{i}]" for i in range(n_br)))
        order = []
        bi = 0
        prev = 0.0
        for wi, (oA, oB, cA) in enumerate(wins):
            fc.append(f"[vb{bi}]trim=start={prev:.3f}:end={cA:.3f},setpts=PTS-STARTPTS[p{wi}a]")
            order.append(f"[p{wi}a]")
            bi += 1
            nfrz = int(round(cA * 30))
            fc.append(f"[vb{bi}]select='eq(n\\,{nfrz})',loop=loop=999:size=1,"
                      f"setpts=PTS-STARTPTS,fps=30,trim=start=0:end={b_eff:.3f},"
                      f"setpts=PTS-STARTPTS[f{wi}]")
            bi += 1
            order.append(f"[f{wi}]")
            prev = cA
        fc.append(f"[vb{bi}]trim=start={prev:.3f}:end={od:.3f},setpts=PTS-STARTPTS[p{len(wins)}a]")
        order.append(f"[p{len(wins)}a]")
        # баннер: ширина и вертикаль гуляют у каждого рендера, иначе
        # баннерный кусок пиксель-в-пиксель одинаков во всех клипах пачки
        _bn_w = random.randint(1000, 1022)
        _bn_y = random.randint(758, 794)
        _bn_key = _banner_key_vf(bpath)
        fc.append(f"[2:v]scale={_bn_w}:-2,fps=30,{_bn_key}format=yuva420p,"
                  f"colorchannelmixer=aa=0.95,setpts=PTS/{BANNER_SPEED:.3f}[bn0]")
        fc.append(f"[bn0]split={len(wins)}" + "".join(f"[bb{i}]" for i in range(len(wins))))
        for wi, (oA, oB, cA) in enumerate(wins):
            fc.append(f"[f{wi}][bb{wi}]overlay=(W-w)/2:{_bn_y}:eof_action=pass[m{wi}]")
            order[1 + wi * 2] = f"[m{wi}]"
        fc.append("".join(order) + f"concat=n={len(order)}:v=1:a=0[vcat]")
        vlink = "[vcat]"
    else:
        fc.append(f"{vlink}null[vcat]")
        vlink = "[vcat]"
    # --- качество + субтитры + CTA ---
    fc.append(f"{vlink}hqdn3d=5:4:3:3,unsharp=5:5:0.5:5:5:0.0{vf_ad}[vqual]")
    asspath = out + ".ass"
    ass_ok = False
    if captions:
        try:
            ass_ok = _ass_captions(segs, start, spd, wins, b_eff, asspath,
                                   str(moment.get("part_title") or ""))
        except Exception:
            ass_ok = False
    if ass_ok:
        fc.append(f"[vqual]subtitles={_fesc(asspath)}[vsub]")
    else:
        fc.append("[vqual]null[vsub]")
        try:
            if os.path.exists(asspath):
                os.remove(asspath)
        except OSError:
            pass
    if moment.get("cta") and out_dur > 3:
        _cff = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if not os.path.exists(_cff):
            _cff = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        fc.append(f"[vsub]drawtext=fontfile={_fesc(_cff)}:"
                  f"text='ЖДИ ЧАСТЬ 2 • ПОДПИШИСЬ':fontsize=62:"
                  f"fontcolor=#FFD60A:borderw=3:bordercolor=black:"
                  f"x=(w-text_w)/2:y=300:"
                  f"enable='between(t,{out_dur - 2.5:.2f},{out_dur:.2f})'[vout]")
    else:
        fc.append("[vsub]null[vout]")
    # --- звук ---
    has_src_a = _has_audio_stream(src)
    if has_src_a:
        fc.append(f"[0:a]atrim=start={off:.3f}:end={off + dur:.3f},asetpts=(PTS-STARTPTS)/{spd:.4f},"
                  f"aresample=44100,atempo={spd:.4f}{af_ad},"
                  f"aformat=sample_fmts=fltp:channel_layouts=stereo[aa]")
    if use_banner and has_src_a:
        n_br = 1 + len(wins)
        fc.append(f"[aa]asplit={n_br}" + "".join(f"[aa{i}]" for i in range(n_br)))
        aorder = []
        prev = 0.0
        for wi, (oA, oB, cA) in enumerate(wins):
            fc.append(f"[aa{wi}]atrim=start={prev:.3f}:end={cA:.3f},"
                      f"asetpts=PTS-STARTPTS[a{wi}]")
            aorder.append(f"[a{wi}]")
            prev = cA
        fc.append(f"[aa{len(wins)}]atrim=start={prev:.3f}:end={od:.3f},"
                  f"asetpts=PTS-STARTPTS[a{len(wins)}]")
        aorder.append(f"[a{len(wins)}]")
        if _banner_has_audio(bpath):
            fc.append(f"[2:a]atempo={BANNER_SPEED:.3f},aresample=44100,"
                      f"volume=0.5,aformat=sample_fmts=fltp:channel_layouts=stereo[ba]")
            aorder2 = []
            for wi in range(len(wins)):
                aorder2.append(aorder[wi * 2] if wi * 2 < len(aorder) else "")
                aorder2.append("[ba]")
            aorder2.append(aorder[-1])
            aorder = [x for x in aorder2 if x]
        fc.append("".join(aorder) + f"concat=n={len(aorder)}:v=0:a=1[acat]")
        fc.append(f"[acat]{_LOUDNORM},volume={random.uniform(0.95, 1.05):.3f},"
                  f"{_TIKTOK_AF_TAIL}[aout]")
        amap = ["-map", "[aout]"]
    elif has_src_a:
        fc.append(f"[aa]{_LOUDNORM},volume={random.uniform(0.95, 1.05):.3f},"
                  f"{_TIKTOK_AF_TAIL}[aout]")
        amap = ["-map", "[aout]"]
    else:
        amap = []
    cmd = [FFMPEG, "-y", "-v", "error", "-progress", "pipe:1", "-nostdin",
           "-ss", str(round(kf, 3)), "-i", src,
           "-stream_loop", "-1", "-ss", str(round(goff, 3)), "-t", str(round(od, 3)),
           "-i", gp]
    if use_banner:
        cmd += ["-i", bpath]
    cmd += ["-filter_complex", ";".join(fc),
            "-map", "[vout]"] + amap + _rand_meta() + _video_codec_args() + [
                "-pix_fmt", "yuv420p", "-r", "30",
                *_AUDIO_OUT, "-movflags", "+faststart",
                "-t", str(round(out_dur, 2)), out]
    try:
        rc = _run_ffmpeg_progress(cmd, out_dur, progress)
    finally:
        try:
            if os.path.exists(asspath):
                os.remove(asspath)
        except OSError:
            pass
    if rc != 0 or not _has_video(out):
        raise RuntimeError("рендер не удался")
    return out


def _with_part2_cta(frame, font):
    """Плашка «ЖДИ ЧАСТЬ 2 • ПОДПИШИСЬ» вверху кадра — механика open loop,
    главный драйвер подписок и досмотров у сериальных нарезок."""
    from PIL import Image, ImageDraw
    try:
        img = Image.fromarray(frame)
        dr = ImageDraw.Draw(img)
        txt = "ЖДИ ЧАСТЬ 2 • ПОДПИШИСЬ"
        bb = dr.textbbox((0, 0), txt, font=font, stroke_width=4)
        tw = bb[2] - bb[0]
        dr.text(((frame.shape[1] - tw) / 2, 300), txt, font=font,
                fill="#FFD60A", stroke_width=4, stroke_fill="black")
        return __import__("numpy").asarray(img)
    except Exception:
        return frame


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
    if not _has_video(out):
        raise RuntimeError("нарезка не удалась")
    return out


def _dhash_9x8(gray):
    """dHash 64 бит: устойчив к пережиму/ресайзу, чувствителен к содержанию."""
    import numpy as np
    g = np.asarray(gray, dtype=np.uint8).reshape(8, 9)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(bool(b))
    return v


def _ham(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


SIG_PATH = os.path.join(BASE, "ad_sigs.json")
_SIG_CACHE = (0, [])


def _load_sigs():
    """Известные рекламные креативы: [{name, box, hashes[], thr}]."""
    import time
    global _SIG_CACHE
    now = time.time()
    if now - _SIG_CACHE[0] < 60:
        return _SIG_CACHE[1]
    try:
        sigs = json.load(open(SIG_PATH, encoding="utf-8"))
        if not isinstance(sigs, list):
            sigs = []
    except Exception:
        sigs = []
    _SIG_CACHE = (now, sigs)
    return sigs


def learn_sig(path: str, name: str, s: float, e: float,
              box=(0.0, 0.25, 1.0, 0.45), thr: int = 10, n: int = 7):
    """Выучить креатив: хеши области в диапазоне [s,e] -> ad_sigs.json.
    Кадры — точным проходом ffmpeg (сики OpenCV врут и учат не то)."""
    import cv2
    import numpy as np
    r = subprocess.run(
        [FFMPEG, "-v", "error", "-i", path,
         "-vf", "fps=4,scale=168:90,format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, timeout=600)
    fs = 168 * 90
    nf = len(r.stdout) // fs
    if nf < 8:
        raise RuntimeError("не читается видео")
    frames = np.frombuffer(r.stdout[:nf * fs], dtype=np.uint8).reshape(nf, 90, 168)
    H, W = 90, 168
    x0, y0 = int(box[0] * W), int(box[1] * H)
    x1, y1 = int((box[0] + box[2]) * W), int((box[1] + box[3]) * H)
    kk = [k for k in range(nf) if s <= k * 0.25 <= e]
    if len(kk) < 3:
        raise RuntimeError("мало кадров для сигнатуры")
    idx = sorted(set(round(k * (len(kk) - 1) / (n - 1)) for k in range(n)))
    hashes = []
    for k in [kk[i] for i in idx]:
        crop = cv2.resize(frames[k][max(y0, 0):max(y1, 1),
                                    max(x0, 0):max(x1, 1)], (9, 8))
        hashes.append(_dhash_9x8(crop))
    try:
        sigs = json.load(open(SIG_PATH, encoding="utf-8"))
        if not isinstance(sigs, list):
            sigs = []
    except Exception:
        sigs = []
    sigs = [x for x in sigs if x.get("name") != name]
    sigs.append({"name": name, "box": list(box),
                 "hashes": [int(x) for x in hashes], "thr": int(thr)})
    with open(SIG_PATH, "w", encoding="utf-8") as f:
        json.dump(sigs, f, ensure_ascii=False)
    global _SIG_CACHE
    _SIG_CACHE = (0, [])
    return {"name": name, "hashes": len(hashes)}


def _match_sigs(frames, step: float, dur: float):
    """Прогон сигнатур по кадрам: точные совпадения известных креативов."""
    import cv2
    import numpy as np
    found = []
    sigs = _load_sigs()
    if not sigs or len(frames) < 8:
        return found
    H, W = frames[0].shape[:2]
    for sig in sigs:
        try:
            box = sig.get("box", (0, 0.25, 1, 0.45))
            thr = int(sig.get("thr", 10))
            hs = [int(x) for x in sig.get("hashes", [])]
            if not hs:
                continue
            x0, y0 = int(box[0] * W), int(box[1] * H)
            x1, y1 = int((box[0] + box[2]) * W), int((box[1] + box[3]) * H)
            hits = []
            for k in range(len(frames)):
                crop = cv2.resize(frames[k][max(y0, 0):max(y1, 1),
                                            max(x0, 0):max(x1, 1)], (9, 8))
                if min(_ham(_dhash_9x8(crop), h) for h in hs) <= thr:
                    hits.append(k * step)
            runs, cur = [], None
            for t in hits:
                if cur is None:
                    cur = [t, t]
                elif t - cur[1] <= 1.0:
                    cur[1] = t
                else:
                    runs.append(tuple(cur))
                    cur = [t, t]
            if cur is not None:
                runs.append(tuple(cur))
            for a, b in runs:
                if b - a + step >= 1.5 and a >= 1.0 and b <= dur - 1.0:
                    found.append({"name": str(sig.get("name", "?")),
                                  "s": round(a - 0.25, 2), "e": round(b + 0.5, 2)})
        except Exception:
            continue
    found.sort(key=lambda o: (o["s"], o["e"]))
    return found[:6]


def analyze_inserts(path: str, min_d: float = 1.5, max_d: float = 20.0,
                    edge: float = 3.0, step: float = 0.25):
    """Ищет чужие вставки четырёх типов: freeze (статичная картинка),
    fullscreen (движущаяся вставка во весь кадр), overlay (статичный блок —
    бегущая строка/плашка — поверх движущегося видео), rect (начало и конец
    вставки — одновременный срез компактного прямоугольника; ловит и статичные,
    и анимированные вставки типа FloatUP).
    Возвращает {"freeze": [(s,e)], "fullscreen": [...],
    "overlay": [{"s","e","box":(x,y,w,h) 0..1}],
    "rect": [{"s","e","box"}], "base": float, "dur": float}."""
    import cv2
    import numpy as np
    res = {"freeze": [], "fullscreen": [], "overlay": [], "rect": [], "sig": [],
           "base": 0.0, "dur": 0.0}
    GX, GY, CBX, CBY = 168, 90, 12, 9
    bw, bh = GX // CBX, GY // CBY
    try:
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 0
        if n <= 0 or fps <= 0:
            cap.release()
            return res
        dur = n / fps
        res["dur"] = round(dur, 2)

        def _corr(a, b):
            try:
                a = a.astype(float).ravel()
                b = b.astype(float).ravel()
                sa, sb = a.std(), b.std()
                if sa < 1e-6 or sb < 1e-6:
                    return 1.0 if abs(a.mean() - b.mean()) < 1e-6 else 0.0
                return float(np.corrcoef(a, b)[0, 1])
            except Exception:
                return 0.0

        prev_g, prev_blocks = None, None
        ts, ds, corrs, bds = [], [], [], []
        # точный ряд без сиков: один проход ffmpeg. Seek-джиттер OpenCV
        # в этой сборке врёт на секунды и размазывает замеры (доказано тестом).
        if dur > 900:
            step = max(step, 1.0)
        sfps = max(1, int(round(1.0 / step)))
        r = subprocess.run(
            [FFMPEG, "-v", "error", "-i", path,
             "-vf", f"fps={sfps},scale={GX}:{GY},format=gray",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, timeout=900)
        fs = GX * GY
        nf = len(r.stdout) // fs
        cap.release()
        if nf < 8:
            return res
        frames = np.frombuffer(r.stdout[:nf * fs], dtype=np.uint8).reshape(nf, GY, GX)
        dur = nf / sfps
        res["dur"] = round(dur, 2)
        for k in range(1, nf):
            g, p = frames[k], frames[k - 1]
            t = k * step
            ds.append(float(np.abs(g.astype(int) - p.astype(int)).mean()))
            ts.append(round(t, 3))
            bds.append([float(np.abs(
                g[y * bh:(y + 1) * bh, x * bw:(x + 1) * bw].astype(int)
                - p[y * bh:(y + 1) * bh, x * bw:(x + 1) * bw].astype(int)).mean())
                for y in range(CBY) for x in range(CBX)])
            h0 = np.histogram(g[:GY // 2], bins=16, range=(0, 256))[0].astype(float)
            h1 = np.histogram(g[GY // 2:], bins=16, range=(0, 256))[0].astype(float)
            corrs.append(_corr(h0, h1))
        if len(ds) < 8:
            return res
        try:
            # точные совпадения известных креативов — режут без вопросов
            res["sig"] = _match_sigs(frames, step, dur)
        except Exception:
            res["sig"] = []
        base = float(np.median(ds)) or 1.0
        res["base"] = round(base, 2)
        fz = max(1.5, 0.35 * base)      # уровень «стоит» относительно фона

        def _local_spike(s: float, e: float) -> bool:
            """Переходы в/из вставки заметны на ЛОКАЛЬНОМ фоне, а не на
            среднем по всему видео (на дёрганом геймплее глобальный порог
            слепнет и пропускает плашки с мягкими переходами)."""
            win = [d for t, d in zip(ts, ds) if (s - 3 <= t < s) or (e < t <= e + 3)]
            lb = float(np.median(win)) if win else base
            pre = max([d for t, d in zip(ts, ds) if s - 1 <= t <= s] or [0.0])
            post = max([d for t, d in zip(ts, ds) if e <= t <= e + 1] or [0.0])
            return max(pre, post) >= max(4.0, 1.2 * lb)

        def _runs(flags):
            runs, cur = [], None
            for k, low in enumerate(flags):
                if low and cur is None:
                    cur = k
                elif not low and cur is not None:
                    runs.append((cur, k))
                    cur = None
            if cur is not None:
                runs.append((cur, len(flags)))
            return runs

        taken = []
        # --- freeze ---
        out = []
        for a, b in _runs([d < fz for d in ds]):
            s = ts[a] - step
            e = ts[b - 1] if b - 1 < len(ts) else dur
            if not (min_d <= e - s <= max_d):
                continue
            if s < edge or e > dur - edge:
                continue
            if float(np.median(ds[a:b])) >= fz:
                continue
            if not _local_spike(s, e):
                continue  # без заметного стыка — просто спокойная сцена, не вставка
            out.append((round(s, 2), round(e, 2)))
        out.sort(key=lambda r: -(r[1] - r[0]))
        res["freeze"] = out[:6]
        taken += res["freeze"]
        # --- fullscreen: верх и низ кадра долго похожи + стыки по краям ---
        bc = float(np.median(corrs)) if corrs else 0.0
        cthr = min(0.95, max(0.80, bc + 0.15))
        full = []
        for a, b in _runs([c >= cthr for c in corrs]):
            s = ts[a] - step
            e = ts[b - 1] if b - 1 < len(ts) else dur
            if not (min_d <= e - s <= max_d):
                continue
            if s < edge or e > dur - edge:
                continue
            if any(not (e <= x or s >= y) for x, y in taken):
                continue
            if not _local_spike(s, e):
                continue
            full.append((round(s, 2), round(e, 2)))
        full.sort(key=lambda r: -(r[1] - r[0]))
        res["fullscreen"] = full[:4]
        taken += res["fullscreen"]
        # --- overlay: блок спокойнее фона (тонкая плашка делит блоки
        # с фоном — чистая абсолютная тишина не требуется) ---
        W = 12
        masks = []
        for k in range(len(ds)):
            lo, hi = max(0, k - W // 2), min(len(ds), k + W // 2 + 1)
            gmed = float(np.median(ds[lo:hi]))
            mask = []
            for bi in range(CBX * CBY):
                bmed = float(np.median([bds[j][bi] for j in range(lo, hi)]))
                mask.append(bmed < gmed * 0.5 and gmed > 5.0)
            masks.append(mask)
        groups, cur, curbits = [], None, None
        for k, mask in enumerate(masks):
            if any(mask) and cur is None:
                cur, curbits = k, [0] * (CBX * CBY)
            if cur is not None:
                for bi, on in enumerate(mask):
                    if on:
                        curbits[bi] += 1
                if not any(mask):
                    groups.append((cur, k, curbits))
                    cur, curbits = None, None
        if cur is not None:
            groups.append((cur, len(masks), curbits))
        ovs = []
        for a, b, bits in groups:
            s = ts[a] - step
            e = ts[min(b, len(ts) - 1)]
            # вычитаем занятое заморозками/фулскрином — остаток живёт сам
            rest = [(s, e)]
            for x, y in sorted(taken):
                nxt = []
                for a2, b2 in rest:
                    if b2 <= x or a2 >= y:
                        nxt.append((a2, b2))
                    else:
                        if a2 < x - 0.2:
                            nxt.append((a2, x))
                        if b2 > y + 0.2:
                            nxt.append((y, b2))
                rest = nxt
            for s, e in rest:
                if e - s < 2.5 or e - s > min(20.0, dur * 0.6):
                    continue
                if s < edge or e > dur - edge:
                    continue
                # ядро: блоки статичны почти всё время (осевший фон мерцает и отсеется)
                core = [bi for bi, c in enumerate(bits) if c >= (b - a) * 0.8]
                if len(core) / (CBX * CBY) < 0.04:
                    continue
                xs = [bi % CBX for bi in core]
                ys = [bi // CBX for bi in core]
                fill = len(core) / ((max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1))
                if fill < 0.4:
                    continue
                # появление/исчезновение — заметным стыком на локальном фоне
                if not _local_spike(s, e):
                    continue
                area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1) / (CBX * CBY)
                if area < 0.06 or area > 0.5:
                    continue
                ovs.append({"s": round(s, 2), "e": round(e, 2),
                            "box": (round(min(xs) / CBX, 3), round(min(ys) / CBY, 3),
                                    round((max(xs) - min(xs) + 1) / CBX, 3),
                                    round((max(ys) - min(ys) + 1) / CBY, 3))})
        ovs.sort(key=lambda o: (o["s"], o["e"]))
        merged = []
        for o in ovs:
            if merged and o["s"] - merged[-1]["e"] < 1.0 and _iou(
                    merged[-1]["box"], o["box"]) > 0.2:
                m = merged[-1]
                m["e"] = max(m["e"], o["e"])
                x0 = min(m["box"][0], o["box"][0])
                y0 = min(m["box"][1], o["box"][1])
                x1 = max(m["box"][0] + m["box"][2], o["box"][0] + o["box"][2])
                y1 = max(m["box"][1] + m["box"][3], o["box"][1] + o["box"][3])
                m["box"] = (round(x0, 3), round(y0, 3),
                            round(x1 - x0, 3), round(y1 - y0, 3))
            else:
                merged.append(dict(o))
        merged.sort(key=lambda o: -(o["e"] - o["s"]))
        res["overlay"] = merged[:4]
        # --- rect: начало и конец вставки — одновременный срез компактного
        # прямоугольника (остальное продолжает двигаться). Ловит и статичные,
        # и анимированные вставки: у всех есть границы, даже если внутри движение.
        try:
            nthr = []
            for bi in range(CBX * CBY):
                bb = float(np.median([bds[j][bi] for j in range(len(ds))])) or 1.0
                nthr.append(max(8.0, 1.4 * bb))
            events = []  # (t, bbox)
            NB = CBX * CBY
            for k in range(len(ds)):
                sp = {bi for bi in range(NB) if bds[k][bi] >= nthr[bi]}
                if not sp or len(sp) > int(NB * 0.85):
                    continue  # тишина или глобальный монтажный стык — не вставка
                # связные компоненты (4-связность)
                seen_c, comps = set(), []
                for seed in sorted(sp):
                    if seed in seen_c:
                        continue
                    stack, comp = [seed], []
                    seen_c.add(seed)
                    while stack:
                        cur_b = stack.pop()
                        comp.append(cur_b)
                        x0, y0 = cur_b % CBX, cur_b // CBX
                        for nx, ny in ((x0 - 1, y0), (x0 + 1, y0),
                                       (x0, y0 - 1), (x0, y0 + 1)):
                            nb = ny * CBX + nx
                            if (0 <= nx < CBX and 0 <= ny < CBY
                                    and nb in sp and nb not in seen_c):
                                seen_c.add(nb)
                                stack.append(nb)
                    if len(comp) >= 4:
                        xs = [c % CBX for c in comp]
                        ys = [c // CBX for c in comp]
                        bwq = max(xs) - min(xs) + 1
                        bhq = max(ys) - min(ys) + 1
                        if (len(comp) / (bwq * bhq) >= 0.45
                                and 0.05 <= bwq * bhq / NB <= 0.75):
                            events.append((ts[k], (min(xs) / CBX, min(ys) / CBY,
                                                   bwq / CBX, bhq / CBY)))
            rects = []
            used_e = set()

            def _thumb(t0, t1):
                idx = [k for k in range(1, nf) if t0 <= k * step <= t1]
                if not idx:
                    return None
                m = np.mean([frames[k].astype(float) for k in idx], axis=0)
                return cv2.resize(m.astype(np.uint8), (48, 27))
            for i, (t0, b0) in enumerate(events):
                cands = []
                for j in range(i + 1, len(events)):
                    t1, b1 = events[j]
                    if t1 - t0 > 20.0:
                        break
                    if j in used_e:
                        continue
                    if t1 - t0 < 1.5:
                        continue
                    q = _iou(b0, b1)
                    if q >= 0.45:
                        cands.append((q, j, t1, b1))
                # сначала самое похожее место стыка, а не первое попавшееся
                cands.sort(key=lambda c: -c[0])
                done = False
                for _q, j, t1, b1 in cands:
                    x0 = min(b0[0], b1[0])
                    y0 = min(b0[1], b1[1])
                    x1 = max(b0[0] + b0[2], b1[0] + b1[2])
                    y1 = max(b0[1] + b0[3], b1[1] + b1[3])
                    # вставка-захват: банда во всю ширину кадра (мидроллы и
                    # фулскрины рисуют от края до края; куски геймплея — нет)
                    if not (x0 <= 0.05 and x1 >= 0.95):
                        continue
                    s, e = t0 - step, t1
                    if e - s < 1.5 or e - s > 20.0:
                        continue
                    if s < edge or e > dur - edge:
                        continue
                    if any(not (e <= x or s >= y) for x, y in taken):
                        continue
                    # содержимое внутри — чужое (далеко от обоих краёв)
                    th_in = _thumb(s + 0.5, e - 0.5)
                    th_pre = _thumb(s - 2.5, s - 0.5)
                    th_post = _thumb(e + 0.5, e + 2.5)
                    if th_in is None or th_pre is None or th_post is None:
                        continue
                    d_foreign = min(
                        float(np.abs(th_in.astype(int) - th_pre.astype(int)).mean()),
                        float(np.abs(th_in.astype(int) - th_post.astype(int)).mean()))
                    if d_foreign < 14.0:
                        continue
                    used_e.add(j)
                    rects.append({"s": round(s, 2), "e": round(e, 2),
                                  "box": (round(x0, 3), round(y0, 3),
                                          round(x1 - x0, 3), round(y1 - y0, 3))})
                    taken.append((round(s, 2), round(e, 2)))
                    done = True
                    break
                if done:
                    continue
            rects.sort(key=lambda o: -(o["e"] - o["s"]))
            res["rect"] = rects[:4]
        except Exception:
            pass
    except Exception:
        pass
    return res


def _iou(a, b):
    try:
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
        inter = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
        u = a[2] * a[3] + b[2] * b[3] - inter
        return inter / u if u > 0 else 0.0
    except Exception:
        return 0.0


def detect_banner_freezes(path: str, min_d: float = 2.0,
                           max_d: float = 12.0, edge: float = 3.0,
                           step: float = 0.25, thresh: float = 4.0):
    """Совместимость: только заморозки (см. analyze_inserts)."""
    try:
        return analyze_inserts(path, min_d, max_d, edge, step)["freeze"]
    except Exception:
        return []


def _probe_wh_fps(path: str):
    import cv2
    c = cv2.VideoCapture(path)
    w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(c.get(cv2.CAP_PROP_FPS) or 0) or 0
    c.release()
    return w, h, n, fps


def _has_audio_stream(path: str) -> bool:
    try:
        r = subprocess.run([FFMPEG, "-i", path],
                           capture_output=True, text=True, timeout=30)
        return " Audio:" in r.stderr
    except Exception:
        return False


def remove_silence(src: str, out: str, min_sil: float = 0.8,
                   thresh_db: float = -40.0, pad: float = 0.25,
                   progress=None) -> str:
    """Аккуратно вырезает тишину: паузы от 0.8с тише -40дБ, по краям речи
    подушка 0.25с, короткие огрызки не трогаем. Без пауз — возвращает src
    как есть (без перекода)."""
    import shutil
    W, H, N, FPS = _probe_wh_fps(src)
    if N <= 0 or FPS <= 0:
        raise RuntimeError("не читается видео")
    dur = N / FPS
    r = subprocess.run(
        [FFMPEG, "-v", "info", "-i", src,
         "-af", f"silencedetect=noise={thresh_db}dB:d={min_sil}",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=600)
    sils = []
    cur_s = None
    for line in (r.stderr or "").splitlines():
        m1 = re.search(r"silence_start:\s*([\d.]+)", line)
        m2 = re.search(r"silence_end:\s*([\d.]+)", line)
        if m1:
            cur_s = float(m1.group(1))
        elif m2 and cur_s is not None:
            sils.append((cur_s, float(m2.group(1))))
            cur_s = None
    cuts = [(max(s + pad, 0.0), min(e - pad, dur))
            for s, e in sils if e - pad > s + pad + 0.3]
    if not cuts:
        if progress:
            progress(100, "тишины нет — как есть")
        return src
    keep, cur = [], 0.0
    for s, e in sorted(cuts):
        if s > cur + 0.1:
            keep.append((cur, s))
        cur = max(cur, e)
    if cur < dur - 0.1:
        keep.append((cur, dur))
    keep = [(s, e) for s, e in keep if e - s > 0.3]
    if not keep:
        return src
    if progress:
        progress(20, f"режу {len(cuts)} пауз…")
    fc = []
    for i, (s, e) in enumerate(keep):
        fc.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]")
    vcat = "".join(f"[v{i}]" for i in range(len(keep)))
    fc.append(f"{vcat}concat=n={len(keep)}:v=1:a=0[vout]")
    if _has_audio_stream(src):
        for i, (s, e) in enumerate(keep):
            fc.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
        acat = "".join(f"[a{i}]" for i in range(len(keep)))
        fc.append(f"{acat}concat=n={len(keep)}:v=0:a=1[aout]")
        amap = ["-map", "[aout]"]
    else:
        amap = []
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-i", src,
         "-filter_complex", ";".join(fc),
         "-map", "[vout]"] + amap + _rand_meta() + _video_codec_args() + [
             *_AUDIO_OUT, "-movflags", "+faststart", out],
        timeout=1800, check=True)
    if progress:
        progress(100, "готово")
    if not _has_video(out):
        raise RuntimeError("чистка тишины не удалась")
    return out


def swap_banner(src: str, out: str, banner_path: str | None = None,
                remove: bool = True, progress=None,
                at: float | None = None,
                cutranges: list | None = None) -> dict:
    """Замена чужого баннера: вырезает найденные вставки (или заданные
    cutranges=[(s,e)] — ручной отрезок) и ставит свой баннер
    (по умолчанию assets/banner.mp4) СТРОГО по середине итога либо в at.
    Возвращает {removed, mid, total}."""
    import os as _os
    bpath = banner_path if banner_path and _os.path.exists(banner_path) else BANNER_FILE
    if not _os.path.exists(bpath):
        raise RuntimeError("нет баннера для вставки")
    W, H, N, FPS = _probe_wh_fps(src)
    if W <= 0 or H <= 0 or N <= 0 or FPS <= 0:
        raise RuntimeError("не читается видео")
    dur = N / FPS
    bw, bh, bn, bfps = _probe_wh_fps(bpath)
    b_eff = (bn / max(bfps, 1)) / BANNER_SPEED if bn and bfps else 4.0
    freezes = analyze_inserts(src) if (remove and cutranges is None) else {
        "freeze": [], "fullscreen": [], "overlay": [], "rect": [], "sig": []}
    if cutranges is not None:
        # ручной отрезок: режем ровно его, свой баннер — на его начало
        cuts = sorted((float(s), float(e)) for s, e in cutranges
                      if float(e) > float(s) + 0.2)
        if at is None and cuts:
            at = cuts[0][0]
    else:
        cuts = sorted(freezes["freeze"] + freezes["fullscreen"]
                      + [(o["s"], o["e"]) for o in freezes.get("sig", [])])
    try:
        with open(out + ".analysis.json", "w", encoding="utf-8") as _af:
            _af.write(json.dumps(
                {"freeze": freezes["freeze"],
                 "fullscreen": freezes["fullscreen"],
                 "overlay": freezes["overlay"],
                 "rect": freezes.get("rect", []),
                 "sig": freezes.get("sig", []),
                 "base": freezes.get("base", 0),
                 "dur": round(dur, 2)}, ensure_ascii=False))
    except Exception:
        pass
    kept = []
    cur = 0.0
    for s, e in cuts:
        if s > cur + 0.2:
            kept.append((cur, min(s, dur)))
        cur = max(cur, e)
    if cur < dur - 0.2:
        kept.append((cur, dur))
    if not kept:
        raise RuntimeError("после вырезания ничего не осталось")
    kept_dur = sum(e - s for s, e in kept)
    total = kept_dur + b_eff
    oA = total / 2  # старт своего баннера — строго середина итога
    if at is not None:
        try:
            oA = min(max(float(at), 0.0), max(total - b_eff, 0.0))
        except (TypeError, ValueError):
            pass
    ins = min(oA, kept_dur)  # точка вставки в координатах kept (баннеров до неё нет)
    # режем kept-кусок под точкой вставки, чтобы баннер встал ровно на total/2
    new_kept, done, acc, src_mid = [], False, 0.0, kept[0][0]
    for s, e in kept:
        if not done and acc + (e - s) >= ins - 1e-6:
            cut = min(max(s + (ins - acc), s), e)
            if cut > s + 0.2:
                new_kept.append((s, cut))
            new_kept.append("MID")
            if e - cut > 0.2:
                new_kept.append((cut, e))
            src_mid = min(max(cut, 0.0), max(dur - 0.1, 0.0))
            done = True
        else:
            new_kept.append((s, e))
        acc += e - s
    if not done:
        new_kept.append("MID")
        src_mid = max(dur - 0.1, 0.0)
    kept = new_kept
    # накрытия бегущих вставок: источник -> kept -> финал (сдвиг за мид-баннером)
    def _to_kept(t):
        off = 0.0
        for s, e in cuts:
            if t >= e - 1e-6:
                off += e - s
            elif t > s + 1e-6:
                return None
        return t - off
    covers = []
    budget = 0.4 * max(kept_dur, 1.0)  # накрыть можно max 40% хронометража
    used = 0.0
    for ov in sorted(freezes["overlay"], key=lambda o: -(o["e"] - o["s"])):
        a, b = _to_kept(ov["s"]), _to_kept(ov["e"])
        if a is None or b is None or b - a < 2.0:
            continue
        if used + (b - a) > budget:
            continue
        covers.append({"s": a, "e": b, "ss": ov["s"], "se": ov["e"],
                       "box": ov["box"]})
        used += b - a
        if len(covers) >= 2:
            break
    covf = []
    for ov in covers[:2]:
        s, e = ov["s"], ov["e"]
        # кусок через точку вставки делим: хвост уехал на b_eff вместе с контентом
        if s < ins - 1e-6 and e > ins + 1e-6:
            parts = [(s, ins), (ins + b_eff, e + b_eff)]
        elif s >= ins - 1e-6:
            parts = [(s + b_eff, e + b_eff)]
        else:
            parts = [(s, e)]
        for ps, pe in parts:
            if pe - ps >= 1.0:
                covf.append((ps, pe, ov["box"]))
    if progress:
        progress(10, f"нашёл: вырезать {len(cuts)}, накрыть {len(covf)}…")
    frame_png = out + ".freeze.png"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-ss", f"{src_mid:.2f}", "-i", src,
         "-frames:v", "1", frame_png],
        timeout=120, check=True)
    # свой баннер вписываем в кадр (центр); зелёный фон хромакей-файлов
    # вычитается ключом (_sw_key), сквозь него видно исходное видео
    sw = min(bw or W, W)
    sh = max(int(round((bh or H) * sw / max(bw or W, 1))), 1)
    if sh > H:
        sh = H
        sw = max(int(round((bw or W) * sh / max(bh or H, 1))), 1)
    ox, oy = (W - sw) // 2, (H - sh) // 2
    vparts = []
    vi = 0
    for part in kept:
        if part == "MID":
            continue
        s, e = part
        vparts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{vi}]")
        vi += 1
    fps_i = max(int(round(FPS)), 1)
    vparts.append(f"[1:v]trim=start=0:end={b_eff:.3f},setpts=PTS-STARTPTS[vf]")
    vparts.append(f"[2:v]split={1 + len(covf)}"
                  + "".join(f"[cbs{i}]" for i in range(1 + len(covf))))
    _sw_key = _banner_key_vf(bpath)
    bfs = f"[cbs0]scale={sw}:{sh},fps={fps_i},{_sw_key}format=yuva420p,colorchannelmixer=aa=0.95"
    bfs += ",setpts=PTS/{:.3f}[vb]".format(BANNER_SPEED)
    vparts.append(bfs)
    vparts.append(f"[vf][vb]overlay={ox}:{oy}:eof_action=pass[vmid]")
    # итоговая склейка: куски + середина с баннером ровно на total/2
    order = []
    vi = 0
    for part in kept:
        if part == "MID":
            order.append("[vmid]")
        else:
            order.append(f"[v{vi}]")
            vi += 1
    vparts.append("".join(order) + f"concat=n={len(order)}:v=1:a=0[vcat]")
    cur = "[vcat]"
    for ci, (s, e, box) in enumerate(covf):
        bx, by, bwq, bhq = box
        rx = max(int(bx * W - 0.02 * W), 0)
        ry = max(int(by * H - 0.02 * H), 0)
        rw = min(int(bwq * W + 0.04 * W), W - rx)
        rh = min(int(bhq * H + 0.04 * H), H - ry)
        csw = min(max(rw, 2), W)
        csh = max(int(round((bh or H) * csw / max(bw or W, 1))), 1)
        if csh > H:
            csh, csw = H, max(int(round((bw or W) * H / max(bh or H, 1))), 1)
        ox2 = min(max(rx + rw // 2 - csw // 2, 0), max(W - csw, 0))
        oy2 = min(max(ry + rh // 2 - csh // 2, 0), max(H - csh, 0))
        vparts.append(
            f"{cur}drawbox=x={rx}:y={ry}:w={rw}:h={rh}:color=black@0.85:t=fill:"
            f"enable='between(t,{s:.3f},{e:.3f})'[d{ci}]")
        vparts.append(
            f"[cbs{ci + 1}]scale={csw}:{csh},fps={fps_i},{_sw_key}format=yuva420p,"
            f"colorchannelmixer=aa=0.95,"
            f"setpts=PTS-STARTPTS+{s:.3f}/TB[cb{ci}]")
        vparts.append(
            f"[d{ci}][cb{ci}]overlay={ox2}:{oy2}:eof_action=pass:"
            f"enable='between(t,{s:.3f},{e:.3f})'[o{ci}]")
        cur = f"[o{ci}]"
    vparts.append(f"{cur}copy[vout]")
    src_audio = _has_audio_stream(src)
    ban_audio = _has_audio_stream(bpath)
    if src_audio:
        aparts = []
        ai = 0
        for part in kept:
            if part == "MID":
                continue
            s, e = part
            aparts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{ai}]")
            ai += 1
        aorder = []
        ai = 0
        for part in kept:
            if part == "MID":
                aorder.append("[asil]" if not ban_audio else "[amix]")
            else:
                aorder.append(f"[a{ai}]")
                ai += 1
        fc = ";".join(vparts) + ";"
        fc += ";".join(aparts) + ";"
        fc += f"anullsrc=r=48000:cl=stereo:d={b_eff:.3f}[asil];"
        if ban_audio:
            fc += (f"[2:a]atempo={BANNER_SPEED:.3f},atrim=0:{b_eff:.3f},"
                   f"asetpts=PTS-STARTPTS,"
                   f"volume=0.5,aresample=48000,"
                   f"aformat=channel_layouts=stereo[ba];"
                   f"[asil][ba]amix=inputs=2:duration=first[amix];")
        fc += "".join(aorder) + f"concat=n={len(aorder)}:v=0:a=1[aout]"
        if covf:
            duck = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e, _ in covf)
            fc += f";[aout]volume=0:enable='{duck}'[aout2]"
            amap = ["-map", "[aout2]"]
        else:
            amap = ["-map", "[aout]"]
    else:
        fc = ";".join(vparts)
        amap = []
    if progress:
        progress(40, "режу и вставляю баннер…")
    cmd = [FFMPEG, "-y", "-v", "error", "-i", src,
           "-loop", "1", "-framerate", str(fps_i), "-t", f"{b_eff:.3f}",
           "-i", frame_png, "-i", bpath,
           "-filter_complex", fc,
           "-map", "[vout]"] + amap + _rand_meta() + _video_codec_args() + [
               *_AUDIO_OUT, "-movflags", "+faststart", out]
    subprocess.run(cmd, timeout=1800, check=True)
    try:
        if os.path.exists(frame_png):
            os.remove(frame_png)
    except OSError:
        pass
    if progress:
        progress(100, "готово")
    if not _has_video(out):
        raise RuntimeError("замена баннера не удалась")
    return {"removed": cuts,
            "covered": [{"s": c["ss"], "e": c["se"]} for c in covers],
            "mid": round(oA, 2), "total": round(total, 2),
            "analysis": {"freeze": freezes["freeze"],
                         "fullscreen": freezes["fullscreen"],
                         "overlay": freezes["overlay"],
                         "rect": freezes.get("rect", []),
                         "sig": freezes.get("sig", []),
                         "base": freezes.get("base", 0),
                         "dur": round(dur, 2)}}


def uniqualize_video(src: str, out: str, opts: dict | None = None,
                     progress=None) -> str:
    """Уникализатор: делает копию видео с другим хешем и отпечатком —
    лёгкие необратимые правки картинки и звука против Content ID и
    антиплагиата соцсетей. Глаз разницы почти не видит.
    По умолчанию уже злое: скорость +1%, питч, подрезка краёв.
    ВАЖНО для TikTok: каждый прогон ОБЯЗАН отличаться (скорость/зум/цвет
    рандомятся), иначе повторная заливка чужого видео с вотермаркой = 0."""
    import random
    opts = opts or {}
    try:
        _spd0 = float(opts.get("speed", 1.01))
    except (TypeError, ValueError):
        _spd0 = 1.01
    try:
        _zm0 = float(opts.get("zoom", 2.0))
    except (TypeError, ValueError):
        _zm0 = 2.0
    # дефолтные вызовы (цепочка unique_in и форма без явных настроек):
    # скорость и зум обязаны гулять, иначе все файлы пачки с тем же темпом
    _auto = (abs(_spd0 - 1.01) < 1e-9 or abs(_spd0 - 1.0) < 1e-9) and abs(_zm0 - 2.0) < 1e-9
    speed = _spd0
    if _auto:
        # цепочка (скачал чужое -> вставил баннер -> уникализировал):
        # скорость обязана гулять, иначе все файлы одной пачки с тем же темпом
        speed = round(random.uniform(1.015, 1.04), 3)
    speed = min(1.05, max(0.95, speed))
    mirror = bool(opts.get("mirror", False))
    color = str(opts.get("color", "medium"))
    zoom_pct = _zm0
    if _auto:
        # зум гуляет шире: заодно подъедает края с тикток-вотермаркой оригинала
        zoom_pct = round(random.uniform(2.5, 4.5), 1)
    zoom_pct = min(6.0, max(0.0, zoom_pct))
    grain = bool(opts.get("grain", True))
    pitch = bool(opts.get("pitch", True))
    # подрезка краёв: ломает точное совпадение длины и стартовых кадров.
    # Расширили до 0.3-0.8с с каждой стороны: длительность гуляет ±1.5с.
    th, te = 0.0, 0.0
    _w, _h = 0, 0
    try:
        _w, _h, _n, _f = _probe_wh_fps(src)
        _dur = _n / max(_f, 1) if _n and _f else 0
    except Exception:
        _dur = 0
    if _dur > 5:
        th = round(random.uniform(0.3, 0.8), 2)
        te = round(_dur - random.uniform(0.3, 0.8), 2)
    # сила цветокора по пресету + лёгкий джиттер (каждый прогон уникален)
    if color == "light":
        cc = (1.02, 1.04, 1.05, 1.09, 0.004, 0.010, -2, 2, 2, 3)
    elif color == "strong":
        cc = (1.07, 1.12, 1.14, 1.20, 0.012, 0.025, -7, 7, 4, 7)
    else:
        cc = (1.04, 1.08, 1.09, 1.15, 0.008, 0.018, -4, 4, 2, 5)
    vf = []
    if th > 0:
        vf.append(f"trim=start={th:.2f}:end={te:.2f},setpts=PTS-STARTPTS")
    if mirror:
        vf.append("hflip")
    if abs(speed - 1.0) > 1e-4:
        vf.append(f"setpts={1.0 / speed:.4f}*PTS")  # видео догоняет звук
    z = 1.0 + zoom_pct / 100.0
    if z > 1.0001 and _w > 0 and _h > 0:
        # случайный центр кропа (±3%): одинаковый исходник даёт разный кадр
        jx = random.uniform(-0.03, 0.03)
        jy = random.uniform(-0.03, 0.03)
        vf.append(f"scale=iw*{z:.3f}:ih*{z:.3f},"
                  f"crop={_w}:{_h}:x='(in_w-{_w})/2+({jx:.4f}*in_w)':"
                  f"y='(in_h-{_h})/2+({jy:.4f}*in_h)'")
    vf.append(
        f"eq=contrast={random.uniform(cc[0], cc[1]):.3f}:"
        f"saturation={random.uniform(cc[2], cc[3]):.3f}:"
        f"brightness={random.uniform(cc[4], cc[5]):.3f},"
        f"hue=h={random.uniform(cc[6], cc[7]):.1f}"
        + (f",noise=alls={random.randint(cc[8], cc[9])}:allf=t" if grain else ""))
    vf.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    af = []
    if th > 0:
        af.append(f"atrim=start={th:.2f}:end={te:.2f},asetpts=PTS-STARTPTS")
    if abs(speed - 1.0) > 1e-4:
        af.append(f"atempo={speed:.4f}")
    if pitch:
        k = random.uniform(1.01, 1.035)
        af.append(f"aresample=44100,asetrate=44100*{k:.4f},"
                  f"aresample=44100,atempo={1 / k:.4f}")
    af.append(_LOUDNORM)
    # лёгкий джиттер громкости после нормализации: у каждого файла свой уровень
    af.append(f"volume={random.uniform(0.95, 1.05):.3f}")
    af_str = _tiktok_af(*af)
    if progress:
        progress(5, "уникализация…")
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-i", src,
         "-vf", ",".join(vf),
         "-af", af_str,
         "-map", "0:v", "-map", "0:a?"] + _rand_meta()
        + _video_codec_args()
        + [*_AUDIO_OUT, "-movflags", "+faststart", out],
        timeout=1800, check=True)
    if progress:
        progress(100, "готово")
    if not _has_video(out):
        raise RuntimeError("уникализация не удалась")
    return out


def compute_parts(segs, min_len: float = 45.0, max_len: float = 59.0):
    """Режет ВСЁ видео на последовательные части 45–59с по границам слов.
    Возвращает [{start, end}] без пропусков и наложений."""
    import bisect
    if not segs:
        return []
    t0, t1 = float(segs[0]["start"]), float(segs[-1]["end"])
    total = t1 - t0
    if total <= 0:
        return []
    if total <= max_len:
        return [{"start": round(t0, 1), "end": round(t1, 1)}]
    n = max(1, round(total / 52.0))
    for _ in range(5):
        if total / n <= max_len:
            break
        n += 1
    bounds = _word_bounds(segs)
    ends = sorted({we for _, we in bounds})
    out = []
    prev = t0
    for i in range(1, n + 1):
        if i >= n:
            e = t1
        else:
            raw = t0 + total * i / n
            j = bisect.bisect_left(ends, raw)
            cands = [x for x in ends[max(0, j - 6):j + 6] if abs(x - raw) <= 3.0]
            e = min(cands, key=lambda x: abs(x - raw)) if cands else raw
            e = min(max(e, prev + 5), t1)
        s = prev
        if e - s >= 5:
            out.append({"start": round(s, 1), "end": round(e, 1)})
        prev = e
        if prev >= t1 - 0.5:
            break
    return out


def render_blur9(src: str, moment: dict, segs, out: str,
                 captions: bool = True, progress=None,
                 antidetect: bool = False, mirror: bool = False):
    """Блюр 9:16 под кражу у конкурентов: фон — блюр копия кадра,
    центр — резкий fit (края с вотермарками режутся), зеркало + зум +
    сдвиг цвета против Content ID. Сабы наши, звук оригинала."""
    import cv2
    import random as _brnd
    from PIL import Image, ImageDraw
    start, end = float(moment["start"]), float(moment["end"])
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fps = round(min(float(fps or 30), 30), 2)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if nfr and fps > 0:
        end = min(end, nfr / fps)
    if end - start < 1:
        cap.release()
        raise RuntimeError("момент за пределами видео")
    dur = end - start
    font = _font(64)
    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1920",
           "-r", str(fps),
           "-i", "pipe:0",
           "-ss", str(start), "-t", str(round(dur, 2)), "-i", src,
           "-map", "0:v", "-map", "1:a?"]
    qvf = _quality_vf()
    if antidetect:
        vf, af = _antidetect_filters()
        qvf += "," + vf
        cmd += ["-vf", qvf, "-af", _tiktok_af(
            af, _LOUDNORM, f"volume={_brnd.uniform(0.95, 1.05):.3f}")]
    else:
        # always-on: лёгкий питч + громкость, иначе звук 1:1 с оригиналом
        _bk = _brnd.uniform(1.005, 1.012)
        cmd += ["-vf", qvf, "-af", _tiktok_af(
            f"aresample=44100,asetrate=44100*{_bk:.4f},"
            f"aresample=44100,atempo={1 / _bk:.4f}",
            _LOUDNORM, f"volume={_brnd.uniform(0.95, 1.05):.3f}")]
    cmd += _video_codec_args() + ["-pix_fmt", "yuv420p",
            *_AUDIO_OUT, "-movflags", "+faststart",
            "-t", str(round(dur, 2)), out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    _warm_start(cap)
    total = max(int(round(dur * fps)), 1)
    last = None
    # always-on наезд: у каждого рендера свой кроп центра
    if antidetect:
        _bzw, _bzh = 1040, 1850
    else:
        _bzw, _bzh = _brnd.randint(1000, 1020), _brnd.randint(1780, 1815)
    try:
        for k in range(total):
            ok, fr = cap.read()
            if not ok:
                if last is None:
                    break
                fr = last
            else:
                if mirror:
                    fr = cv2.flip(fr, 1)
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            last = fr
            t = start + k / fps
            h, w = fr.shape[:2]
            # фон: блюр-копия на весь кадр, затемнена
            small = cv2.resize(fr, (135, 240), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), 8)
            bg = cv2.resize(small, (1080, 1920), interpolation=cv2.INTER_LINEAR)
            bg = cv2.convertScaleAbs(bg, alpha=0.6)
            # центр: fit внутрь (режем края с вотермарками) + свой наезд у каждого
            sc = min(_bzw / max(w, 1), _bzh / max(h, 1))
            fw, fh = max(int(round(w * sc)), 1), max(int(round(h * sc)), 1)
            fg = cv2.resize(fr, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
            x0, y0 = (1080 - fw) // 2, (1920 - fh) // 2
            frame = bg
            frame[y0:y0 + fh, x0:x0 + fw] = fg
            if captions:
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
                        y = min(1330 - block, 1560)
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
            if moment.get("cta") and t > end - 2.5:
                frame = _with_part2_cta(frame, font)
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
    if proc.returncode != 0 or not _has_video(out):
        raise RuntimeError("рендер не удался")
    return out


def render_clip(src: str, moment: dict, segs, out: str, facetrack: bool = True,
                captions: bool = True, progress=None, vmode: str = "face",
                antidetect: bool = False, mirror: bool = False,
                banner: bool = True, gameplay: str = "subway",
                max_total: float | None = None, banner_path: str | None = None):
    if vmode == "split":
        return render_split_clip(src, moment, segs, out, facetrack=facetrack,
                                 captions=captions, progress=progress,
                                 antidetect=antidetect, mirror=mirror,
                                 banner=banner, gameplay=gameplay,
                                 banner_path=banner_path,
                                 max_total=max_total)
    if vmode == "blur9":
        return render_blur9(src, moment, segs, out,
                            captions=captions, progress=progress,
                            antidetect=antidetect, mirror=mirror)
    import cv2
    from PIL import Image, ImageDraw
    start, end = float(moment["start"]), float(moment["end"])
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fps = round(min(float(fps or 30), 30), 2)
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
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1920", "-r", str(fps),
           "-i", "pipe:0",
           "-ss", str(start), "-t", str(round(dur, 2)), "-i", src,
           "-map", "0:v", "-map", "1:a?"] + _rand_meta()
    qvf = _quality_vf()
    if antidetect:
        vf, af = _antidetect_filters()
        qvf += "," + vf
        cmd += ["-vf", qvf, "-af", _tiktok_af(
            af, _LOUDNORM, f"volume={_rnd.uniform(0.95, 1.05):.3f}")]
    else:
        # Базовый always-on слой: даже без чекбокса звук чуть отличается
        # от оригинала (иначе Content ID по аудио = 0 просмотров).
        _ck = _rnd.uniform(1.005, 1.012)
        cmd += ["-vf", qvf, "-af", _tiktok_af(
            f"aresample=44100,asetrate=44100*{_ck:.4f},"
            f"aresample=44100,atempo={1 / _ck:.4f}",
            _LOUDNORM, f"volume={_rnd.uniform(0.95, 1.05):.3f}")]
    cmd += _video_codec_args() + ["-pix_fmt", "yuv420p",
            *_AUDIO_OUT, "-movflags", "+faststart",
            "-t", str(round(dur, 2)), out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    _warm_start(cap)  # первый кадр не чёрный
    total = max(int(round(dur * fps)), 1)
    last = None
    import random as _rnd
    zm = _rnd.uniform(1.03, 1.06) if antidetect else _rnd.uniform(1.01, 1.02)
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
                if antidetect:  # наезд 3-6%: прячет края, лого и ватермарки
                    h0, w0 = fr.shape[:2]
                    fr = cv2.resize(fr, (int(w0 * zm), int(h0 * zm)),
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
            if moment.get("cta") and t > end - 2.5:
                frame = _with_part2_cta(frame, font)
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
    if proc.returncode != 0 or not _has_video(out):
        raise RuntimeError("рендер не удался")
    return out
