"""ClipSaw web — ссылка на YouTube -> пачка вертикальных шортсов.
Только нарезка сериалов/фильмов: сцены -> транскрибация -> LLM-скоринг -> 9:16 -> субтитры."""
import json
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

os.environ["PATH"] = (os.path.expanduser("~/.deno/bin") + ":"
                       + os.path.expanduser("~/.local/bin") + ":"
                       + "/opt/homebrew/bin:/usr/local/bin:"
                       + os.environ.get("PATH", ""))

import ai_clips as ac
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "shorts")
TMP_DIR = os.path.join(BASE, "tmp")
LEGACY_DIR = os.path.join(BASE, "legacy")
THUMB_DIR = os.path.join(BASE, "thumbs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(LEGACY_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)
UNIQ_IN = os.path.join(BASE, "unique_in")
UNIQ_OUT = os.path.join(BASE, "unique")
os.makedirs(UNIQ_IN, exist_ok=True)
os.makedirs(UNIQ_OUT, exist_ok=True)
UNIQ_JOBS = {}
FFMPEG = os.path.expanduser("~/.local/bin/ffmpeg")

DB = os.path.join(BASE, "data.db")
CFG = os.path.join(BASE, "config.json")
DEFAULT_CFG = {"gemini_key": "", "drex_key": "", "want": 8, "gameplay": "subway", "lens": "40-59", "limit_min": 0,
               "vmode": "split", "captions": True, "facetrack": True,
               "antidetect": False, "mirror": False, "banner": True,
               "asr_model": "base", "max_workers": 2}


DB_VERSION = 6


def _connect():
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


def init_db():
    """Идемпотентные миграции: можно вызывать при каждом старте."""
    c = _connect()
    try:
        with c:
            c.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
            row = c.execute("SELECT v FROM meta WHERE k='version'").fetchone()
            ver = int(row["v"]) if row else 0
            if ver < 1:
                c.execute("""CREATE TABLE IF NOT EXISTS jobs(
                  id TEXT PRIMARY KEY, title TEXT, url TEXT, status TEXT,
                  created REAL, finished REAL, clips TEXT, total_size INTEGER)""")
                c.execute("""CREATE TABLE IF NOT EXISTS clips(
                  job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                  idx INTEGER NOT NULL, title TEXT, score INTEGER,
                  verdict TEXT, hook TEXT, dur REAL, size INTEGER,
                  views INTEGER NOT NULL DEFAULT 0, created REAL,
                  PRIMARY KEY (job_id, idx))""")
                c.execute("CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs(status)")
                c.execute("CREATE INDEX IF NOT EXISTS ix_jobs_finished ON jobs(finished)")
                c.execute("CREATE INDEX IF NOT EXISTS ix_clips_job ON clips(job_id)")
            if ver < 2:
                # бэкфилл: разложить clips-JSON старых задач в таблицу clips
                have = {r["job_id"] for r in
                        c.execute("SELECT DISTINCT job_id FROM clips").fetchall()}
                for r in c.execute("SELECT id, clips FROM jobs").fetchall():
                    if r["id"] in have:
                        continue
                    try:
                        arr = json.loads(r["clips"] or "[]")
                    except Exception:
                        continue
                    for k, x in enumerate(arr):
                        try:
                            c.execute(
                                "INSERT OR IGNORE INTO clips"
                                "(job_id,idx,title,score,verdict,hook,dur,size,views,created)"
                                " VALUES (?,?,?,?,?,?,?,?,0,?)",
                                (r["id"], int(x.get("i", k)),
                                 str(x.get("title", ""))[:200],
                                 int(x.get("score", 0) or 0),
                                 str(x.get("verdict", ""))[:240],
                                 str(x.get("hook", ""))[:40],
                                 float(x.get("dur", 0) or 0),
                                 int(x.get("size", 0) or 0), time.time()))
                        except Exception:
                            continue
            if ver < 3:
                cols = [r[1] for r in c.execute("PRAGMA table_info(jobs)").fetchall()]
                if "scorer" not in cols:
                    c.execute("ALTER TABLE jobs ADD COLUMN scorer TEXT NOT NULL DEFAULT 'heuristic'")
            if ver < 4:
                cols = [r[1] for r in c.execute("PRAGMA table_info(clips)").fetchall()]
                for col in ("story", "part", "of"):
                    if col not in cols:
                        c.execute(f"ALTER TABLE clips ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")
                c.execute("UPDATE clips SET \"of\"=1 WHERE \"of\"=0")
            if ver < 5:
                c.execute("""CREATE TABLE IF NOT EXISTS refs(
                  id TEXT PRIMARY KEY, url TEXT, title TEXT, dur REAL,
                  hook TEXT, wps REAL, top TEXT, created REAL)""")
                c.execute("CREATE INDEX IF NOT EXISTS ix_refs_created ON refs(created)")
            if ver < 6:
                cols = [r[1] for r in c.execute("PRAGMA table_info(refs)").fetchall()]
                for col, typ in (("views", "INTEGER NOT NULL DEFAULT 0"),
                                 ("viral", "REAL NOT NULL DEFAULT 0.5"),
                                 ("hookp", "REAL NOT NULL DEFAULT 0.5"),
                                 ("whole", "REAL NOT NULL DEFAULT 0.5"),
                                 ("peak", "REAL NOT NULL DEFAULT 0.5"),
                                 ("score", "INTEGER NOT NULL DEFAULT 0")):
                    if col not in cols:
                        c.execute(f"ALTER TABLE refs ADD COLUMN {col} {typ}")
            c.execute("INSERT OR REPLACE INTO meta(k,v) VALUES ('version',?)",
                      (str(DB_VERSION),))
    finally:
        c.close()


def db():
    return _connect()


def log_job(job_id, title, url, status, clips=None, scorer="heuristic"):
    clips = clips or []
    try:
        c = _connect()
        try:
            with lock:
                created = jobs.get(job_id, {}).get("started", time.time())
            with c:
                c.execute(
                    "INSERT OR REPLACE INTO jobs"
                    "(id,title,url,status,created,finished,clips,total_size,scorer)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (job_id, title or "", url or "", status, created, time.time(),
                     json.dumps(clips, ensure_ascii=False),
                     sum(int(x.get("size", 0) or 0) for x in clips), scorer))
                c.execute("DELETE FROM clips WHERE job_id=?", (job_id,))
                c.executemany(
                    "INSERT INTO clips"
                    "(job_id,idx,title,score,verdict,hook,dur,size,views,created,story,part,\"of\")"
                    " VALUES (?,?,?,?,?,?,?,?,0,?,?,?,?)",
                    [(job_id, int(x.get("i", k)), str(x.get("title", ""))[:200],
                      int(x.get("score", 0) or 0), str(x.get("verdict", ""))[:300],
                      str(x.get("hook", ""))[:40], float(x.get("dur", 0) or 0),
                      int(x.get("size", 0) or 0), time.time(),
                      int(x.get("story", 0) or 0), int(x.get("part", 0) or 0),
                      int(x.get("of", 1) or 1))
                     for k, x in enumerate(clips)])
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()


def get_settings():
    try:
        with open(CFG, encoding="utf-8") as f:
            return {**DEFAULT_CFG, **json.load(f)}
    except Exception:
        return dict(DEFAULT_CFG)


def save_settings(d):
    cur = get_settings()
    for k in DEFAULT_CFG:
        if k in d:
            cur[k] = d[k]
    try:
        cur["max_workers"] = min(4, max(1, int(cur.get("max_workers", 2))))
    except Exception:
        cur["max_workers"] = 2
    with open(CFG, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False)
    return cur


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total

jobs = {}
lock = threading.RLock()  # рекурсивный: wait_slot -> queue_pos берут его вложенно
active = 0
MAX_WORKERS = 2


def queue_pos(job_id: str) -> int:
    with lock:
        mine = jobs.get(job_id, {}).get("started", 0)
        return sum(1 for jid, j in jobs.items()
                   if j.get("status") in ("queued",) and j.get("started", 0) < mine)


def wait_slot(job_id):
    global active
    while True:
        with lock:
            mx = get_settings().get("max_workers", 2)
            try:
                mx = min(4, max(1, int(mx)))
            except Exception:
                mx = 2
            if active < mx:
                active += 1
                return
            pos = queue_pos(job_id)
            if job_id in jobs:
                jobs[job_id].update(stage=f"в очереди… (перед тобой: {pos})" if pos else "в очереди…")
        time.sleep(2)


def free_slot():
    global active
    with lock:
        active = max(0, active - 1)


def fetch_source(url: str, progress=None, limit_min: int = 0) -> str:
    from yt_dlp import YoutubeDL

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes", 0)
            k = (done / total) if total else 0
            sp = (d.get("_speed_str") or "").strip()
            progress(k, f"качаю видео… {int(k * 100)}% {sp}".strip())

    tmp = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
            "socket_timeout": 30, "ffmpeg_location": FFMPEG,
            "progress_hooks": [hook],
            "concurrent_fragment_downloads": 4,
            "retries": 10, "fragment_retries": 10,
            # вис ниже 100 КБ/с — убивать и ретраить, а не висеть вечно
            "throttled_rate": 100000,
            # БЕЗ extractor_args: дефолтные клиенты отдают 1440p, а android/ios
            # под SABR-гейтом дают только 360p — проверяли, это и душило качество
            # aria2c: 16 коннектов вместо одного — ровная скорость без просадок
            "external_downloader": {"default": "aria2c"},
            "external_downloader_args": {"aria2c": ["-x", "16", "-s", "16", "-k", "1M",
                                                    "--file-allocation=none", "-q"]},
            # качаем до 1440p: даунскейл в 1080 чище, чем апскейл из 720p.
            # vbr первым — из одинаковых берём самый жирный по битрейту
            # 30fps-варианты первыми: выход всё равно max 30fps, а 60fps = x2 байт впустую
            "format": ("bv*[height<=1440][fps<=30]+ba/b[height<=1440][fps<=30]"
                       "/bv*[height<=1440]+ba/b[height<=1440]"),
            "format_sort": ["vbr"],
            "merge_output_format": "mp4", "outtmpl": tmp}
    try:
        limit_min = max(0, int(limit_min or 0))
    except (TypeError, ValueError):
        limit_min = 0
    if limit_min > 0:
        # нужен только кусок: тянем секцию, а не всё видео
        # (HLS/DASH — только нужные фрагменты, остальное как раньше)
        opts["download_sections"] = [f"*00:00-{limit_min:02d}:00"]
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = (info or {}).get("title", "видео")
    path = tmp if os.path.exists(tmp) else None
    if not path:
        cands = [os.path.join(TMP_DIR, f) for f in os.listdir(TMP_DIR)]
        path = max(cands, key=os.path.getmtime) if cands else None
    if not path:
        raise RuntimeError("не скачалось")
    if not ac._readable_video(path):
        try:
            os.remove(path)
        except OSError:
            pass
        raise RuntimeError("скачался битый файл — попробуй ещё раз")
    return path, title


def run_job(job_id: str, url: str, opts: dict):
    def prog(pct, stage):
        with lock:
            if job_id in jobs:
                jobs[job_id].update(pct=pct, stage=stage)

    wait_slot(job_id)
    src = None
    try:
        import shutil
        free_gb = shutil.disk_usage(BASE).free / 1e9
        if free_gb < 1.5:
            raise RuntimeError(f"Нет места на диске ({free_gb:.1f} ГБ) — почисти Библиотеку")
        with lock:
            jobs[job_id].update(status="work", stage="качаю видео…", pct=1)
        src, title = fetch_source(
            url, progress=lambda k, s: prog(int(1 + k * 7), s),
            limit_min=opts.get("limit_min", 0))
        with lock:
            jobs[job_id].update(title=title)
        mode = opts.get("vmode", "blur")
        if mode == "parts":
            # режим частей: сцены и DREX не нужны вообще.
            # Сначала чистим паузы, потом ОДНА транскрибация, потом резка.
            prog(6, "чищу паузы…")
            clean = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
            try:
                stage_in = ac.remove_silence(
                    src, clean,
                    progress=lambda p, s: prog(int(6 + p * 0.04), "чищу паузы…"))
            except Exception:
                stage_in = src
            if stage_in != src:
                try:
                    if os.path.exists(src):
                        os.remove(src)
                except OSError:
                    pass
                src = stage_in
            scenes = []
            prog(10, "транскрибация…")
            segs, lang = ac.transcribe(src, opts.get("limit_min", 0),
                                       progress=lambda p, s: prog(int(10 + p * 0.7), s),
                                       model_name=opts.get("asr_model", "base"))
            prog(82, "режу на части…")
            bounds = ac.compute_parts(segs, 40.0, 59.0)
            nb = len(bounds)
            moments = [{"start": b["start"], "end": b["end"], "score": 0,
                        "title": f"Часть {k + 1}/{nb}",
                        "part_title": f"ЧАСТЬ {k + 1}/{nb}",
                        "verdict": f"Часть {k + 1} из {nb}",
                        "hook": f"Часть {k + 1}/{nb}",
                        "story": 0, "part": k, "of": nb}
                       for k, b in enumerate(bounds)]
            scorer = "parts"
            if not moments:
                raise RuntimeError("не вышло нарезать на части")
        else:
            prog(6, "ищу сцены…")
            try:
                scenes = ac.detect_scenes(src)
            except Exception:
                scenes = []
            prog(10, "транскрибация…")
            segs, lang = ac.transcribe(src, opts.get("limit_min", 0),
                                        progress=lambda p, s: prog(int(10 + p * 0.7), s),
                                        model_name=opts.get("asr_model", "base"))
        stamps = ac.parse_stamps(opts.get("stamps", ""))
        if mode == "parts":
            pass  # моменты-части уже готовы выше
        elif stamps:
            scorer = "manual"
            moments = [{**m, "title": m.get("title") or f"Момент {k + 1}"}
                       for k, m in enumerate(stamps[:20])]
        else:
            prog(82, "оценка моментов…")
            moments, scorer = ac.pick_moments(
                segs, lang, opts.get("api_key", ""),
                opts.get("want", 8),
                opts.get("min_len", 20), opts.get("max_len", 60),
                scenes=scenes, min_gap=opts.get("min_gap", 45))
        with lock:
            if job_id in jobs:
                jobs[job_id]["scorer"] = scorer
        if not moments:
            raise RuntimeError("не нашёл моментов — попробуй ручные таймкоды")
        clips = []
        fast = opts.get("fast", False)
        done = {"n": 0}

        def one(i_m):
            i, m = i_m
            out = os.path.join(OUT_DIR, f"{job_id}_{i + 1}.mp4")
            total_n = len(moments)

            def _pp(p, _i=i):
                with lock:
                    if job_id in jobs:
                        # прогресс только вперёд: параллельные воркеры
                        # перезаписывали друг друга и бар дёргался 1/8↔2/8
                        npct = int(84 + (_i + max(0.0, min(1.0, p))) / total_n * 15)
                        if npct > jobs[job_id].get("pct", 0):
                            jobs[job_id].update(
                                stage=f"рендер… {_i + 1}/{total_n}", pct=npct)

            if mode == "parts":
                ac.render_split_clip(src, m, segs, out,
                                     facetrack=False,
                                     captions=opts.get("captions", True),
                                     antidetect=opts.get("antidetect", False),
                                     mirror=opts.get("mirror", False),
                                     banner=opts.get("banner", True),
                                     banner_path=opts.get("banner_path"),
                                     gameplay=opts.get("gameplay", "subway"),
                                     max_total=float(opts.get("max_len", 59) or 59),
                                     progress=_pp)
            elif fast:
                ac.fast_cut(src, m, out)
            else:
                ac.render_clip(src, m, segs, out,
                               facetrack=opts.get("facetrack", True),
                               captions=opts.get("captions", True),
                               vmode=mode, antidetect=opts.get("antidetect", False),
                               mirror=opts.get("mirror", False),
                               banner=opts.get("banner", True),
                               banner_path=opts.get("banner_path"),
                               gameplay=opts.get("gameplay", "subway"),
                               max_total=float(opts.get("max_len", 60) or 60),
                               progress=_pp)
            with lock:
                done["n"] += 1
                n = done["n"]
                if job_id in jobs:
                    jobs[job_id].update(
                        stage=f"рендер… {n}/{len(moments)}",
                        pct=int(84 + n / len(moments) * 15))
            return {"i": i, "title": m.get("title", f"Клип {i + 1}"),
                    "score": m.get("score", 0),
                    "verdict": str(m.get("verdict", ""))[:300],
                    "hook": str(m.get("hook", ""))[:40],
                    "story": int(m.get("story", 0)), "part": int(m.get("part", 0)),
                    "of": int(m.get("of", 1)),
                    "dur": ac.media_duration(out) or round(float(m["end"]) - float(m["start"]), 1),
                    "size": os.path.getsize(out)}

        # 2 клипа параллельно — на M2 даёт ~x1.7 к скорости (кодирует железо,
        # Python-цикл один поток не насыщает все ядра)
        with ThreadPoolExecutor(max_workers=1 if fast else 2) as ex:
            for c in ex.map(one, list(enumerate(moments))):
                clips.append(c)
        clips.sort(key=lambda c: c["i"])
        with lock:
            jobs[job_id].update(status="done", pct=100, stage="готово", clips=clips)
            title = jobs[job_id].get("title", url)
        log_job(job_id, title, url, "done", clips,
                jobs.get(job_id, {}).get("scorer", "heuristic"))
        # исходник отъел своё — удаляем, иначе ночная пачка из 10 видео сожрёт диск
        try:
            if src and os.path.exists(src):
                os.remove(src)
        except OSError:
            pass
    except Exception as e:
        with lock:
            jobs[job_id].update(status="error", error=str(e)[:300])
        log_job(job_id, url, url, "error",
                scorer=jobs.get(job_id, {}).get("scorer", "heuristic"))
    finally:
        try:
            if src and os.path.exists(src):
                os.remove(src)
        except OSError:
            pass
        free_slot()


@app.get("/")
def index():
    return send_file(os.path.join(BASE, "index.html"))


@app.post("/api/start")
def api_start():
    data = request.get_json(force=True, silent=True) or {}
    urls = [u.strip() for u in (data.get("urls") or "").splitlines() if u.strip()]
    stamps = data.get("stamps", "")
    if not urls:
        return jsonify({"ok": False, "error": "Вставь хоть одну ссылку"}), 400
    if stamps.strip() and len(urls) > 1:
        return jsonify({"ok": False, "error": "Таймкоды работают только с одним видео"}), 400
    cfg = get_settings()
    if not data.get("api_key"):
        data["api_key"] = cfg.get("drex_key", "")
    for k in ("want", "gameplay", "limit_min", "vmode", "captions", "facetrack", "antidetect", "mirror", "banner", "asr_model", "drex_key"):
        if k not in data or data[k] in ("", None):
            data[k] = cfg.get(k)
    if not data.get("min_len") or not data.get("max_len"):
        try:
            mn, mx = str(data.get("lens", cfg.get("lens", "40-59"))).split("-")
            data["min_len"], data["max_len"] = int(mn), int(mx)
        except Exception:
            data["min_len"], data["max_len"] = 40, 59
    try:
        want = max(1, min(50, int(data.get("want", cfg.get("want", 8)) or 8)))
    except (TypeError, ValueError):
        want = 8
    ids = []
    for url in urls[:10]:
        job_id = uuid.uuid4().hex[:8]
        with lock:
            jobs[job_id] = {"id": job_id, "url": url, "title": url,
                            "status": "queued", "pct": 0, "stage": "в очереди…",
                            "started": time.time()}
        threading.Thread(target=run_job, args=(job_id, url, {
            "api_key": data.get("api_key", ""), "want": want,
            "min_len": int(data.get("min_len", 20)), "max_len": int(data.get("max_len", 60)),
            "limit_min": int(data.get("limit_min", 0)), "min_gap": 20,
            "captions": bool(data.get("captions", True)),
            "facetrack": bool(data.get("facetrack", True)),
            "antidetect": bool(data.get("antidetect", False)),
            "mirror": bool(data.get("mirror", False)),
            "banner": bool(data.get("banner", True)),
            "banner_path": resolve_banner(data.get("banner_file", "")),
            "gameplay": data.get("gameplay", "subway") if data.get("gameplay") in ("subway", "minecraft") else "subway",
            "asr_model": data.get("asr_model", "base") if data.get("asr_model") in ("base", "small", "giga") else "base",
            "fast": bool(data.get("fast", False)),
            "vmode": data.get("vmode", "blur"), "stamps": stamps}), daemon=True).start()
        ids.append(job_id)
    return jsonify({"ok": True, "jobs": ids})


def fetch_twitch(url: str, progress=None, limit_min: int = 0):
    """Качает VOD/клип Twitch в лучшем качестве (без внешнего даунлоадера —
    у Twitch HLS, его качает сам yt-dlp)."""
    from yt_dlp import YoutubeDL

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes", 0)
            k = (done / total) if total else 0
            sp = (d.get("_speed_str") or "").strip()
            progress(k, f"качаю стрим… {int(k * 100)}% {sp}".strip())

    tmp = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
            "socket_timeout": 30, "ffmpeg_location": FFMPEG,
            "progress_hooks": [hook],
            "concurrent_fragment_downloads": 12,
            "retries": 10, "fragment_retries": 10,
            # 30fps-варианты первыми: выход всё равно max 30fps, а 60fps = x2 байт впустую
            "format": ("bv*[height<=1080][fps<=30]+ba/b[height<=1080][fps<=30]"
                       "/bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b"),
            "format_sort": ["res", "vbr", "size"],
            "merge_output_format": "mp4", "outtmpl": tmp}
    try:
        limit_min = max(0, int(limit_min or 0))
    except (TypeError, ValueError):
        limit_min = 0
    if limit_min > 0:
        # у Twitch HLS — качаются только нужные сегменты
        opts["download_sections"] = [f"*00:00-{limit_min:02d}:00"]
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = (info or {}).get("title", "стрим")
    path = tmp if os.path.exists(tmp) else None
    if not path:
        cands = [os.path.join(TMP_DIR, f) for f in os.listdir(TMP_DIR)]
        path = max(cands, key=os.path.getmtime) if cands else None
    if not path:
        raise RuntimeError("не скачалось")
    if not ac._readable_video(path):
        try:
            os.remove(path)
        except OSError:
            pass
        raise RuntimeError("скачался битый файл — попробуй ещё раз")
    return path, title


def run_twitch_job(job_id: str, url: str, opts: dict):
    """Twitch-версия run_job: VOD -> сцены -> транскрибация -> DREX
    (домен streams) -> вертикаль 30/70 (вебка сверху). Очередь, история
    и раздача файлов — общие (jobs, log_job, /api/clip)."""

    def prog(pct, stage):
        with lock:
            if job_id in jobs:
                jobs[job_id].update(pct=pct, stage=stage)

    wait_slot(job_id)
    src = None
    try:
        import shutil
        free_gb = shutil.disk_usage(BASE).free / 1e9
        if free_gb < 1.5:
            raise RuntimeError(f"Нет места на диске ({free_gb:.1f} ГБ) — почисти Библиотеку")
        with lock:
            jobs[job_id].update(status="work", stage="качаю стрим…", pct=1)
        src, title = fetch_twitch(
            url, progress=lambda k, s: prog(int(1 + k * 7), s),
            limit_min=opts.get("limit_min", 0))
        with lock:
            jobs[job_id].update(title=title)
        prog(6, "ищу сцены…")
        try:
            scenes = ac.detect_scenes(src)
        except Exception:
            scenes = []
        prog(10, "транскрибация…")
        segs, lang = ac.transcribe(src, opts.get("limit_min", 0),
                                   progress=lambda p, s: prog(int(10 + p * 0.7), s),
                                   model_name=opts.get("asr_model", "base"))
        prog(82, "оценка моментов…")
        moments, scorer = ac.pick_moments(
            segs, lang, opts.get("api_key", ""),
            opts.get("want", 8),
            opts.get("min_len", 40), opts.get("max_len", 59),
            scenes=scenes, min_gap=opts.get("min_gap", 20),
            domain="streams")
        with lock:
            if job_id in jobs:
                jobs[job_id]["scorer"] = scorer
        if not moments:
            raise RuntimeError("не нашёл моментов")
        clips = []
        done = {"n": 0}

        def one(i_m):
            i, m = i_m
            out = os.path.join(OUT_DIR, f"{job_id}_{i + 1}.mp4")
            total_n = len(moments)

            def _pp(p, _i=i):
                with lock:
                    if job_id in jobs:
                        # прогресс только вперёд: параллельные воркеры
                        # перезаписывали друг друга и бар дёргался 1/8↔2/8
                        npct = int(84 + (_i + max(0.0, min(1.0, p))) / total_n * 15)
                        if npct > jobs[job_id].get("pct", 0):
                            jobs[job_id].update(
                                stage=f"рендер… {_i + 1}/{total_n}", pct=npct)

            ac.render_twitch_clip(src, m, segs, out,
                                  captions=opts.get("captions", True),
                                  antidetect=opts.get("antidetect", False),
                                  mirror=opts.get("mirror", False),
                                  banner=opts.get("banner", True),
                                  banner_path=opts.get("banner_path"),
                                  max_total=float(opts.get("max_len", 59) or 59),
                                  progress=_pp)
            with lock:
                done["n"] += 1
                n = done["n"]
                if job_id in jobs:
                    jobs[job_id].update(
                        stage=f"рендер… {n}/{len(moments)}",
                        pct=int(84 + n / len(moments) * 15))
            return {"i": i, "title": m.get("title", f"Клип {i + 1}"),
                    "score": m.get("score", 0),
                    "verdict": str(m.get("verdict", ""))[:300],
                    "hook": str(m.get("hook", "") or m.get("title", ""))[:40],
                    "story": 0, "part": 0, "of": 1,
                    "dur": ac.media_duration(out) or round(float(m["end"]) - float(m["start"]), 1),
                    "size": os.path.getsize(out)}

        with ThreadPoolExecutor(max_workers=2) as ex:
            for c in ex.map(one, list(enumerate(moments))):
                clips.append(c)
        clips.sort(key=lambda c: c["i"])
        with lock:
            jobs[job_id].update(status="done", pct=100, stage="готово", clips=clips)
            title = jobs[job_id].get("title", url)
        log_job(job_id, title, url, "done", clips,
                jobs.get(job_id, {}).get("scorer", "drex"))
        try:
            if src and os.path.exists(src):
                os.remove(src)
        except OSError:
            pass
    except Exception as e:
        with lock:
            jobs[job_id].update(status="error", error=str(e)[:300])
        log_job(job_id, url, url, "error",
                scorer=jobs.get(job_id, {}).get("scorer", "drex"))
    finally:
        try:
            if src and os.path.exists(src):
                os.remove(src)
        except OSError:
            pass
        free_slot()


@app.post("/api/twitch/start")
def api_twitch_start():
    data = request.get_json(force=True, silent=True) or {}
    urls = [u.strip() for u in (data.get("urls") or "").splitlines() if u.strip()]
    if not urls:
        return jsonify({"ok": False, "error": "Вставь хоть одну ссылку на Twitch"}), 400
    bad = [u for u in urls if "twitch.tv" not in u]
    if bad:
        return jsonify({"ok": False, "error": "Только ссылки twitch.tv"}), 400
    cfg = get_settings()
    if not data.get("api_key"):
        data["api_key"] = cfg.get("drex_key", "")
    if not data.get("min_len") or not data.get("max_len"):
        try:
            mn, mx = str(data.get("lens", "40-59")).split("-")
            data["min_len"], data["max_len"] = int(mn), int(mx)
        except Exception:
            data["min_len"], data["max_len"] = 40, 59
    try:
        want = max(1, min(50, int(data.get("want", 8) or 8)))
    except (TypeError, ValueError):
        want = 8
    try:
        limit_min = max(0, int(data.get("limit_min", 0) or 0))
    except (TypeError, ValueError):
        limit_min = 0
    ids = []
    for url in urls[:5]:
        job_id = uuid.uuid4().hex[:8]
        with lock:
            jobs[job_id] = {"id": job_id, "url": url, "title": url,
                            "status": "queued", "pct": 0, "stage": "в очереди…",
                            "started": time.time()}
        threading.Thread(target=run_twitch_job, args=(job_id, url, {
            "api_key": data.get("api_key", ""), "want": want,
            "min_len": int(data.get("min_len", 40)), "max_len": int(data.get("max_len", 59)),
            "limit_min": limit_min, "min_gap": 20,
            "captions": True,
            "antidetect": bool(data.get("antidetect", False)),
            "mirror": bool(data.get("mirror", False)),
            "banner": bool(data.get("banner", True)),
            "banner_path": resolve_banner(data.get("banner_file", "")),
            "asr_model": "base"}), daemon=True).start()
        ids.append(job_id)
    return jsonify({"ok": True, "jobs": ids})


@app.get("/api/job/<job_id>")
def api_job(job_id):
    with lock:
        j = jobs.get(job_id)
    if not j:
        return jsonify({"ok": False, "error": "нет задачи"}), 404
    out = dict(j)
    out["ok"] = True
    return jsonify(out)


@app.get("/api/clip/<job_id>/<int:idx>")
def api_clip(job_id, idx):
    p = os.path.join(OUT_DIR, f"{job_id}_{idx + 1}.mp4")
    if not os.path.exists(p):
        return jsonify({"ok": False}), 404
    with lock:
        j = jobs.get(job_id, {})
    clips = j.get("clips", [])
    name = (clips[idx]["title"][:50] + ".mp4") if idx < len(clips) else f"short{idx + 1}.mp4"
    return send_file(p, as_attachment=True, download_name=name)


@app.get("/api/tiktok-check/<job_id>/<int:idx>")
def api_tiktok_check(job_id, idx):
    """Предзаливная проверка клипа: ок ли файл для TikTok.
    Файл с ok=False почти наверняка получит 0 просмотров."""
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", job_id or ""):
        return jsonify({"ok": False}), 404
    p = os.path.join(OUT_DIR, f"{job_id}_{idx + 1}.mp4")
    if not os.path.exists(p):
        return jsonify({"ok": False, "error": "клип не найден"}), 404
    try:
        res = ac.tiktok_safety_check(p)
        res["ok_api"] = True
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 500


@app.get("/api/stats")
def api_stats():
    try:
        c = db()
        try:
            videos = c.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='done'").fetchone()[0]
            errs = c.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='error'").fetchone()[0]
            n_clips = c.execute(
                "SELECT COUNT(*) FROM clips WHERE job_id IN"
                " (SELECT id FROM jobs WHERE status='done')").fetchone()[0]
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        videos = errs = n_clips = 0
    n_files = 0
    for d in (OUT_DIR, LEGACY_DIR):
        try:
            n_files += sum(1 for f in os.listdir(d)
                           if os.path.isfile(os.path.join(d, f))
                           and not f.endswith((".part", ".ytdl")))
        except OSError:
            pass
    return jsonify({"ok": True, "videos": videos, "clips": n_clips, "errors": errs,
                    "files": n_files, "bytes_shorts": dir_size(OUT_DIR),
                    "bytes_tmp": dir_size(TMP_DIR)})


@app.get("/api/history")
def api_history():
    try:
        lim = max(1, min(200, int(request.args.get("limit", 50))))
    except (TypeError, ValueError):
        lim = 50
    try:
        c = db()
        try:
            rows = c.execute(
                "SELECT id,title,url,status,created,finished,total_size,scorer"
                " FROM jobs ORDER BY finished DESC LIMIT ?", (lim,)).fetchall()
            items = []
            for r in rows:
                cr = c.execute(
                    "SELECT idx,title,score,verdict,hook,dur,size,views,story,part,\"of\""
                    " FROM clips WHERE job_id=? ORDER BY idx", (r["id"],)).fetchall()
                items.append({
                    "id": r["id"], "title": r["title"], "url": r["url"],
                    "status": r["status"], "created": r["created"],
                    "finished": r["finished"], "total_size": r["total_size"],
                    "scorer": r["scorer"] or "heuristic",
                    "clips": [{"i": x["idx"], "title": x["title"], "score": x["score"],
                               "verdict": x["verdict"] or "", "hook": x["hook"] or "",
                               "dur": x["dur"], "size": x["size"],
                               "views": x["views"], "story": x["story"] or 0,
                               "part": x["part"] or 0, "of": x["of"] or 1}
                              for x in cr]})
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        items = []
    return jsonify({"ok": True, "items": items})


def _thumb_path(kind, name):
    import hashlib
    safe = f"{kind}_{hashlib.md5(name.encode('utf-8')).hexdigest()[:12]}.jpg"
    return os.path.join(THUMB_DIR, safe)


def _drop_thumb(kind, name):
    try:
        tp = _thumb_path(kind, name)
        if os.path.exists(tp):
            os.remove(tp)
    except OSError:
        pass


@app.get("/api/zip/<job_id>")
def api_zip(job_id):
    """Все нарезки серии одним ZIP (распакуется папкой)."""
    import re
    import zipfile
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", job_id or ""):
        return jsonify({"ok": False}), 404
    try:
        files = sorted(
            f for f in os.listdir(OUT_DIR)
            if f.startswith(job_id + "_") and f.endswith(".mp4")
            and os.path.isfile(os.path.join(OUT_DIR, f)))
    except OSError:
        files = []
    if not files:
        return jsonify({"ok": False, "error": "клипы не найдены"}), 404
    zp = os.path.join(TMP_DIR, f"all_{job_id}.zip")
    try:
        newest = max(os.path.getmtime(os.path.join(OUT_DIR, f)) for f in files)
    except OSError:
        newest = 0
    if (not os.path.exists(zp) or os.path.getmtime(zp) < newest
            or os.path.getsize(zp) == 0):
        with lock:
            title = (jobs.get(job_id, {}).get("title") or job_id) if job_id in jobs else job_id
        if title == job_id:
            try:
                c = db()
                try:
                    r = c.execute("SELECT title FROM jobs WHERE id=?",
                                  (job_id,)).fetchone()
                    if r and r["title"]:
                        title = r["title"]
                finally:
                    c.close()
            except Exception:
                pass
        base = re.sub(r"[^\w\- ]+", "", title, flags=re.U).strip()[:50] or job_id
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
            for n, f in enumerate(files, 1):
                arc = f"{base}/{n:02d}_{f}"
                z.write(os.path.join(OUT_DIR, f), arc)
    return send_file(zp, mimetype="application/zip", as_attachment=True,
                     download_name=f"{job_id[:6]}_clips.zip")


@app.get("/api/zip-multi")
def api_zip_multi():
    """Один ZIP на несколько задач: ?jobs=id1,id2. Внутри папка на серию."""
    import re
    import zipfile
    raw = (request.args.get("jobs") or "")
    ids = [j.strip() for j in raw.split(",") if re.fullmatch(r"[A-Za-z0-9_-]{1,32}", j.strip())][:10]
    if not ids:
        return jsonify({"ok": False, "error": "нет задач"}), 404
    zp = os.path.join(TMP_DIR, "all_" + "_".join(i[:6] for i in ids) + ".zip")
    made, newest = [], 0
    try:
        for job_id in ids:
            files = sorted(
                f for f in os.listdir(OUT_DIR)
                if f.startswith(job_id + "_") and f.endswith(".mp4")
                and os.path.isfile(os.path.join(OUT_DIR, f)))
            if not files:
                continue
            title = job_id
            try:
                c = db()
                try:
                    r = c.execute("SELECT title FROM jobs WHERE id=?", (job_id,)).fetchone()
                    if r and r["title"]:
                        title = r["title"]
                finally:
                    c.close()
            except Exception:
                pass
            base = re.sub(r"[^\w\- ]+", "", title, flags=re.U).strip()[:50] or job_id
            for n, f in enumerate(files, 1):
                p = os.path.join(OUT_DIR, f)
                made.append((p, f"{base}/{n:02d}_{f}"))
                try:
                    newest = max(newest, os.path.getmtime(p))
                except OSError:
                    pass
    except OSError:
        pass
    if not made:
        return jsonify({"ok": False, "error": "клипы не найдены"}), 404
    if (not os.path.exists(zp) or os.path.getmtime(zp) < newest
            or os.path.getsize(zp) == 0):
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
            for p, arc in made:
                z.write(p, arc)
    return send_file(zp, mimetype="application/zip", as_attachment=True,
                     download_name="clips_all.zip")


@app.get("/api/debug/stacks")
def api_debug_stacks():
    import sys
    import traceback
    out = []
    for tid, fr in sys._current_frames().items():
        out.append(f"--- thread {tid} ---\n"
                   + "".join(traceback.format_stack(fr))[-1200:])
    return app.response_class("\n".join(out), mimetype="text/plain")


def _clip_id(name: str):
    """job_id, idx из имени файла <job>_<i>.mp4."""
    import re
    m = re.fullmatch(r"([A-Za-z0-9_-]{1,32})_(\d+)\.mp4", name or "")
    return (m.group(1), int(m.group(2)) - 1) if m else (None, None)


@app.post("/api/views")
def api_views():
    """Ручная пометка просмотров клипа: {"kind","name","views"}. Учит ранкер."""
    data = request.get_json(force=True, silent=True) or {}
    kind, name = (data.get("kind") or ""), (data.get("name") or "")
    try:
        views = max(0, int(data.get("views", 0)))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "число нужно"}), 400
    job_id, idx = _clip_id(name)
    if not job_id or kind not in ("short", "legacy"):
        return jsonify({"ok": False, "error": "не клип"}), 400
    try:
        c = db()
        try:
            with c:
                c.execute("UPDATE clips SET views=? WHERE job_id=? AND idx=?",
                          (views, job_id, idx))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False}), 500
    return jsonify({"ok": True, "views": views})


@app.get("/api/refs")
def api_refs_list():
    try:
        c = db()
        try:
            rows = c.execute(
                "SELECT id,url,title,dur,hook,wps,top,created,views,score FROM refs"
                " ORDER BY created DESC LIMIT 100").fetchall()
            items = [dict(r) for r in rows]
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        items = []
    return jsonify({"ok": True, "items": items})


@app.delete("/api/refs/<rid>")
def api_refs_del(rid):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", rid or ""):
        return jsonify({"ok": False}), 404
    try:
        c = db()
        try:
            with c:
                c.execute("DELETE FROM refs WHERE id=?", (rid,))
        finally:
            c.close()
    except Exception:
        pass
    return jsonify({"ok": True})


def _parse_views(s: str) -> int:
    """150000, 150k, 1.5M, 1,2 млн, 500 тыс -> число."""
    import re
    t = (s or "").strip().lower().replace(",", ".").replace(" ", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(к|k|тыс|m|м|млн|mln)?\.?", t)
    if not m:
        return 0
    v, suf = float(m.group(1)), (m.group(2) or "")
    if suf in ("m", "м", "млн", "mln"):
        v *= 1000000
    elif suf in ("к", "k", "тыс"):
        v *= 1000
    return int(v)


def _analyze_ref(url: str, views: int = 0):
    """Скачать -> транскрибировать -> DREX-разбор. Файл удаляем, профиль возвращаем."""
    import ai_clips as ac
    import uuid
    src, title = fetch_source(url)
    try:
        segs, lang = ac.transcribe(src, 0)
        words = sum(len(s["text"].split()) for s in segs)
        dur = segs[-1]["end"] - segs[0]["start"] if segs else 0
        wps = round(words / max(dur, 1), 2)
        hook = " ".join(segs[0]["text"].split()[:12]) if segs else ""
        top, probs, score = "", {}, 0
        key = get_settings().get("drex_key", "")
        if key and segs:
            try:
                cands = ac._drex_score(segs, key, 3, 10, 60)
                if cands:
                    m = cands[0]
                    top = (f"{m['score']}: {m['title'][:100]} "
                           f"[{m['start']:.0f}-{m['end']:.0f}с]")
                    score = int(m["score"])
                    _pv = {"viral": r"виральность (\d\.\d+)", "hook": r"хук (\d\.\d+)",
                           "whole": r"цельность (\d\.\d+)", "peak": r"пик (\d\.\d+)"}
                    for f, pat in _pv.items():
                        mm = __import__("re").search(pat, m.get("verdict", ""))
                        probs[f] = float(mm.group(1)) if mm else 0.5
            except Exception:
                pass
    finally:
        try:
            if src and os.path.exists(src):
                os.remove(src)
        except OSError:
            pass
    return {"id": uuid.uuid4().hex[:8], "url": url, "title": title,
            "dur": round(dur, 1), "hook": hook, "wps": wps, "top": top,
            "views": views, "score": score,
            "viral": probs.get("viral", 0.5), "hookp": probs.get("hook", 0.5),
            "whole": probs.get("whole", 0.5), "peak": probs.get("peak", 0.5),
            "created": time.time()}


REF_BATCHES = {}


def _refs_bulk_worker(bid: str, items):
    st = REF_BATCHES.get(bid)
    if not st:
        return
    for it in items:
        if st.get("stop"):
            break
        try:
            r = _analyze_ref(it["url"], it["views"])
            try:
                c = db()
                try:
                    with c:
                        c.execute("INSERT INTO refs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                  (r["id"], r["url"], r["title"], r["dur"], r["hook"],
                                   r["wps"], r["top"], r["created"], r["views"],
                                   r["viral"], r["hookp"], r["whole"], r["peak"], r["score"]))
                finally:
                    c.close()
            except Exception:
                pass
            st["done"] += 1
            st["last"] = (r["title"] or r["url"])[:60]
        except Exception as e:
            st["done"] += 1
            st["errors"] = st.get("errors", 0) + 1
            st["last"] = f"Ошибка: {str(e)[:80]}"
    st["finished"] = True


@app.post("/api/refs/bulk")
def api_refs_bulk():
    """Пачка: {"items": [{"url", "views"}]} -> фоновый разбор, статус по id."""
    import uuid
    data = request.get_json(force=True, silent=True) or {}
    items = []
    for it in (data.get("items") or [])[:50]:
        if isinstance(it, dict) and (it.get("url") or "").strip():
            items.append({"url": it["url"].strip(),
                          "views": _parse_views(str(it.get("views", "")))})
        elif isinstance(it, str) and it.strip():
            parts = it.strip().rsplit(None, 1)
            if len(parts) == 2 and _parse_views(parts[1]) > 0:
                items.append({"url": parts[0], "views": _parse_views(parts[1])})
            else:
                items.append({"url": it.strip(), "views": 0})
    if not items:
        return jsonify({"ok": False, "error": "Пусто"}), 400
    bid = uuid.uuid4().hex[:8]
    REF_BATCHES[bid] = {"total": len(items), "done": 0, "finished": False, "last": ""}
    threading.Thread(target=_refs_bulk_worker, args=(bid, items), daemon=True).start()
    return jsonify({"ok": True, "batch": bid, "total": len(items)})


@app.post("/api/refs/hunt")
def api_refs_hunt():
    """Автоохота: поиск шортсов по запросу -> топ по просмотрам -> в обучение.
    TikTok с этого IP забанен, охотимся на YouTube Shorts (тот же контент)."""
    from yt_dlp import YoutubeDL
    import uuid
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("q") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Дай тег или запрос"}), 400
    try:
        limit = max(3, min(20, int(data.get("limit", 10))))
    except (TypeError, ValueError):
        limit = 10
    try:
        max_dur = float(data.get("max_dur", 0) or 0)
    except (TypeError, ValueError):
        max_dur = 0
    try:
        with YoutubeDL({"quiet": True, "no_warnings": True,
                        "socket_timeout": 30, "extract_flat": True,
                        "noplaylist": False}) as ydl:
            info = ydl.extract_info(f"ytsearch{limit * 3}:{q}", download=False)
        cands = []
        for e in (info.get("entries") or []):
            vid = (e or {}).get("id")
            vc = (e or {}).get("view_count") or 0
            if vid and vc:
                cands.append({"url": f"https://www.youtube.com/watch?v={vid}",
                              "views": vc,
                              "title": (e.get("title") or "")[:80],
                              "dur": (e.get("duration") or 0)})
        cands.sort(key=lambda x: -x["views"])
        if max_dur > 0:
            # добираем длительности точечно, берём топ по просмотрам
            with YoutubeDL({"quiet": True, "no_warnings": True,
                            "socket_timeout": 30}) as ydl:
                sized = []
                for c in cands[:limit * 2]:
                    d = c.get("dur") or 0
                    if not d:
                        try:
                            inf = ydl.extract_info(c["url"], download=False)
                            d = inf.get("duration") or 0
                        except Exception:
                            d = 0
                    if d <= max_dur:
                        sized.append(c)
                    if len(sized) >= limit:
                        break
            cands = sized
        else:
            cands = cands[:limit]
        if not cands:
            return jsonify({"ok": False, "error": "Ничего не нашлось"}), 404
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 400
    bid = uuid.uuid4().hex[:8]
    REF_BATCHES[bid] = {"total": len(cands), "done": 0, "finished": False,
                        "last": "", "hunt": q}
    threading.Thread(target=_refs_bulk_worker, args=(bid, cands),
                     daemon=True).start()
    top = [{"title": c["title"], "views": c["views"]} for c in cands[:5]]
    return jsonify({"ok": True, "batch": bid, "total": len(cands), "top": top})


@app.get("/api/refs/bulk/<bid>")
def api_refs_bulk_status(bid):
    st = REF_BATCHES.get(bid)
    if not st:
        return jsonify({"ok": False, "error": "нет пачки"}), 404
    return jsonify({"ok": True, **{k: v for k, v in st.items() if k != "stop"}})
@app.post("/api/refs")
def api_refs_add():
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "Дай ссылку"}), 400
    try:
        r = _analyze_ref(url, _parse_views(str(data.get("views", ""))))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]}), 400
    try:
        c = db()
        try:
            with c:
                c.execute("INSERT INTO refs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (r["id"], r["url"], r["title"], r["dur"], r["hook"],
                           r["wps"], r["top"], r["created"], r["views"],
                           r["viral"], r["hookp"], r["whole"], r["peak"], r["score"]))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False}), 500
    return jsonify({"ok": True, "id": r["id"], "title": r["title"], "dur": r["dur"],
                    "wps": r["wps"], "hook": r["hook"], "top": r["top"]})


@app.get("/api/learn")
def api_learn():
    import learn as learn_mod
    return jsonify({"ok": True, **learn_mod.status()})


@app.post("/api/learn/train")
def api_learn_train():
    import learn as learn_mod
    res = learn_mod.train()
    if not res:
        return jsonify({"ok": False,
                        "error": "Мало меток: нужно 6+ клипов с просмотрами и 2+ удалённых"}), 400
    return jsonify({"ok": True, **res})


@app.get("/api/library")
def api_library():
    from urllib.parse import quote
    items = []
    for d, kind in ((OUT_DIR, "short"), (LEGACY_DIR, "legacy")):
        try:
            files = os.listdir(d)
        except OSError:
            continue
        for f in files:
            p = os.path.join(d, f)
            if os.path.isfile(p) and not f.endswith((".part", ".ytdl", ".temp")):
                items.append({"name": f, "kind": kind, "size": os.path.getsize(p),
                              "mtime": os.path.getmtime(p),
                              "url": f"/dl/{kind}/{quote(f)}",
                              "thumb": f"/api/thumb/{kind}/{quote(f)}"})
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify({"ok": True, "files": items[:300]})


@app.get("/api/thumb/<kind>/<name>")
def api_thumb(kind, name):
    """Постер для сетки (кэш в thumbs/, реген если видео новее)."""
    import ai_clips as ac
    d = OUT_DIR if kind == "short" else LEGACY_DIR if kind == "legacy" else None
    if not d or ".." in name or "/" in name:
        return jsonify({"ok": False}), 404
    p = os.path.join(d, name)
    if not os.path.isfile(p):
        return jsonify({"ok": False}), 404
    os.makedirs(THUMB_DIR, exist_ok=True)
    tp = _thumb_path(kind, name)
    try:
        if (not os.path.exists(tp)
                or os.path.getmtime(tp) < os.path.getmtime(p)
                or os.path.getsize(tp) == 0):
            ac.make_thumb(p, tp)
        if os.path.exists(tp) and os.path.getsize(tp) > 0:
            return send_file(tp, mimetype="image/jpeg")
    except Exception:
        pass
    return jsonify({"ok": False}), 404


def _download_social(url: str, dest_base: str, progress=None):
    """Качает видео по ссылке (TikTok и др.) в максимальном качестве."""
    from yt_dlp import YoutubeDL

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes", 0)
            k = (done / total) if total else 0
            progress(int(k * 25), f"качаю… {int(k * 100)}%")

    opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
            "socket_timeout": 30, "ffmpeg_location": FFMPEG,
            "progress_hooks": [hook],
            "concurrent_fragment_downloads": 8,
            "retries": 10, "fragment_retries": 10,
            "format": "bv*+ba/b",
            "format_sort": ["res", "vbr", "size"],
            "merge_output_format": "mp4", "outtmpl": dest_base + ".%(ext)s"}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = (((info or {}).get("title") or "tiktok")[:60])
    import glob
    cands = [p for p in glob.glob(dest_base + ".*")
             if os.path.isfile(p) and p.rsplit(".", 1)[-1].lower()
             in ("mp4", "mov", "mkv", "webm")]
    path = max(cands, key=os.path.getmtime) if cands else None
    if not path or not ac._readable_video(path):
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        raise RuntimeError("не скачалось — проверь ссылку")
    return path, title


# Оптимум уникализации: глаз не видит, хеши ломаются.
UNIQ_OPTIMAL = {"speed": 1.01, "mirror": False, "color": "medium",
                "zoom": 2.0, "grain": True, "pitch": True}


def _link_worker(dlid: str, url: str):
    """Шаг 1: скачал в максималке -> почистил тишину -> СТОП, ждём человека:
    статус ready + исходник для выбора отрезка ползунком."""

    def prog(pct, stage="качаю…"):
        s = UNIQ_JOBS.get(dlid)
        if s is not None:
            s.update(pct=pct, stage=stage)

    banner_path = (UNIQ_JOBS.get(dlid) or {}).get("banner_path")

    try:
        import re
        raw_base = os.path.join(UNIQ_IN, f"{dlid}_src")
        path, title = _download_social(
            url, raw_base,
            progress=lambda p, s="качаю…": prog(min(int(p), 60), s))
        safe = (re.sub(r"[^\w\-. ]+", "", title.strip())[:40] or "tiktok")
        # тишина — аккуратно, до выбора отрезка (тайминги дальше едут по чистому)
        silenced = os.path.join(UNIQ_IN, f"{dlid}_nosil.mp4")
        try:
            stage0 = ac.remove_silence(
                path, silenced,
                progress=lambda p, s="чищу тишину…": prog(min(60 + int(p * 0.3), 90), s))
        except Exception:
            stage0 = path
        # исходник для выбора отрезка ползунком (живёт 7 дней)
        import shutil
        src_keep = os.path.join(UNIQ_IN, f"{dlid}_{safe}_src.mp4")
        try:
            if stage0 != src_keep:
                shutil.copy(stage0, src_keep)
        except OSError:
            src_keep = stage0 if os.path.exists(stage0) else path
        try:
            _w, _h, _n, _f = ac._probe_wh_fps(src_keep)
            src_dur = round(_n / max(_f, 1), 1) if _n and _f else 0
        except Exception:
            src_dur = 0
        s = UNIQ_JOBS.get(dlid)
        if s is not None:
            s.update(status="ready", pct=100, stage="выбери отрезок чужого баннера",
                     title=safe,
                     src_url="/dluniqsrc/" + os.path.basename(src_keep),
                     src_dur=src_dur)
    except Exception as e:
        s = UNIQ_JOBS.get(dlid)
        if s is not None:
            s.update(status="error", error=str(e)[:300])
    finally:
        try:
            import glob as _g
            for p in (_g.glob(os.path.join(UNIQ_IN, f"{dlid}_src.*"))
                      + [os.path.join(UNIQ_IN, f"{dlid}_nosil.mp4")]):
                try:
                    if os.path.exists(p) and os.path.basename(p) != os.path.basename(src_keep):
                        os.remove(p)
                except (OSError, NameError):
                    pass
        except Exception:
            pass


def _finish_worker(jid: str, path: str, base: str, cut, banner_path=None,
                   mirror: bool = False):
    """Шаг 2: вырезать выбранный отрезок (или авто), вставить свой баннер,
    уникализировать -> готовый файл."""

    def prog(pct, stage="режу…"):
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(pct=pct, stage=stage)

    try:
        swapped = os.path.join(UNIQ_IN, f"{jid}_swap.mp4")
        if cut is not None:
            info = ac.swap_banner(path, swapped, banner_path=banner_path,
                                  remove=False, cutranges=cut,
                                  progress=lambda p, s="режу…": prog(int(p * 0.5), s))
        else:
            info = ac.swap_banner(path, swapped, banner_path=banner_path, remove=True,
                                  progress=lambda p, s="режу…": prog(int(p * 0.5), s))
        out = os.path.join(UNIQ_OUT, f"{jid}_{base}_final.mp4")
        opts = dict(UNIQ_OPTIMAL)
        if mirror:
            opts["mirror"] = True
        # хук-титр для первых секунд: берём из имени (там заголовок видоса)
        opts["hook"] = base
        ac.uniqualize_video(swapped, out, opts,
                            progress=lambda p, s="уникализация…": prog(min(50 + int(p * 0.5), 100), s))
        try:
            with open(os.path.join(UNIQ_OUT, f"{jid}_{base}_analysis.json"),
                      "w", encoding="utf-8") as _af:
                _af.write(json.dumps(info.get("analysis", {}), ensure_ascii=False))
        except Exception:
            pass
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(status="done", pct=100, stage="готово", title=base,
                      url="/dluniq/" + os.path.basename(out),
                      size=os.path.getsize(out),
                      removed=info.get("removed", []))
    except Exception as e:
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(status="error", error=str(e)[:300])
    finally:
        try:
            if os.path.exists(swapped):
                os.remove(swapped)
        except OSError:
            pass


@app.post("/api/uniq-finish")
def api_uniq_finish():
    import re
    data = request.get_json(force=True, silent=True) or {}
    src = (data.get("src") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_][\w\-. ]*\.mp4", src or "") or ".." in src or "/" in src:
        return jsonify({"ok": False, "error": "нет исходника"}), 400
    path = os.path.join(UNIQ_IN, src)
    if not os.path.isfile(path):
        return jsonify({"ok": False, "error": "исходник удалён (живёт 7 дней)"}), 404
    cut = None
    if data.get("s") is not None and data.get("e") is not None:
        try:
            s, e = float(data.get("s")), float(data.get("e"))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "отрезок такой?"}), 400
        if e - s < 0.5:
            return jsonify({"ok": False, "error": "отрезок короче 0.5с"}), 400
        cut = [(s, e)]
    jid = uuid.uuid4().hex[:8]
    base = (src.rsplit("_src.mp4", 1)[0][-40:] or "video")
    base = re.sub(r"^[a-f0-9]{8}_", "", base)[:40] or "video"
    parent = UNIQ_JOBS.get(src.split("_")[0]) or {}
    banner_path = parent.get("banner_path")
    mirror = bool(parent.get("mirror", False))
    UNIQ_JOBS[jid] = {"id": jid, "base": base, "kind": "finish",
                      "status": "work", "pct": 0, "stage": "старт…",
                      "started": time.time()}
    threading.Thread(target=_finish_worker,
                     args=(jid, path, base, cut, banner_path, mirror),
                     daemon=True).start()
    return jsonify({"ok": True, "id": jid})


@app.post("/api/uniq-link")
def api_uniq_link():
    data = request.get_json(force=True, silent=True) or {}
    raw_urls = data.get("urls") or ""
    if isinstance(raw_urls, (list, tuple)):
        urls = [str(u).strip() for u in raw_urls if str(u).strip()][:5]
    else:
        urls = [u.strip() for u in str(raw_urls).splitlines()
                if u.strip()][:5]
    if not urls:
        return jsonify({"ok": False, "error": "Вставь ссылку на TikTok"}), 400
    banner_path = resolve_banner(data.get("banner_file", ""))
    mirror = bool(data.get("mirror", False))
    made = []
    for url in urls:
        dlid = uuid.uuid4().hex[:8]
        UNIQ_JOBS[dlid] = {"id": dlid, "kind": "link", "url": url,
                           "status": "work", "pct": 0, "stage": "в очереди…",
                           "started": time.time(), "children": [],
                           "banner_path": banner_path, "mirror": mirror}
        threading.Thread(target=_link_worker, args=(dlid, url),
                         daemon=True).start()
        made.append(dlid)
    return jsonify({"ok": True, "jobs": made})


def _uniq_worker(jid: str, path: str, opts: dict):
    def prog(pct, stage="уникализация…"):
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(pct=pct, stage=stage)
    try:
        base = UNIQ_JOBS.get(jid, {}).get("base", "video")
        out = os.path.join(UNIQ_OUT, f"{jid}_{base}_uniq.mp4")
        ac.uniqualize_video(path, out, opts,
                            progress=lambda p, s="уникализация…": prog(int(p), s))
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(status="done", pct=100, stage="готово",
                      url="/dluniq/" + os.path.basename(out),
                      size=os.path.getsize(out))
    except Exception as e:
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(status="error", error=str(e)[:300])
    finally:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


@app.post("/api/uniq")
def api_uniq():
    import re
    f = request.files.get("file")
    if f is None or not (f.filename or ""):
        return jsonify({"ok": False, "error": "Приложи видеофайл"}), 400
    name = re.sub(r"[^\w\-. ]+", "", (f.filename or "").strip())[:60] or "video.mp4"
    if "." not in name or name.rsplit(".", 1)[1].lower() not in (
            "mp4", "mov", "mkv", "webm", "m4v"):
        return jsonify({"ok": False, "error": "Нужно видео: mp4/mov/mkv/webm"}), 400
    jid = uuid.uuid4().hex[:8]
    path = os.path.join(UNIQ_IN, f"{jid}_{name}")
    try:
        f.save(path)
    except Exception:
        return jsonify({"ok": False, "error": "Не сохранилось"}), 400
    try:
        if os.path.getsize(path) > 800 * 1024 * 1024:
            os.remove(path)
            return jsonify({"ok": False, "error": "Файл больше 800 МБ"}), 400
    except OSError:
        pass
    form = request.form or {}

    def _f(key, default):
        try:
            return float(form.get(key, default))
        except (TypeError, ValueError):
            return default

    opts = {"speed": min(1.05, max(0.95, _f("speed", 1.0))),
            "mirror": form.get("mirror") == "1",
            "color": form.get("color", "medium") if form.get("color") in (
                "light", "medium", "strong") else "medium",
            "zoom": min(5.0, max(0.0, _f("zoom", 2.0))),
            "grain": form.get("grain", "1") == "1",
            "pitch": form.get("pitch") == "1"}
    base = (name.rsplit(".", 1)[0] or "video")[:40]
    UNIQ_JOBS[jid] = {"id": jid, "base": base, "status": "work",
                      "pct": 0, "stage": "старт…", "started": time.time()}
    threading.Thread(target=_uniq_worker, args=(jid, path, opts), daemon=True).start()
    return jsonify({"ok": True, "id": jid})


def _cs2_worker(jid: str, path: str, title: str, opts: dict):
    """Рестайл «КС 2, но я...»: чёрные полосы -> блюр, пилюля, уникализация."""

    def prog(pct, stage="рестайл…"):
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(pct=int(pct), stage=stage)

    try:
        base = UNIQ_JOBS.get(jid, {}).get("base", "video")
        out = os.path.join(UNIQ_OUT, f"{jid}_{base}_cs2.mp4")
        info = ac.restyle_cs2(
            path, out, title,
            mirror=bool(opts.get("mirror", False)),
            progress=lambda p, s="рестайл…": prog(int(p * 100), s))
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            box = info.get("box") or [0, 0]
            bar = "полосы вырезаны" if box[0] > 2 else "полос не было — весь кадр контент"
            st.update(status="done", pct=100, stage="готово",
                      url="/dluniq/" + os.path.basename(out),
                      size=os.path.getsize(out),
                      removed=[f"верх {box[0]}пх", f"низ {box[1]}пх", bar])
    except Exception as e:
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(status="error", error=str(e)[:300])
    finally:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


@app.post("/api/cs2")
def api_cs2():
    """Загрузка видео + заголовок пилюли -> рестайл «КС 2, но я...»."""
    import re
    f = request.files.get("file")
    if f is None or not (f.filename or ""):
        return jsonify({"ok": False, "error": "Приложи видеофайл"}), 400
    name = re.sub(r"[^\w\-. ]+", "", (f.filename or "").strip())[:60] or "video.mp4"
    if "." not in name or name.rsplit(".", 1)[1].lower() not in (
            "mp4", "mov", "mkv", "webm", "m4v"):
        return jsonify({"ok": False, "error": "Нужно видео: mp4/mov/mkv/webm"}), 400
    form = request.form or {}
    title = re.sub(r"\s+", " ", (form.get("title") or "")).strip()[:120]
    if not title:
        return jsonify({"ok": False, "error": "Впиши заголовок для пилюли"}), 400
    jid = uuid.uuid4().hex[:8]
    path = os.path.join(UNIQ_IN, f"{jid}_{name}")
    try:
        f.save(path)
    except Exception:
        return jsonify({"ok": False, "error": "Не сохранилось"}), 400
    try:
        if os.path.getsize(path) > 800 * 1024 * 1024:
            os.remove(path)
            return jsonify({"ok": False, "error": "Файл больше 800 МБ"}), 400
    except OSError:
        pass
    opts = {"mirror": form.get("mirror") == "1"}
    base = (name.rsplit(".", 1)[0] or "video")[:40]
    UNIQ_JOBS[jid] = {"id": jid, "base": base, "kind": "cs2",
                      "status": "work", "pct": 0, "stage": "старт…",
                      "started": time.time(), "title": title}
    threading.Thread(target=_cs2_worker,
                     args=(jid, path, title, opts), daemon=True).start()
    return jsonify({"ok": True, "id": jid})


@app.get("/api/uniq/<jid>")
def api_uniq_status(jid):
    st = UNIQ_JOBS.get(jid)
    if not st:
        return jsonify({"ok": False, "error": "нет задачи"}), 404
    out = dict(st)
    out["ok"] = True
    return jsonify(out)


@app.get("/api/uniq-lib")
def api_uniq_lib():
    from urllib.parse import quote
    items = []
    try:
        files = os.listdir(UNIQ_OUT)
    except OSError:
        files = []
    for fn in files:
        p = os.path.join(UNIQ_OUT, fn)
        if os.path.isfile(p) and fn.endswith(".mp4"):
            items.append({"name": fn, "size": os.path.getsize(p),
                          "mtime": os.path.getmtime(p),
                          "url": "/dluniq/" + quote(fn)})
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify({"ok": True, "files": items[:200]})


def _swap_worker(jid: str, path: str, banner: str | None, remove: bool):
    def prog(pct, stage="замена баннера…"):
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(pct=pct, stage=stage)
    try:
        base = UNIQ_JOBS.get(jid, {}).get("base", "video")
        out = os.path.join(UNIQ_OUT, f"{jid}_{base}_swap.mp4")
        info = ac.swap_banner(
            path, out, banner_path=banner, remove=remove,
            progress=lambda p, s="замена баннера…": prog(int(p), s))
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(status="done", pct=100, stage="готово",
                      url="/dluniq/" + os.path.basename(out),
                      size=os.path.getsize(out),
                      removed=info.get("removed", []),
                      covered=info.get("covered", []),
                      mid=info.get("mid", 0), total=info.get("total", 0))
    except Exception as e:
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(status="error", error=str(e)[:300])
    finally:
        for p in (path, banner):
            try:
                if p and os.path.exists(p) and p.startswith(UNIQ_IN):
                    os.remove(p)
            except OSError:
                pass


@app.post("/api/banner-swap")
def api_banner_swap():
    import re
    f = request.files.get("file")
    if f is None or not (f.filename or ""):
        return jsonify({"ok": False, "error": "Приложи видеофайл"}), 400
    name = re.sub(r"[^\w\-. ]+", "", (f.filename or "").strip())[:60] or "video.mp4"
    if "." not in name or name.rsplit(".", 1)[1].lower() not in (
            "mp4", "mov", "mkv", "webm", "m4v"):
        return jsonify({"ok": False, "error": "Нужно видео: mp4/mov/mkv/webm"}), 400
    jid = uuid.uuid4().hex[:8]
    path = os.path.join(UNIQ_IN, f"{jid}_{name}")
    try:
        f.save(path)
    except Exception:
        return jsonify({"ok": False, "error": "Не сохранилось"}), 400
    try:
        if os.path.getsize(path) > 800 * 1024 * 1024:
            os.remove(path)
            return jsonify({"ok": False, "error": "Файл больше 800 МБ"}), 400
    except OSError:
        pass
    banner = None
    bf = request.files.get("banner")
    if bf is not None and (bf.filename or ""):
        bname = re.sub(r"[^\w\-. ]+", "", bf.filename.strip())[:60] or "banner.mp4"
        banner = os.path.join(UNIQ_IN, f"{jid}_ban_{bname}")
        try:
            bf.save(banner)
        except Exception:
            banner = None
    form = request.form or {}
    remove = form.get("cut", "1") == "1"
    base = (name.rsplit(".", 1)[0] or "video")[:40]
    UNIQ_JOBS[jid] = {"id": jid, "base": base, "kind": "swap", "status": "work",
                      "pct": 0, "stage": "старт…", "started": time.time()}
    threading.Thread(target=_swap_worker, args=(jid, path, banner, remove),
                     daemon=True).start()
    return jsonify({"ok": True, "id": jid})


def _manual_worker(jid: str, path: str, at: float, base: str):
    """Ручная точка баннера: замена на исходнике + уникализация (как в цепочке)."""

    def prog(pct, stage="ставлю баннер…"):
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(pct=pct, stage=stage)

    try:
        swapped = os.path.join(UNIQ_IN, f"{jid}_swap.mp4")
        info = ac.swap_banner(path, swapped, banner_path=None, remove=True, at=at,
                              progress=lambda p, s="режу…": prog(int(p * 0.5), s))
        out = os.path.join(UNIQ_OUT, f"{jid}_{base}_manual.mp4")
        ac.uniqualize_video(swapped, out, dict(UNIQ_OPTIMAL),
                            progress=lambda p, s="уникализация…": prog(min(50 + int(p * 0.5), 100), s))
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(status="done", pct=100, stage="готово", title=base,
                      url="/dluniq/" + os.path.basename(out),
                      size=os.path.getsize(out),
                      removed=info.get("removed", []))
    except Exception as e:
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(status="error", error=str(e)[:300])
    finally:
        try:
            if os.path.exists(swapped):
                os.remove(swapped)
        except OSError:
            pass


@app.post("/api/uniq-manual")
def api_uniq_manual():
    import re
    data = request.get_json(force=True, silent=True) or {}
    src = (data.get("src") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_][\w\-. ]*\.mp4", src or "") or ".." in src or "/" in src:
        return jsonify({"ok": False, "error": "нет исходника"}), 400
    path = os.path.join(UNIQ_IN, src)
    if not os.path.isfile(path):
        return jsonify({"ok": False, "error": "исходник удалён (живёт 7 дней)"}), 404
    try:
        at = max(0.0, float(data.get("at", 0) or 0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "время такое?"}), 400
    jid = uuid.uuid4().hex[:8]
    base = (src.rsplit("_src.mp4", 1)[0][-40:] or "video")
    base = re.sub(r"^[a-f0-9]{8}_", "", base)[:40] or "video"
    UNIQ_JOBS[jid] = {"id": jid, "base": base, "kind": "manual",
                      "status": "work", "pct": 0, "stage": "старт…",
                      "started": time.time()}
    threading.Thread(target=_manual_worker, args=(jid, path, at, base), daemon=True).start()
    return jsonify({"ok": True, "id": jid})


@app.get("/dluniqsrc/<name>")
def serve_uniq_src(name):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_][\w\-. ]*\.mp4", name or "") or ".." in name or "/" in name:
        return jsonify({"ok": False}), 404
    p = os.path.join(UNIQ_IN, name)
    if not os.path.isfile(p):
        return jsonify({"ok": False, "error": "файл удалён"}), 404
    return send_file(p)


def _banner_dir():
    return os.path.join(BASE, "assets")


def list_banners():
    """Доступные баннеры из assets/*.mp4: [{id, name, dur, size}]."""
    import glob
    items = []
    try:
        files = sorted(glob.glob(os.path.join(BASE, "assets", "*.mp4")))
    except Exception:
        files = []
    for p in files:
        try:
            _w, _h, _n, _f = ac._probe_wh_fps(p)
            dur = round(_n / max(_f, 1), 1) if _n and _f else 0
        except Exception:
            dur = 0
        base = os.path.basename(p)
        name = base.rsplit(".", 1)[0]
        if base == os.path.basename(ac.BANNER_FILE):
            name += " (по умолчанию)"
        try:
            green = bool(ac._banner_green(p))
        except Exception:
            green = False
        if green:
            name += " · хромакей"
        try:
            size = os.path.getsize(p)
        except OSError:
            size = 0
        items.append({"id": base, "name": name, "dur": dur, "size": size,
                      "green": green})
    return items


def resolve_banner(name):
    """Имя файла -> путь в assets или None (= дефолтный Мусордроп)."""
    base = os.path.basename(str(name or "").strip())
    if not base or ".." in base or "/" in base or not base.endswith(".mp4"):
        return None
    p = os.path.join(BASE, "assets", base)
    return p if os.path.isfile(p) else None


@app.get("/api/banners")
def api_banners():
    return jsonify({"ok": True, "banners": list_banners()})


@app.get("/dluniq/<name>")
def serve_uniq(name):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_][\w\-. ]*\.mp4", name or "") or ".." in name or "/" in name:
        return jsonify({"ok": False}), 404
    p = os.path.join(UNIQ_OUT, name)
    if not os.path.isfile(p):
        return jsonify({"ok": False, "error": "файл удалён"}), 404
    return send_file(p, as_attachment=True)


@app.delete("/api/uniq/<name>")
def api_uniq_del(name):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_][\w\-. ]*\.mp4", name or "") or ".." in name or "/" in name:
        return jsonify({"ok": False}), 404
    p = os.path.join(UNIQ_OUT, name)
    try:
        if os.path.isfile(p):
            os.remove(p)
            return jsonify({"ok": True})
    except OSError:
        pass
    return jsonify({"ok": False}), 404


@app.get("/dl/<kind>/<name>")
def serve_file(kind, name):
    d = OUT_DIR if kind == "short" else LEGACY_DIR if kind == "legacy" else None
    if not d or ".." in name or "/" in name:
        return jsonify({"ok": False}), 404
    p = os.path.join(d, name)
    if not os.path.isfile(p):
        return jsonify({"ok": False, "error": "файл удалён или не докачался"}), 404
    return send_file(p, as_attachment=True)


def _mark_rejected(kind, name):
    """Удалённый клип = негативная метка для обучения ранкера."""
    job_id, idx = _clip_id(name)
    if not job_id or kind not in ("short", "legacy"):
        return
    try:
        c = db()
        try:
            with c:
                c.execute("UPDATE clips SET views=-1 WHERE job_id=? AND idx=?",
                          (job_id, idx))
        finally:
            c.close()
    except Exception:
        pass


@app.delete("/api/del/<kind>/<name>")
def api_del(kind, name):
    d = OUT_DIR if kind == "short" else LEGACY_DIR if kind == "legacy" else None
    if not d or ".." in name or "/" in name:
        return jsonify({"ok": False}), 404
    p = os.path.join(d, name)
    if os.path.isfile(p):
        os.remove(p)
        _drop_thumb(kind, name)
        _mark_rejected(kind, name)
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 404


@app.delete("/api/del-bulk")
def api_del_bulk():
    """Удаление пачкой: {"items": [{"kind": "short", "name": "..."}]}."""
    data = request.get_json(force=True, silent=True) or {}
    items = data.get("items") or []
    removed, failed = 0, []
    for it in items[:500]:
        kind = (it.get("kind") or "") if isinstance(it, dict) else ""
        name = (it.get("name") or "") if isinstance(it, dict) else ""
        d = OUT_DIR if kind == "short" else LEGACY_DIR if kind == "legacy" else None
        if not d or not name or ".." in name or "/" in name:
            failed.append(name)
            continue
        p = os.path.join(d, name)
        try:
            if os.path.isfile(p):
                os.remove(p)
                _drop_thumb(kind, name)
                _mark_rejected(kind, name)
                removed += 1
            else:
                failed.append(name)
        except OSError:
            failed.append(name)
    return jsonify({"ok": True, "removed": removed, "failed": failed})


@app.get("/api/settings")
def api_settings_get():
    return jsonify({"ok": True, "settings": get_settings()})


@app.post("/api/settings")
def api_settings_post():
    return jsonify({"ok": True,
                    "settings": save_settings(request.get_json(force=True, silent=True) or {})})


@app.post("/api/cleanup")
def api_cleanup():
    removed, freed = 0, 0
    now = time.time()
    try:
        for f in os.listdir(TMP_DIR):
            p = os.path.join(TMP_DIR, f)
            if os.path.isfile(p) and now - os.path.getmtime(p) > 24 * 3600:
                freed += os.path.getsize(p)
                os.remove(p)
                removed += 1
    except OSError:
        pass
    try:
        # обломки докачек (фрагменты HLS/DASH) — им и 2 часов много
        for f in os.listdir(TMP_DIR):
            p = os.path.join(TMP_DIR, f)
            if not os.path.isfile(p):
                continue
            if now - os.path.getmtime(p) <= 2 * 3600:
                continue
            if f.endswith((".part", ".ytdl")) or ".part-Frag" in f or f.endswith(".temp"):
                freed += os.path.getsize(p)
                os.remove(p)
                removed += 1
    except OSError:
        pass
    try:
        # забракованный геймплей старше 2 часов — в утиль
        for f in os.listdir(GAMEPLAY_DIR):
            p = os.path.join(GAMEPLAY_DIR, f)
            if (os.path.isfile(p) and f.endswith(".bad")
                    and now - os.path.getmtime(p) > 2 * 3600):
                freed += os.path.getsize(p)
                os.remove(p)
                removed += 1
    except OSError:
        pass
    try:
        for f in os.listdir(UNIQ_IN):
            p = os.path.join(UNIQ_IN, f)
            if os.path.isfile(p) and now - os.path.getmtime(p) > 7 * 24 * 3600:
                freed += os.path.getsize(p)
                os.remove(p)
                removed += 1
    except OSError:
        pass
    return jsonify({"ok": True, "removed": removed, "freed": freed})


@app.get("/health")
def health():
    return jsonify({"ok": True})


init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=False)
