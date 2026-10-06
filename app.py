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
FFMPEG = os.path.expanduser("~/.local/bin/ffmpeg")

DB = os.path.join(BASE, "data.db")
CFG = os.path.join(BASE, "config.json")
DEFAULT_CFG = {"gemini_key": "", "drex_key": "", "want": 8, "gameplay": "subway", "lens": "20-60", "limit_min": 0,
               "vmode": "split", "captions": True, "facetrack": True,
               "antidetect": False, "mirror": False, "banner": True,
               "asr_model": "base", "max_workers": 2}


DB_VERSION = 4


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


def fetch_source(url: str, progress=None) -> str:
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
            "format": ("bv*[height<=1440]+ba/b[height<=1440]"),
            "format_sort": ["vbr"],
            "merge_output_format": "mp4", "outtmpl": tmp}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = (info or {}).get("title", "видео")
    path = tmp if os.path.exists(tmp) else None
    if not path:
        cands = [os.path.join(TMP_DIR, f) for f in os.listdir(TMP_DIR)]
        path = max(cands, key=os.path.getmtime) if cands else None
    if not path:
        raise RuntimeError("не скачалось")
    return path, title


def run_job(job_id: str, url: str, opts: dict):
    def prog(pct, stage):
        with lock:
            if job_id in jobs:
                jobs[job_id].update(pct=pct, stage=stage)

    wait_slot(job_id)
    try:
        import shutil
        free_gb = shutil.disk_usage(BASE).free / 1e9
        if free_gb < 1.5:
            raise RuntimeError(f"Нет места на диске ({free_gb:.1f} ГБ) — почисти Библиотеку")
        with lock:
            jobs[job_id].update(status="work", stage="качаю видео…", pct=1)
        src, title = fetch_source(
            url, progress=lambda k, s: prog(int(1 + k * 7), s))
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
        stamps = ac.parse_stamps(opts.get("stamps", ""))
        scorer = "manual"
        if stamps:
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
        mode, fast = opts.get("vmode", "blur"), opts.get("fast", False)
        done = {"n": 0}

        def one(i_m):
            i, m = i_m
            out = os.path.join(OUT_DIR, f"{job_id}_{i + 1}.mp4")
            if fast:
                ac.fast_cut(src, m, out)
            else:
                ac.render_clip(src, m, segs, out,
                               facetrack=opts.get("facetrack", True),
                               captions=opts.get("captions", True),
                               vmode=mode, antidetect=opts.get("antidetect", False),
                               mirror=opts.get("mirror", False),
                               banner=opts.get("banner", True),
                               gameplay=opts.get("gameplay", "subway"),
                               max_total=float(opts.get("max_len", 60) or 60))
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
            mn, mx = str(data.get("lens", cfg.get("lens", "20-60"))).split("-")
            data["min_len"], data["max_len"] = int(mn), int(mx)
        except Exception:
            data["min_len"], data["max_len"] = 20, 60
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
            "gameplay": data.get("gameplay", "subway") if data.get("gameplay") in ("subway", "minecraft") else "subway",
            "asr_model": data.get("asr_model", "base") if data.get("asr_model") in ("base", "small", "giga") else "base",
            "fast": bool(data.get("fast", False)),
            "vmode": data.get("vmode", "blur"), "stamps": stamps}), daemon=True).start()
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


@app.get("/dl/<kind>/<name>")
def serve_file(kind, name):
    d = OUT_DIR if kind == "short" else LEGACY_DIR if kind == "legacy" else None
    if not d or ".." in name or "/" in name:
        return jsonify({"ok": False}), 404
    p = os.path.join(d, name)
    if not os.path.isfile(p):
        return jsonify({"ok": False, "error": "файл удалён или не докачался"}), 404
    return send_file(p, as_attachment=True)


@app.delete("/api/del/<kind>/<name>")
def api_del(kind, name):
    d = OUT_DIR if kind == "short" else LEGACY_DIR if kind == "legacy" else None
    if not d or ".." in name or "/" in name:
        return jsonify({"ok": False}), 404
    p = os.path.join(d, name)
    if os.path.isfile(p):
        os.remove(p)
        _drop_thumb(kind, name)
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
    return jsonify({"ok": True, "removed": removed, "freed": freed})


@app.get("/health")
def health():
    return jsonify({"ok": True})


init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=False)
