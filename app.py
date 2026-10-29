"""Lipps by huakly web — ссылка на YouTube -> пачка вертикальных шортсов.
Только нарезка сериалов/фильмов: сцены -> транскрибация -> LLM-скоринг -> 9:16 -> субтитры."""
import csv
import io
import json
import os
import re
import sqlite3
import subprocess
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


def _safe_stem(name: str) -> str:
    s = re.sub(r"[^\w\-. ]+", "", str(name or ""), flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip(" ._-")
    if s.lower().endswith(".mp4"):
        s = s[:-4].rstrip(" ._-")
    return s[:60]


def _clip_filename(job_id: str, idx: int, out_name: str = "") -> str:
    stem = _safe_stem(out_name)
    if stem:
        return f"{stem}_{job_id}_{int(idx) + 1}.mp4"
    return f"{job_id}_{int(idx) + 1}.mp4"


def _clip_path(job_id: str, idx: int):
    canonical = os.path.join(OUT_DIR, f"{job_id}_{int(idx) + 1}.mp4")
    if os.path.isfile(canonical):
        return canonical
    suffix = f"_{job_id}_{int(idx) + 1}.mp4"
    try:
        names = os.listdir(OUT_DIR)
    except OSError:
        return None
    found = [os.path.join(OUT_DIR, f) for f in names
             if f.endswith(suffix) and os.path.isfile(os.path.join(OUT_DIR, f))]
    return max(found, key=os.path.getmtime) if found else None


def _job_clip_files(job_id: str) -> list[str]:
    try:
        names = os.listdir(OUT_DIR)
    except OSError:
        return []
    pat = re.compile(rf"(?:^|_){re.escape(job_id)}_\d+(?:v\d+)?\.mp4$")
    return sorted(f for f in names
                  if pat.search(f) and os.path.isfile(os.path.join(OUT_DIR, f)))


LEGACY_DIR = os.path.join(BASE, "legacy")
THUMB_DIR = os.path.join(BASE, "thumbs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(LEGACY_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)
UNIQ_IN = os.path.join(BASE, "unique_in")
UNIQ_OUT = os.path.join(BASE, "unique")
GAMEPLAY_DIR = os.path.join(BASE, "gameplay")
os.makedirs(UNIQ_IN, exist_ok=True)
os.makedirs(UNIQ_OUT, exist_ok=True)
os.makedirs(GAMEPLAY_DIR, exist_ok=True)
UNIQ_JOBS = {}
FFMPEG = os.path.expanduser("~/.local/bin/ffmpeg")

DB = os.path.join(BASE, "data.db")
CFG = os.path.join(BASE, "config.json")
DEFAULT_CFG = {"gemini_key": "", "drex_key": "", "want": 8, "gameplay": "subway.mp4", "lens": "40-59", "limit_min": 0,
               "vmode": "split", "captions": True, "facetrack": True,
               "antidetect": True, "mirror": False, "banner": True,
               "clip_music": True, "remove_pauses": True, "tt_cookies": "",
               "asr_model": "giga", "asr_api_key": "",
               "asr_api_base": "https://gw.lunaroute.com/v1", "max_workers": 2}


DB_VERSION = 11


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
            if ver < 7:
                cols = [r[1] for r in c.execute("PRAGMA table_info(clips)").fetchall()]
                if "hidden" not in cols:
                    c.execute("ALTER TABLE clips ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0")
            if ver < 8:
                c.execute("""CREATE TABLE IF NOT EXISTS tags(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT UNIQUE NOT NULL,
                  theme TEXT NOT NULL DEFAULT '',
                  max_videos INTEGER NOT NULL DEFAULT 0,
                  account_id INTEGER,
                  created REAL)""")
                c.execute("""CREATE TABLE IF NOT EXISTS clip_tags(
                  job_id TEXT NOT NULL, idx INTEGER NOT NULL,
                  tag_id INTEGER NOT NULL,
                  PRIMARY KEY(job_id, idx, tag_id))""")
                c.execute("CREATE INDEX IF NOT EXISTS ix_clip_tags_tag ON clip_tags(tag_id)")
                c.execute("""CREATE TABLE IF NOT EXISTS accounts(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  nick TEXT UNIQUE NOT NULL,
                  created REAL, synced REAL,
                  followers INTEGER NOT NULL DEFAULT 0,
                  likes INTEGER NOT NULL DEFAULT 0,
                  video_count INTEGER NOT NULL DEFAULT 0,
                  total_views INTEGER NOT NULL DEFAULT 0,
                  source TEXT NOT NULL DEFAULT '',
                  error TEXT NOT NULL DEFAULT '')""")
                c.execute("""CREATE TABLE IF NOT EXISTS account_videos(
                  account_id INTEGER NOT NULL, video_id TEXT NOT NULL,
                  title TEXT, views INTEGER NOT NULL DEFAULT 0,
                  likes INTEGER NOT NULL DEFAULT 0,
                  comments INTEGER NOT NULL DEFAULT 0,
                  url TEXT, published REAL,
                  PRIMARY KEY(account_id, video_id))""")
            if ver < 9:
                cols = [r[1] for r in c.execute("PRAGMA table_info(accounts)").fetchall()]
                if "following" not in cols:
                    c.execute("ALTER TABLE accounts ADD COLUMN following INTEGER NOT NULL DEFAULT 0")
            if ver < 10:
                cols = [r[1] for r in c.execute("PRAGMA table_info(account_videos)").fetchall()]
                for col, typ in (("clip_job", "TEXT"), ("clip_idx", "INTEGER"),
                                 ("link_source", "TEXT NOT NULL DEFAULT ''"),
                                 ("first_seen", "REAL"), ("last_seen", "REAL"),
                                 ("views_updated", "REAL")):
                    if col not in cols:
                        c.execute(f"ALTER TABLE account_videos ADD COLUMN {col} {typ}")
                c.execute("CREATE INDEX IF NOT EXISTS ix_account_videos_clip"
                          " ON account_videos(clip_job, clip_idx)")
                tcols = [r[1] for r in c.execute("PRAGMA table_info(tags)").fetchall()]
                if "hashtags" not in tcols:
                    c.execute("ALTER TABLE tags ADD COLUMN hashtags TEXT NOT NULL DEFAULT ''")
                c.execute("""CREATE TABLE IF NOT EXISTS tag_accounts(
                  tag_id INTEGER NOT NULL, account_id INTEGER NOT NULL,
                  PRIMARY KEY(tag_id, account_id))""")
                c.execute("CREATE INDEX IF NOT EXISTS ix_tag_accounts_acc"
                          " ON tag_accounts(account_id)")
                c.execute("""INSERT OR IGNORE INTO tag_accounts(tag_id, account_id)
                  SELECT id, account_id FROM tags WHERE account_id IS NOT NULL""")
                c.execute("""CREATE TABLE IF NOT EXISTS video_metrics(
                  account_id INTEGER NOT NULL, video_id TEXT NOT NULL,
                  ts REAL NOT NULL, views INTEGER NOT NULL DEFAULT 0,
                  likes INTEGER NOT NULL DEFAULT 0,
                  comments INTEGER NOT NULL DEFAULT 0,
                  PRIMARY KEY(account_id, video_id, ts))""")
                c.execute("CREATE INDEX IF NOT EXISTS ix_video_metrics_acc"
                          " ON video_metrics(account_id, ts)")
                c.execute("CREATE INDEX IF NOT EXISTS ix_video_metrics_vid"
                          " ON video_metrics(account_id, video_id, ts)")
            if ver < 11:
                # пресеты: варианты описаний (через строку '---') и наборы
                # тегов (по строке) — один пресет на все нарезки серии
                c.execute("""CREATE TABLE IF NOT EXISTS presets(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT UNIQUE NOT NULL,
                  descs TEXT NOT NULL DEFAULT '',
                  tagsets TEXT NOT NULL DEFAULT '',
                  created REAL)""")
                # очередь публикаций: файл -> пресет -> аккаунт -> время
                c.execute("""CREATE TABLE IF NOT EXISTS posts(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  file TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'short',
                  title TEXT NOT NULL DEFAULT '',
                  preset_id INTEGER, caption TEXT NOT NULL DEFAULT '',
                  account TEXT NOT NULL DEFAULT '1',
                  scheduled REAL NOT NULL,
                  status TEXT NOT NULL DEFAULT 'queued',
                  stage TEXT NOT NULL DEFAULT '',
                  error TEXT NOT NULL DEFAULT '',
                  pid INTEGER,
                  created REAL, started REAL, finished REAL)""")
                c.execute("CREATE INDEX IF NOT EXISTS ix_posts_due"
                          " ON posts(status, scheduled)")
                c.execute("CREATE INDEX IF NOT EXISTS ix_posts_file"
                          " ON posts(file, kind)")
                jcols = [r[1] for r in c.execute("PRAGMA table_info(jobs)").fetchall()]
                if "preset_id" not in jcols:
                    c.execute("ALTER TABLE jobs ADD COLUMN preset_id INTEGER")
            c.execute("INSERT OR REPLACE INTO meta(k,v) VALUES ('version',?)",
                      (str(DB_VERSION),))
    finally:
        c.close()


def db():
    return _connect()


def log_job(job_id, title, url, status, clips=None, scorer="heuristic", tags=None,
            preset_id=None):
    clips = clips or []
    try:
        c = _connect()
        try:
            with lock:
                created = jobs.get(job_id, {}).get("started", time.time())
            with c:
                c.execute(
                    "INSERT OR REPLACE INTO jobs"
                    "(id,title,url,status,created,finished,clips,total_size,scorer,preset_id)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (job_id, title or "", url or "", status, created, time.time(),
                     json.dumps(clips, ensure_ascii=False),
                     sum(int(x.get("size", 0) or 0) for x in clips), scorer,
                     _preset_id_value(preset_id)))
                c.execute("DELETE FROM clips WHERE job_id=?", (job_id,))
                # повторная запись задачи — новые клипы: старые теги и
                # привязки к публикациям к ним уже не относятся
                c.execute("DELETE FROM clip_tags WHERE job_id=?", (job_id,))
                c.execute("UPDATE account_videos SET clip_job=NULL, clip_idx=NULL,"
                          " link_source='' WHERE clip_job=?", (job_id,))
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
                tids = []
                for t in (tags or []):
                    try:
                        tids.append(int(t))
                    except Exception:
                        pass
                if tids:
                    q = ",".join("?" * len(tids))
                    valid = [r[0] for r in c.execute(
                        f"SELECT id FROM tags WHERE id IN ({q})", tids).fetchall()]
                    rows = []
                    for k, x in enumerate(clips):
                        idx = int(x.get("i", k))
                        rows.extend((job_id, idx, tid) for tid in valid)
                    if rows:
                        c.executemany(
                            "INSERT OR IGNORE INTO clip_tags"
                            "(job_id,idx,tag_id) VALUES(?,?,?)", rows)
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()


def get_settings():
    try:
        with open(CFG, encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        saved = {}
    out = {**DEFAULT_CFG, **saved}
    if saved and "remove_pauses" not in saved:
        # конфиг прошлой версии: включаем новые дефолты один раз,
        # дальше уважаем выбор пользователя в настройках
        out["antidetect"] = True
        out["remove_pauses"] = True
        try:
            with open(CFG, "w", encoding="utf-8") as f:
                json.dump({k: out[k] for k in DEFAULT_CFG if k in out},
                          f, ensure_ascii=False)
        except OSError:
            pass
    out["gameplay"] = ac.gameplay_value(out.get("gameplay"))
    return out


def save_settings(d):
    cur = get_settings()
    for k in DEFAULT_CFG:
        if k in d:
            cur[k] = d[k]
    cur["gameplay"] = ac.gameplay_value(cur.get("gameplay"))
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


class _YdlLogger:
    """Ошибки yt-dlp — в server.log, а не в пустоту (иначе 'Conversion
    failed' не расследуешь). Прогресс не дублируем — он идёт через hook."""

    def debug(self, msg):
        pass

    def warning(self, msg):
        print(f"[yt-dlp] {msg}", flush=True)

    def error(self, msg):
        print(f"[yt-dlp] {msg}", flush=True)


# Запасной формат без слияния экзотики: avc+aac в родном mp4.
# Основной формат иногда падает на постобработке (битый фрагмент,
# странный кодек) — тогда докачиваем этим, он собирается всегда.
_SAFE_FORMAT = ("bv*[height<=1080][fps<=30][ext=mp4]+ba[ext=m4a]"
                "/b[height<=1080][ext=mp4]/b")


def fetch_source(url: str, progress=None, limit_min: int = 0) -> str:
    """Качает всё видео целиком быстрым aria2c. БЕЗ download_sections:
    проверено живьём — для YouTube DASH секции всё равно тянут весь
    файл (1.2 ГБ ради 10 минут), только медленным нативным качальщиком
    и через хрупкую постобработку (отсюда 'Conversion failed').
    limit_min режется локально за секунды (_slice_copy ниже)."""
    from yt_dlp import YoutubeDL

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes", 0)
            k = (done / total) if total else 0
            sp = (d.get("_speed_str") or "").strip()
            progress(k, f"качаю видео… {int(k * 100)}% {sp}".strip())

    try:
        limit_min = max(0, int(limit_min or 0))
    except (TypeError, ValueError):
        limit_min = 0

    def _base_opts(out):
        return {"quiet": True, "no_warnings": True, "noplaylist": True,
                "socket_timeout": 30, "ffmpeg_location": FFMPEG,
                "logger": _YdlLogger(),
                "progress_hooks": [hook],
                "concurrent_fragment_downloads": 4,
                "retries": 10, "fragment_retries": 10,
                # вис ниже 100 КБ/с — убивать и ретраить, а не висеть вечно
                "throttled_rate": 100000,
                # БЕЗ extractor_args: дефолтные клиенты отдают 1440p, а android/ios
                # под SABR-гейтом дают только 360p — проверяли, это и душило качество
                # aria2c: 16 коннектов вместо одного — ровная скорость без просадок
                "external_downloader": {"default": "aria2c"},
                "external_downloader_args": {
                    "aria2c": ["-x", "16", "-s", "16", "-k", "1M",
                               "--file-allocation=none", "-q"]},
                # качаем до 1440p: даунскейл в 1080 чище, чем апскейл из 720p.
                # vbr первым — из одинаковых берём самый жирный по битрейту
                # 30fps-варианты первыми: выход всё равно max 30fps, а 60fps = x2 байт впустую
                "format": ("bv*[height<=1440][fps<=30]+ba/b[height<=1440][fps<=30]"
                           "/bv*[height<=1440]+ba/b[height<=1440]"),
                "format_sort": ["vbr"],
                "merge_output_format": "mp4", "outtmpl": out}

    def _grabber(opts):
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return (info or {}).get("title", "видео")

    tmp = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
    try:
        title = _grabber(_base_opts(tmp))
    except Exception as e1:
        # постобработка/слияние упало — второй шанс merge-safe форматом
        print(f"[fetch] основной формат не собрался ({str(e1)[:150]}), "
              f"пробую { _SAFE_FORMAT.split('+')[0]}…", flush=True)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        tmp = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
        so = _base_opts(tmp)
        so["format"] = _SAFE_FORMAT
        try:
            title = _grabber(so)
        except Exception as e2:
            raise RuntimeError(f"не скачалось: {str(e2)[:200]}") from e2
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
    if limit_min > 0:
        # только первые N минут: мгновенный локальный вырез вместо секций
        cut = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
        try:
            _slice_copy(path, 0.0, limit_min * 60.0, cut)
            try:
                os.remove(path)
            except OSError:
                pass
            path = cut
        except Exception:
            try:
                if os.path.exists(cut):
                    os.remove(cut)
            except OSError:
                pass
    return path, title


def _fmt_ts(t: float) -> str:
    """Секунды -> ЧЧ:ММ:СС для подписей."""
    t = max(0, int(t))
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def _story_manual_moments(stamps, kind, total_dur=0.0, segs=None):
    """Ручные таймкоды -> моменты в формате историй.
    kind='parts' нумерует части и вешает «продолжение» только на последнюю;
    kind='moments' оставляет самостоятельные моменты без нумерации."""
    ranges = []
    for st in (stamps or [])[:20]:
        try:
            s, e = float(st.get("start", 0)), float(st.get("end", 0))
        except (TypeError, ValueError):
            continue
        if s < 0 or e <= s:
            continue
        # границы — к целым словам, иначе ручной таймкод режет полслова
        s, e = ac.snap_words(s, e, segs)
        if total_dur:
            if s >= total_dur:
                continue
            e = min(e, total_dur)
        if e - s < 1.5:
            continue
        ranges.append((s, e))
    parts = kind != "moments"
    n = len(ranges)
    out = []
    for k, (s, e) in enumerate(ranges):
        stamp = f"{_fmt_ts(s)}–{_fmt_ts(e)}"
        title = (f"Часть {k + 1}/{n}" if parts else f"Момент {k + 1}") + " · " + stamp
        out.append({
            "start": round(s, 3), "end": round(e, 3), "score": 100,
            "title": title, "verdict": f"Вручную {stamp}",
            "hook": title[:40],
            "part_title": f"ЧАСТЬ {k + 1}/{n}" if parts else "",
            "story": 1 if parts else 0,
            "part": k if parts else 0,
            "of": n if parts else 1,
            "last": (k == n - 1) if parts else True,
        })
    return out


def _slice_copy(src: str, s: float, dur: float, out: str) -> str:
    """Мгновенный вырез куска без перекода (-c copy, секунды).
    Границы по кейфреймам (±пару сек) — не страшно: дальше всё
    (паузы, транскрибация, рендер) работает в локальных координатах
    куска, а запас pad покрывает сдвиг."""
    import subprocess as _sp
    _sp.run([FFMPEG, "-y", "-v", "error",
             "-ss", str(round(max(s, 0.0), 2)), "-t", str(round(max(dur, 1.0), 2)),
             "-i", src, "-map", "0:v", "-map", "0:a?",
             "-c", "copy", "-movflags", "+faststart", out],
            timeout=600, check=True)
    if not ac._readable_video(out):
        raise RuntimeError("вырез куска не удался")
    return out


def _manual_story_cut(src, m, segs):
    """История по своим таймкодам: точный вырез отрезка + паузы без речи
    (по словам и silero VAD — даже поверх музыки). Возвращает локальные
    (файл, момент, сегменты) и временные файлы на удаление."""
    s, e = float(m["start"]), float(m["end"])
    dur = max(e - s, 1.0)
    cut = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
    _slice_copy(src, s, dur, cut)
    words = []
    for sg in (segs or []):
        for w in (sg.get("words") or []):
            if w["e"] > s and w["s"] < e:
                words.append({"w": w["w"],
                              "s": max(w["s"] - s, 0.0),
                              "e": min(w["e"] - s, dur)})
    words.sort(key=lambda w: w["s"])
    gaps = ac.speech_gaps(words, 0.0, dur) if words else []
    gaps += ac.vad_gaps(cut, 0.0, dur)
    keeps = []
    clean = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
    try:
        got = ac.remove_silence(cut, clean, window=(0.0, dur),
                                silences=gaps, keep_out=keeps)
    except Exception as ex:
        print("[story pauses]", ex)
        try:
            if os.path.exists(clean):
                os.remove(clean)
        except OSError:
            pass
        got, keeps = cut, [(0.0, dur)]
    local = ac.remap_segs([{"start": 0.0, "end": dur, "words": words}], keeps)
    mm = dict(m)
    mm["start"] = 0.0
    mm["end"] = ac.media_duration(got) or round(dur, 1)
    return got, mm, local, [cut, clean]


def run_manual_job(job_id: str, url: str, opts: dict, stamps: list, prog) -> list:
    """Ручные таймкоды — короткий путь без DREX, сцен и полной
    транскрибации. YouTube не отдаёт куски для 1080p+ (проверено:
    --download-sections всё равно тянет весь файл, а N таких скачек
    только хуже), поэтому: качаем 1 раз целиком, мгновенно режем
    на куски локально, дальше — только твои минуты (паузы, субтитры,
    рендер). Возвращает clips-мета для log_job."""
    total = len(stamps)
    mode = opts.get("vmode", "split")
    # --- фаза 1: одна скачка (2→20%) + мгновенная нарезка кусков ---
    prog(2, "качаю видео…")
    full, vtitle = fetch_source(
        url, progress=lambda k, s: prog(int(2 + k * 18), "качаю видео…"))
    with lock:
        if job_id in jobs:
            jobs[job_id].update(title=vtitle)
    raws = []
    for i, m in enumerate(stamps):
        s, e = float(m["start"]), float(m["end"])
        p = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
        _slice_copy(full, max(s - 8.0, 0.0), (e - s) + 16.0, p)
        raws.append(p)
    try:
        if os.path.exists(full):
            os.remove(full)  # дальше нужны только куски
    except OSError:
        pass
    # --- фаза 2: паузы -> субтитры минут -> рендер (20→99%) ---
    clips = []
    for i, (p, m) in enumerate(zip(raws, stamps)):
        f0 = 20 + 79 * i / total
        f1 = 20 + 79 * (i + 1) / total

        def _pp(q, _a=f0, _b=f1, _i=i):
            with lock:
                if job_id in jobs:
                    npct = int(_a + max(0.0, min(1.0, q)) * (_b - _a))
                    if npct > jobs[job_id].get("pct", 0):
                        jobs[job_id].update(
                            stage=f"момент {_i + 1}/{total}…", pct=npct)

        s, e = float(m["start"]), float(m["end"])
        # окно внутри слайса: слайс начинается с s-8, поэтому окно =
        # (pre, pre + длина), иначе в клип попадали лишние 8 секунд до таймкода
        pre = s - max(s - 8.0, 0.0)
        clip_len = max(e - s, 1.0)
        # ручной таймкод часто падает в середину слова — клип начинался
        # или заканчивался обрубком. Тянем границы к целым словам по
        # пословной транскрибации слайса (в его локальной шкале).
        segs_p = []
        try:
            segs_p, _lp = ac.transcribe(
                p, 0, progress=lambda q, st: _pp(q / 34.0 * 0.12),
                model_name=opts.get("asr_model", "giga"),
                api_key=opts.get("asr_api_key", ""),
                api_base=opts.get("asr_api_base", ""))
            ws, we = ac.snap_words(pre, pre + clip_len, segs_p)
            pdur = ac.media_duration(p) or 0.0
            if pdur > 0:
                ws, we = max(0.0, ws), min(pdur, we)
            if we - ws >= 0.5:
                pre, clip_len = ws, we - ws
        except Exception as ex:
            print("[manual words]", ex)
        want_pauses = bool(opts.get("remove_pauses", True))
        # паузы без речи: по словам + silero VAD (слова ASR иногда
        # растягиваются через паузу — VAD слушает звук и страхует)
        gaps = []
        if want_pauses:
            gaps = ac.speech_gaps(
                [w for sg in segs_p for w in (sg.get("words") or [])],
                pre, pre + clip_len)
            gaps += ac.vad_gaps(p, pre, pre + clip_len)
        keeps = []
        clean = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
        try:
            got = ac.remove_silence(
                p, clean,
                window=(pre, pre + clip_len),
                detect=want_pauses, silences=gaps, keep_out=keeps,
                progress=lambda q, st: _pp(q * 0.10, ))
        except Exception as ex:
            print("[manual pauses]", ex)
            try:
                if os.path.exists(clean):
                    os.remove(clean)
            except OSError:
                pass
            # паузы не почистились — хотя бы точное окно, без 8с запаса
            trim = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
            try:
                _slice_copy(p, pre, clip_len, trim)
                got = trim
                keeps = [(pre, pre + clip_len)]
            except Exception as ex2:
                print("[manual trim]", ex2)
                try:
                    if os.path.exists(trim):
                        os.remove(trim)
                except OSError:
                    pass
                got = p
        stage_in = got
        # субтитры в шкале после вырезания пауз — переносим сегменты,
        # а не транскрибируем второй раз
        segs_i = (ac.remap_segs(segs_p, keeps)
                  if mode != "fast" and opts.get("captions", True) else [])
        dur = ac.media_duration(stage_in) or round(clip_len, 1)
        title_m = f"Момент {i + 1} · {_fmt_ts(s)}–{_fmt_ts(e)}"
        mm = {"start": 0.0, "end": dur, "score": 100, "title": title_m,
              "verdict": f"Вручную {_fmt_ts(s)}–{_fmt_ts(e)}",
              "hook": title_m[:40]}
        out = os.path.join(OUT_DIR, _clip_filename(job_id, i, opts.get("out_name", "")))
        if mode == "fast":
            ac.fast_cut(stage_in, {"start": 0.0, "end": dur}, out)
        else:
            _bgm = ac.pick_clip_music() if opts.get("clip_music", True) else None
            ac.render_clip(
                stage_in, mm, segs_i, out,
                facetrack=opts.get("facetrack", True),
                captions=opts.get("captions", True),
                vmode=mode, antidetect=opts.get("antidetect", False),
                mirror=opts.get("mirror", False),
                banner=opts.get("banner", True),
                banner_path=opts.get("banner_path"),
                gameplay=opts.get("gameplay", "subway"),
                max_total=float(opts.get("max_len", 60) or 60),
                bgm=_bgm,
                hook=opts.get("hook", ""),
                progress=lambda q: _pp(0.35 + max(0.0, min(1.0, q)) * 0.65))
        clips.append({
            "i": i, "title": title_m, "score": 100,
            "verdict": f"Вручную {_fmt_ts(s)}–{_fmt_ts(e)}",
            "hook": title_m[:40], "story": 0, "part": 0, "of": 1,
            "file": os.path.basename(out),
            "dur": ac.media_duration(out) or round(dur, 1),
            "size": os.path.getsize(out) if os.path.exists(out) else 0})
        _pp(1.0)
        # слайсы отъели своё — удаляем сразу, а не копим в tmp
        for f in {p, stage_in}:
            try:
                if f and os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass
        try:
            if os.path.exists(clean) and clean not in (p, stage_in):
                os.remove(clean)
        except OSError:
            pass
    return clips


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
        mode = opts.get("vmode", "blur")
        raw_stamps = (opts.get("stamps") or "")
        stamps = ac.parse_stamps(raw_stamps)
        if raw_stamps.strip() and not stamps:
            # текст таймкодов есть, но ничего не распарсилось — раньше такое
            # МОЛЧА уходило в авто-режим с DREX. Теперь честная ошибка.
            raise RuntimeError(
                "таймкоды не распознались — пиши 'мм:сс-мм:сс', по одному на строку")
        if mode not in ("parts", "story") and stamps:
            # ручной путь: DREX/сцены/полное видео не нужны вообще —
            # качаем только моменты, субтитры только на эти минуты
            # истории обрабатываются ниже: им нужен src и story-рендер
            with lock:
                jobs[job_id].update(status="work", stage="качаю моменты…", pct=1)
            clips = run_manual_job(job_id, url, opts, stamps[:20], prog)
            with lock:
                jobs[job_id].update(status="done", pct=100, stage="готово",
                                    clips=clips, scorer="manual")
                title = jobs[job_id].get("title", url)
            log_job(job_id, title, url, "done", clips, "manual",
                    tags=opts.get("tags"), preset_id=opts.get("preset_id"))
            return
        with lock:
            jobs[job_id].update(status="work", stage="качаю видео…", pct=1)
        src, title = fetch_source(
            url, progress=lambda k, s: prog(int(1 + k * 7), s),
            limit_min=opts.get("limit_min", 0))
        with lock:
            jobs[job_id].update(title=title)
        # белая полоса снизу (poki у геймплейных рипов) — срезаем сразу,
        # дальше все режимы работают с чистым кадром
        src = ac.strip_bar_file(src)
        mode = opts.get("vmode", "blur")
        # паузы — один раз ДО сцен/транскрибации: дальше сцены, сегменты
        # и моменты живут в одной шкале чистого видео (без рассинхрона).
        # parts/story чистят паузы у себя, fast — только режет без перекода.
        if (mode not in ("parts", "story", "fast")
                and not opts.get("fast")
                and opts.get("remove_pauses", True)):
            prog(6, "чищу паузы…")
            clean = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
            try:
                stage_in = ac.remove_silence(
                    src, clean,
                    progress=lambda p, s: prog(int(6 + p * 0.03), "чищу паузы…"))
            except Exception as ex:
                print("[pauses]", ex)
                try:
                    if os.path.exists(clean):
                        os.remove(clean)
                except OSError:
                    pass
                stage_in = src
            if stage_in != src:
                try:
                    if os.path.exists(src):
                        os.remove(src)
                except OSError:
                    pass
                src = stage_in
        if mode == "story":
            # истории: русская озвучка уже в исходнике. Авто-режим чистит
            # паузы и ищет обрывы через DREX; ручные таймкоды режут ровно
            # выбранные отрезки по исходной шкале времени.
            manual = bool(stamps)
            if not manual:
                if opts.get("story_remove_pauses", True):
                    prog(6, "чищу паузы…")
                    clean = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
                    try:
                        stage_in = ac.remove_silence(
                            src, clean, min_sil=0.40, thresh_db=-38.0, pad=0.14,
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
            kind = opts.get("story_kind", "parts")
            scenes = []
            if kind == "moments" and not manual:
                prog(10, "ищу сцены…")
                try:
                    scenes = ac.detect_scenes(src)
                except Exception:
                    scenes = []
            prog(10, "транскрибация…")
            segs, lang = ac.transcribe(
                src, opts.get("limit_min", 0),
                progress=lambda p, s: prog(int(10 + p * 0.7), s),
                model_name=opts.get("asr_model", "giga"),
                api_key=opts.get("asr_api_key", ""),
                api_base=opts.get("asr_api_base", ""))
            if not segs:
                raise RuntimeError("не расслышал русскую озвучку в видео")
            if manual:
                moments = _story_manual_moments(
                    stamps, kind, total_dur=ac.media_duration(src) or 0.0,
                    segs=segs)
                if not moments:
                    raise RuntimeError(
                        "таймкоды вне видео или короче 1.5 сек — проверь минуты")
                scorer = "manual"
            elif kind == "moments":
                prog(82, "DREX: ищу интересные моменты…")
                moments, scorer = ac.pick_moments(
                    segs, lang, opts.get("api_key", ""),
                    int(opts.get("want", 8) or 8),
                    float(opts.get("min_len", 40) or 40),
                    float(opts.get("max_len", 59) or 59),
                    scenes=scenes, min_gap=opts.get("min_gap", 30),
                    domain="series")
                if not moments:
                    raise RuntimeError("DREX не нашёл моментов — проверь ключ")
                for m in moments:
                    m["last"] = True
                    m["part_title"] = ""
                    m["story"] = 0
                    m["part"] = 0
                    m["of"] = 1
            else:
                prog(82, "DREX 1.5: ищу обрывы…")
                moments = ac.pick_anime_parts(
                    segs, opts.get("api_key", ""),
                    min_len=float(opts.get("min_len", 180) or 180),
                    max_len=float(opts.get("max_len", 240) or 240))
                scorer = "drex"
                if not moments:
                    raise RuntimeError("DREX 1.5 не нашёл обрывов — проверь ключ")
        elif mode == "parts":
            # режим частей: сцены и DREX не нужны вообще.
            # Сначала чистим паузы, потом ОДНА транскрибация, потом резка.
            if opts.get("remove_pauses", True):
                prog(6, "чищу паузы…")
                clean = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
                try:
                    stage_in = ac.remove_silence(
                        src, clean,
                        progress=lambda p, s: prog(int(6 + p * 0.04), "чищу паузы…"))
                except Exception:
                    try:
                        if os.path.exists(clean):
                            os.remove(clean)
                    except OSError:
                        pass
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
                                       model_name=opts.get("asr_model", "giga"),
                                       api_key=opts.get("asr_api_key", ""),
                                       api_base=opts.get("asr_api_base", ""))
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
                                        model_name=opts.get("asr_model", "giga"),
                                        api_key=opts.get("asr_api_key", ""),
                                        api_base=opts.get("asr_api_base", ""))
        # Обычные клипы с ручными таймкодами сюда не доходят: их увёл ранний
        # возврат в run_manual_job. Истории со stamps собираются ниже.
        if mode not in ("parts", "story"):
            prog(82, "оценка моментов…")
            domain = ac.pick_domain(segs, opts.get("domain", "auto"))
            moments, scorer = ac.pick_moments(
                segs, lang, opts.get("api_key", ""),
                opts.get("want", 8),
                opts.get("min_len", 20), opts.get("max_len", 60),
                scenes=scenes, min_gap=opts.get("min_gap", 45),
                domain=domain)
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
            out = os.path.join(OUT_DIR, _clip_filename(job_id, i, opts.get("out_name", "")))
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

            # фон нарезки: свой трек на клип (ротация пула без повторов)
            clip_bgm = (ac.pick_clip_music() if opts.get("clip_music", True)
                        else None)
            extra = []
            if mode == "story":
                src_i, m_i, segs_i, story_tmp = src, m, segs, []
                if manual and opts.get("story_remove_pauses", True):
                    try:
                        src_i, m_i, segs_i, story_tmp = _manual_story_cut(
                            src, m, segs)
                    except Exception as ex:
                        print("[story cut]", ex)
                        src_i, m_i, segs_i, story_tmp = src, m, segs, []
                try:
                    nvars = max(1, min(3, int(opts.get("story_variants", 1) or 1)))
                    for _vi in range(nvars):
                        _o = out if _vi == 0 else out[:-4] + f"v{_vi + 1}.mp4"
                        raw = _o + ".raw.mp4" if opts.get("story_banner", True) else _o
                        _bgm = ac.pick_story_music() if opts.get("story_music", True) else None
                        ac.render_story_part(
                            src_i, m_i, segs_i, raw,
                            captions=opts.get("captions", True),
                            blur_subs=opts.get("story_blur_subs", True),
                            unique=opts.get("story_unique", True),
                            mirror=opts.get("story_mirror", opts.get("mirror", False)),
                            cta=opts.get("story_cta", True),
                            bgm=_bgm,
                            hq=bool(opts.get("story_banner", True)),
                            progress=_pp)
                        if raw != _o:
                            try:
                                ac.insert_banners(raw, _o,
                                                  banner_path=opts.get("banner_path"),
                                                  middle=True)
                            except Exception as e:
                                print("[banner]", e)
                                try:
                                    os.replace(raw, _o)
                                except OSError:
                                    pass
                            finally:
                                try:
                                    if os.path.exists(raw):
                                        os.remove(raw)
                                except OSError:
                                    pass
                        if _vi:
                            extra.append((_vi, _o))
                finally:
                    for f in story_tmp:
                        try:
                            if f and os.path.exists(f):
                                os.remove(f)
                        except OSError:
                            pass
            elif mode == "parts":
                ac.render_split_clip(src, m, segs, out,
                                     facetrack=False,
                                     captions=opts.get("captions", True),
                                     antidetect=opts.get("antidetect", False),
                                     mirror=opts.get("mirror", False),
                                     banner=opts.get("banner", True),
                                     banner_path=opts.get("banner_path"),
                                     gameplay=opts.get("gameplay", "subway"),
                                     max_total=float(opts.get("max_len", 59) or 59),
                                     bgm=clip_bgm,
                                     hook=opts.get("hook", ""),
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
                               bgm=clip_bgm,
                               hook=opts.get("hook", ""),
                               progress=_pp)
            with lock:
                done["n"] += 1
                n = done["n"]
                if job_id in jobs:
                    jobs[job_id].update(
                        stage=f"рендер… {n}/{len(moments)}",
                        pct=int(84 + n / len(moments) * 15))
            base = {"i": i, "title": m.get("title", f"Клип {i + 1}"),
                    "score": m.get("score", 0),
                    "verdict": str(m.get("verdict", ""))[:300],
                    "hook": str(m.get("hook", ""))[:40],
                    "story": int(m.get("story", 0)), "part": int(m.get("part", 0)),
                    "of": int(m.get("of", 1)),
                    "file": os.path.basename(out),
                    "dur": ac.media_duration(out) or round(float(m["end"]) - float(m["start"]), 1),
                    "size": os.path.getsize(out)}
            if not extra:
                return base
            res = [base]
            for _vi, _o in extra:
                d = dict(base)
                d["i"] = i + _vi / 10.0
                d["title"] = base["title"] + f" · вар.{_vi + 1}"
                d["file"] = os.path.basename(_o)
                d["dur"] = ac.media_duration(_o) or base["dur"]
                try:
                    d["size"] = os.path.getsize(_o)
                except OSError:
                    d["size"] = base["size"]
                res.append(d)
            return res

        # 2 клипа параллельно — кодирует железо, ffprobe/ffmpeg одного
        # потока все ядра не насыщают. as_completed вместо map: зависший
        # клип больше не блокирует выдачу готовых (map ждал по порядку —
        # один вставший ffmpeg держал всю пачку на «рендер… 2/8»).
        from concurrent.futures import as_completed as _ac
        ex = ThreadPoolExecutor(max_workers=1 if fast else 2)
        try:
            futs = {ex.submit(one, (i, m)): i for i, m in enumerate(moments)}
            for fut in _ac(futs):
                _r = fut.result()
                clips.extend(_r if isinstance(_r, list) else [_r])
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        clips.sort(key=lambda c: c["i"])
        with lock:
            jobs[job_id].update(status="done", pct=100, stage="готово", clips=clips)
            title = jobs[job_id].get("title", url)
        log_job(job_id, title, url, "done", clips,
                jobs.get(job_id, {}).get("scorer", "heuristic"),
                tags=opts.get("tags"), preset_id=opts.get("preset_id"))
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
                scorer=jobs.get(job_id, {}).get("scorer", "heuristic"),
                preset_id=opts.get("preset_id"))
    finally:
        try:
            if src and os.path.exists(src):
                os.remove(src)
        except OSError:
            pass
        free_slot()


@app.get("/")
def index():
    r = send_file(os.path.join(BASE, "index.html"))
    r.headers["Cache-Control"] = "no-store"  # фронт правим часто — без залипшего кэша
    return r


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
    story = str(data.get("vmode") or "") == "story"
    if not data.get("api_key"):
        data["api_key"] = cfg.get("drex_key", "")
    for k in ("want", "gameplay", "limit_min", "vmode", "captions", "facetrack", "antidetect", "mirror", "banner", "clip_music", "remove_pauses", "asr_model", "asr_api_key", "asr_api_base", "drex_key"):
        if k not in data or data[k] in ("", None):
            data[k] = cfg.get(k)
    data["gameplay"] = ac.gameplay_value(data.get("gameplay"))
    out_name = _safe_stem(data.get("out_name"))
    if story:
        kind = str(data.get("story_kind") or "parts")
        if kind not in ("parts", "moments"):
            kind = "parts"
        data["story_kind"] = kind
        try:
            mn = int(data.get("min_len") or (40 if kind == "moments" else 180))
            mx = int(data.get("max_len") or (59 if kind == "moments" else 240))
        except (TypeError, ValueError):
            mn, mx = (40, 59) if kind == "moments" else (180, 240)
        if kind == "moments":
            data["min_len"] = max(5, min(300, mn))
            data["max_len"] = max(8, min(300, mx))
        else:
            data["min_len"] = max(45, min(600, mn))
            data["max_len"] = max(60, min(600, mx))
        try:
            nv = int(data.get("story_variants") or 1)
        except (TypeError, ValueError):
            nv = 1
        data["story_variants"] = max(1, min(3, nv))
        if data.get("asr_model") not in ("base", "small", "giga", "whisper-large-v3"):
            data["asr_model"] = "giga"
        data["facetrack"] = True
    elif not data.get("min_len") or not data.get("max_len"):
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
                            "started": time.time(),
                            "kind": "story" if story else "clip"}
        threading.Thread(target=run_job, args=(job_id, url, {
            "api_key": data.get("api_key", ""), "want": want,
            "min_len": int(data.get("min_len", 20)), "max_len": int(data.get("max_len", 60)),
            "limit_min": int(data.get("limit_min", 0)), "min_gap": 20,
            "captions": bool(data.get("captions", True)),
            "facetrack": bool(data.get("facetrack", True)),
            "antidetect": bool(data.get("antidetect", False)),
            "mirror": bool(data.get("mirror", False)),
            "clip_music": bool(data.get("clip_music", True)),
            "remove_pauses": bool(data.get("remove_pauses", True)),
            "banner": bool(data.get("banner", True)),
            "banner_path": resolve_banner(data.get("banner_file", "")),
            "gameplay": data["gameplay"],
            "out_name": out_name,
            "hook": str(data.get("hook") or "").strip()[:120],
            "tags": _parse_tag_ids(data.get("tags")),
            "preset_id": _preset_id_value(data.get("preset_id")),
            "asr_model": data.get("asr_model", "giga") if data.get("asr_model") in ("base", "small", "giga", "whisper-large-v3") else "giga",
            "asr_api_key": data.get("asr_api_key", ""),
            "asr_api_base": data.get("asr_api_base", ""),
            "fast": bool(data.get("fast", False)),
            "vmode": data.get("vmode", "blur"), "stamps": stamps,
            "story_blur_subs": bool(data.get("story_blur_subs", True)),
            "story_unique": bool(data.get("story_unique", True)),
            "story_mirror": bool(data.get("story_mirror", False)),
            "story_cta": bool(data.get("story_cta", True)),
            "story_remove_pauses": bool(data.get("story_remove_pauses", True)),
            "story_kind": data.get("story_kind", "parts"),
            "story_banner": bool(data.get("story_banner", True)),
            "story_music": bool(data.get("story_music", True)),
            "story_variants": data.get("story_variants", 1),
            "domain": data.get("domain", "auto") if data.get("domain") in ("auto", "series", "streams") else "auto"}), daemon=True).start()
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
        # белая полоса снизу — срезаем сразу, дальше все режимы с чистым кадром
        src = ac.strip_bar_file(src)
        # паузы: стример молчит между катками — вырезаем до сцен/транскрибации,
        # чтобы все координаты дальше были по одному чистому таймлайну
        if opts.get("remove_pauses", True):
            prog(6, "чищу паузы…")
            clean = os.path.join(TMP_DIR, uuid.uuid4().hex + ".mp4")
            try:
                stage_in = ac.remove_silence(
                    src, clean,
                    progress=lambda p, s: prog(int(6 + p * 0.03), "чищу паузы…"))
            except Exception as ex:
                print("[twitch pauses]", ex)
                try:
                    if os.path.exists(clean):
                        os.remove(clean)
                except OSError:
                    pass
                stage_in = src
            if stage_in != src:
                try:
                    if os.path.exists(src):
                        os.remove(src)
                except OSError:
                    pass
                src = stage_in
        prog(6, "ищу сцены…")
        try:
            scenes = ac.detect_scenes(src)
        except Exception:
            scenes = []
        prog(10, "транскрибация…")
        segs, lang = ac.transcribe(src, opts.get("limit_min", 0),
                                   progress=lambda p, s: prog(int(10 + p * 0.7), s),
                                   model_name=opts.get("asr_model", "giga"),
                                   api_key=opts.get("asr_api_key", ""),
                                   api_base=opts.get("asr_api_base", ""))
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
            out = os.path.join(OUT_DIR, _clip_filename(job_id, i, opts.get("out_name", "")))
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
                                  bgm=(ac.pick_clip_music()
                                       if opts.get("clip_music", True) else None),
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
                    "file": os.path.basename(out),
                    "dur": ac.media_duration(out) or round(float(m["end"]) - float(m["start"]), 1),
                    "size": os.path.getsize(out)}

        from concurrent.futures import as_completed as _ac2
        ex = ThreadPoolExecutor(max_workers=2)
        try:
            futs = {ex.submit(one, (i, m)): i for i, m in enumerate(moments)}
            for fut in _ac2(futs):
                clips.append(fut.result())
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        clips.sort(key=lambda c: c["i"])
        with lock:
            jobs[job_id].update(status="done", pct=100, stage="готово", clips=clips)
            title = jobs[job_id].get("title", url)
        log_job(job_id, title, url, "done", clips,
                jobs.get(job_id, {}).get("scorer", "drex"),
                tags=opts.get("tags"), preset_id=opts.get("preset_id"))
        try:
            if src and os.path.exists(src):
                os.remove(src)
        except OSError:
            pass
    except Exception as e:
        with lock:
            jobs[job_id].update(status="error", error=str(e)[:300])
        log_job(job_id, url, url, "error",
                scorer=jobs.get(job_id, {}).get("scorer", "drex"),
                preset_id=opts.get("preset_id"))
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
    for k in ("asr_model", "asr_api_key", "asr_api_base"):
        if not data.get(k):
            data[k] = cfg.get(k, "")
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
                            "started": time.time(), "kind": "twitch"}
        threading.Thread(target=run_twitch_job, args=(job_id, url, {
            "api_key": data.get("api_key", ""), "want": want,
            "min_len": int(data.get("min_len", 40)), "max_len": int(data.get("max_len", 59)),
            "limit_min": limit_min, "min_gap": 20,
            "captions": True,
            "antidetect": bool(data.get("antidetect", False)),
            "mirror": bool(data.get("mirror", False)),
            "banner": bool(data.get("banner", True)),
            "banner_path": resolve_banner(data.get("banner_file", "")),
            "clip_music": bool(data.get("clip_music", cfg.get("clip_music", True))),
            "remove_pauses": bool(data.get("remove_pauses", cfg.get("remove_pauses", True))),
            "tags": _parse_tag_ids(data.get("tags")),
            "preset_id": _preset_id_value(data.get("preset_id")),
            "asr_model": data.get("asr_model", "giga") if data.get("asr_model") in ("base", "small", "giga", "whisper-large-v3") else "giga",
            "asr_api_key": data.get("asr_api_key", ""),
            "asr_api_base": data.get("asr_api_base", "")}), daemon=True).start()
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


@app.get("/api/jobs")
def api_jobs():
    """Задачи для восстановления очереди после F5 и после перезапуска
    сервера: память сессии + последние готовые задачи из истории."""
    keys = ("id", "url", "title", "status", "pct", "stage",
            "started", "error", "kind", "scorer")
    with lock:
        items = []
        known = set()
        for j in jobs.values():
            it = {k: j.get(k) for k in keys}
            it["clips"] = [c for c in (j.get("clips") or [])
                           if not c.get("hidden")]
            items.append(it)
            known.add(j.get("id"))
    try:
        c = _connect()
        try:
            rows = c.execute(
                "SELECT id,title,url,status,created,finished,scorer"
                " FROM jobs ORDER BY finished DESC LIMIT 12").fetchall()
            for r in rows:
                if r["id"] in known:
                    continue
                cr = c.execute(
                    "SELECT idx,title,score,verdict,hook,dur,size,views,story,part,\"of\""
                    " FROM clips WHERE job_id=? AND hidden=0 ORDER BY idx",
                    (r["id"],)).fetchall()
                story = any((x["story"] or 0) for x in cr)
                items.append({
                    "id": r["id"], "url": r["url"], "title": r["title"],
                    "status": r["status"], "pct": 100,
                    "stage": "готово", "started": r["created"],
                    "error": "" if r["status"] == "done" else "задача из истории завершилась с ошибкой — детали в логе сервера",
                    "kind": "story" if story else "clip",
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
    items.sort(key=lambda x: x.get("started") or 0)
    tmap = _clip_tag_map()
    for it in items:
        for cl in (it.get("clips") or []):
            cl["tags"] = tmap.get((it.get("id"), int(cl.get("i", -1))), [])
    return jsonify({"ok": True, "items": items})


@app.delete("/api/job/<job_id>")
def api_job_delete(job_id):
    """Удалить задачу: память + история + файлы клипов на диске."""
    with lock:
        j = jobs.get(job_id)
        if j and j.get("status") in ("queued", "work"):
            return jsonify({"ok": False,
                            "error": "Задача ещё выполняется — удалить можно после завершения"}), 409
        jobs.pop(job_id, None)
    removed = 0
    try:
        for name in _job_clip_files(job_id):
            try:
                os.remove(os.path.join(OUT_DIR, name))
                removed += 1
            except OSError:
                pass
            _drop_thumb("short", name)
    except Exception:
        pass
    try:
        c = _connect()
        try:
            with c:
                c.execute("DELETE FROM clip_tags WHERE job_id=?", (job_id,))
                c.execute("UPDATE account_videos SET clip_job=NULL, clip_idx=NULL,"
                          " link_source='' WHERE clip_job=?", (job_id,))
                c.execute("DELETE FROM clips WHERE job_id=?", (job_id,))
                c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
    return jsonify({"ok": True, "removed": removed})


def _short_path(name: str):
    """Файл шортса по имени (варианты историй живут как _1v2.mp4 и в int-роут
    не попадают). Только basename из shorts/legacy — без обхода путей."""
    raw = str(name or "")
    safe = os.path.basename(raw)
    if not safe or safe != raw or not safe.lower().endswith(".mp4"):
        return None, ""
    for d in (OUT_DIR, LEGACY_DIR):
        p = os.path.join(d, safe)
        if os.path.isfile(p):
            return p, safe
    return None, safe


@app.get("/api/short/<path:name>")
def api_short(name):
    p, safe = _short_path(name)
    if not p:
        return jsonify({"ok": False, "error": "нет файла"}), 404
    return send_file(p, as_attachment=True, download_name=safe)


@app.delete("/api/short/<path:name>")
def api_short_delete(name):
    p, safe = _short_path(name)
    if not p:
        return jsonify({"ok": False, "error": "нет файла"}), 404
    try:
        os.remove(p)
    except OSError:
        return jsonify({"ok": False, "error": "файл не удалился"}), 500
    _drop_thumb("short", safe)
    with lock:
        for j in jobs.values():
            cl = j.get("clips") or []
            for c in list(cl):
                if c.get("file") == safe:
                    cl.remove(c)
    return jsonify({"ok": True})


@app.get("/api/clip/<job_id>/<int:idx>")
def api_clip(job_id, idx):
    p = _clip_path(job_id, idx)
    if not p:
        return jsonify({"ok": False}), 404
    base = os.path.basename(p)
    with lock:
        j = jobs.get(job_id, {})
    clips = j.get("clips", [])
    if f"_{job_id}_" in base:
        name = base
    else:
        name = (clips[idx]["title"][:50] + ".mp4") if idx < len(clips) else f"short{idx + 1}.mp4"
    return send_file(p, as_attachment=True, download_name=name)


@app.post("/api/clip/<job_id>/<int:idx>/hide")
def api_clip_hide(job_id, idx):
    """Скрыть/вернуть готовый клип: файл остаётся на диске, из списков
    (очередь, история, клипы) он пропадает. Возврат — кнопкой во вкладке
    Клипы (галочка «скрытые»)."""
    data = request.get_json(force=True, silent=True) or {}
    hid = 1 if data.get("hidden", True) else 0
    with lock:
        j = jobs.get(job_id)
        if j:
            for c in (j.get("clips") or []):
                if int(c.get("i", -1)) == idx:
                    c["hidden"] = hid
    try:
        c = _connect()
        try:
            with c:
                c.execute("UPDATE clips SET hidden=? WHERE job_id=? AND idx=?",
                          (hid, job_id, idx))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
    return jsonify({"ok": True, "hidden": bool(hid)})


@app.delete("/api/clip/<job_id>/<int:idx>")
def api_clip_delete(job_id, idx):
    """Удалить один готовый клип: файл, постер и запись в истории."""
    p = _clip_path(job_id, idx)
    name = os.path.basename(p) if p else f"{job_id}_{idx + 1}.mp4"
    try:
        if p and os.path.exists(p):
            os.remove(p)
    except OSError:
        pass
    _drop_thumb("short", name)
    with lock:
        j = jobs.get(job_id)
        if j:
            j["clips"] = [c for c in (j.get("clips") or [])
                          if int(c.get("i", -1)) != idx]
    try:
        c = _connect()
        try:
            with c:
                c.execute("DELETE FROM clip_tags WHERE job_id=? AND idx=?",
                          (job_id, idx))
                c.execute("DELETE FROM clips WHERE job_id=? AND idx=?",
                          (job_id, idx))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
    return jsonify({"ok": True})


def _clip_tag_map():
    out = {}
    try:
        c = _connect()
        try:
            for r in c.execute("SELECT job_id, idx, tag_id FROM clip_tags"):
                out.setdefault((r["job_id"], r["idx"]), []).append(r["tag_id"])
        finally:
            c.close()
    except Exception:
        pass
    return out


def _norm_hashtags(value):
    """'#драма, истории' -> '#драма #истории' (без дублей)."""
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = re.split(r"[\s,]+", str(value or ""))
    out, seen = [], set()
    for p in parts:
        p = re.sub(r"[^\w\-а-яё]+", "", str(p or "").strip().lstrip("#"),
                   flags=re.I)[:40]
        if not p:
            continue
        h = "#" + p
        if h.lower() not in seen:
            seen.add(h.lower())
            out.append(h)
        if len(out) >= 15:
            break
    return " ".join(out)


def _tag_accounts_map(c, ids=None):
    q = ("SELECT ta.tag_id, a.id, a.nick FROM tag_accounts ta"
         " JOIN accounts a ON a.id=ta.account_id")
    if ids:
        q += " WHERE ta.tag_id IN (" + ",".join("?" * len(ids)) + ")"
    rows = c.execute(q, tuple(ids or ())).fetchall()
    out = {}
    for r in rows:
        out.setdefault(r["tag_id"], []).append({"id": r["id"], "nick": r["nick"]})
    for v in out.values():
        v.sort(key=lambda x: (x["nick"] or "").lower())
    return out


def _tag_json(r, count=0, posted=0, views=0, accounts=None):
    accounts = accounts or []
    acc_ids = [a["id"] for a in accounts]
    primary = r["account_id"] if r["account_id"] in acc_ids else (acc_ids[0] if acc_ids else None)
    mx = int(r["max_videos"] or 0)
    return {"id": r["id"], "name": r["name"], "theme": r["theme"],
            "hashtags": r["hashtags"] or "",
            "max_videos": mx, "account_id": primary,
            "account_ids": acc_ids, "accounts": accounts,
            "account_nick": ", ".join("@" + a["nick"] for a in accounts),
            "created": r["created"], "count": count, "posted": posted,
            "remaining": (max(0, mx - posted) if mx else None),
            "views": views}


@app.get("/api/tags")
def api_tags():
    """Теги + клипы, публикации, просмотры и остаток лимита."""
    items = []
    try:
        c = _connect()
        try:
            rows = c.execute(
                "SELECT t.*,"
                " COUNT(x.job_id) AS clips_cnt,"
                " COALESCE(SUM(x.posted),0) AS posted_cnt,"
                " COALESCE(SUM(x.views),0) AS views_sum"
                " FROM tags t"
                " LEFT JOIN ("
                "   SELECT ct.tag_id, ct.job_id, ct.idx,"
                "          CASE WHEN COUNT(av.video_id)>0 THEN 1 ELSE 0 END posted,"
                "          COALESCE(SUM(av.views),0) views"
                "   FROM clip_tags ct"
                "   JOIN clips cl ON cl.job_id=ct.job_id AND cl.idx=ct.idx"
                "                 AND cl.hidden=0"
                "   LEFT JOIN account_videos av"
                "     ON av.clip_job=ct.job_id AND av.clip_idx=ct.idx"
                "   GROUP BY ct.tag_id, ct.job_id, ct.idx"
                " ) x ON x.tag_id=t.id"
                " GROUP BY t.id ORDER BY t.theme, t.name").fetchall()
            amap = _tag_accounts_map(c)
            for r in rows:
                items.append(_tag_json(r, r["clips_cnt"] or 0,
                                       r["posted_cnt"] or 0, r["views_sum"] or 0,
                                       amap.get(r["id"], [])))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
    return jsonify({"ok": True, "items": items})


def _account_ids_from_body(d, c):
    """account_ids/account_id из тела запроса + валидация существования."""
    raw = d.get("account_ids", d.get("account_id"))
    if raw is None:
        raw = []
    elif isinstance(raw, str):
        raw = [x for x in re.split(r"[,\s]+", raw) if x]
    elif not isinstance(raw, (list, tuple, set)):
        raw = [raw]
    ids = []
    for x in raw:
        try:
            n = int(x)
        except Exception:
            return None, f"Некорректный аккаунт: {x}"
        if n > 0 and n not in ids:
            ids.append(n)
    if ids:
        q = ",".join("?" * len(ids))
        valid = {r[0] for r in c.execute(
            f"SELECT id FROM accounts WHERE id IN ({q})", ids).fetchall()}
        missing = [n for n in ids if n not in valid]
        if missing:
            return None, "Аккаунт не найден: " + ", ".join(map(str, missing))
    return ids, ""


def _set_tag_accounts(c, tid, ids):
    c.execute("DELETE FROM tag_accounts WHERE tag_id=?", (tid,))
    if ids:
        c.executemany("INSERT OR IGNORE INTO tag_accounts(tag_id,account_id)"
                      " VALUES(?,?)", [(tid, a) for a in ids])
    c.execute("UPDATE tags SET account_id=? WHERE id=?",
              (ids[0] if ids else None, tid))


@app.post("/api/tags")
def api_tag_create():
    d = request.get_json(force=True, silent=True) or {}
    name = (d.get("name") or "").strip().lstrip("#").strip()[:40]
    if not name:
        return jsonify({"ok": False, "error": "Введите название тега"}), 400
    theme = (d.get("theme") or "").strip()[:40]
    hashtags = _norm_hashtags(d.get("hashtags"))
    try:
        mx = max(0, int(d.get("max_videos") or 0))
    except Exception:
        mx = 0
    try:
        c = _connect()
        try:
            acc_ids, err = _account_ids_from_body(d, c)
            if acc_ids is None:
                return jsonify({"ok": False, "error": err}), 400
            with c:
                c.execute("INSERT INTO tags(name,theme,max_videos,account_id,"
                          "hashtags,created) VALUES(?,?,?,?,?,?)",
                          (name, theme, mx, acc_ids[0] if acc_ids else None,
                           hashtags, time.time()))
                r = c.execute("SELECT * FROM tags WHERE name=?", (name,)).fetchone()
                _set_tag_accounts(c, r["id"], acc_ids)
            accounts = _tag_accounts_map(c, [r["id"]]).get(r["id"], [])
        finally:
            c.close()
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "Такой тег уже есть"}), 409
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось создать тег"}), 500
    return jsonify({"ok": True, "tag": _tag_json(r, accounts=accounts)})


@app.patch("/api/tags/<int:tid>")
def api_tag_update(tid):
    d = request.get_json(force=True, silent=True) or {}
    try:
        c = _connect()
        try:
            r = c.execute("SELECT * FROM tags WHERE id=?", (tid,)).fetchone()
            if not r:
                return jsonify({"ok": False, "error": "Тег не найден"}), 404
            name = d.get("name", r["name"])
            name = (name or "").strip().lstrip("#").strip()[:40] or r["name"]
            theme = ((d.get("theme", r["theme"])) or "").strip()[:40]
            hashtags = (_norm_hashtags(d.get("hashtags", r["hashtags"]))
                        if "hashtags" in d else (r["hashtags"] or ""))
            try:
                mx = max(0, int(d.get("max_videos", r["max_videos"]) or 0))
            except Exception:
                mx = r["max_videos"]
            acc_ids = None
            if "account_ids" in d or "account_id" in d:
                acc_ids, err = _account_ids_from_body(d, c)
                if acc_ids is None:
                    return jsonify({"ok": False, "error": err}), 400
            with c:
                c.execute("UPDATE tags SET name=?,theme=?,max_videos=?,hashtags=?"
                          " WHERE id=?", (name, theme, mx, hashtags, tid))
                if acc_ids is not None:
                    _set_tag_accounts(c, tid, acc_ids)
            r = c.execute("SELECT * FROM tags WHERE id=?", (tid,)).fetchone()
            accounts = _tag_accounts_map(c, [tid]).get(tid, [])
        finally:
            c.close()
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "Такой тег уже есть"}), 409
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось сохранить тег"}), 500
    return jsonify({"ok": True, "tag": _tag_json(r, accounts=accounts)})


@app.delete("/api/tags/<int:tid>")
def api_tag_delete(tid):
    try:
        c = _connect()
        try:
            with c:
                c.execute("DELETE FROM tag_accounts WHERE tag_id=?", (tid,))
                c.execute("DELETE FROM clip_tags WHERE tag_id=?", (tid,))
                c.execute("DELETE FROM tags WHERE id=?", (tid,))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
    return jsonify({"ok": True})


def _parse_tag_ids(vals):
    if isinstance(vals, str):
        vals = vals.split(",")
    ids = set()
    for x in (vals or []):
        try:
            ids.add(int(x))
        except Exception:
            pass
    return sorted(ids)


def _do_clip_tags(c, job_id, idx, ids, replace=True):
    valid = {r[0] for r in c.execute("SELECT id FROM tags").fetchall()}
    ids = sorted(i for i in ids if i in valid)
    if replace:
        c.execute("DELETE FROM clip_tags WHERE job_id=? AND idx=?", (job_id, idx))
    if ids:
        c.executemany("INSERT OR IGNORE INTO clip_tags(job_id,idx,tag_id)"
                      " VALUES(?,?,?)", [(job_id, idx, i) for i in ids])
    return ids


@app.post("/api/clip/<job_id>/<int:idx>/tags")
def api_clip_tags_set(job_id, idx):
    """Точный набор тегов одного клипа (галочки в модалке)."""
    d = request.get_json(force=True, silent=True) or {}
    ids = _parse_tag_ids(d.get("tags"))
    try:
        c = _connect()
        try:
            with c:
                ids = _do_clip_tags(c, job_id, idx, ids, True)
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
    with lock:
        j = jobs.get(job_id)
        if j:
            for cl in (j.get("clips") or []):
                if int(cl.get("i", -1)) == idx:
                    cl["tags"] = ids
    return jsonify({"ok": True, "tags": ids})


@app.post("/api/job/<job_id>/tags")
def api_job_tags_add(job_id):
    """Добавить теги сразу всем клипам задачи (для «тег на всё видео»)."""
    d = request.get_json(force=True, silent=True) or {}
    ids = _parse_tag_ids(d.get("tags"))
    with lock:
        j = jobs.get(job_id)
        idxs = [int(c.get("i", -1)) for c in (j.get("clips") or [])] if j else []
    try:
        c = _connect()
        try:
            if not idxs:
                idxs = [r["idx"] for r in c.execute(
                    "SELECT idx FROM clips WHERE job_id=? AND hidden=0",
                    (job_id,)).fetchall()]
            with c:
                for i in idxs:
                    _do_clip_tags(c, job_id, i, ids, False)
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
    with lock:
        j = jobs.get(job_id)
        if j:
            for cl in (j.get("clips") or []):
                cl["tags"] = sorted(set(cl.get("tags") or []) | set(ids))
    return jsonify({"ok": True, "tags": ids, "clips": len(idxs)})


@app.get("/api/tag/<int:tid>/clips")
def api_tag_clips(tid):
    items = []
    try:
        c = _connect()
        try:
            rows = c.execute(
                "SELECT ct.job_id, ct.idx, cl.title, cl.score, cl.dur, cl.size,"
                " cl.story, cl.part, cl.\"of\", j.title AS job_title, j.url AS job_url,"
                " (SELECT COUNT(*) FROM account_videos av"
                "   WHERE av.clip_job=ct.job_id AND av.clip_idx=ct.idx) AS posts"
                " FROM clip_tags ct"
                " JOIN clips cl ON cl.job_id=ct.job_id AND cl.idx=ct.idx"
                " LEFT JOIN jobs j ON j.id=ct.job_id"
                " WHERE ct.tag_id=? AND cl.hidden=0"
                " ORDER BY COALESCE(j.finished,0) DESC, cl.score DESC LIMIT 300",
                (tid,)).fetchall()
            for r in rows:
                fp = _clip_path(r["job_id"], r["idx"])
                items.append({"job_id": r["job_id"], "i": r["idx"],
                              "file": os.path.basename(fp) if fp else "",
                              "title": r["title"] or "", "score": r["score"] or 0,
                              "dur": r["dur"] or 0, "size": r["size"] or 0,
                              "story": r["story"] or 0, "part": r["part"] or 0,
                              "of": r["of"] or 1, "posted": (r["posts"] or 0) > 0,
                              "job_title": r["job_title"] or r["job_url"] or r["job_id"]})
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
    return jsonify({"ok": True, "items": items})


def _clip_caption(title, tag_name, hashtags):
    text = re.sub(r"\s+", " ", (title or "")).strip()
    tags = hashtags or _norm_hashtags(tag_name)
    caption = (text + ("\n\n" + tags if tags else "")).strip()
    return caption[:2200]


@app.get("/api/tag/<int:tid>/zip")
def api_tag_zip(tid):
    """Пакет публикации тега: клипы + captions.csv/txt.
    max_videos — реальный стоп-лист: в пакет попадают только ещё не
    опубликованные клипы, и не больше остатка лимита. ?all=1 — всё."""
    import hashlib
    import zipfile
    all_mode = (request.args.get("all") or "").lower() in ("1", "true", "yes")
    try:
        c = _connect()
        try:
            t = c.execute("SELECT * FROM tags WHERE id=?", (tid,)).fetchone()
            rows = c.execute(
                "SELECT ct.job_id, ct.idx, cl.title, cl.score, j.finished,"
                " (SELECT COUNT(*) FROM account_videos av"
                "   WHERE av.clip_job=ct.job_id AND av.clip_idx=ct.idx) AS posts"
                " FROM clip_tags ct"
                " JOIN clips cl ON cl.job_id=ct.job_id AND cl.idx=ct.idx"
                " LEFT JOIN jobs j ON j.id=ct.job_id"
                " WHERE ct.tag_id=? AND cl.hidden=0"
                " ORDER BY COALESCE(j.finished,0) DESC, ct.idx ASC",
                (tid,)).fetchall()
            accounts = _tag_accounts_map(c, [tid]).get(tid, [])
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось собрать архив"}), 500
    if not t:
        return jsonify({"ok": False, "error": "Тег не найден"}), 404

    posted_clips = {(r["job_id"], int(r["idx"])) for r in rows if r["posts"]}
    posted = len(posted_clips)
    mx = int(t["max_videos"] or 0)
    if all_mode:
        selected = rows
        package_name = "all"
    else:
        unposted = [r for r in rows if not r["posts"]]
        if mx:
            remaining = max(0, mx - posted)
            if remaining <= 0:
                return jsonify({
                    "ok": False,
                    "error": f"Лимит тега исчерпан: опубликовано {posted}/{mx}. "
                             "Отвяжи лишнюю публикацию или увеличь лимит."}), 409
            selected = unposted[:remaining]
        else:
            selected = unposted
        if not selected:
            return jsonify({
                "ok": False,
                "error": "Все клипы тега уже опубликованы. "
                         "Кнопка «Всё» скачает архив без учёта публикаций."}), 409
        package_name = "package"

    def _safe(s):
        s = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", s or "")
        s = re.sub(r"\s+", " ", s).strip()
        return s[:60]

    base = _safe(t["name"]) or f"tag_{tid}"
    entries = []
    for r in selected:
        p = _clip_path(r["job_id"], r["idx"])
        if p and os.path.isfile(p):
            entries.append((p, r, os.path.getmtime(p), os.path.getsize(p)))
    if not entries:
        return jsonify({"ok": False,
                        "error": "К этому тегу пока не привязано ни одного файла"}), 404

    acc_nicks = ", ".join("@" + a["nick"] for a in accounts)
    hashtags = t["hashtags"] or _norm_hashtags(t["name"])
    captions = []
    for n, (p, r, _, _) in enumerate(entries, 1):
        nm = _safe(r["title"]) or os.path.splitext(os.path.basename(p))[0]
        fname = f"{n:02d}_{nm}.mp4"
        cap = _clip_caption(r["title"], t["name"], hashtags)
        captions.append({"file": fname, "title": r["title"] or "",
                         "caption": cap, "tag": t["name"],
                         "theme": t["theme"] or "", "hashtags": hashtags,
                         "accounts": acc_nicks})
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf, lineterminator="\n")
    writer.writerow(["file", "title", "caption", "tag", "theme",
                     "hashtags", "accounts"])
    for x in captions:
        writer.writerow([x["file"], x["title"], x["caption"], x["tag"],
                         x["theme"], x["hashtags"], x["accounts"]])
    csv_text = "\ufeff" + csv_buf.getvalue()
    txt_text = "\n\n".join(f"{x['file']}\n{x['caption']}" for x in captions) + "\n"

    sig = hashlib.sha1(
        (f"{tid}|{package_name}|{base}|{all_mode}|{mx}|{posted}|"
         f"{hashtags}|{t['theme']}|{acc_nicks}|"
         + "|".join(f"{r['job_id']}:{r['idx']}:{r['title']}:{m}:{s}"
                    for _, r, m, s in entries)).encode()).hexdigest()
    zp = os.path.join(TMP_DIR, f"tag_{tid}_{package_name}.zip")
    sf = zp + ".sig"
    cached = False
    try:
        cached = (os.path.exists(zp) and os.path.getsize(zp) > 0
                  and os.path.exists(sf)
                  and open(sf, encoding="utf-8").read().strip() == sig)
    except OSError:
        cached = False
    if not cached:
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
            z.writestr(f"{base}/captions.csv", csv_text)
            z.writestr(f"{base}/captions.txt", txt_text)
            for (p, r, _, _), x in zip(entries, captions):
                z.write(p, f"{base}/{x['file']}")
        with open(sf, "w", encoding="utf-8") as f:
            f.write(sig)
    return send_file(zp, mimetype="application/zip", as_attachment=True,
                     download_name=f"{base}_{package_name}.zip")


TT_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TT_BROWSERS = ("chrome", "safari", "firefox", "edge", "brave", "chromium")


def _tt_nick(raw):
    s = (raw or "").strip()
    m = re.search(r"tiktok\.com/@([\w.\-]+)", s)
    if m:
        s = m.group(1)
    s = s.lstrip("@").strip()
    return s if re.fullmatch(r"[\w.\-]{1,64}", s) else ""


def _tt_profile(nick):
    """Подписчики/лайки/кол-во видео из SSR-страницы профиля."""
    try:
        p = subprocess.run(["curl", "-sL", "--max-time", "25", "-A", TT_UA,
                            f"https://www.tiktok.com/@{nick}"],
                           capture_output=True, text=True, timeout=35)
        h = p.stdout or ""
        m = re.search(
            r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"'
            r' type="application/json">(.*?)</script>', h, re.S)
        if not m:
            return None
        d = json.loads(m.group(1))
        ui = (d.get("__DEFAULT_SCOPE__", {})
               .get("webapp.user-detail", {})
               .get("userInfo") or {})
        user = ui.get("user") or {}
        st = ui.get("stats") or {}
        if not user:
            return None
        return {"nick": user.get("uniqueId") or nick,
                "followers": int(st.get("followerCount") or 0),
                "following": int(st.get("followingCount") or 0),
                "likes": int(st.get("heartCount") or st.get("heart") or 0),
                "video_count": int(st.get("videoCount") or 0)}
    except Exception:
        return None


def _tt_parse_entries(j, nick):
    out = []
    for e in (j.get("entries") or []):
        if not e:
            continue
        vid = str(e.get("id") or "")
        if not vid:
            continue
        views = e.get("view_count")
        if views is None:
            st = e.get("stats") or {}
            views = st.get("playCount") or st.get("viewCount")
        try:
            views = int(views or 0)
        except Exception:
            views = 0
        try:
            likes = int(e.get("like_count") or 0)
        except Exception:
            likes = 0
        try:
            comments = int(e.get("comment_count") or 0)
        except Exception:
            comments = 0
        out.append({
            "video_id": vid,
            "title": (e.get("title") or "").strip(),
            "views": views, "likes": likes, "comments": comments,
            "url": e.get("url") or f"https://www.tiktok.com/@{nick}/video/{vid}",
            "published": e.get("timestamp") or 0})
    return out


def _tt_videos(nick, progress=None):
    """Видео аккаунта с просмотрами через yt-dlp.
    --flat-playlist часто отдаёт view_count=0. Если весь список нулевой,
    повторяем без flat — иначе хорошая статистика молча перезапишется
    нулями (в БД нули всё равно не затирают старые значения)."""
    browser = (get_settings().get("tt_cookies") or "").strip()
    url = f"https://www.tiktok.com/@{nick}"

    def _p(stage, pct):
        if progress:
            progress(stage, pct)

    def _cmd(flat, imp):
        cmd = ["yt-dlp", "--no-warnings", "-J", "--extractor-retries", "3"]
        if flat:
            cmd.append("--flat-playlist")
        if browser in TT_BROWSERS:
            cmd[1:1] = ["--cookies-from-browser", browser]
        if imp:
            cmd[1:1] = ["--impersonate", imp]
        cmd.append(url)
        return cmd

    def _run(flat, imp, timeout):
        try:
            p = subprocess.run(_cmd(flat, imp), capture_output=True,
                               text=True, timeout=timeout)
            return _tt_parse_entries(json.loads(p.stdout or "{}"), nick)
        except Exception:
            return []

    flat_out = []
    for imp in ("chrome", ""):
        _p("видео · быстрый список", 30)
        out = _run(True, imp, 300)
        if out:
            flat_out = out
            break
    if flat_out and any(v["views"] > 0 for v in flat_out):
        return flat_out, "flat"

    # нули (или пусто) — пробуем полную выдачу, она отдаёт просмотры
    for imp in ("chrome", ""):
        _p("видео · полные данные", 45)
        out = _run(False, imp, 600)
        if not out:
            continue
        if flat_out:
            merged = {v["video_id"]: v for v in flat_out}
            merged.update({v["video_id"]: v for v in out})
            return list(merged.values()), "full"
        return out, "full"
    return flat_out, ("flat-zero" if flat_out else "")



REFRESH = {}
REFRESH_LOCK = threading.Lock()


def _norm_title(s):
    """Нормализация заголовка для авто-привязки видео к клипу."""
    s = (s or "").lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"#\S+", " ", s)
    s = re.sub(r"[^\wа-яё]+", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def _auto_match(c, aid, now):
    """Аккуратная авто-привязка: точное совпадение или токены >= 0.6.
    Ручные привязки не трогаем."""
    clips = c.execute(
        "SELECT job_id, idx, title FROM clips"
        " WHERE hidden=0 AND COALESCE(title,'')!=''").fetchall()
    exact, ambiguous = {}, set()
    for cl in clips:
        key = _norm_title(cl["title"])
        if len(key) < 8:
            continue
        if key in exact:
            ambiguous.add(key)
        else:
            exact[key] = (cl["job_id"], int(cl["idx"]))
    keys = [(k, v) for k, v in exact.items() if k not in ambiguous]
    vids = c.execute(
        "SELECT video_id, title FROM account_videos"
        " WHERE account_id=? AND (clip_job IS NULL OR clip_job='')"
        " AND COALESCE(link_source,'')!='manual-off'",
        (aid,)).fetchall()
    for v in vids:
        key = _norm_title(v["title"])
        if len(key) < 8:
            continue
        match = exact.get(key) if key not in ambiguous else None
        if not match:
            contains = []
            for ck, pos in keys:
                if len(key) >= 12 and (key in ck or ck in key):
                    contains.append((ck, pos))
            if len(contains) == 1:
                match = contains[0][1]
        if not match:
            tokens = set(key.split())
            best, best_score = None, 0.0
            for ck, pos in keys:
                ct = set(ck.split())
                inter = len(tokens & ct)
                if inter < 3:
                    continue
                score = inter / max(1, len(tokens | ct))
                if score >= 0.6 and score > best_score:
                    best, best_score = pos, score
            match = best
        if match:
            c.execute("UPDATE account_videos SET clip_job=?, clip_idx=?,"
                      " link_source='auto' WHERE account_id=? AND video_id=?",
                      (match[0], match[1], aid, v["video_id"]))


def _baselines(c, account_id=None):
    """Последние снапшоты не новее 24ч/72ч -> {key: {'24': views, ...}}."""
    now = time.time()
    cuts = {"24": now - 86400, "72": now - 259200}
    q = "SELECT account_id, video_id, ts, views FROM video_metrics"
    args = ()
    if account_id is not None:
        q += " WHERE account_id=?"
        args = (account_id,)
    res = {}
    for r in c.execute(q, args):
        key = (r["account_id"], r["video_id"])
        slot = res.setdefault(key, {})
        for k, cut in cuts.items():
            if r["ts"] <= cut:
                cur = slot.get(k)
                if not cur or r["ts"] > cur[0]:
                    slot[k] = (r["ts"], r["views"])
    return {k: {kk: vv[1] for kk, vv in v.items()}
            for k, v in res.items()}


def _account_json(aid):
    try:
        c = _connect()
        try:
            r = c.execute("SELECT * FROM accounts WHERE id=?", (aid,)).fetchone()
            if not r:
                return None
            vids = c.execute(
                "SELECT av.*, cl.title AS clip_title, cl.hidden AS clip_hidden"
                " FROM account_videos av"
                " LEFT JOIN clips cl ON cl.job_id=av.clip_job"
                "   AND cl.idx=av.clip_idx"
                " WHERE av.account_id=?"
                " ORDER BY av.views DESC, av.published DESC", (aid,)).fetchall()
            bases = _baselines(c, aid)
        finally:
            c.close()
    except Exception:
        return None
    vlist = []
    d24 = d72 = 0
    n24 = n72 = 0
    for v in vids:
        base = bases.get((aid, v["video_id"]), {})
        cur = v["views"] or 0
        x24 = (cur - base["24"]) if "24" in base else None
        x72 = (cur - base["72"]) if "72" in base else None
        if x24 is not None:
            d24 += x24
            n24 += 1
        if x72 is not None:
            d72 += x72
            n72 += 1
        vlist.append({
            "video_id": v["video_id"], "title": v["title"] or "",
            "views": cur, "likes": v["likes"] or 0,
            "comments": v["comments"] or 0, "url": v["url"],
            "published": v["published"] or 0,
            "clip_job": v["clip_job"] or "", "clip_idx": v["clip_idx"],
            "clip_title": v["clip_title"] or "",
            "link_source": v["link_source"] or "",
            "delta24": x24, "delta72": x72,
            "first_seen": v["first_seen"], "last_seen": v["last_seen"],
            "views_updated": v["views_updated"]})
    n = len(vlist)
    total_views = r["total_views"] or sum(v["views"] for v in vlist)
    return {"id": r["id"], "nick": r["nick"], "created": r["created"],
            "synced": r["synced"], "followers": r["followers"],
            "following": r["following"] or 0,
            "likes": r["likes"], "video_count": r["video_count"],
            "total_views": total_views, "source": r["source"] or "",
            "error": r["error"] or "",
            "videos_total": n,
            "sum_likes": sum(v["likes"] or 0 for v in vlist),
            "sum_comments": sum(v["comments"] or 0 for v in vlist),
            "avg_views": int(total_views / n) if n else 0,
            "delta24": d24 if n24 else None,
            "delta72": d72 if n72 else None,
            "delta24_videos": n24, "delta72_videos": n72,
            "videos": vlist}


@app.get("/api/accounts")
def api_accounts():
    items = []
    try:
        c = _connect()
        try:
            ids = [r[0] for r in c.execute(
                "SELECT id FROM accounts ORDER BY id").fetchall()]
        finally:
            c.close()
    except Exception:
        ids = []
    for aid in ids:
        a = _account_json(aid)
        if a:
            with REFRESH_LOCK:
                st = dict(REFRESH.get(aid) or {})
            if st:
                a["refresh"] = st
            items.append(a)
    return jsonify({"ok": True, "items": items})


@app.post("/api/accounts")
def api_account_create():
    d = request.get_json(force=True, silent=True) or {}
    nick = _tt_nick(d.get("nick"))
    if not nick:
        return jsonify({"ok": False, "error": "Укажи ник или ссылку на профиль TikTok"}), 400
    try:
        c = _connect()
        try:
            with c:
                c.execute("INSERT INTO accounts(nick,created,error)"
                          " VALUES(?,?,'')", (nick, time.time()))
                aid = c.execute("SELECT id FROM accounts WHERE nick=?",
                                (nick,)).fetchone()[0]
        finally:
            c.close()
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "Этот аккаунт уже добавлен"}), 409
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось добавить аккаунт"}), 500
    return jsonify({"ok": True, "account": _account_json(aid)})


@app.delete("/api/accounts/<int:aid>")
def api_account_delete(aid):
    try:
        c = _connect()
        try:
            with c:
                c.execute("DELETE FROM tag_accounts WHERE account_id=?", (aid,))
                c.execute("UPDATE tags SET account_id=(SELECT account_id FROM"
                          " tag_accounts ta WHERE ta.tag_id=tags.id"
                          " ORDER BY account_id LIMIT 1) WHERE account_id=?", (aid,))
                c.execute("DELETE FROM video_metrics WHERE account_id=?", (aid,))
                c.execute("DELETE FROM account_videos WHERE account_id=?", (aid,))
                c.execute("DELETE FROM accounts WHERE id=?", (aid,))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
    with REFRESH_LOCK:
        REFRESH.pop(aid, None)
    return jsonify({"ok": True})


def _save_refresh(aid, a, prof, vids, source):
    """Слить новые метрики со старыми: нули не затирают просмотры,
    история пишется в video_metrics, связи с клипами сохраняются."""
    now = time.time()
    c = _connect()
    try:
        with c:
            if not c.execute("SELECT 1 FROM accounts WHERE id=?", (aid,)).fetchone():
                return
            old = {r["video_id"]: r for r in c.execute(
                "SELECT * FROM account_videos WHERE account_id=?", (aid,)).fetchall()}
            fresh = 0
            for v in vids:
                o = old.get(v["video_id"])
                views = int(v["views"] or 0)
                likes = int(v["likes"] or 0)
                comments = int(v["comments"] or 0)
                if o:
                    if views <= 0 and (o["views"] or 0) > 0:
                        views = o["views"]
                    if likes <= 0 and (o["likes"] or 0) > 0:
                        likes = o["likes"]
                    if comments <= 0 and (o["comments"] or 0) > 0:
                        comments = o["comments"]
                if views > 0:
                    fresh += 1
                title = (v["title"] or (o["title"] if o else "") or "")
                url = (v["url"] or (o["url"] if o else "") or "")
                published = (v["published"] or (o["published"] if o else 0) or 0)
                clip_job = o["clip_job"] if o else None
                clip_idx = o["clip_idx"] if o else None
                link_source = (o["link_source"] if o else "") or ""
                first_seen = (o["first_seen"] if o and o["first_seen"] else now)
                views_updated = (now if int(v["views"] or 0) > 0
                                 else (o["views_updated"] if o else None))
                if int(v["views"] or 0) > 0:
                    c.execute(
                        "INSERT OR IGNORE INTO video_metrics"
                        "(account_id,video_id,ts,views,likes,comments)"
                        " VALUES(?,?,?,?,?,?)",
                        (aid, v["video_id"], now, int(v["views"] or 0),
                         int(v["likes"] or 0), int(v["comments"] or 0)))
                c.execute(
                    "INSERT INTO account_videos"
                    "(account_id,video_id,title,views,likes,comments,url,published,"
                    " clip_job,clip_idx,link_source,first_seen,last_seen,views_updated)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(account_id,video_id) DO UPDATE SET"
                    " title=excluded.title, views=excluded.views,"
                    " likes=excluded.likes, comments=excluded.comments,"
                    " url=excluded.url, published=excluded.published,"
                    " clip_job=excluded.clip_job, clip_idx=excluded.clip_idx,"
                    " link_source=excluded.link_source, last_seen=excluded.last_seen,"
                    " views_updated=excluded.views_updated",
                    (aid, v["video_id"], title[:500], views, likes, comments,
                     url[:500], published, clip_job, clip_idx, link_source,
                     first_seen, now, views_updated))
            c.execute(
                "UPDATE account_videos SET clip_job=NULL, clip_idx=NULL,"
                " link_source='' WHERE account_id=? AND clip_job IS NOT NULL"
                " AND NOT EXISTS (SELECT 1 FROM clips cl"
                "   WHERE cl.job_id=account_videos.clip_job"
                "   AND cl.idx=account_videos.clip_idx)", (aid,))
            c.execute("DELETE FROM video_metrics WHERE account_id=? AND ts<?",
                      (aid, now - 120 * 86400))
            _auto_match(c, aid, now)
            total = c.execute(
                "SELECT COALESCE(SUM(views),0) FROM account_videos"
                " WHERE account_id=?", (aid,)).fetchone()[0]
            n_vids = c.execute(
                "SELECT COUNT(*) FROM account_videos WHERE account_id=?",
                (aid,)).fetchone()[0]
            if vids:
                err = ""
                if fresh == 0:
                    err = ("TikTok отдал нули — просмотры сохранены "
                           "с прошлого сбора" if total else
                           "TikTok отдал нули по просмотрам — включи cookies")
                c.execute(
                    "UPDATE accounts SET synced=?, followers=?, following=?,"
                    " likes=?, video_count=?, total_views=?, source=?, error=?"
                    " WHERE id=?",
                    (now,
                     prof["followers"] if prof else (a["followers"] or 0),
                     prof["following"] if prof else (a["following"] or 0),
                     prof["likes"] if prof else (a["likes"] or 0),
                     max(prof["video_count"] if prof else 0, n_vids),
                     total, source or "yt-dlp", err, aid))
            elif prof:
                c.execute(
                    "UPDATE accounts SET synced=?, followers=?, following=?,"
                    " likes=?, video_count=?, source='profile', error=?"
                    " WHERE id=?",
                    (now, prof["followers"], prof["following"], prof["likes"],
                     prof["video_count"],
                     "просмотры по видео TikTok не отдал — "
                     "включи cookies браузера в настройках вкладки", aid))
            else:
                c.execute("UPDATE accounts SET error=? WHERE id=?",
                          ("TikTok не отдал данные — проверь ник "
                           "или включи cookies браузера", aid))
    finally:
        c.close()


def _refresh_worker(aid):
    def state(**kw):
        with REFRESH_LOCK:
            st = REFRESH.setdefault(aid, {})
            st.update(kw)
            st["updated"] = time.time()

    try:
        c = _connect()
        try:
            a = c.execute("SELECT * FROM accounts WHERE id=?", (aid,)).fetchone()
        finally:
            c.close()
        if not a:
            raise RuntimeError("аккаунт удалён")
        state(status="work", stage="подписчики и профиль", pct=5, error="")
        prof = _tt_profile(a["nick"])
        state(stage="видео · быстрый список", pct=25)
        vids, source = _tt_videos(
            a["nick"], progress=lambda stage, pct: state(stage=stage, pct=pct))
        state(stage="сохраняю метрики", pct=80)
        c = _connect()
        try:
            alive = c.execute("SELECT 1 FROM accounts WHERE id=?",
                              (aid,)).fetchone()
        finally:
            c.close()
        if not alive:
            raise RuntimeError("аккаунт удалён")
        _save_refresh(aid, a, prof, vids, source)
        state(status="done", stage="готово", pct=100, finished=time.time(),
              videos=len(vids), source=source)
    except Exception as e:
        import traceback
        traceback.print_exc()
        msg = str(e)[:300]
        state(status="error", stage="ошибка", error=msg,
              finished=time.time())
        try:
            c = _connect()
            try:
                with c:
                    c.execute("UPDATE accounts SET error=? WHERE id=?", (msg, aid))
            finally:
                c.close()
        except Exception:
            pass


@app.post("/api/accounts/<int:aid>/refresh")
def api_account_refresh(aid):
    """Асинхронный сбор аналитики. HTTP отвечает сразу, прогресс — GET."""
    c = _connect()
    try:
        a = c.execute("SELECT id FROM accounts WHERE id=?", (aid,)).fetchone()
    finally:
        c.close()
    if not a:
        return jsonify({"ok": False, "error": "Аккаунт не найден"}), 404
    with REFRESH_LOCK:
        st = REFRESH.get(aid)
        if st and st.get("status") == "work":
            return jsonify({"ok": True, "state": dict(st)})
        REFRESH[aid] = {"status": "work", "stage": "в очереди", "pct": 0,
                        "started": time.time(), "finished": 0, "error": "",
                        "videos": 0}
        out = dict(REFRESH[aid])
    threading.Thread(target=_refresh_worker, args=(aid,), daemon=True).start()
    return jsonify({"ok": True, "state": out})


@app.get("/api/accounts/<int:aid>/refresh")
def api_account_refresh_status(aid):
    with REFRESH_LOCK:
        st = dict(REFRESH.get(aid) or {"status": "idle", "stage": "",
                                       "pct": 0, "error": ""})
    return jsonify({"ok": True, "state": st})


@app.post("/api/account-videos/<int:aid>/<video_id>/link")
def api_account_video_link(aid, video_id):
    """Ручная привязка видео TikTok к клипу: {job_id, idx} или пустой job_id."""
    d = request.get_json(force=True, silent=True) or {}
    job_id = str(d.get("job_id") or "").strip()
    try:
        idx = int(d.get("idx"))
    except Exception:
        idx = -1
    try:
        c = _connect()
        try:
            if not c.execute("SELECT 1 FROM accounts WHERE id=?", (aid,)).fetchone():
                return jsonify({"ok": False, "error": "Аккаунт не найден"}), 404
            with c:
                if job_id:
                    if idx < 0 or not c.execute(
                            "SELECT 1 FROM clips WHERE job_id=? AND idx=?",
                            (job_id, idx)).fetchone():
                        return jsonify({"ok": False,
                                        "error": "Клип не найден"}), 404
                    c.execute(
                        "UPDATE account_videos SET clip_job=?, clip_idx=?,"
                        " link_source='manual' WHERE account_id=? AND video_id=?",
                        (job_id, idx, aid, video_id))
                else:
                    c.execute(
                        "UPDATE account_videos SET clip_job=NULL, clip_idx=NULL,"
                        " link_source='manual-off' WHERE account_id=? AND video_id=?",
                        (aid, video_id))
                changed = c.total_changes
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось сохранить привязку"}), 500
    if not changed:
        return jsonify({"ok": False, "error": "Видео не найдено"}), 404
    return jsonify({"ok": True})


@app.get("/api/analytics/tags")
def api_analytics_tags():
    """Просмотры по тегам и тематикам + дельты 24ч/72ч."""
    try:
        c = _connect()
        try:
            tag_rows = c.execute(
                "SELECT * FROM tags ORDER BY theme, name").fetchall()
            rows = c.execute(
                "SELECT t.id AS tag_id, t.name, t.theme, t.max_videos,"
                " t.hashtags, ct.job_id, ct.idx, cl.title, cl.score,"
                " av.account_id, av.video_id, av.views, av.likes, av.comments"
                " FROM clip_tags ct"
                " JOIN tags t ON t.id=ct.tag_id"
                " JOIN clips cl ON cl.job_id=ct.job_id AND cl.idx=ct.idx"
                "   AND cl.hidden=0"
                " LEFT JOIN account_videos av"
                "   ON av.clip_job=ct.job_id AND av.clip_idx=ct.idx"
                " ORDER BY t.theme, t.name").fetchall()
            bases = _baselines(c)
            amap = _tag_accounts_map(c)
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось посчитать статистику"}), 500

    def empty_tag(t):
        mx = int(t["max_videos"] or 0)
        return {"tag_id": t["id"], "name": t["name"], "theme": t["theme"] or "",
                "hashtags": t["hashtags"] or "", "max_videos": mx,
                "account_ids": [a["id"] for a in amap.get(t["id"], [])],
                "accounts": amap.get(t["id"], []),
                "clips": set(), "posts": set(), "posted_clips": set(),
                "views": 0, "likes": 0,
                "comments": 0, "d24": 0, "d72": 0, "n24": 0, "n72": 0}

    tags = {t["id"]: empty_tag(t) for t in tag_rows}
    themes = {}
    for t in tag_rows:
        th = t["theme"] or "Без тематики"
        st = themes.setdefault(th, {"theme": th, "tags": set(), "clips": set(),
                                    "posts": set(), "posted_clips": set(),
                                    "views": 0, "likes": 0,
                                    "comments": 0, "d24": 0, "d72": 0,
                                    "n24": 0, "n72": 0})
        st["tags"].add(t["id"])
    theme_posts = {}

    for r in rows:
        tag = tags.get(r["tag_id"])
        if not tag:
            continue
        clip_key = (r["job_id"], int(r["idx"]))
        th = r["theme"] or "Без тематики"
        theme = themes[th]
        tag["clips"].add(clip_key)
        theme["clips"].add(clip_key)
        if not r["video_id"]:
            continue
        aid, vid = r["account_id"], r["video_id"]
        views = int(r["views"] or 0)
        likes = int(r["likes"] or 0)
        comments = int(r["comments"] or 0)
        base = bases.get((aid, vid), {})
        x24 = (views - base["24"]) if "24" in base else None
        x72 = (views - base["72"]) if "72" in base else None
        tag["posts"].add((aid, vid))
        tag["posted_clips"].add(clip_key)
        tag["views"] += views
        tag["likes"] += likes
        tag["comments"] += comments
        if x24 is not None:
            tag["d24"] += x24
            tag["n24"] += 1
        if x72 is not None:
            tag["d72"] += x72
            tag["n72"] += 1
        pkey = (th, clip_key, aid, vid)
        if pkey not in theme_posts:
            theme_posts[pkey] = (views, likes, comments, x24, x72)
            theme["posts"].add(pkey)
            theme["posted_clips"].add(clip_key)
            theme["views"] += views
            theme["likes"] += likes
            theme["comments"] += comments
            if x24 is not None:
                theme["d24"] += x24
                theme["n24"] += 1
            if x72 is not None:
                theme["d72"] += x72
                theme["n72"] += 1

    def public(st, key):
        posted = len(st["posted_clips"])
        mx = int(st.get("max_videos") or 0)
        out = {key: st.get("name") or st.get("theme"),
               "theme": st.get("theme") or "",
               "clips": len(st["clips"]), "posted": posted,
               "views": st["views"], "likes": st["likes"],
               "comments": st["comments"],
               "avg_views": int(st["views"] / posted) if posted else 0,
               "delta24": st["d24"] if st["n24"] else None,
               "delta72": st["d72"] if st["n72"] else None,
               "delta24_posts": st["n24"], "delta72_posts": st["n72"]}
        if key == "tag_id":
            out.update({"tag_id": st["tag_id"], "hashtags": st["hashtags"],
                        "max_videos": mx,
                        "remaining": (max(0, mx - posted) if mx else None),
                        "account_ids": st["account_ids"],
                        "accounts": st["accounts"]})
        else:
            out["tags"] = len(st["tags"])
        return out

    return jsonify({
        "ok": True,
        "themes": [public(x, "theme") for x in
                   sorted(themes.values(), key=lambda x: x["theme"].lower())],
        "tags": [public(x, "tag_id") for x in
                 sorted(tags.values(), key=lambda x: (x["theme"].lower(),
                                                      x["name"].lower()))]})


@app.get("/api/tiktok-check/<job_id>/<int:idx>")
def api_tiktok_check(job_id, idx):
    """Предзаливная проверка клипа: ок ли файл для TikTok.
    Файл с ok=False почти наверняка получит 0 просмотров."""
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", job_id or ""):
        return jsonify({"ok": False}), 404
    p = _clip_path(job_id, idx)
    if not p:
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
                    " FROM clips WHERE job_id=? AND hidden=0 ORDER BY idx",
                    (r["id"],)).fetchall()
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
    files = _job_clip_files(job_id)
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
            files = _job_clip_files(job_id)
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
    """job_id, idx из <job>_<i>.mp4 или <название>_<job>_<i>.mp4."""
    import re
    m = re.fullmatch(r"(?:.*_)?([A-Za-z0-9_-]{1,32})_(\d+)\.mp4", name or "")
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
    hidden = set()
    try:
        c = _connect()
        try:
            for r in c.execute(
                    "SELECT job_id, idx FROM clips WHERE hidden=1").fetchall():
                p = _clip_path(r["job_id"], r["idx"])
                hidden.add(os.path.basename(p) if p
                           else f"{r['job_id']}_{r['idx'] + 1}.mp4")
        finally:
            c.close()
    except Exception:
        pass
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
                              "hidden": (kind == "short" and f in hidden),
                              "url": f"/dl/{kind}/{quote(f)}",
                              "thumb": f"/api/thumb/{kind}/{quote(f)}"})
    items.sort(key=lambda x: x["mtime"], reverse=True)
    meta_map = _clip_meta_map()
    post_map = _post_state_map()
    for it in items:
        if it["kind"] == "short":
            m = _clip_meta_for(it["name"], meta_map)
            if m:
                it["job_id"] = m["job_id"]
                it["idx"] = m["idx"]
                it["title"] = m["title"]
                it["job_title"] = m["job_title"]
                it["preset_id"] = m["preset_id"]
        it["post_status"] = post_map.get((it["name"], it["kind"]), "")
    return jsonify({"ok": True, "files": items[:300]})


# ---------- автопостинг в TikTok Studio через живые профили Camoufox ----------
# очередь крутится на сервере: в назначенное время открывается настоящий
# браузер нужного профиля, файл и текст уже стоят в студии — остаётся
# нажать «Опубликовать». Пароли/капчу не трогаем, публикацию не жмём сами.
AUTOPOST_DIR = os.path.join(TMP_DIR, "autopost")
os.makedirs(AUTOPOST_DIR, exist_ok=True)
TIKTOKUA_DIR = os.path.expanduser("~/Library/Application Support/TikTokUA")
AUTOPOST_PROFILES = ("1", "2", "3")
AUTOPOST_BULK_MAX = 30  # студия принимает до 30 видео за раз
AUTOPOST_ACTIVE = ("starting", "running", "waiting", "login", "manual")
AUTOPOST_FINAL = ("done", "closed", "error", "canceled")
AUTOPOST_LOCK = threading.Lock()
AUTOPOST_THREAD = None
AUTOPOST_PROCS = {}


def _preset_id_value(v):
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _preset_variants(text):
    """Варианты описания разделяются строкой из дефисов (---) или --- в тексте."""
    parts = re.split(r"^\s*-{3,}\s*$|\s+-{3,}\s+", str(text or ""),
                     flags=re.M)
    return [p.strip() for p in parts if p.strip()]


def _tagset_variants(text):
    """Каждая непустая строка — отдельный набор тегов."""
    return [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]


def _preset_json(r):
    descs = _preset_variants(r["descs"])
    tagsets = _tagset_variants(r["tagsets"])
    return {"id": r["id"], "name": r["name"], "descs": r["descs"] or "",
            "tagsets": r["tagsets"] or "", "created": r["created"],
            "desc_count": len(descs), "tag_count": len(tagsets),
            "desc_variants": descs, "tag_variants": tagsets}


def _compose_caption(preset, k, title=""):
    """Описание и теги идут по кругу — у соседних видео текст разный."""
    if not preset:
        return ""
    descs = _preset_variants(preset["descs"])
    tagsets = _tagset_variants(preset["tagsets"])
    t = (title or "").strip()
    desc = descs[k % len(descs)] if descs else ""
    tag_line = ""
    if tagsets:
        tag_line = _norm_hashtags(
            tagsets[k % len(tagsets)].replace("{title}", t))
    if t:
        desc = desc.replace("{title}", t)
    return "\n\n".join(x for x in (desc.strip(), tag_line.strip()) if x)[:2200]


def _clip_meta_map():
    out = {}
    try:
        c = _connect()
        try:
            for r in c.execute(
                    "SELECT c.job_id, c.idx, c.title, c.hidden,"
                    " j.title AS job_title, j.preset_id"
                    " FROM clips c JOIN jobs j ON j.id=c.job_id"):
                out[(r["job_id"], int(r["idx"]))] = {
                    "job_id": r["job_id"], "idx": int(r["idx"]),
                    "title": r["title"] or "", "job_title": r["job_title"] or "",
                    "preset_id": r["preset_id"], "hidden": int(r["hidden"] or 0)}
        finally:
            c.close()
    except Exception:
        pass
    return out


def _clip_meta_for(name, meta_map):
    m = re.search(r"(?:^|_)([0-9a-zA-Z]{4,32})_(\d+)(?:v\d+)?\.mp4$",
                  str(name or ""))
    if m:
        hit = meta_map.get((m.group(1), int(m.group(2)) - 1))
        if hit:
            return hit
    for (jid, idx), meta in meta_map.items():
        if str(name or "").endswith(f"_{jid}_{idx + 1}.mp4"):
            return meta
    return None


def _post_state_map():
    out = {}
    try:
        c = _connect()
        try:
            for r in c.execute(
                    "SELECT file, kind, status FROM posts ORDER BY id"):
                out[(r["file"], r["kind"])] = r["status"]
        finally:
            c.close()
    except Exception:
        pass
    return out


def _post_target(name, kind):
    name = os.path.basename(str(name or ""))
    if not name or kind not in ("short", "legacy"):
        return None
    d = OUT_DIR if kind == "short" else LEGACY_DIR
    p = os.path.join(d, name)
    return p if os.path.isfile(p) else None


def _autopost_paths(pid):
    base = os.path.join(AUTOPOST_DIR, str(int(pid)))
    return base + ".json", base + ".status.json", base + ".log"


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _autopost_read_status(pid):
    try:
        with open(_autopost_paths(pid)[1], encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _autopost_pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _autopost_alive(post_id, pid):
    p = AUTOPOST_PROCS.get(post_id)
    if p is not None:
        return p.poll() is None
    return _autopost_pid_alive(pid)


def _autopost_stop(pid):
    """Погасить раннер вместе с окном Camoufox (он в своей сессии)."""
    try:
        import signal
        p = int(pid)
        try:
            os.killpg(os.getpgid(p), signal.SIGTERM)
        except Exception:
            os.kill(p, signal.SIGTERM)
        return p
    except Exception:
        return None


def _autopost_kill_later(pid):
    time.sleep(2.5)
    try:
        import signal
        if _autopost_pid_alive(pid):
            try:
                os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
            except Exception:
                os.kill(int(pid), signal.SIGKILL)
    except Exception:
        pass


def _autopost_json(r):
    from urllib.parse import quote
    path = _post_target(r["file"], r["kind"])
    return {"id": r["id"], "file": r["file"], "kind": r["kind"],
            "title": r["title"] or "", "preset_id": r["preset_id"],
            "caption": r["caption"] or "", "account": r["account"],
            "scheduled": r["scheduled"], "status": r["status"],
            "stage": r["stage"] or "", "error": r["error"] or "",
            "created": r["created"], "started": r["started"],
            "finished": r["finished"], "pid": r["pid"],
            "exists": bool(path),
            "url": f"/dl/{r['kind']}/{quote(r['file'])}",
            "thumb": f"/api/thumb/{r['kind']}/{quote(r['file'])}"}


def _parse_when(v, default):
    if v in (None, ""):
        return default
    try:
        n = float(v)
        if n > 1e11:  # миллисекунды из JS Date
            n /= 1000.0
        return n
    except (TypeError, ValueError):
        pass
    try:
        import datetime as _dt
        return _dt.datetime.fromisoformat(
            str(v).strip().replace("Z", "+00:00")).timestamp()
    except Exception:
        return default


def _autopost_spawn(c, rows):
    """Открыть окно профиля и загрузить пачку видео через bulk-режим.
    В одну пачку идут все queued-видео одного аккаунта (до 30)."""
    rows = list(rows)[:AUTOPOST_BULK_MAX]
    if not rows:
        return None, "пустая очередь"
    acc = str(rows[0]["account"] or "1")
    if acc not in AUTOPOST_PROFILES:
        return None, f"нет профиля {acc}"
    venv_py = os.path.join(TIKTOKUA_DIR, ".venv", "bin", "python")
    runner = os.path.join(BASE, "autopost_runner.py")
    if not os.path.isfile(venv_py):
        return None, "не найден Python TikTokUA (Camoufox)"
    if not os.path.isfile(runner):
        return None, "не найден autopost_runner.py"
    profile = os.path.join(TIKTOKUA_DIR, "profiles", f"account-{acc}")
    if not os.path.isdir(profile):
        return None, f"нет профиля account-{acc} — открой TikTok UA {acc} раз"

    items, broken = [], []
    for r in rows:
        src = _post_target(r["file"], r["kind"])
        if not src:
            broken.append((r["id"], "файл не найден в библиотеке"))
            continue
        _cfg, status_path, _log = _autopost_paths(r["id"])
        try:
            if os.path.exists(status_path):
                os.remove(status_path)
        except OSError:
            pass
        items.append({
            "post_id": r["id"], "file": src,
            "name": os.path.basename(src),
            "caption": r["caption"] or "",
            "publish_at": float(r["scheduled"] or 0),
            "status_file": status_path})
        if len(items) >= AUTOPOST_BULK_MAX:
            break
    for pid, err in broken:
        with c:
            c.execute("UPDATE posts SET status='error', stage='не запустилось',"
                      " error=?, finished=? WHERE id=?",
                      (err, time.time(), pid))
    if not items:
        return None, broken[0][1] if broken else "нет видео для загрузки"

    leader = items[0]["post_id"]
    cfg_path, _status_path, log_path = _autopost_paths(leader)
    try:
        if os.path.isfile(log_path) and os.path.getsize(log_path) > 3 * 1024 * 1024:
            os.remove(log_path)
    except OSError:
        pass
    _atomic_json(cfg_path, {
        "batch": len(items) > 1, "account": acc, "profile_dir": profile,
        "log_file": log_path, "items": items})
    try:
        with open(log_path, "ab", buffering=0) as logf:
            proc = subprocess.Popen(
                [venv_py, runner, "--config", cfg_path],
                stdout=logf, stderr=subprocess.STDOUT,
                cwd=BASE, start_new_session=True)
    except Exception as ex:
        with c:
            for it in items:
                c.execute("UPDATE posts SET status='error',"
                          " stage='не запустилось', error=?, finished=?"
                          " WHERE id=?",
                          (f"не удалось запустить браузер: {ex}",
                           time.time(), it["post_id"]))
        return None, f"не удалось запустить браузер: {ex}"
    now = time.time()
    with c:
        for it in items:
            c.execute("UPDATE posts SET status='starting',"
                      " stage='запускаю браузер…', error='', pid=?, started=?"
                      " WHERE id=?", (proc.pid, now, it["post_id"]))
    AUTOPOST_PROCS[leader] = proc
    for it in items[1:]:
        AUTOPOST_PROCS[it["post_id"]] = proc
    return proc.pid, ""


def _autopost_tick():
    now = time.time()
    c = _connect()
    try:
        rows = c.execute("SELECT * FROM posts WHERE status IN (?,?,?,?,?)",
                         AUTOPOST_ACTIVE).fetchall()
        for r in rows:
            st = _autopost_read_status(r["id"])
            new = str(st.get("status") or "")
            stage = str(st.get("stage") or r["stage"] or "")
            err = str(st.get("error") or "")
            if new == "queued":
                # раннер вернул видео в очередь (bulk не сложился)
                with c:
                    c.execute("UPDATE posts SET status='queued', stage='',"
                              " error='', pid=NULL, started=NULL WHERE id=?",
                              (r["id"],))
                AUTOPOST_PROCS.pop(r["id"], None)
                continue
            target = ""
            if new in ("published", "scheduled"):
                target = "done"
            elif new in ("closed", "canceled", "error"):
                target = new
            elif new in AUTOPOST_ACTIVE and (
                    new != r["status"] or (stage and stage != r["stage"])):
                target = new
            if target in AUTOPOST_FINAL:
                with c:
                    c.execute("UPDATE posts SET status=?, stage=?, error=?,"
                              " finished=?, pid=NULL WHERE id=?",
                              (target, stage, err or r["error"], now, r["id"]))
                AUTOPOST_PROCS.pop(r["id"], None)
                continue
            if target:
                with c:
                    c.execute("UPDATE posts SET status=?, stage=?, error=?"
                              " WHERE id=?", (target, stage, err or r["error"],
                                              r["id"]))
                continue
            if not r["pid"]:
                continue
            if not _autopost_alive(r["id"], r["pid"]):
                final = "closed" if new in ("waiting", "login", "manual") else "error"
                err2 = err or ("окно закрыто" if final == "closed"
                               else "раннер завершился без статуса")
                with c:
                    c.execute("UPDATE posts SET status=?, stage=?, error=?,"
                              " finished=?, pid=NULL WHERE id=?",
                              (final, stage or ("ждал Опубликовать" if final == "closed"
                                                else "ошибка"), err2, now, r["id"]))
                AUTOPOST_PROCS.pop(r["id"], None)
        busy = {str(r["account"]) for r in c.execute(
            "SELECT account FROM posts WHERE status IN (?,?,?,?,?)",
            AUTOPOST_ACTIVE).fetchall()}
        if len(busy) >= len(AUTOPOST_PROFILES):
            return
        due = c.execute("SELECT * FROM posts WHERE status='queued'"
                        " ORDER BY scheduled, id").fetchall()
        groups = {}
        for q in due:
            acc = str(q["account"] or "1")
            if acc not in AUTOPOST_PROFILES or acc in busy:
                continue
            groups.setdefault(acc, []).append(q)
        for acc, batch_rows in groups.items():
            if len(busy) >= len(AUTOPOST_PROFILES):
                break
            pid, err = _autopost_spawn(c, batch_rows)
            if not pid:
                with c:
                    for b in batch_rows:
                        c.execute("UPDATE posts SET status='error',"
                                  " stage='не запустилось', error=?, finished=?"
                                  " WHERE id=?",
                                  (err[:300], time.time(), b["id"]))
                continue
            busy.add(acc)
    finally:
        c.close()


def _autopost_loop():
    while True:
        try:
            _autopost_tick()
        except Exception:
            import traceback
            traceback.print_exc()
        time.sleep(5)


def start_autopost_scheduler():
    global AUTOPOST_THREAD
    if os.environ.get("CLIPSAW_NO_AUTOPOST"):
        return
    with AUTOPOST_LOCK:
        if AUTOPOST_THREAD and AUTOPOST_THREAD.is_alive():
            return
        AUTOPOST_THREAD = threading.Thread(target=_autopost_loop,
                                           name="autopost", daemon=True)
        AUTOPOST_THREAD.start()


@app.get("/api/presets")
def api_presets():
    items = []
    try:
        c = _connect()
        try:
            for r in c.execute("SELECT * FROM presets ORDER BY name"):
                items.append(_preset_json(r))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
    return jsonify({"ok": True, "items": items})


def _preset_body(d):
    name = (d.get("name") or "").strip()[:60]
    descs = str(d.get("descs") or "")[:20000]
    tagsets = str(d.get("tagsets") or "")[:20000]
    if not name:
        return None, "Введите название пресета"
    if not _preset_variants(descs) and not _tagset_variants(tagsets):
        return None, "Добавь хотя бы описание или теги"
    return (name, descs, tagsets), ""


@app.post("/api/presets")
def api_preset_create():
    d = request.get_json(force=True, silent=True) or {}
    body, err = _preset_body(d)
    if not body:
        return jsonify({"ok": False, "error": err}), 400
    try:
        c = _connect()
        try:
            with c:
                cur = c.execute("INSERT INTO presets(name,descs,tagsets,created)"
                                " VALUES(?,?,?,?)", (*body, time.time()))
                r = c.execute("SELECT * FROM presets WHERE id=?",
                              (cur.lastrowid,)).fetchone()
        finally:
            c.close()
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "Такой пресет уже есть"}), 409
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось сохранить пресет"}), 500
    return jsonify({"ok": True, "preset": _preset_json(r)})


@app.patch("/api/presets/<int:pid>")
def api_preset_update(pid):
    d = request.get_json(force=True, silent=True) or {}
    try:
        c = _connect()
        try:
            r = c.execute("SELECT * FROM presets WHERE id=?", (pid,)).fetchone()
            if not r:
                return jsonify({"ok": False, "error": "Пресет не найден"}), 404
            merged = {"name": d.get("name", r["name"]),
                      "descs": d.get("descs", r["descs"]),
                      "tagsets": d.get("tagsets", r["tagsets"])}
            body, err = _preset_body(merged)
            if not body:
                return jsonify({"ok": False, "error": err}), 400
            with c:
                c.execute("UPDATE presets SET name=?, descs=?, tagsets=? WHERE id=?",
                          (*body, pid))
                r = c.execute("SELECT * FROM presets WHERE id=?", (pid,)).fetchone()
        finally:
            c.close()
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "Такой пресет уже есть"}), 409
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось сохранить пресет"}), 500
    return jsonify({"ok": True, "preset": _preset_json(r)})


@app.delete("/api/presets/<int:pid>")
def api_preset_delete(pid):
    try:
        c = _connect()
        try:
            with c:
                c.execute("UPDATE posts SET preset_id=NULL WHERE preset_id=?", (pid,))
                c.execute("DELETE FROM presets WHERE id=?", (pid,))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось удалить пресет"}), 500
    return jsonify({"ok": True})


@app.get("/api/posts")
def api_posts():
    limit = 200
    try:
        limit = max(1, min(500, int(request.args.get("limit") or 200)))
    except (TypeError, ValueError):
        pass
    items = []
    try:
        c = _connect()
        try:
            rows = c.execute(
                "SELECT p.*, pr.name AS preset_name FROM posts p"
                " LEFT JOIN presets pr ON pr.id=p.preset_id"
                " ORDER BY p.id DESC LIMIT ?", (limit,)).fetchall()
        finally:
            c.close()
        for r in rows:
            it = _autopost_json(r)
            it["preset_name"] = r["preset_name"] or ""
            items.append(it)
    except Exception:
        import traceback
        traceback.print_exc()
    items.sort(key=lambda x: (
        0 if x["status"] in AUTOPOST_ACTIVE else
        1 if x["status"] == "queued" else 2,
        x["scheduled"] if x["status"] in AUTOPOST_ACTIVE + ("queued",) else -x["scheduled"]))
    profiles = []
    for n in AUTOPOST_PROFILES:
        profiles.append({"id": n, "label": f"Аккаунт {n}",
                         "exists": os.path.isdir(os.path.join(
                             TIKTOKUA_DIR, "profiles", f"account-{n}"))})
    return jsonify({"ok": True, "items": items, "profiles": profiles,
                    "server_time": time.time(),
                    "runner": os.path.isfile(os.path.join(
                        TIKTOKUA_DIR, ".venv", "bin", "python"))})


@app.post("/api/posts/bulk")
def api_posts_bulk():
    import random
    d = request.get_json(force=True, silent=True) or {}
    raw = d.get("files") or []
    if not isinstance(raw, list) or not raw:
        return jsonify({"ok": False, "error": "Выбери видео в библиотеке"}), 400
    mode = d.get("preset_id", "auto")
    if mode in (None, "", "auto"):
        preset_mode = "auto"
    elif str(mode) == "0":
        preset_mode = "none"
    else:
        preset_mode = _preset_id_value(mode)
        if preset_mode is None:
            return jsonify({"ok": False, "error": "Некорректный пресет"}), 400
    try:
        interval = max(5, min(1440, int(float(d.get("interval_min") or 90))))
    except (TypeError, ValueError):
        interval = 90
    try:
        jitter = max(0, min(60, int(float(
            d.get("jitter_min") if d.get("jitter_min") is not None else 3))))
    except (TypeError, ValueError):
        jitter = 3
    start = _parse_when(d.get("start"), time.time() + 90 * 60)
    accs = []
    for a in (d.get("accounts") or []):
        a = str(a).strip()
        if a in AUTOPOST_PROFILES and a not in accs:
            accs.append(a)
    if not accs:
        accs = ["1"]
    caption_body = str(d.get("caption") or "")[:2200]
    meta_map = _clip_meta_map()
    c = _connect()
    try:
        preset_row = None
        base_rotation = 0
        if preset_mode not in ("auto", "none"):
            preset_row = c.execute("SELECT * FROM presets WHERE id=?",
                                   (preset_mode,)).fetchone()
            if not preset_row:
                return jsonify({"ok": False, "error": "Пресет не найден"}), 404
            base_rotation = c.execute(
                "SELECT COUNT(*) FROM posts WHERE preset_id=?",
                (preset_mode,)).fetchone()[0]
        added, skipped, ids, prev_slot = 0, 0, [], 0.0
        with c:
            for x in raw[:200]:
                if isinstance(x, dict):
                    name = os.path.basename(str(x.get("name") or ""))
                    kind = str(x.get("kind") or "short")
                else:
                    name = os.path.basename(str(x or ""))
                    kind = "short"
                if kind not in ("short", "legacy"):
                    kind = "short"
                if not _post_target(name, kind):
                    skipped += 1
                    continue
                dup = c.execute(
                    "SELECT 1 FROM posts WHERE file=? AND kind=? AND status"
                    " NOT IN ('error','canceled','closed') LIMIT 1",
                    (name, kind)).fetchone()
                if dup:
                    skipped += 1
                    continue
                meta = _clip_meta_for(name, meta_map) or {}
                pid = None
                if preset_mode == "auto":
                    pid = _preset_id_value(meta.get("preset_id"))
                elif preset_mode != "none":
                    pid = preset_mode
                pr = preset_row if (pid and preset_row and pid == preset_mode) else (
                    c.execute("SELECT * FROM presets WHERE id=?", (pid,)).fetchone()
                    if pid else None)
                if pid and not pr:
                    pid = None
                title = (meta.get("title") or meta.get("job_title")
                         or os.path.splitext(name)[0])
                k = base_rotation + added
                caption = _compose_caption(pr, k, title)
                if not caption and caption_body:
                    caption = caption_body.replace("{title}", title)[:2200]
                account = accs[added % len(accs)]
                slot = start + added * interval * 60
                if jitter:
                    slot += random.uniform(-jitter * 60, jitter * 60)
                if prev_slot:
                    slot = max(slot, prev_slot + 60)
                # студия принимает расписание минимум за 15 минут
                slot = max(slot, time.time() + 16 * 60)
                cur = c.execute(
                    "INSERT INTO posts(file,kind,title,preset_id,caption,account,"
                    "scheduled,status,stage,created) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (name, kind, str(title)[:200], pid, caption, account,
                     slot, "queued", "в очереди", time.time()))
                ids.append(cur.lastrowid)
                prev_slot = slot
                added += 1
    finally:
        c.close()
    return jsonify({"ok": True, "added": added, "skipped": skipped, "ids": ids})


@app.post("/api/posts/<int:pid>/now")
def api_post_now(pid):
    try:
        c = _connect()
        try:
            r = c.execute("SELECT * FROM posts WHERE id=?", (pid,)).fetchone()
            if not r:
                return jsonify({"ok": False, "error": "Публикация не найдена"}), 404
            if r["status"] in AUTOPOST_ACTIVE:
                return jsonify({"ok": False, "error": "Уже запущено"}), 400
            with c:
                c.execute("UPDATE posts SET status='queued', stage='в очереди',"
                          " scheduled=?, error='', started=NULL, finished=NULL,"
                          " pid=NULL WHERE id=?", (time.time() + 20 * 60, pid))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось"}), 500
    return jsonify({"ok": True})


@app.post("/api/posts/<int:pid>/cancel")
def api_post_cancel(pid):
    proc_pid = None
    try:
        c = _connect()
        try:
            r = c.execute("SELECT * FROM posts WHERE id=?", (pid,)).fetchone()
            if not r:
                return jsonify({"ok": False, "error": "Публикация не найдена"}), 404
            if r["status"] not in AUTOPOST_ACTIVE:
                return jsonify({"ok": False,
                                "error": "Уже не выполняется — просто удали"}), 400
            proc_pid = r["pid"]
            sib = [pid]
            if proc_pid:
                sib = [x["id"] for x in c.execute(
                    "SELECT id FROM posts WHERE pid=? AND status IN (?,?,?,?,?)",
                    (proc_pid,) + AUTOPOST_ACTIVE).fetchall()]
                if pid not in sib:
                    sib.append(pid)
            with c:
                for sid in sib:
                    c.execute("UPDATE posts SET status='canceled',"
                              " stage='отменено', finished=?, pid=NULL"
                              " WHERE id=?", (time.time(), sid))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось отменить"}), 500
    for sid in sib:
        AUTOPOST_PROCS.pop(sid, None)
    if proc_pid:
        _autopost_stop(proc_pid)
        threading.Thread(target=_autopost_kill_later, args=(proc_pid,),
                         daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/posts/<int:pid>/done")
def api_post_done(pid):
    try:
        c = _connect()
        try:
            with c:
                c.execute("UPDATE posts SET status='done', stage='отмечено вручную',"
                          " finished=?, pid=NULL WHERE id=?", (time.time(), pid))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось"}), 500
    return jsonify({"ok": True})


@app.delete("/api/posts/<int:pid>")
def api_post_delete(pid):
    try:
        c = _connect()
        try:
            r = c.execute("SELECT status FROM posts WHERE id=?", (pid,)).fetchone()
            if not r:
                return jsonify({"ok": False, "error": "Публикация не найдена"}), 404
            if r["status"] in AUTOPOST_ACTIVE:
                return jsonify({"ok": False,
                                "error": "Сначала отмени запущенную"}), 400
            with c:
                c.execute("DELETE FROM posts WHERE id=?", (pid,))
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось удалить"}), 500
    return jsonify({"ok": True})


@app.post("/api/posts/clear")
def api_posts_clear():
    try:
        c = _connect()
        try:
            with c:
                cur = c.execute("DELETE FROM posts WHERE status IN"
                                " ('done','closed','error','canceled')")
            removed = cur.rowcount
        finally:
            c.close()
    except Exception:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "Не удалось очистить"}), 500
    return jsonify({"ok": True, "removed": removed})


@app.get("/api/posts/<int:pid>/log")
def api_post_log(pid):
    _cfg, _status, log_path = _autopost_paths(pid)
    if not os.path.isfile(log_path):
        # видео из пачки пишут один общий лог лидера
        try:
            c = _connect()
            try:
                r = c.execute("SELECT pid FROM posts WHERE id=?", (pid,)).fetchone()
                if r and r["pid"]:
                    sib = c.execute(
                        "SELECT id FROM posts WHERE pid=? AND id!=?",
                        (r["pid"], pid)).fetchall()
                    for x in sib:
                        p = _autopost_paths(x["id"])[2]
                        if os.path.isfile(p):
                            log_path = p
                            break
            finally:
                c.close()
        except Exception:
            pass
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            txt = f.read().decode("utf-8", "replace")
    except OSError:
        txt = ""
    return txt, 200, {"Content-Type": "text/plain; charset=utf-8"}


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
                   mirror: bool = False, caps: bool = True):
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
        opts["caps"] = caps
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
    caps = bool(parent.get("caps", True))
    UNIQ_JOBS[jid] = {"id": jid, "base": base, "kind": "finish",
                      "status": "work", "pct": 0, "stage": "старт…",
                      "started": time.time()}
    threading.Thread(target=_finish_worker,
                     args=(jid, path, base, cut, banner_path, mirror, caps),
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
    caps = bool(data.get("caps", True))
    made = []
    for url in urls:
        dlid = uuid.uuid4().hex[:8]
        UNIQ_JOBS[dlid] = {"id": dlid, "kind": "link", "url": url,
                           "status": "work", "pct": 0, "stage": "в очереди…",
                           "started": time.time(), "children": [],
                           "banner_path": banner_path, "mirror": mirror,
                           "caps": caps}
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


def _bnov_worker(jid: str, vpath: str, bpath: str, base: str,
                 cleanup_b: bool = True, speed: float | None = None,
                 force_key: bool | None = None):
    """Своё видео + баннер (свой файл или из готовых): ИИ уникализирует
    баннер, ускоряет (своё «х» или стандарт), вычитает хромакей,
    ставит 95% непрозрачности и запускает ровно с середины видео."""

    def prog(pct, stage="готовлю…"):
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(pct=int(pct), stage=stage)

    try:
        out = os.path.join(UNIQ_OUT, f"{jid}_{base}_banner.mp4")
        info = ac.overlay_banner_mid(vpath, out, banner_path=bpath,
                                     progress=prog, speed=speed,
                                     force_key=force_key)
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(status="done", pct=100, stage="готово", title=base,
                      url="/dluniq/" + os.path.basename(out),
                      size=os.path.getsize(out),
                      removed=[f"видео {info['dur']}с -> баннер ровно с {info['mid']}с",
                               f"баннер ускорен x{info['speed']} со звуком, без обрезок",
                               "баннер до краёв экрана, по центру",
                               "видео на паузе под баннер",
                               "без прозрачности",
                               "видео баннера уникализировано"])
    except Exception as e:
        st = UNIQ_JOBS.get(jid)
        if st is not None:
            st.update(status="error", error=str(e)[:300])
    finally:
        for p, do in ((vpath, True), (bpath, cleanup_b)):
            try:
                # баннер из assets — общий файл, не удаляем
                if do and p and os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


@app.post("/api/banner-ov")
def api_banner_ov():
    """Загрузка своего видео + баннер (файлом или из готовых) ->
    оверлей с середины. speed — своё ускорение «х», key=1 — силой
    вычесть зелёный хромакей."""
    import re
    fv = request.files.get("file")
    fb = request.files.get("banner")
    if fv is None or not (fv.filename or ""):
        return jsonify({"ok": False, "error": "Приложи видеофайл"}), 400
    form = request.form or {}
    vname = re.sub(r"[^\w\-. ]+", "", (fv.filename or "").strip())[:60] or "video.mp4"
    if "." not in vname or vname.rsplit(".", 1)[1].lower() not in (
            "mp4", "mov", "mkv", "webm", "m4v"):
        return jsonify({"ok": False, "error": "Нужно видео: mp4/mov/mkv/webm"}), 400
    jid = uuid.uuid4().hex[:8]
    vpath = os.path.join(UNIQ_IN, f"{jid}_v_{vname}")
    try:
        fv.save(vpath)
    except Exception:
        return jsonify({"ok": False, "error": "Не сохранилось"}), 400
    try:
        if os.path.getsize(vpath) > 800 * 1024 * 1024:
            raise OSError("Файл больше 800 МБ")
    except OSError as e:
        try:
            if os.path.exists(vpath):
                os.remove(vpath)
        except OSError:
            pass
        return jsonify({"ok": False, "error": str(e)}), 400
    # баннер: свой файл важнее, иначе — выбранный из готовых (assets)
    cleanup_b = False
    if fb is not None and (fb.filename or "").strip():
        bname = re.sub(r"[^\w\-. ]+", "", (fb.filename or "").strip())[:60] or "banner.mp4"
        if "." not in bname or bname.rsplit(".", 1)[1].lower() not in (
                "mp4", "mov", "mkv", "webm", "m4v"):
            try:
                os.remove(vpath)
            except OSError:
                pass
            return jsonify({"ok": False, "error": "Баннер: нужно видео mp4/mov/mkv/webm"}), 400
        bpath = os.path.join(UNIQ_IN, f"{jid}_b_{bname}")
        try:
            fb.save(bpath)
        except Exception:
            try:
                os.remove(vpath)
            except OSError:
                pass
            return jsonify({"ok": False, "error": "Баннер не сохранился"}), 400
        try:
            if os.path.getsize(bpath) > 800 * 1024 * 1024:
                raise OSError("Баннер больше 800 МБ")
        except OSError as e:
            for p in (vpath, bpath):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
            return jsonify({"ok": False, "error": str(e)}), 400
        cleanup_b = True
    else:
        bpath = resolve_banner(form.get("banner_file", ""))  # None = дефолтный
    try:
        speed = float(form.get("speed"))
    except (TypeError, ValueError):
        speed = None
    force_key = {"1": True, "0": False}.get((form.get("key") or "").strip())
    base = (vname.rsplit(".", 1)[0] or "video")[:40]
    UNIQ_JOBS[jid] = {"id": jid, "base": base, "kind": "banner-ov",
                      "status": "work", "pct": 0, "stage": "старт…",
                      "started": time.time()}
    threading.Thread(target=_bnov_worker,
                     args=(jid, vpath, bpath, base, cleanup_b, speed, force_key),
                     daemon=True).start()
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
    """Доступные баннеры из assets/*.{mp4,webm}: [{id, name, dur, size, green}]."""
    import glob
    items = []
    try:
        files = sorted(glob.glob(os.path.join(BASE, "assets", "*.mp4"))
                       + glob.glob(os.path.join(BASE, "assets", "*.webm")))
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
        if ac._banner_shard(p):
            name += " · хромакей · 90% · 1.2x"
        elif green:
            name += " · хромакей · 95%"
        elif ac._banner_trading(p):
            name += " · 70%"
        else:
            name += " · 95%"
        try:
            size = os.path.getsize(p)
        except OSError:
            size = 0
        items.append({"id": base, "name": name, "dur": dur, "size": size,
                      "green": green})
    return items


def resolve_banner(name):
    """Имя файла -> путь в assets или None (= RubUP по умолчанию)."""
    base = os.path.basename(str(name or "").strip())
    if not base or ".." in base or "/" in base:
        return None
    if not base.endswith((".mp4", ".webm")):
        return None
    p = os.path.join(BASE, "assets", base)
    return p if os.path.isfile(p) else None


@app.get("/api/banners")
def api_banners():
    return jsonify({"ok": True, "banners": list_banners()})


@app.get("/api/gameplays")
def api_gameplays():
    return jsonify({"ok": True, "gameplays": ac.gameplay_items()})


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
    start_autopost_scheduler()
    app.run(host="127.0.0.1", port=5003, debug=False)
