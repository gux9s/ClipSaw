"""Автопостинг Lipps: открывает живой профиль Camoufox, загружает пачку
видео через bulk-режим TikTok Studio, вписывает описания и время выхода,
а финальную кнопку «Post (N)» нажимает человек.

Ничего не публикует, не решает капчу и не вводит пароли. Профиль, отпечаток
и локация — те же, что у обычного запуска TikTok UA, поэтому для TikTok это
тот же самый браузер, а не новый «робот».
"""
import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time
import traceback

STUDIO_URL = "https://www.tiktok.com/tiktokstudio/upload"
LOGIN_URL = "https://www.tiktok.com/login?lang=uk-UA"

EDITOR_SELECTORS = (
    '[data-e2e="caption-input"] [contenteditable="true"]',
    'div[contenteditable="true"].public-DraftEditor-content',
    '[contenteditable="true"][role="textbox"]',
    'div[contenteditable="true"]',
)
UPLOAD_BUTTONS = (
    '[data-e2e="upload-icon"]',
    'button:has-text("Select video")',
    'button:has-text("Select videos")',
    'button:has-text("Upload video")',
    'button:has-text("Вибрати відео")',
    'button:has-text("Завантажити")',
    'button:has-text("Выбрать видео")',
    '[role="button"]:has-text("Select video")',
)
SUCCESS_WORDS = (
    "your video is being uploaded",
    "your video has been uploaded",
    "your videos have been uploaded",
    "your video was uploaded",
    "video published",
    "video has been published",
    "being uploaded to tiktok",
    "being uploaded",
    "posted",
    "опубліковано",
    "опубликовано",
    "відео завантажується",
    "видео загружается",
)
# финальное расписание: эти слова проверяем только когда панель Post (N)
# уже исчезла — свой тост после выбора времени в строке так не поймать
SCHEDULED_DONE_WORDS = (
    "have been scheduled",
    "has been scheduled",
    "video scheduled",
    "videos scheduled",
    "заплановано на",
    "запланировано на",
    "відкладено на",
    "отложено на",
)
# для расписания: эти слова ищем только в тостах/диалогах, не в теле
# страницы (иначе вкладка «Scheduled» даст ложное срабатывание)
SCHEDULED_WORDS = (
    "scheduled",
    "заплановано",
    "запланировано",
    "відкладено",
    "отложено",
)

# включаем режим расписания: только label / radio / switch, никаких
# button — среди кнопок есть финальная «Schedule», её жмёт человек
SCHEDULE_OPTION_SELECTORS = (
    'label:has-text("Schedule")',
    'label:has-text("Запланировать")',
    'label:has-text("Розклад")',
    '[role="radio"]:has-text("Schedule")',
    '[role="radio"]:has-text("Запланировать")',
    '[role="radio"]:has-text("Розклад")',
    '[role="switch"]:has-text("Schedule")',
    '[role="switch"]:has-text("Запланировать")',
)

# bulk-страница студии: один файловый input, таблица видео и одна кнопка
# «Post (N)» — её и жмёт человек
BULK_BAR = ".bulk-publish-post-bar"
BULK_TABLE = "table.video-task-table"
BULK_ROW = "tr.task-draggable-row"
CAPTION_CELL = ".caption-editor-text"
CAPTION_TEXT = ".caption-text-content"
CAPTION_EDITOR = '.editor-container [contenteditable="true"], .editor-container textarea'
SCHED_BUTTON = ".scheduler-input-container .input-button"
SCHED_VALUE = ".scheduler-input-container .input-value-text"
CALENDAR = ".calendar-wrapper"
TIMER = ".tiktok-timer-time-picker-container"

BULK_UI_TIMEOUT = 240        # ждём появления bulk-страницы после выбора файлов
BULK_FILL_TIMEOUT = 3 * 3600  # загрузка + обработка всех видео
WATCH_TIMEOUT = 6 * 3600     # ждём, пока человек нажмёт Post (N)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def write_status(path, status, stage="", error="", extra=None):
    data = {"status": status, "stage": stage, "error": error or "",
            "ts": time.time()}
    if extra:
        data.update(extra)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass
    log(f"{status}: {stage}" + (f" ({error})" if error else ""))


def write_all(items, status, stage, error=""):
    for it in items:
        write_status(it["status_file"], status, stage, error)


def load_fingerprint(profile_dir):
    from camoufox.fingerprints import generate_fingerprint
    path = os.path.join(profile_dir, "fingerprint.json")
    if not os.path.exists(path):
        os.makedirs(profile_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(generate_fingerprint(os="macos"), f, indent=2)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_login(page):
    try:
        if "/login" in page.url:
            return True
    except Exception:
        return False
    try:
        return page.locator(
            '[data-e2e="login-button"], button[data-e2e="login-button"]'
        ).count() > 0
    except Exception:
        return False


def has_file_input(page):
    try:
        return page.locator('input[type="file"]').count() > 0
    except Exception:
        return False


def find_editor(page):
    for sel in EDITOR_SELECTORS:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible():
                return loc
        except Exception:
            continue
    return None


def success_visible(page, body_fallback=True, words=None):
    """Ищем тост/диалог успеха только после ухода из редактора.
    body_fallback=False — доверяем только тостам (для промежуточных
    проверок, где в теле страницы есть служебные слова)."""
    words = words or (SUCCESS_WORDS + SCHEDULED_WORDS)
    try:
        nodes = page.locator(
            '[role="dialog"], [data-e2e*="success"], [data-e2e="toast"],'
            ' [class*="toast"], [role="alert"]')
        for i in range(min(nodes.count(), 4)):
            try:
                txt = (nodes.nth(i).inner_text(timeout=1500) or "").lower()
            except Exception:
                continue
            if any(w in txt for w in words):
                return True
    except Exception:
        pass
    if not body_fallback:
        return False
    try:
        body = page.locator("body").inner_text(timeout=2500).lower()
        return any(w in body for w in SUCCESS_WORDS)
    except Exception:
        return False


def type_text(page, text, fast=False):
    for ch in text:
        if ch == "\n":
            page.keyboard.press("Enter")
            page.wait_for_timeout(random.randint(40, 110))
        else:
            page.keyboard.type(ch)
            page.wait_for_timeout(random.randint(6, 22) if fast
                                  else random.randint(12, 58))


def clear_editor(page):
    page.keyboard.press("Meta+A")
    page.wait_for_timeout(120)
    page.keyboard.press("Backspace")
    page.wait_for_timeout(random.randint(150, 350))


def type_caption(page, editor, caption):
    editor.click()
    page.wait_for_timeout(random.randint(450, 1000))
    clear_editor(page)
    type_text(page, caption)
    # пробел закрывает выпадашку хэштегов, чтобы она не мешала кнопке
    page.keyboard.type(" ")
    page.wait_for_timeout(random.randint(300, 800))


def plan_time(publish_at):
    """Время публикации из очереди. Студия принимает минимум +15 минут,
    шаг выбора — 5 минут; протухшее время сдвигаем на +20 минут."""
    try:
        ts = float(publish_at or 0)
    except (TypeError, ValueError):
        return 0
    if ts <= 0:
        return 0
    ts = int(ts // 300) * 300
    if ts < time.time() + 16 * 60:
        ts = int((time.time() + 20 * 60) // 300 + 1) * 300
    return ts


def fmt_target(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%d.%m %H:%M")


def _find_schedule_option(page):
    for sel in SCHEDULE_OPTION_SELECTORS:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible():
                return loc
        except Exception:
            continue
    return None


# ---------- календарь и колёсико времени студии ----------

def _pick_calendar_date(page, d):
    try:
        cal = page.locator(CALENDAR).last
        if not cal.count():
            return False
    except Exception:
        return False
    today = dt.date.today()
    delta = (d.year - today.year) * 12 + (d.month - today.month)
    try:
        arrows = cal.locator(".month-header-wrapper .arrow")
        for _ in range(min(abs(delta), 14)):
            arrows.nth(1 if delta > 0 else 0).click(timeout=3000)
            page.wait_for_timeout(random.randint(250, 500))
    except Exception:
        return False
    try:
        days = cal.locator(".day.valid")
        target = days.filter(has_text=re.compile(rf"^0?{d.day}$"))
        if not target.count():
            return False
        target.first.click(timeout=4000)
        page.wait_for_timeout(random.randint(300, 650))
        return True
    except Exception:
        return False


def _option_index(loc, value):
    try:
        n = loc.count()
    except Exception:
        return -1
    best = (-1, 10 ** 9)
    for i in range(n):
        try:
            txt = loc.nth(i).inner_text(timeout=1200) or ""
        except Exception:
            continue
        digits = re.sub(r"\D", "", txt)
        if not digits:
            continue
        v = int(digits)
        if v == value:
            return i
        dist = abs(v - value)
        if dist < best[1]:
            best = (i, dist)
    return best[0] if best[1] <= 3 else -1


def _pick_timer(page, d):
    try:
        box = page.locator(TIMER).last
        if not box.count():
            return False
        hours = box.locator(".tiktok-timer-option-text.tiktok-timer-left")
        mins = box.locator(".tiktok-timer-option-text.tiktok-timer-right")
        hi = _option_index(hours, d.hour)
        mi = _option_index(mins, d.minute)
        if hi < 0:
            return False
        hours.nth(hi).click(timeout=3000)
        page.wait_for_timeout(random.randint(180, 400))
        if mi >= 0:
            mins.nth(mi).click(timeout=3000)
            page.wait_for_timeout(random.randint(200, 450))
        return mi >= 0
    except Exception:
        return False


def _schedule_reflected(page, d):
    """Проверяем, что в полях студии реально стоит наша дата и время."""
    try:
        vals = page.evaluate(
            """() => [...document.querySelectorAll('input')].map(i => ({
                 type: i.type || '', value: i.value || '',
                 hint: (i.placeholder || '') + ' ' +
                       (i.getAttribute('aria-label') || '')}))""")
    except Exception:
        return False
    dates = {d.strftime("%Y-%m-%d"), d.strftime("%d.%m.%Y"),
             d.strftime("%m/%d/%Y"), d.strftime("%d/%m/%Y"),
             d.strftime("%d-%m-%Y"), d.strftime("%Y/%m/%d"),
             d.strftime("%d.%m.%y"), d.strftime("%m/%d/%y"),
             d.strftime("%d-%m-%y")}
    times = {d.strftime("%H:%M"), d.strftime("%I:%M %p").lstrip("0"),
             d.strftime("%I:%M%p").lstrip("0"), d.strftime("%H.%M")}
    date_ok = time_ok = False
    for v in vals:
        val = (v.get("value") or "").strip().lower()
        if not val:
            continue
        if any(x.lower() in val for x in dates):
            date_ok = True
        if any(x.lower() in val for x in times):
            time_ok = True
    return date_ok and time_ok


def _native_set(page, selector, value):
    return page.evaluate(
        """([sel, val]) => {
          const el = document.querySelector(sel);
          if (!el) return null;
          const proto = el.tagName === 'TEXTAREA'
            ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
          try {
            Object.getOwnPropertyDescriptor(proto, 'value')
              .set.call(el, val);
          } catch (e) { el.value = val; }
          el.dispatchEvent(new Event('input', {bubbles: true}));
          el.dispatchEvent(new Event('change', {bubbles: true}));
          return el.value;
        }""", [selector, value])


def _fill_input(page, sel, value):
    loc = page.locator(sel).first
    try:
        if not loc.count():
            return False
    except Exception:
        return False
    try:
        loc.fill(value, timeout=5000)
        return True
    except Exception:
        pass
    try:
        return bool(_native_set(page, sel, value))
    except Exception:
        return False


def apply_schedule(page, target_ts):
    """Ставит дату/время в редакторе студии. В новом интерфейсе это
    календарь + колёсико времени; в старом — нативные input.
    Финальную кнопку не жмёт. Не вышло — False (впишет человек)."""
    d = dt.datetime.fromtimestamp(target_ts)
    iso = d.strftime("%Y-%m-%dT%H:%M")
    date_iso = d.strftime("%Y-%m-%d")
    time_iso = d.strftime("%H:%M")
    try:
        has_dt = page.locator('input[type="datetime-local"]').count() > 0
        has_d = page.locator('input[type="date"]').count() > 0
        has_t = page.locator('input[type="time"]').count() > 0
    except Exception:
        has_dt = has_d = has_t = False
    if has_dt or (has_d and has_t):
        filled = False
        if has_dt:
            filled = _fill_input(page, 'input[type="datetime-local"]', iso)
        else:
            fd = _fill_input(page, 'input[type="date"]', date_iso)
            ft = _fill_input(page, 'input[type="time"]', time_iso)
            filled = fd and ft
        if filled:
            try:
                page.evaluate("() => { if (document.activeElement) "
                              "document.activeElement.blur(); }")
            except Exception:
                pass
            page.wait_for_timeout(random.randint(600, 1200))
            if _schedule_reflected(page, d):
                return True

    # новый студийный пикер: включаем «Schedule» и выбираем дату/время
    opt = _find_schedule_option(page)
    if opt is None:
        try:
            # возможно, режим уже включён
            cont = page.locator('[data-e2e="SCHEDULE_CONTAINER"]')
            if not cont.count():
                return False
        except Exception:
            return False
    else:
        try:
            opt.click(timeout=4000)
        except Exception:
            return False
        page.wait_for_timeout(random.randint(700, 1300))
        try:
            _accept_schedule_ack(page)
        except Exception:
            pass
        page.wait_for_timeout(random.randint(400, 900))
    try:
        cont = page.locator('[data-e2e="SCHEDULE_CONTAINER"]')
        inputs = cont.locator("input[readonly]")
        if inputs.count() < 2:
            inputs = cont.locator("input")
        if inputs.count() < 2:
            return False
        inputs.nth(0).click(timeout=4000)
        page.wait_for_timeout(random.randint(500, 900))
        if not _pick_calendar_date(page, d):
            return False
        page.wait_for_timeout(random.randint(300, 700))
        try:
            inputs.nth(1).click(timeout=4000)
        except Exception:
            pass
        page.wait_for_timeout(random.randint(500, 900))
        _pick_timer(page, d)
        try:
            page.evaluate("() => { if (document.activeElement) "
                          "document.activeElement.blur(); }")
        except Exception:
            pass
        page.wait_for_timeout(random.randint(500, 1000))
        return _schedule_reflected(page, d)
    except Exception:
        return False


# ---------- bulk-страница ----------

def set_files(page, paths):
    """Выбираем файлы в скрытом input. Возвращает (ok, multiple)."""
    try:
        inp = page.locator('input[type="file"]').first
        multiple = bool(inp.evaluate("el => !!el.multiple"))
    except Exception:
        inp, multiple = None, False
    if inp is not None:
        try:
            inp.set_input_files(paths if multiple else paths[0],
                                timeout=30000)
            return True, multiple
        except Exception as ex:
            log(f"input не принялся: {ex}")
    for sel in UPLOAD_BUTTONS:
        try:
            btn = page.locator(sel).first
            if not btn.count() or not btn.is_visible():
                continue
            with page.expect_file_chooser(timeout=4000) as fc:
                btn.click()
            fc.value.set_files(paths if multiple else paths[0])
            return True, multiple
        except Exception:
            continue
    return False, multiple


def bulk_rows(page):
    return page.locator(f"{BULK_TABLE} {BULK_ROW}")


def wait_bulk_page(page, count, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if page.locator(BULK_BAR).count():
                n = bulk_rows(page).count()
                if n >= min(count, 1):
                    return n
        except Exception:
            pass
        page.wait_for_timeout(1500)
    return 0


def _row_caption_text(row):
    try:
        t = row.locator(CAPTION_TEXT).first
        if t.count():
            return (t.inner_text(timeout=1500) or "").strip()
    except Exception:
        pass
    try:
        ed = row.locator(CAPTION_EDITOR).first
        if ed.count() and ed.is_visible():
            return (ed.inner_text(timeout=1500) or "").strip()
    except Exception:
        pass
    return ""


def _norm(txt):
    return re.sub(r"\s+", " ", (txt or "").strip())


def fill_row_caption(page, row, text):
    try:
        cell = row.locator(CAPTION_CELL).first
        if not cell.count():
            return False
        cell.click(timeout=4000)
        page.wait_for_timeout(random.randint(450, 950))
    except Exception:
        return False
    editor = None
    for sel in (CAPTION_EDITOR.replace(", ", ",").split(",") +
                ['div[contenteditable="true"]', "textarea"]):
        try:
            loc = row.locator(sel.strip()).first
            if loc.count() and loc.is_visible():
                editor = loc
                break
        except Exception:
            continue
    if editor is None:
        return False
    try:
        editor.click()
        page.wait_for_timeout(random.randint(250, 550))
        clear_editor(page)
        if text.strip():
            type_text(page, text, fast=True)
        page.wait_for_timeout(random.randint(250, 550))
        # onBlur в студии сохраняет подпись
        page.evaluate("() => { if (document.activeElement) "
                      "document.activeElement.blur(); }")
        page.wait_for_timeout(random.randint(500, 900))
        try:
            btn = row.locator(".editor-container button").last
            if btn.count():
                btn.click(timeout=2500)
                page.wait_for_timeout(random.randint(300, 600))
        except Exception:
            pass
    except Exception:
        return False
    if not text.strip():
        return True
    got = _norm(_row_caption_text(row))
    want = _norm(text)
    probe = want[:24]
    return bool(got) and (got == want or probe in got)


def _accept_schedule_ack(page):
    """Студия один раз показывает подтверждение расписания — принимаем."""
    dialogs = page.locator(
        '[role="dialog"]:visible, .tux-modal:visible, [class*="modal"]:visible')
    try:
        n = dialogs.count()
    except Exception:
        return False
    for i in range(n):
        dlg = dialogs.nth(i)
        try:
            txt = (dlg.inner_text(timeout=800) or "").lower()
        except Exception:
            continue
        if not any(w in txt for w in ("schedul", "розклад", "расписан",
                                      "plan", "план", "acknowledge",
                                      "got it", "зрозуміло")):
            continue
        try:
            btns = dlg.locator("button")
            if btns.count():
                btns.last.click(timeout=3000)
                return True
        except Exception:
            continue
    return False


def _ensure_picker_open(page):
    """Открывает календарь, принимая подтверждение расписания, если оно есть."""
    for _ in range(10):
        try:
            if page.locator(CALENDAR).count():
                return True
        except Exception:
            pass
        clicked = False
        try:
            clicked = _accept_schedule_ack(page)
        except Exception:
            pass
        page.wait_for_timeout(900 if clicked else 500)
    return False


def _row_schedule_ok(row, d):
    try:
        txt = (row.locator(SCHED_VALUE).first.inner_text(timeout=2000) or "")
    except Exception:
        return False
    low = txt.strip().lower()
    if not low:
        return False
    bad = ("post now", "зараз", "сейчас", "опублікувати зараз",
           "опубликовать сейчас")
    if any(b in low for b in bad):
        return False
    day_ok = re.search(rf"\b0?{d.day}\b", txt) is not None
    time_ok = any(t in txt for t in (d.strftime("%H:%M"), d.strftime("%H.%M")))
    if not time_ok:
        t12 = d.strftime("%I:%M %p").lstrip("0").lower()
        time_ok = t12 in low or t12.replace(" ", "") in low.replace(" ", "")
    return day_ok and (time_ok or f":{d.minute:02d}" in txt)


def set_row_schedule(page, row, target_ts):
    d = dt.datetime.fromtimestamp(target_ts)
    try:
        btn = row.locator(SCHED_BUTTON).first
        if not btn.count():
            return False
        btn.click(timeout=5000)
        page.wait_for_timeout(random.randint(600, 1100))
    except Exception:
        return False
    if not _ensure_picker_open(page):
        return False
    if not _pick_calendar_date(page, d):
        return False
    page.wait_for_timeout(random.randint(250, 600))
    if not _pick_timer(page, d):
        return False
    page.wait_for_timeout(random.randint(250, 600))
    try:
        foot = page.locator(
            '.picker-footer:visible, .post-scheduler-footer:visible').last
        btns = foot.locator(".operation-confirm button")
        if not btns.count():
            return False
        btns.last.click(timeout=4000)
        page.wait_for_timeout(random.randint(700, 1200))
    except Exception:
        return False
    return _row_schedule_ok(row, d)


def _assign_rows(page, items):
    """Строки идут в порядке выбранных файлов; страхуемся по имени файла.
    Позиционный запасной вариант — только если строк ровно столько же,
    сколько видео: иначе легко перепутать чужие строки."""
    rows = bulk_rows(page)
    n = rows.count()
    used = set()
    keys = []
    for i in range(n):
        keys.append(re.sub(r"[^0-9a-zа-яіїєґ]+", "",
                           _row_caption_text(rows.nth(i)).lower()))
    mapping = []
    for it in items:
        stem = os.path.splitext(it["name"])[0].lower()
        key = re.sub(r"[^0-9a-zа-яіїєґ]+", "", stem)[:16]
        hit = None
        if key:
            for i in range(n):
                if i in used:
                    continue
                if key in keys[i]:
                    hit = i
                    break
        if hit is None and n == len(items):
            for i in range(n):
                if i not in used:
                    hit = i
                    break
        if hit is None:
            mapping.append((it, None))
            continue
        used.add(hit)
        mapping.append((it, rows.nth(hit)))
    return mapping


def bulk_fill(page, items):
    """Заполняем описание и время в каждой строке bulk-таблицы.
    Возвращает список видео, где не удалось выставить время."""
    failed = []
    t0 = time.time()
    for idx, it in enumerate(items):
        left = BULK_FILL_TIMEOUT - (time.time() - t0)
        if left <= 0:
            it["failed"] = True
            failed.append(it)
            continue
        row = it.get("row")
        if row is None:
            it["failed"] = True
            failed.append(it)
            continue
        # ждём, пока студия доделает обработку именно этой строки
        ready = False
        t1 = time.time()
        while time.time() - t1 < min(left, 1800):
            try:
                if row.locator(CAPTION_CELL).count():
                    ready = True
                    break
            except Exception:
                pass
            page.wait_for_timeout(2000)
        if not ready:
            it["failed"] = True
            write_status(it["status_file"], "manual",
                         f"строка {idx + 1}: видео не догрузилось — проверь")
            failed.append(it)
            continue
        stage = f"пачка {idx + 1}/{len(items)}: вписываю описание…"
        write_status(it["status_file"], "running", stage)
        cap_ok = True
        if it["caption"].strip():
            cap_ok = fill_row_caption(page, row, it["caption"])
        target = plan_time(it.get("publish_at"))
        it["target"] = target
        sched_ok = True
        if target:
            write_status(it["status_file"], "running",
                         f"пачка {idx + 1}/{len(items)}: ставлю время "
                         f"{fmt_target(target)}…")
            sched_ok = set_row_schedule(page, row, target)
        if cap_ok and sched_ok:
            it["filled"] = True
            it["wait_msg"] = (
                f"пачка {idx + 1}/{len(items)} готова: {fmt_target(target)}"
                if target else f"пачка {idx + 1}/{len(items)} готова")
            write_status(it["status_file"], "running", it["wait_msg"])
        else:
            it["failed"] = True
            if not sched_ok and target:
                write_status(it["status_file"], "manual",
                             f"строка {idx + 1}: не выставил время "
                             f"{fmt_target(target)} — поставь вручную "
                             "или убери видео")
            else:
                write_status(it["status_file"], "manual",
                             f"строка {idx + 1}: не вписал описание — "
                             "проверь и нажми Post")
            failed.append(it)
    return failed


def wait_post_button(page, timeout):
    """Ждём, пока загрузка всех файлов закончится и Post (N) станет активной."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            btn = page.locator(".bulk-publish-button button").last
            if not btn.count():
                btn = page.locator('[data-e2e="POST_VIDEO_BUTTON"]').last
            if btn.count() and btn.is_enabled():
                return True
        except Exception:
            pass
        page.wait_for_timeout(2500)
    return False


def finish_item(status_file, target_ts, guessed=False):
    label = fmt_target(target_ts) if target_ts else ""
    if target_ts:
        stage = (f"запланировано на {label}" if not guessed else
                 f"похоже, запланировано на {label} — проверь")
        write_status(status_file, "scheduled", stage)
    else:
        stage = ("опубликовано — закрываю окно" if not guessed else
                 "похоже, опубликовано — проверь")
        write_status(status_file, "published", stage)


def finish_success(page, status_file, target_ts, guessed=False):
    """Фиксируем финал. Публикацию/расписание жмёт человек — мы только
    записываем, что окно ушло из редактора."""
    finish_item(status_file, target_ts, guessed)
    try:
        page.wait_for_timeout(8000)
    except Exception:
        pass


def wait_batch(page, items, status="waiting", message=None):
    """Держим окно открытым, пока человек не нажмёт Post (N).
    Финалом считаем только уход со страницы загрузки или строгий тост
    студии: пропажа панели из-за «Edit detail» успехом не считается."""
    started = time.time()
    left_run = 0
    last_beat = 0
    while time.time() - started < WATCH_TIMEOUT:
        if page.is_closed():
            for it in items:
                write_status(it["status_file"], "closed", "окно закрыто")
            return
        try:
            saw = success_visible(page, body_fallback=False,
                                  words=SUCCESS_WORDS)
        except Exception:
            saw = False
        if not saw:
            try:
                bar_there = page.locator(BULK_BAR).count() > 0
            except Exception:
                bar_there = True
            if not bar_there:
                try:
                    saw = success_visible(page, body_fallback=False,
                                          words=SCHEDULED_DONE_WORDS)
                except Exception:
                    saw = False
        if not saw:
            try:
                url = page.url
            except Exception:
                url = STUDIO_URL
            if "/upload" not in url and "/tiktokstudio" in url:
                left_run += 1
            else:
                left_run = 0
        if saw or left_run >= 3:
            for it in items:
                if it.get("failed"):
                    continue
                finish_item(it["status_file"], it.get("target") or 0,
                            guessed=not saw)
            try:
                page.wait_for_timeout(4000)
            except Exception:
                pass
            return
        if time.time() - last_beat > 12:
            last_beat = time.time()
            for it in items:
                if it.get("failed"):
                    continue
                write_status(it["status_file"], status,
                             message or it.get("wait_msg")
                             or "нажми Post в окне")
        page.wait_for_timeout(3000)
    for it in items:
        if not it.get("failed"):
            write_status(it["status_file"], "closed",
                         "окно осталось открытым")


def bulk_run(page, items):
    """Одна пачка: выбираем все файлы, заполняем каждую строку, отдаём
    финальную кнопку человеку. Если bulk не сложился — False."""
    paths = [it["file"] for it in items]
    write_all(items, "running", f"загружаю {len(paths)} видео в студию…")
    ok, multiple = set_files(page, paths)
    if not ok or not multiple:
        if ok and not multiple:
            write_all(items, "running", "bulk недоступен — гружу по одному")
        return False
    rows = wait_bulk_page(page, len(paths), BULK_UI_TIMEOUT)
    if not rows:
        return False
    # ждём появления всех строк (каждый файл стартует свою загрузку)
    t0 = time.time()
    while time.time() - t0 < 600:
        try:
            if bulk_rows(page).count() >= len(items):
                break
        except Exception:
            pass
        page.wait_for_timeout(1500)
    mapping = _assign_rows(page, items)
    missing = [it for it, row in mapping if row is None]
    for it in missing:
        it["failed"] = True
        write_status(it["status_file"], "manual",
                     f"видео «{(it.get('name') or '')[:40]}» не появилось "
                     "в пачке — проверь окно")
    items = [(it, row) for it, row in mapping if row is not None]
    if not items:
        return False
    for it, row in items:
        it["row"] = row
        write_status(it["status_file"], "running",
                     f"в пачке из {len(items)} — жду обработку…")
    failed = bulk_fill(page, [it for it, _ in items])
    if failed:
        log(f"не сложились строки: {[it['post_id'] for it in failed]}")
    ok_items = [it for it, _ in items if not it.get("failed")]
    if not ok_items:
        msg = "не смог заполнить пачку — заполни строки вручную и жми Post"
        for it, _ in items:
            it["failed"] = False
            it.setdefault("target", plan_time(it.get("publish_at")))
            it["wait_msg"] = msg
        write_all([it for it, _ in items], "manual", msg)
        wait_batch(page, [it for it, _ in items], status="manual")
        return True
    ready = wait_post_button(page, BULK_FILL_TIMEOUT)
    total = len(ok_items)
    for it in ok_items:
        when = fmt_target(it["target"]) if it.get("target") else ""
        if not ready:
            it["wait_msg"] = "загрузка ещё идёт — дождись и жми Post"
        elif when:
            it["wait_msg"] = f"всё заполнено — выход {when}, жми Post ({total})"
        else:
            it["wait_msg"] = f"всё заполнено — жми Post ({total}) в окне"
        write_status(it["status_file"], "waiting", it["wait_msg"])
    wait_batch(page, [it for it, _ in items])
    return True


def run_single(page, item):
    """Старый одиночный путь: одно видео, редактор, расписание."""
    status_file = item["status_file"]
    caption = item.get("caption") or ""
    target_ts = plan_time(item.get("publish_at"))
    write_status(status_file, "running", "загружаю видео…")
    ok, _multiple = set_files(page, [item["file"]])
    if not ok:
        write_status(status_file, "manual",
                     "не нашёл поле загрузки — загрузи видео вручную")
        wait_until_closed(page, status_file, "manual",
                          "окно открыто: загрузи видео вручную")
        return 0
    editor = None
    t0 = time.time()
    while time.time() - t0 < 120:
        editor = find_editor(page)
        if editor is not None:
            break
        page.wait_for_timeout(1500)
    if editor is None:
        if is_login(page):
            write_status(status_file, "login",
                         "сессия слетела — войди в аккаунт")
            wait_until_closed(page, status_file, "login",
                              "войди в аккаунт, я жду")
            return 0
        write_status(status_file, "manual",
                     "редактор не появился — допиши описание вручную")
        wait_until_closed(page, status_file, "manual",
                          "окно открыто: допиши описание вручную")
        return 0

    page.wait_for_timeout(random.randint(800, 1800))
    if caption.strip():
        write_status(status_file, "running", "вписываю описание и теги…")
        try:
            type_caption(page, editor, caption)
        except Exception as ex:
            write_status(status_file, "manual",
                         "не смог вписать текст — впиши вручную",
                         str(ex)[:300])
            wait_until_closed(page, status_file, "manual",
                              "окно открыто: впиши текст вручную")
            return 0
    if target_ts:
        label = fmt_target(target_ts)
        write_status(status_file, "running",
                     f"выставляю публикацию на {label}…")
        schedule_ok = apply_schedule(page, target_ts)
        if schedule_ok:
            write_status(status_file, "waiting",
                         f"время {label} выставлено — нажми "
                         "«Запланировать» в окне")
        else:
            write_status(status_file, "manual",
                         f"выставь в студии время {label} и нажми "
                         "«Запланировать» — сам не смог")
            wait_until_closed(page, status_file, "manual",
                              f"окно открыто: поставь {label} и нажми "
                              "«Запланировать»", target_ts)
            return 0
    else:
        write_status(status_file, "waiting",
                     "всё заполнено — нажми «Опубликовать» в окне")

    # ждём публикацию: уход из редактора + тост успеха
    upload_url = page.url
    gone = 0
    saw_success = False
    t0 = time.time()
    while time.time() - t0 < 6 * 3600:
        if page.is_closed():
            write_status(status_file, "closed", "окно закрыто")
            return 0
        try:
            editor_alive = editor.count() > 0 and editor.is_visible()
        except Exception:
            editor_alive = False
        try:
            url_now = page.url
        except Exception:
            url_now = upload_url
        if not editor_alive:
            gone += 1
        else:
            gone = 0
        left_upload = (url_now != upload_url
                       and "/upload" not in url_now
                       and "/tiktokstudio" in url_now)
        if gone >= 2 or left_upload:
            for _ in range(10):
                if page.is_closed():
                    break
                try:
                    if success_visible(
                            page,
                            words=SUCCESS_WORDS + SCHEDULED_DONE_WORDS):
                        saw_success = True
                        break
                except Exception:
                    pass
                page.wait_for_timeout(2000)
            if saw_success:
                finish_success(page, status_file, target_ts)
            elif left_upload and target_ts:
                finish_success(page, status_file, target_ts, guessed=True)
            else:
                write_status(
                    status_file, "closed",
                    "редактор закрылся без подтверждения — проверь, "
                    "вышло ли видео, и отметь вручную")
            return 0
        page.wait_for_timeout(2000)
    write_status(status_file, "closed", "окно осталось открытым")
    return 0


def run(cfg):
    raw_items = cfg.get("items") or []
    if not raw_items and cfg.get("file"):
        raw_items = [{"post_id": cfg.get("post_id"), "file": cfg["file"],
                      "name": os.path.basename(cfg["file"]),
                      "caption": cfg.get("caption") or "",
                      "publish_at": cfg.get("publish_at") or 0,
                      "status_file": cfg["status_file"]}]
    items = []
    for it in raw_items:
        if not it.get("status_file"):
            log(f"нет status_file у {it.get('post_id')}, пропускаю")
            continue
        if not os.path.isfile(it.get("file") or ""):
            write_status(it["status_file"], "error", "файл видео не найден",
                         it.get("file") or "")
            continue
        it["publish_at"] = float(it.get("publish_at") or 0)
        items.append(it)
    if not items:
        return 2

    account = str(cfg.get("account") or "1")
    profile_dir = cfg["profile_dir"]
    write_all(items, "starting", f"открываю профиль {account}…")
    try:
        from camoufox.sync_api import Camoufox
    except Exception as ex:
        write_all(items, "error", "в venv TikTokUA нет camoufox",
                  str(ex)[:300])
        return 3

    kwargs = dict(
        headless=False,
        persistent_context=True,
        user_data_dir=profile_dir,
        fingerprint=load_fingerprint(profile_dir),
        i_know_what_im_doing=True,
        humanize=True,
        locale="uk-UA",
        config={
            "timezone": "Europe/Kyiv",
            "geolocation:latitude": 50.4501,
            "geolocation:longitude": 30.5234,
        },
    )
    try:
        with Camoufox(**kwargs) as context:
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(20000)
            page.bring_to_front()
            write_all(items, "starting", "открываю TikTok Studio…")
            page.goto(STUDIO_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(random.randint(1200, 2600))

            deadline = time.time() + 45 * 60
            while is_login(page) and time.time() < deadline:
                write_all(items, "login",
                          "нужно войти в аккаунт — я жду, окно открыто")
                page.wait_for_timeout(3000)
            if is_login(page):
                write_all(items, "error", "аккаунт так и не вошёл")
                return 4
            if "/tiktokstudio" not in page.url:
                page.goto(STUDIO_URL, wait_until="domcontentloaded")
                page.wait_for_timeout(random.randint(1500, 3000))

            waited = 0
            while not has_file_input(page) and waited < 20000:
                page.wait_for_timeout(1000)
                waited += 1000
                if is_login(page):
                    break
            if is_login(page):
                while is_login(page) and time.time() < deadline:
                    write_all(items, "login",
                              "нужно войти в аккаунт — я жду, окно открыто")
                    page.wait_for_timeout(3000)
                if is_login(page):
                    write_all(items, "error", "аккаунт так и не вошёл")
                    return 4
                page.goto(STUDIO_URL, wait_until="domcontentloaded")
                page.wait_for_timeout(random.randint(1500, 3000))

            bulk_done = False
            if len(items) > 1:
                try:
                    multiple = bool(page.locator(
                        'input[type="file"]').first.evaluate(
                        "el => !!el.multiple"))
                except Exception:
                    multiple = False
                if multiple:
                    bulk_done = bulk_run(page, items)
                    if bulk_done:
                        return 0
                    # bulk не открылся: первую строку могло уже загрузить
                    for it in items[1:]:
                        write_status(it["status_file"], "queued", "")
                    items = items[:1]
                else:
                    for it in items[1:]:
                        write_status(it["status_file"], "queued", "")
                    items = items[:1]
            return run_single(page, items[0])
    except KeyboardInterrupt:
        write_all(items, "canceled", "отменено")
        return 0
    except Exception as ex:
        text = str(ex)
        low = text.lower()
        if "user data dir" in low or "singleton" in low or "profile" in low:
            text = (f"профиль account-{account} уже открыт другим окном — "
                    "закрой TikTok UA и повтори")
        write_all(items, "error", "ошибка браузера", text[:400])
        traceback.print_exc()
        return 5


def wait_until_closed(page, status_file, status, message, target_ts=0):
    """Держим окно открытым, пока человек не закончит. Если он всё-таки
    нажмёт «Запланировать»/«Опубликовать», фиксируем это сами."""
    write_status(status_file, status, message)
    try:
        upload_url = page.url
    except Exception:
        upload_url = ""
    t0 = time.time()
    while time.time() - t0 < 6 * 3600:
        if page.is_closed():
            write_status(status_file, "closed", "окно закрыто")
            return
        try:
            if success_visible(page, body_fallback=False,
                               words=SUCCESS_WORDS + SCHEDULED_DONE_WORDS):
                finish_success(page, status_file, target_ts)
                return
        except Exception:
            pass
        if "/upload" in upload_url:
            try:
                url = page.url
                if "/upload" not in url and "/tiktokstudio" in url:
                    finish_success(page, status_file, target_ts, guessed=True)
                    return
            except Exception:
                pass
        write_status(status_file, status, message)
        page.wait_for_timeout(5000)
    write_status(status_file, "closed", "окно осталось открытым")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="JSON с параметрами публикации")
    ap.add_argument("--check", action="store_true",
                    help="проверить окружение без запуска браузера")
    args = ap.parse_args()
    if args.check:
        out = {"python": sys.executable}
        try:
            import camoufox
            from camoufox.sync_api import Camoufox  # noqa: F401
            out["camoufox"] = str(getattr(camoufox, "__version__", "ok"))
        except Exception as ex:
            out["camoufox_error"] = str(ex)
        try:
            import playwright
            out["playwright"] = "ok"
        except Exception as ex:
            out["playwright_error"] = str(ex)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    if not args.config:
        print("нужен --config", file=sys.stderr)
        return 1
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
