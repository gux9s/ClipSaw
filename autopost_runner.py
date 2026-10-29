"""Автопостинг Lipps: открывает живой профиль Camoufox, кладёт видео и
описание в TikTok Studio и ждёт, пока человек сам нажмёт «Опубликовать».

Ничего не публикует, не решает капчу и не вводит пароли. Профиль, отпечаток
и локация — те же, что у обычного запуска TikTok UA, поэтому для TikTok это
тот же самый браузер, а не новый «робот».
"""
import argparse
import datetime as dt
import json
import os
import random
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
    'button:has-text("Upload video")',
    'button:has-text("Вибрати відео")',
    'button:has-text("Завантажити")',
    'button:has-text("Выбрать видео")',
    '[role="button"]:has-text("Select video")',
)
SUCCESS_WORDS = (
    "your video is being uploaded",
    "your video has been uploaded",
    "your video was uploaded",
    "video published",
    "video has been published",
    "being uploaded to tiktok",
    "опубліковано",
    "опубликовано",
    "відео завантажується",
    "видео загружается",
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


def success_visible(page, body_fallback=True):
    """Ищем тост/диалог успеха только после ухода из редактора.
    body_fallback=False — доверяем только тостам (для промежуточных
    проверок, где в теле страницы есть служебные слова)."""
    try:
        nodes = page.locator(
            '[role="dialog"], [data-e2e*="success"], [data-e2e="toast"],'
            ' [class*="toast"], [role="alert"]')
        for i in range(min(nodes.count(), 4)):
            try:
                txt = (nodes.nth(i).inner_text(timeout=1500) or "").lower()
            except Exception:
                continue
            if any(w in txt for w in SUCCESS_WORDS + SCHEDULED_WORDS):
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


def set_file(page, path):
    """Основной путь — скрытый input (TikTok сам обрабатывает change)."""
    try:
        inp = page.locator('input[type="file"]').first
        inp.set_input_files(path, timeout=20000)
        return True
    except Exception as ex:
        log(f"input не принялся: {ex}")
    for sel in UPLOAD_BUTTONS:
        try:
            btn = page.locator(sel).first
            if not btn.count() or not btn.is_visible():
                continue
            with page.expect_file_chooser(timeout=4000) as fc:
                btn.click()
            fc.value.set_files(path)
            return True
        except Exception:
            continue
    return False


def type_caption(page, editor, caption):
    editor.click()
    page.wait_for_timeout(random.randint(450, 1200))
    for ch in caption:
        if ch == "\n":
            page.keyboard.press("Enter")
            page.wait_for_timeout(random.randint(45, 130))
        else:
            page.keyboard.type(ch)
            page.wait_for_timeout(random.randint(12, 58))
    # пробел закрывает выпадашку хэштегов, чтобы она не мешала кнопке
    page.keyboard.type(" ")
    page.wait_for_timeout(random.randint(300, 800))


def plan_time(publish_at):
    """Время публикации из очереди. Студия принимает минимум +15 минут,
    поэтому протухшее время сдвигаем на +20 минут."""
    try:
        ts = float(publish_at or 0)
    except (TypeError, ValueError):
        return 0
    if ts <= 0:
        return 0
    if ts < time.time() + 16 * 60:
        return time.time() + 20 * 60
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
             d.strftime("%d.%m.%y"), d.strftime("%m/%d/%y")}
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


def apply_schedule(page, target_ts):
    """Включает режим расписания и ставит дату/время в TikTok Studio.
    Финальную кнопку «Schedule» не жмёт. Не вышло — False (впишет человек)."""
    d = dt.datetime.fromtimestamp(target_ts)
    iso = d.strftime("%Y-%m-%dT%H:%M")
    date_iso = d.strftime("%Y-%m-%d")
    time_iso = d.strftime("%H:%M")
    try:
        has_dt = page.locator('input[type="datetime-local"]').count() > 0
        has_d = page.locator('input[type="date"]').count() > 0
        has_t = page.locator('input[type="time"]').count() > 0
    except Exception:
        return False
    if not (has_dt or (has_d and has_t)):
        opt = _find_schedule_option(page)
        if opt is None:
            return False
        try:
            opt.click(timeout=4000)
        except Exception:
            return False
        page.wait_for_timeout(random.randint(900, 1600))
        try:
            has_dt = page.locator('input[type="datetime-local"]').count() > 0
            has_d = page.locator('input[type="date"]').count() > 0
            has_t = page.locator('input[type="time"]').count() > 0
        except Exception:
            return False
    filled = False
    if has_dt:
        filled = _fill_input(page, 'input[type="datetime-local"]', iso)
    elif has_d and has_t:
        fd = _fill_input(page, 'input[type="date"]', date_iso)
        ft = _fill_input(page, 'input[type="time"]', time_iso)
        filled = fd and ft
    if not filled:
        # запасной путь: текстовые поля с подсказкой date/time
        try:
            pairs = page.evaluate(
                """() => [...document.querySelectorAll('input')]
                     .filter(i => !i.type || i.type === 'text')
                     .map(i => ({hint: ((i.placeholder || '') + ' ' +
                       (i.getAttribute('aria-label') || '')).toLowerCase(),
                       id: i.id || ''}))""")
        except Exception:
            pairs = []
        for p in pairs:
            hint = p.get("hint") or ""
            val = None
            if any(w in hint for w in ("date", "дата", "дд", "mm/dd")):
                val = d.strftime("%m/%d/%Y")
            elif any(w in hint for w in ("time", "час", "время")):
                val = time_iso
            if not val:
                continue
            sel = f"#{p['id']}" if p.get("id") else ""
            if sel and _fill_input(page, sel, val):
                filled = True
    if not filled:
        return False
    try:
        page.evaluate("() => { if (document.activeElement) "
                      "document.activeElement.blur(); }")
    except Exception:
        pass
    page.wait_for_timeout(random.randint(700, 1400))
    return _schedule_reflected(page, d)


def finish_success(page, status_file, target_ts, guessed=False):
    """Фиксируем финал. Публикацию/расписание жмёт человек — мы только
    записываем, что окно ушло из редактора."""
    label = fmt_target(target_ts) if target_ts else ""
    if target_ts:
        stage = (f"запланировано на {label}" if not guessed else
                 f"похоже, запланировано на {label} — проверь")
        write_status(status_file, "scheduled", stage)
    else:
        stage = ("опубликовано — закрываю окно" if not guessed else
                 "похоже, опубликовано — проверь")
        write_status(status_file, "published", stage)
    try:
        page.wait_for_timeout(8000)
    except Exception:
        pass


def run(cfg):
    status_file = cfg["status_file"]
    caption = cfg.get("caption") or ""
    src = cfg.get("file") or ""
    account = str(cfg.get("account") or "1")
    profile_dir = cfg["profile_dir"]
    try:
        publish_at = float(cfg.get("publish_at") or 0)
    except (TypeError, ValueError):
        publish_at = 0

    if not os.path.isfile(src):
        write_status(status_file, "error", "файл видео не найден", src)
        return 2

    write_status(status_file, "starting", f"открываю профиль {account}…")
    try:
        from camoufox.sync_api import Camoufox
    except Exception as ex:
        write_status(status_file, "error",
                     "в venv TikTokUA нет camoufox", str(ex)[:300])
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
            write_status(status_file, "starting", "открываю TikTok Studio…")
            page.goto(STUDIO_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(random.randint(1200, 2600))

            deadline = time.time() + 45 * 60
            while is_login(page) and time.time() < deadline:
                write_status(status_file, "login",
                             "нужно войти в аккаунт — я жду, окно открыто")
                page.wait_for_timeout(3000)
            if is_login(page):
                write_status(status_file, "error", "аккаунт так и не вошёл")
                return 4
            if "/tiktokstudio" not in page.url:
                page.goto(STUDIO_URL, wait_until="domcontentloaded")
                page.wait_for_timeout(random.randint(1500, 3000))

            # ждём поле загрузки; если студия ещё не прогрузилась — даём время
            waited = 0
            while not has_file_input(page) and waited < 20000:
                page.wait_for_timeout(1000)
                waited += 1000
                if is_login(page):
                    break
            if is_login(page):
                while is_login(page) and time.time() < deadline:
                    write_status(status_file, "login",
                                 "нужно войти в аккаунт — я жду, окно открыто")
                    page.wait_for_timeout(3000)
                if is_login(page):
                    write_status(status_file, "error", "аккаунт так и не вошёл")
                    return 4
                page.goto(STUDIO_URL, wait_until="domcontentloaded")
                page.wait_for_timeout(random.randint(1500, 3000))

            write_status(status_file, "running", "загружаю видео…")
            if not set_file(page, src):
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
            target_ts = plan_time(publish_at)
            schedule_ok = False
            if target_ts:
                label = fmt_target(target_ts)
                write_status(status_file, "running",
                             f"выставляю публикацию на {label}…")
                schedule_ok = apply_schedule(page, target_ts)
                if schedule_ok:
                    write_status(status_file, "running",
                                 f"время {label} выставлено")
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

            # ждём публикацию: уход из редактора + тост успеха. Окно не
            # закрываем сами, пока не увидим подтверждение.
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
                    # успех ищем только после ухода из редактора: слово
                    # «upload» в самом редакторе — ещё не публикация
                    for _ in range(10):
                        if page.is_closed():
                            break
                        try:
                            if success_visible(page):
                                saw_success = True
                                break
                        except Exception:
                            pass
                        page.wait_for_timeout(2000)
                    if saw_success:
                        finish_success(page, status_file, target_ts)
                    elif left_upload and target_ts:
                        finish_success(page, status_file, target_ts,
                                       guessed=True)
                    else:
                        write_status(
                            status_file, "closed",
                            "редактор закрылся без подтверждения — проверь, "
                            "вышло ли видео, и отметь вручную")
                    return 0
                page.wait_for_timeout(2000)
            write_status(status_file, "closed", "окно осталось открытым")
            return 0
    except KeyboardInterrupt:
        write_status(status_file, "canceled", "отменено")
        return 0
    except Exception as ex:
        text = str(ex)
        low = text.lower()
        if "user data dir" in low or "singleton" in low or "profile" in low:
            text = (f"профиль account-{account} уже открыт другим окном — "
                    "закрой TikTok UA и повтори")
        write_status(status_file, "error", "ошибка браузера", text[:400])
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
            if success_visible(page, body_fallback=False):
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
