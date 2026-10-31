#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""balls_sim.py — «bouncing balls»: шарики падают в круге с вращающимся разрывом.

Порт проекта laiprogramming/bouncing_balls (pygame, step7.py) в безоконный
рендер: чёрный фон, оранжевое кольцо с вращающейся дыркой, гравитация,
упругий отскок и шарики, которые по одному вываливаются наружу. Стиль
исходника сохранён 1:1 (плоские цвета, тонкое кольцо, мелкие шарики),
но за счёт плотности, темпа и «затягивающей» механики дырки смотрится
как готовый шортс, а не как учебный пример.

Каждый запуск — новый ролик: сид, число шариков, палитра, скорость и
направление вращения дырки берутся из random.

    python3 balls_sim.py                      # 60с, 1080x1200, случайный сид
    python3 balls_sim.py --duration 45 --seed 7
    python3 balls_sim.py --count 6            # пачка разных вариантов
    python3 balls_sim.py --preview /tmp/p.png # один кадр для проверки вида

Рендер: numpy + OpenCV + ffmpeg (h264, yuv420p, 30fps). Внешних ассетов нет.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("нужен opencv: pip3 install opencv-python-headless")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    sys.exit("нужен Pillow: pip3 install Pillow")


BASE = os.path.dirname(os.path.abspath(__file__))
GAMEPLAY_DIR = os.path.join(BASE, "gameplay")

# Палитра со скриншота исходника: приглушённые пастельные тона (RGB).
PALETTE = [
    (232, 186, 128),   # peach
    (92, 196, 224),    # cyan
    (150, 124, 220),   # violet
    (92, 172, 152),    # teal
    (118, 182, 116),   # green
    (222, 120, 182),   # magenta
    (96, 156, 232),    # blue
    (204, 222, 116),   # lime
    (224, 152, 198),   # pink
    (168, 206, 168),   # sage
    (230, 158, 96),    # amber
    (140, 200, 232),   # sky
]

ORANGE_RGB = (255, 165, 0)      # цвет кольца, как в исходнике
ORANGE = (0, 165, 255)          # то же в BGR для OpenCV

FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _ffmpeg() -> str:
    for p in (os.path.expanduser("~/.local/bin/ffmpeg"), shutil.which("ffmpeg")):
        if p and os.path.exists(p):
            return p
    return "ffmpeg"


FFMPEG = _ffmpeg()


# ---------------------------------------------------------------- физика ---
class Ball:
    """Шарик. Пока inside — отражается от кольца, после — свободно падает."""

    __slots__ = ("x", "y", "vx", "vy", "color", "r", "inside", "alive", "born")

    def __init__(self, x, y, vx, vy, color, r, inside=True):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.color = color
        self.r = float(r)
        self.inside = inside
        self.alive = True
        self.born = 0.0


class Sim:
    """Кольцо с вращающимся разрывом + шарики под гравитацией."""

    def __init__(self, w, h, cx, cy, radius, cfg, rng):
        self.w, self.h = w, h
        self.cx, self.cy = cx, cy
        self.R = radius
        self.cfg = cfg
        self.rng = rng
        self.balls: list[Ball] = []
        self.gap = rng.uniform(0, math.tau)      # центр разрыва, рад
        self.spin = cfg.spin * (1 if rng.random() < 0.5 else -1)
        self.spin_goal = self.spin
        self.next_turn = rng.uniform(4.0, 9.0)   # когда сменить скорость
        self.escaped = 0
        self.refill = 0.0
        # «вдох» дырки: раз в 8-14с она распахивается и выпускает пачку шаров
        self.gap_w = float(cfg.gap_deg)
        self.wide = 0.0
        self.next_wide = rng.uniform(12.0, 20.0)
        self._prefill()

    # -- стартовая сцена -------------------------------------------------
    def _prefill(self):
        rng, c = self.rng, self.cfg
        n = rng.randint(c.balls_min, c.balls_max)
        colors = [self._color() for _ in range(n)]
        for i in range(n):
            a = rng.uniform(0, math.tau)
            d = self.R * rng.uniform(0.15, 0.80)
            self.balls.append(Ball(
                self.cx + math.cos(a) * d, self.cy + math.sin(a) * d,
                rng.uniform(-1, 1) * self.R * 0.35,
                rng.uniform(-1, 1) * self.R * 0.35,
                colors[i], c.ball_r))

    def _color(self):
        col = self.rng.choice(PALETTE)
        # лёгкий сдвиг, чтобы шарики одного цвета не сливались
        j = self.rng.randint(-12, 12)
        return tuple(int(np.clip(v + j, 0, 255)) for v in col)

    # -- шаг -------------------------------------------------------------
    def step(self, dt: float):
        cfg, rng = self.cfg, self.rng
        # дырка: плавно меняет скорость и иногда разворачивается —
        # именно это держит внимание, а не статичное вращение
        self.next_turn -= dt
        if self.next_turn <= 0:
            self.spin_goal = cfg.spin * rng.choice((-1.0, 1.0)) * rng.uniform(0.6, 1.5)
            self.next_turn = rng.uniform(4.0, 9.0)
        self.spin += (self.spin_goal - self.spin) * min(1.0, dt * 1.6)
        self.gap = (self.gap + self.spin * dt) % math.tau
        # распах дырки — редкий «выдох» арены, держит ритм ролика
        self.next_wide -= dt
        if self.next_wide <= 0:
            self.wide = 0.75
            self.next_wide = rng.uniform(14.0, 24.0)
        if self.inside <= cfg.target_balls - 4:
            self.wide = 0.0          # арена и так полупустая — распах отменяем
        target_w = cfg.gap_deg * (1.55 if self.wide > 0 else 1.0)
        self.wide = max(0.0, self.wide - dt)
        self.gap_w += (target_w - self.gap_w) * min(1.0, dt * 5.0)

        half = self.gap_w * math.pi / 360.0
        lim = self.R - cfg.ball_r
        sub = 4
        h = dt / sub
        for _ in range(sub):
            for b in self.balls:
                if not b.alive:
                    continue
                b.vy += cfg.gravity * h
                b.x += b.vx * h
                b.y += b.vy * h
                dx, dy = b.x - self.cx, b.y - self.cy
                d = math.hypot(dx, dy)
                if not b.inside:
                    # вывалился: живёт своей жизнью, пока не уйдёт за кадр
                    if (b.x < -b.r * 4 or b.x > self.w + b.r * 4
                            or b.y > self.h + b.r * 4):
                        b.alive = False
                    continue
                if d + b.r <= self.R:
                    continue
                nx, ny = dx / d, dy / d
                ang = math.atan2(ny, nx) % math.tau
                if self._in_gap(ang, half):
                    b.inside = False           # проскочил в дырку
                    self.escaped += 1
                    self.refill = min(self.refill, 0.12)   # замена придёт сразу
                    continue
                # отскок от кольца + подкрутка от вращения
                b.x = self.cx + nx * lim
                b.y = self.cy + ny * lim
                dot = b.vx * nx + b.vy * ny
                b.vx -= 2 * dot * nx
                b.vy -= 2 * dot * ny
                b.vx *= cfg.restitution
                b.vy *= cfg.restitution
                kick = self.spin * self.R * 0.10
                b.vx += -ny * kick
                b.vy += nx * kick
                sp = math.hypot(b.vx, b.vy)
                cap = self.R * 2.4
                if sp > cap:            # иначе арена уходит в разнос
                    b.vx *= cap / sp
                    b.vy *= cap / sp
        # подсев: выбывших убираем, арену потихоньку добиваем новыми
        self.balls = [b for b in self.balls if b.alive]
        self.refill -= dt
        if (self.refill <= 0 and self.inside < self.cfg.target_balls
                and len(self.balls) < self.cfg.max_balls):
            self.balls.append(self._drop())
            # полупустую арену заливаем быстрее — она не должна «стыть»
            self.refill = rng.uniform(0.08, 0.28) if self.inside < self.cfg.target_balls - 3 \
                else rng.uniform(0.16, 0.55)

    def _in_gap(self, ang: float, half: float) -> bool:
        a = (ang - self.gap + math.pi) % math.tau - math.pi
        return abs(a) <= half

    def _drop(self) -> Ball:
        """Новый шарик падает сверху — как в исходнике."""
        rng, c = self.rng, self.cfg
        return Ball(self.cx + rng.uniform(-0.18, 0.18) * self.R,
                    self.cy - self.R * 0.78,
                    rng.uniform(-0.6, 0.6) * self.R,
                    rng.uniform(0, 0.5) * self.R,
                    self._color(), c.ball_r)

    @property
    def inside(self) -> int:
        return sum(1 for b in self.balls if b.inside)


# ------------------------------------------------------------- рендеринг ---
def _font(size: int):
    for p in FONT_CANDIDATES:
        if not os.path.exists(p):
            continue
        try:
            f = ImageFont.truetype(p, size)
        except Exception:
            continue
        for name in ("Bold", "Semibold", "Medium"):
            try:
                f.set_variation_by_name(name)
                break
            except Exception:
                continue
        return f
    return ImageFont.load_default()


def _geometry(w, h, cfg):
    """Круг сидит выше центра: внизу остаётся чёрная полоса, на которой
    висит статичная плашка ShardVPN (она всегда идёт под кругом)."""
    area = h - h * cfg.bottom_pad
    radius = min(w * 0.46, area * 0.5 * 0.82)
    return w / 2.0, area * 0.5, radius


def _draw_counter(img, sim, cfg):
    """Счётчик внутри круга — опционально, тем же оранжевым."""
    size = max(14, int(cfg.width * 0.020))
    font = _font(size)
    txt = str(sim.inside)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    tw = d.textlength(txt, font=font)
    d.text((cfg.width - tw - cfg.width * 0.045, cfg.height * 0.035), txt,
           font=font, fill=ORANGE_RGB)
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _frame(sim, cfg, img):
    """Один кадр: чёрный фон, кольцо с разрывом, шарики."""
    img[:] = 0
    thick = max(1, int(round(cfg.width / 540)))
    half_deg = sim.gap_w / 2.0
    g0 = math.degrees(sim.gap) - half_deg
    g1 = math.degrees(sim.gap) + half_deg
    # кольцо — дуга от конца разрыва до его начала
    cv2.ellipse(img, (int(sim.cx), int(sim.cy)), (int(sim.R), int(sim.R)), 0,
                g1, g0 + 360, ORANGE, thick, cv2.LINE_AA)
    for b in sim.balls:
        cv2.circle(img, (int(round(b.x)), int(round(b.y))), int(round(b.r)),
                   (b.color[2], b.color[1], b.color[0]), -1, cv2.LINE_AA)
    if cfg.counter:
        _draw_counter(img, sim, cfg)
    return img


def _prepare(cfg, rng):
    cx, cy, radius = _geometry(cfg.width, cfg.height, cfg)
    sim = Sim(cfg.width, cfg.height, cx, cy, radius, cfg, rng)
    for _ in range(int(cfg.warmup * cfg.fps)):
        sim.step(1.0 / cfg.fps)
    return sim


def render(cfg) -> str:
    W, H = cfg.width, cfg.height
    sim = _prepare(cfg, random.Random(cfg.seed))
    out = cfg.out
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(cfg.fps), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
           "-pix_fmt", "yuv420p", "-g", str(cfg.fps * 2),
           "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = int(cfg.duration * cfg.fps)
    img = np.zeros((H, W, 3), np.uint8)
    t0 = time.time()
    try:
        for fi in range(total):
            proc.stdin.write(_frame(sim, cfg, img).tobytes())
            sim.step(1.0 / cfg.fps)
            if fi % max(1, cfg.fps * 5) == 0:
                el = time.time() - t0
                pct = (fi + 1) / total
                sys.stderr.write(f"\r  {pct * 100:5.1f}%  кадр {fi}/{total}"
                                 f"  в круге {sim.inside}"
                                 f"  выпало {sim.escaped}"
                                 f"  осталось {el / max(pct, 1e-6) - el:4.0f}с")
                sys.stderr.flush()
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    sys.stderr.write(f"\r  готово: {out}  ({os.path.getsize(out) / 1e6:.1f} МБ)"
                     f"  за {time.time() - t0:.0f}с\n")
    return out


def _preview(cfg):
    sim = _prepare(cfg, random.Random(cfg.seed))
    img = np.zeros((cfg.height, cfg.width, 3), np.uint8)
    _frame(sim, cfg, img)
    cv2.imwrite(cfg.preview, img)
    print(f"кадр: {cfg.preview} (в круге {sim.inside}, выпало {sim.escaped})",
          file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bouncing balls: шарики в круге с дыркой")
    ap.add_argument("-o", "--out", default="", help="путь к mp4")
    ap.add_argument("--duration", type=float, default=60.0, help="секунды")
    ap.add_argument("--size", default="1080x1200", help="ШxВ (для сплита 1080x1200)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None, help="сид (по умолчанию случайный)")
    ap.add_argument("--count", type=int, default=1, help="сколько разных роликов сделать")
    ap.add_argument("--balls", default="", help="диапазон стартовых шариков, напр. 8-12")
    ap.add_argument("--target", type=int, default=0, help="сколько держать в круге")
    ap.add_argument("--max", type=int, default=0, help="потолок шариков в кадре")
    ap.add_argument("--gap", type=float, default=0, help="ширина дырки, градусы (60)")
    ap.add_argument("--spin", type=float, default=0, help="скорость вращения, рад/с")
    ap.add_argument("--counter", action="store_true", help="счётчик шариков в круге")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--warmup", type=float, default=2.5, help="секунды прогрева до кадра 0")
    ap.add_argument("--preview", default="", help="сохранить PNG одного кадра и выйти")
    a = ap.parse_args(argv)

    w, h = (int(v) for v in a.size.lower().split("x"))
    a.width, a.height = w, h
    a.bottom_pad = 0.246                   # полоса под плашку внизу кадра
    _cx, _cy, _r = _geometry(w, h, a)
    a.ball_r = max(4.0, _r / 30.0)         # пропорция исходника: r = R/30
    a.gap_deg = a.gap or 32.0
    a.spin = a.spin or 0.45                # как spinning_speed 0.01 * 60fps
    # гравитация исходника: 0.2 px/кадр² при 60fps на 800px → px/с²
    a.gravity = 620.0 * (a.width / 800.0)
    a.restitution = 1.0
    if a.balls:
        lo, _, hi = a.balls.partition("-")
        a.balls_min, a.balls_max = int(lo), int(hi or lo)
    else:
        a.balls_min, a.balls_max = 10, 12
    a.target_balls = a.target or 11
    a.max_balls = a.max or 16

    made = []
    base_out = a.out          # имя-шаблон не должно «наращиваться» от итерации к итерации
    for i in range(max(1, a.count)):
        seed = a.seed if a.seed is not None else random.randrange(1, 10 ** 9)
        if a.count > 1 and a.seed is not None:
            seed += i * 7919
        a.seed = seed
        if base_out:
            out = base_out if a.count == 1 else \
                f"{os.path.splitext(base_out)[0]}_{i + 1:02d}.mp4"
        else:
            os.makedirs(GAMEPLAY_DIR, exist_ok=True)
            k = 1
            while os.path.exists(os.path.join(GAMEPLAY_DIR, f"balls_{k:02d}.mp4")):
                k += 1
            out = os.path.join(GAMEPLAY_DIR, f"balls_{k:02d}.mp4")
        a.out = out
        print(f"сид {seed} → {out}", file=sys.stderr)
        if a.preview:
            _preview(a)
            return 0
        made.append(render(a))
    for m in made:
        print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
