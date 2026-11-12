"""Render the 9:16 (1080x1920) "follow the trail" Isan-monks video.

    python3 video/render.py                # all frames -> $VOUT, then encode with ffmpeg (see video/README.md)
    python3 video/render.py --preview 5 25 # single frames at given seconds
    python3 video/render.py --debug        # map with computed place positions
Needs: pillow, numpy, scipy.
"""
import sys, os, math, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("VOUT", "/tmp/claude-0/s/frames")
W, H, FPS, DUR = 1080, 1920, 30, 140.010958
GOLD = (240, 200, 105); CREAM = (252, 244, 222)
FB = "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf"; FR = "/usr/share/fonts/opentype/tlwg/Loma.otf"
font = lambda s, b=True: ImageFont.truetype(FB if b else FR, s)
EASE = lambda x: x * x * (3 - 2 * x)
clamp = lambda x: max(0.0, min(1.0, x))
find = lambda sub: glob.glob(f"{ROOT}/{sub}")[0]

# ---------- assets ----------
SRC = Image.open(f"{ROOT}/assets/terrain/thailand_satellite.webp").convert("RGB")
MW, MH = SRC.size                       # 941 x 1672 "map units"
UP, PAD = 2, 320                        # upscale factor, reflect padding (map units)
_m = SRC.resize((MW * UP, MH * UP), Image.LANCZOS).filter(ImageFilter.UnsharpMask(3, 60, 2))
MAPIMG = Image.fromarray(np.pad(np.asarray(_m), ((PAD * UP,) * 2, (PAD * UP,) * 2, (0, 0)), mode="reflect"))

def key_edges(im, test):
    """make background (connected to image border and passing `test`) transparent"""
    a = np.asarray(im.convert("RGB")).astype(int)
    lab, _ = ndimage.label(test(a))
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bgm = ndimage.binary_dilation(np.isin(lab, list(edge)), iterations=2)
    al = Image.fromarray(np.where(bgm, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    out = im.convert("RGBA"); out.putalpha(al); return out.crop(out.getbbox())

_p = Image.open(f"{ROOT}/assets/pin/pin_9.webp")
if _p.mode == "RGBA" and _p.getextrema()[3][0] < 10: PIN = _p.crop(_p.split()[3].point(lambda v: 255 if v > 20 else 0).getbbox())
else: PIN = key_edges(_p, lambda a: a.min(2) > 236)

def load_logo():
    pngs = [p for p in glob.glob(f"{ROOT}/Map_real/*logo*.png") + glob.glob(f"{ROOT}/assets/logo/*.png")]
    for p in pngs:                                   # prefer a real transparent PNG if supplied
        im = Image.open(p)
        if im.mode == "RGBA" and im.getextrema()[3][0] < 10: return im.crop(im.getbbox())
    a = np.asarray(Image.open(pngs[0]).convert("RGB")).astype(int)   # else chroma-key the green screen
    g = a[..., 1] - np.maximum(a[..., 0], a[..., 2])
    alpha = np.clip(255 - (g - 40) * 255 / 90, 0, 255).astype(np.uint8)
    a[..., 1] = np.where(g > 0, np.maximum(a[..., 0], a[..., 2]), a[..., 1])
    im = Image.fromarray(np.dstack([a.astype(np.uint8), alpha]), "RGBA")
    return im.crop(im.getbbox())
LOGO = load_logo()
def logo_w(w): return LOGO.resize((w, int(LOGO.height * w / LOGO.width)), Image.LANCZOS)
LOGO_S, LOGO_L = logo_w(400), logo_w(920)

# ---------- places: approximate lon/lat -> map units (illustrative map, not surveyed) ----------
def ll(lon, lat): return (385 + 97 * (lon - 100.5), 735 - 91 * (lat - 13.75))
# (start, end, province, monk, temple, sheet keyword, crop (x0, y0, w) of 3:4 portrait, position)
SEG = [
 (0, 18, None, None, None, None, None, None),
 (18, 37, "นครราชสีมา", "หลวงพ่อคูณ", "วัดบ้านไร่", "หลวงปู่คูณ", (10, 0, 470), ll(101.77, 15.20)),
 (37, 56, "นครพนม", "หลวงปู่คำพันธ์", "วัดธาตุมหาชัย", "หลวงปู่คำพันธ์", (25, 0, 470), ll(104.50, 17.18)),
 (56, 68, "สุรินทร์", "หลวงปู่หงษ์", "สุสานทุ่งมน", "หลวงปู่หงส์", (0, 0, 360), ll(103.40, 14.78)),
 (68, 83, "ศรีสะเกษ", "หลวงปู่หมุน", "วัดบ้านจาน", "หลวงปู่หมุน", (0, 0, 385), ll(104.57, 15.10)),
 (83, 93, "ร้อยเอ็ด", "หลวงปู่ศรี", "วัดป่ากุง", "หลวงปู่ศรี", (40, 0, 480), ll(103.50, 16.03)),
 (93, 103, "ร้อยเอ็ด", "หลวงปู่ทองมา", "วัดสว่างท่าสี", "หลวงปู่ทองมา", (0, 0, 395), ll(103.95, 16.03)),
 (103, 117, "ขอนแก่น", "หลวงปู่ผาง", "วัดอุดมคงคาคีรีเขต", "หลวงปู่ผาง", (0, 0, 395), ll(102.53, 16.13)),
 (117, 128, "อุดรธานี", "หลวงตามหาบัว", "โครงการช่วยชาติ", "หลวงตามหาบัว", (0, 0, 480), ll(102.80, 17.30)),
 (128, 140.02, None, None, None, None, None, None),
]
MONK = [i for i, s in enumerate(SEG) if s[7]]
TRAVEL = 2.2          # seconds the pin flies between places
ISAN = (665, 520)

# ---------- camera ----------
def cam(t):
    """(cx, cy, zoom, tilt) in map units"""
    i = max(k for k, s in enumerate(SEG) if t >= s[0]); s0 = SEG[i][0]
    full = (MW / 2, MH / 2, 1.0, 0.0)
    wide = (ISAN[0], ISAN[1] + 60, 1.75, 0.22)
    def at(k):
        x, y = SEG[k][7]; return (x, y + 95, 2.35, 0.30)
    lerp = lambda a, b, e: tuple(a[j] + (b[j] - a[j]) * e for j in range(4))
    if i == 0:
        return lerp(full, wide, EASE(clamp((t - 6) / 11)))
    if i == len(SEG) - 1:
        return lerp(at(i - 1), (MW / 2, MH / 2 - 40, 1.12, 0.0), EASE(clamp((t - s0) / 4)))
    prev = wide if i == 1 else at(i - 1)
    e = EASE(clamp((t - s0) / TRAVEL))
    c = list(lerp(prev, at(i), e))
    d = math.dist(prev[:2], at(i)[:2])
    c[2] -= min(0.55, d / 450) * math.sin(math.pi * e)      # pull back while flying
    drift = (t - s0 - TRAVEL) / (SEG[i][1] - s0) if t > s0 + TRAVEL else 0
    c[2] += 0.10 * drift                                     # slow push-in while dwelling
    return tuple(c)

def homography(src, dst):
    A = []; b = []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]; b += [u, v]
    return np.linalg.solve(np.array(A, float), np.array(b, float))

def view(t):
    cx, cy, z, tilt = cam(t)
    hw, hh = MW / (2 * z), MH / (2 * z)
    top, bot = 1 + tilt, 1 - tilt * 0.25
    cx = min(max(cx, hw * top), MW - hw * top)               # keep the view inside the map image
    cy = min(max(cy, hh * (1 + tilt * 0.7)), MH - hh)
    src = [(cx - hw * top, cy - hh * (1 + tilt * 0.7)), (cx + hw * top, cy - hh * (1 + tilt * 0.7)),
           (cx + hw * bot, cy + hh), (cx - hw * bot, cy + hh)]
    src = [((x + PAD) * UP, (y + PAD) * UP) for x, y in src]
    co = homography(src, [(0, 0), (W, 0), (W, H), (0, H)])
    img = MAPIMG.transform((W, H), Image.PERSPECTIVE, tuple(co), resample=Image.BICUBIC)
    Hm = np.linalg.inv(np.append(co, 1).reshape(3, 3))
    def proj(p):
        v = Hm @ np.array([(p[0] + PAD) * UP, (p[1] + PAD) * UP, 1.0]); return (v[0] / v[2], v[1] / v[2])
    return img, proj, z

# ---------- overlays ----------
def vignette():
    y, x = np.mgrid[0:H, 0:W]
    d = np.hypot((x - W / 2) / (W / 2), (y - H / 2) / (H / 2))
    a = np.clip((d - 0.75) / 0.65, 0, 1) * 150
    a = np.maximum(a, np.clip(1 - y / 420, 0, 1) * 170)       # darker top for the logo
    return Image.fromarray(np.dstack([np.zeros((H, W, 3), np.uint8), a.astype(np.uint8)]), "RGBA")
VIG = vignette()

CARD_W, CARD_H = 1000, 560
_cards = {}
def card(i):
    if i in _cards: return _cards[i]
    _, _, prov, monk, temple, key, (x0, y0, w), _ = SEG[i]
    f = [p for p in glob.glob(f"{ROOT}/*.png") if key in p][0]
    ph = Image.open(f).convert("RGB").crop((x0, y0, x0 + w, y0 + int(w * 4 / 3))).resize((375, 500), Image.LANCZOS)
    c = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0)); d = ImageDraw.Draw(c)
    d.rounded_rectangle((0, 0, CARD_W - 1, CARD_H - 1), 26, fill=(18, 13, 6, 205), outline=GOLD + (255,), width=3)
    d.rectangle((26, 22, 26 + 375 + 16, 22 + 500 + 16), fill=GOLD + (255,))          # square gold frame
    d.rectangle((31, 27, 31 + 375 + 6, 27 + 500 + 6), fill=(255, 250, 235, 255))
    c.paste(ph, (34, 30))
    tx = 460
    d.text((tx, 46), "จังหวัด", font=font(32, False), fill=GOLD)
    d.text((tx, 92), prov, font=font(60), fill=CREAM)
    d.line((tx, 200, CARD_W - 40, 200), fill=GOLD, width=3)
    fs = 58 if len(monk) <= 10 else 48
    d.text((tx, 232), monk, font=font(fs), fill=GOLD)
    d.text((tx, 232 + fs + 34), temple, font=font(40 if len(temple) < 14 else 34, False), fill=CREAM)
    d.text((tx, CARD_H - 64), "ภาพประกอบ", font=font(24, False), fill=(200, 175, 120))
    _cards[i] = c; return c

def fade(im, a):
    if a >= 0.999: return im
    im = im.copy(); im.putalpha(im.split()[3].point(lambda v: int(v * a))); return im

def pin_state(t):
    """pin position in map units, hop height 0..1, visible"""
    i = max(k for k, s in enumerate(SEG) if t >= s[0])
    if i == 0: return None, 0, i
    if i == len(SEG) - 1: return SEG[MONK[-1]][7], 0, i
    s0 = SEG[i][0]; e = clamp((t - s0) / TRAVEL)
    if i == MONK[0]:
        return SEG[i][7], 1 - EASE(e), i                       # drop in from above
    a, b = SEG[i - 1][7], SEG[i][7]; k = EASE(e)
    return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k), math.sin(math.pi * e) * 0.5, i

def frame(n):
    t = n / FPS
    img, proj, z = view(t)
    img = img.convert("RGBA")
    pos, hop, i = pin_state(t)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    # trail
    visited = [k for k in MONK if t >= SEG[k][0]]
    pts = []
    for j, k in enumerate(visited):
        if j == 0: pts.append(SEG[k][7]); continue
        a, b = SEG[visited[j - 1]][7], SEG[k][7]
        f = 1 if (k != i or i == len(SEG) - 1) else EASE(clamp((t - SEG[k][0]) / TRAVEL))
        pts += [(a[0] + (b[0] - a[0]) * f * q / 24, a[1] + (b[1] - a[1]) * f * q / 24) for q in range(1, 25)]
    sp = [proj(p) for p in pts]
    if len(sp) > 1:
        d.line(sp, fill=(60, 30, 0, 170), width=int(5 * z) + 6, joint="curve")
        d.line(sp, fill=(255, 214, 90, 255), width=int(3 * z) + 2, joint="curve")
    for k in visited:
        if k == i and t < SEG[k][0] + TRAVEL and i != len(SEG) - 1: continue
        x, y = proj(SEG[k][7]); r = 5 * z + 4
        d.ellipse((x - r * 2, y - r * 1.4, x + r * 2, y + r * 1.4), fill=(255, 220, 110, 70))
        d.ellipse((x - r, y - r * 0.75, x + r, y + r * 0.75), fill=(255, 226, 120, 255), outline=(255, 255, 255, 255), width=3)
    img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(0.6)))
    if pos:
        x, y = proj(pos)
        landed = hop < 0.02
        bob = (math.sin(t * 3.2) * 0.5 + 0.5) * 14 if landed else 0
        lift = hop * 330 + bob
        pw = int(70 * z + 20); phh = int(PIN.height * pw / PIN.width)
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0)); sd = ImageDraw.Draw(sh)
        sw = pw * (0.38 - 0.15 * min(1, lift / 330))
        sd.ellipse((x - sw, y - sw * 0.4, x + sw, y + sw * 0.4), fill=(0, 0, 0, 130))
        img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6)))
        if landed:
            ph = (t * 0.9) % 1; rr = pw * (0.3 + 0.7 * ph)
            ImageDraw.Draw(img).ellipse((x - rr, y - rr * 0.6, x + rr, y + rr * 0.6), outline=(255, 235, 150, int(230 * (1 - ph))), width=5)
        p = PIN.resize((pw, phh), Image.LANCZOS)
        img.alpha_composite(p, (int(x - pw / 2), int(y - phh - lift)))
    img.alpha_composite(VIG)
    # logo
    sg = SEG[i]
    if i == 0:
        a = EASE(clamp(t / 1.2)) * (1 - EASE(clamp((t - 12.5) / 3)))
        img.alpha_composite(fade(LOGO_L, a), ((W - 920) // 2, 110))
        img.alpha_composite(fade(LOGO_S, EASE(clamp((t - 14.5) / 3))), ((W - 400) // 2, 40))
    elif i == len(SEG) - 1:
        a = EASE(clamp((t - sg[0] - 2.5) / 1.5))
        img.alpha_composite(fade(LOGO_S, 1 - a), ((W - 400) // 2, 40))
        img.alpha_composite(fade(LOGO_L, a), ((W - 920) // 2, 1120))
    else:
        img.alpha_composite(LOGO_S, ((W - 400) // 2, 40))
        k = EASE(clamp((t - sg[0] - TRAVEL + 0.4) / 0.8)) * (1 - EASE(clamp((t - sg[1] + 0.6) / 0.6)))
        if k > 0:
            img.alpha_composite(fade(card(i), k), ((W - CARD_W) // 2, int(1290 + (1 - k) * 90)))
    return img.convert("RGB")

def work(n): frame(n).save(f"{OUT}/f{n:05d}.jpg", quality=93)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1:2] == ["--preview"]:
        for t in sys.argv[2:]: frame(int(float(t) * FPS)).save(f"{OUT}/prev_{t}.png")
    elif sys.argv[1:2] == ["--debug"]:
        im = SRC.copy(); d = ImageDraw.Draw(im)
        for k in MONK:
            x, y = SEG[k][7]; d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 0, 0)); d.text((x + 8, y - 8), str(k), fill=(255, 255, 255))
        im.save(f"{OUT}/debug_places.png")
        sheet = Image.new("RGB", (8 * 380, 510), (0, 0, 0))
        for j, k in enumerate(MONK): sheet.paste(card(k).convert("RGB").crop((30, 26, 30 + 380, 26 + 510)), (j * 380, 0))
        sheet.save(f"{OUT}/debug_crops.png")
    else:
        with Pool(os.cpu_count()) as p: p.map(work, range(int(DUR * FPS)), chunksize=20)
