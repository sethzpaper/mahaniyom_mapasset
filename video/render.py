"""Render the 9:16 (1080x1920) Isan-monks video: python3 video/render.py [--preview T ...]"""
import sys, os, math, glob, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("VOUT", "/tmp/claude-0/s/frames")
W, H, FPS, DUR = 1080, 1920, 30, 140.010958
GOLD = (233, 196, 106); CREAM = (250, 240, 215)
FB = "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf"; FR = "/usr/share/fonts/opentype/tlwg/Loma.otf"
font = lambda s, b=True: ImageFont.truetype(FB if b else FR, s)

def find(sub): return glob.glob(f"{ROOT}/{sub}")[0]
MAP = Image.open(find("Map_real/bb2c930e-maprealname.png")).convert("RGBA")
def _key_checker(im):
    from scipy import ndimage
    a = np.asarray(im).astype(int)
    mx, mn = a[..., :3].max(2), a[..., :3].min(2)
    cand = (mn > 175) & (mx - mn < 14)          # white / light-gray checker squares
    lab, n = ndimage.label(cand)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bgm = np.isin(lab, list(edge))
    bgm = ndimage.binary_dilation(bgm, iterations=1)
    al = np.where(bgm, 0, 255).astype(np.uint8)
    al = np.asarray(Image.fromarray(al).filter(ImageFilter.GaussianBlur(0.8)))
    out = np.asarray(im).copy(); out[..., 3] = al
    return Image.fromarray(out, "RGBA")
MAP = _key_checker(MAP)
LOGO = Image.open(find("Map_real/txt_*logogold.png")).convert("RGB")
# chroma-key the green screen
a = np.asarray(LOGO).astype(int)
g = (a[..., 1] - np.maximum(a[..., 0], a[..., 2]))
alpha = np.clip(255 - (g - 40) * 255 / 90, 0, 255).astype(np.uint8)
alpha[g < 40] = 255
rgb = a.copy()
sp = (g > 0)  # despill
rgb[..., 1] = np.where(sp, np.minimum(rgb[..., 1], np.maximum(rgb[..., 0], rgb[..., 2])), rgb[..., 1])
LOGO = Image.fromarray(np.dstack([rgb.clip(0, 255).astype(np.uint8), alpha]), "RGBA")
LOGO = LOGO.crop(LOGO.getbbox())

# (start, end, province, monk, temple, sheet-file keyword, crop(x0,y0,w), map pin)
SEG = [
 (0, 18, None, None, None, None, None, None),
 (18, 37, "นครราชสีมา", "หลวงพ่อคูณ", "วัดบ้านไร่", "หลวงปู่คูณ", (0, 0, 500), (614, 640)),
 (37, 56, "นครพนม", "หลวงปู่คำพันธ์", "วัดธาตุมหาชัย", "หลวงปู่คำพันธ์", (0, 0, 500), (848, 385)),
 (56, 68, "สุรินทร์", "หลวงปู่หงษ์", "สุสานทุ่งมน", "หลวงปู่หงส์", (0, 0, 405), (772, 690)),
 (68, 83, "ศรีสะเกษ", "หลวงปู่หมุน", "วัดบ้านจาน", "หลวงปู่หมุน", (0, 0, 405), (850, 692)),
 (83, 93, "ร้อยเอ็ด", "หลวงปู่ศรี", "วัดป่ากุง", "หลวงปู่ศรี", (50, 0, 480), (782, 560)),
 (93, 103, "ร้อยเอ็ด", "หลวงปู่ทองมา", "วัดสว่างท่าสี", "หลวงปู่ทองมา", (0, 0, 405), (800, 590)),
 (103, 117, "ขอนแก่น", "หลวงปู่ผาง", "วัดอุดมคงคาคีรีเขต", "หลวงปู่ผาง", (30, 0, 405), (613, 497)),
 (117, 128, "อุดรธานี", "หลวงตามหาบัว", "โครงการช่วยชาติ", "หลวงตามหาบัว", (0, 0, 500), (690, 372)),
 (128, 140.02, None, None, None, None, None, None),
]
PINS = [s[7] for s in SEG if s[7]]
EASE = lambda x: x * x * (3 - 2 * x)
clamp = lambda x: max(0.0, min(1.0, x))

# static layers
def bg():
    y = np.linspace(0, 1, H)[:, None, None]
    c = np.array([22, 17, 10]) * (1 - y) + np.array([40, 28, 12]) * y
    im = Image.fromarray(np.broadcast_to(c, (H, W, 3)).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((16, 16, W - 17, H - 17), 28, outline=GOLD, width=3)
    return im
BG = bg()
MAPY0, MAPH = 330, 760
card_cache = {}
def card(seg):
    if seg in card_cache: return card_cache[seg]
    _, _, prov, monk, temple, key, (x0, y0, w), _ = SEG[seg]
    f = [p for p in glob.glob(f"{ROOT}/*.png") if key in p][0]
    im = Image.open(f).convert("RGB").crop((x0, y0, x0 + w, y0 + int(w * 4 / 3)))
    im = im.resize((520, 693), Image.LANCZOS)
    card_cache[seg] = im; return im

def cam(seg_i, t):
    """camera center/scale in map coords"""
    def target(i):
        s = SEG[i]
        if s[7] is None:
            return (512, 768, 0.48) if i in (0,) else (600, 640, 0.48)
        return (s[7][0], s[7][1], 1.45)
    i = seg_i; s0 = SEG[i][0]
    cur = target(i)
    if i == 0:
        e = EASE(clamp((t - 9) / 8.0))
        a, b = target(0), (740, 560, 1.15)
        return tuple(a[k] + (b[k] - a[k]) * e for k in range(3))
    prev = target(i - 1) if i > 1 else (740, 560, 1.15)
    if i == len(SEG) - 1:
        prev = target(i - 1); cur = (512, 768, 0.48)
        e = EASE(clamp((t - s0) / 3.0)); return tuple(prev[k] + (cur[k] - prev[k]) * e for k in range(3))
    e = EASE(clamp((t - s0) / 1.4))
    return tuple(prev[k] + (cur[k] - prev[k]) * e for k in range(3))

def draw_map(img, t, seg_i):
    cx, cy, s = cam(seg_i, t)
    vw, vh = W / s, MAPH / s
    box = (cx - vw / 2, cy - vh / 2, cx + vw / 2, cy + vh / 2)
    panel = Image.new("RGBA", (W, MAPH), (26, 21, 12, 255))
    # crop with padding handled by transform
    tile = MAP.transform((W, MAPH), Image.AFFINE,
        (vw / W, 0, box[0], 0, vh / MAPH, box[1]), resample=Image.BICUBIC)
    panel.alpha_composite(tile)
    d = ImageDraw.Draw(panel)
    sx = lambda p: ((p[0] - box[0]) * s, (p[1] - box[1]) * s)
    visited = []
    for i, sg in enumerate(SEG):
        if sg[7] is None: continue
        if t >= sg[0]: visited.append((i, sg[7]))
    pts = [sx(p) for _, p in visited]
    if len(pts) > 1:
        # route line, last leg animates
        for k in range(1, len(pts)):
            a, b = pts[k - 1], pts[k]
            si = visited[k][0]
            f = EASE(clamp((t - SEG[si][0]) / 1.4)) if k == len(pts) - 1 else 1
            b = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            d.line([a, b], fill=(200, 40, 40, 255), width=max(4, int(4 * s)))
    for k, (i, p) in enumerate(visited):
        x, y = sx(p); y -= 22 * s ** 0.5; cur = (k == len(visited) - 1) and seg_i == i
        r = 9 * s + 4
        if cur:
            ph = (t * 1.2) % 1
            rr = r * (1.5 + 2.5 * ph)
            d.ellipse((x - rr, y - rr, x + rr, y + rr), outline=(255, 70, 60, int(220 * (1 - ph))), width=4)
            pin_r = r * 1.1
            d.ellipse((x - pin_r, y - pin_r, x + pin_r, y + pin_r), fill=(215, 35, 35, 255), outline=(255, 255, 255, 255), width=4)
        else:
            d.ellipse((x - r, y - r, x + r, y + r), fill=(233, 196, 106, 255), outline=(90, 50, 10, 255), width=3)
    # vignette edges
    img.paste(panel.convert("RGB"), (0, MAPY0))
    ImageDraw.Draw(img).rounded_rectangle((0, MAPY0, W - 1, MAPY0 + MAPH), 0, outline=GOLD, width=3)

def center_text(d, y, txt, fnt, fill, x=W // 2):
    d.text((x, y), txt, font=fnt, fill=fill, anchor="ma")

def frame(n):
    t = n / FPS
    seg_i = max(i for i, s in enumerate(SEG) if t >= s[0])
    img = BG.copy()
    draw_map(img, t, seg_i)
    d = ImageDraw.Draw(img)
    # header logo
    lw = 640 if seg_i not in (0, len(SEG) - 1) else 640
    lg = LOGO.resize((lw, int(LOGO.height * lw / LOGO.width)), Image.LANCZOS)
    big = seg_i in (0, len(SEG) - 1)
    if not big:
        lg2 = LOGO.resize((380, int(LOGO.height * 380 / LOGO.width)), Image.LANCZOS)
        img.paste(lg2, ((W - 380) // 2, 36), lg2)
    else:
        # fade title in/out
        sg = SEG[seg_i]
        a = EASE(clamp((t - sg[0]) / 1.2)) if seg_i == 0 else EASE(clamp((t - sg[0] - 2) / 1.2))
        if seg_i == 0: a *= 1 - EASE(clamp((t - 14) / 3))
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        big_lg = LOGO.resize((900, int(LOGO.height * 900 / LOGO.width)), Image.LANCZOS)
        ly = 40 if seg_i == 0 else 330
        lay.paste(big_lg, ((W - 900) // 2, ly), big_lg)
        if seg_i == 0:
            ld = ImageDraw.Draw(lay)
        r, g, b, al = lay.split(); al = al.point(lambda v: int(v * a)); lay.putalpha(al)
        img.paste(lay, (0, 0), lay)
        if seg_i == 0:
            # small logo persists subtly after intro fade
            a2 = EASE(clamp((t - 15) / 3))
            lg2 = LOGO.resize((380, int(LOGO.height * 380 / LOGO.width)), Image.LANCZOS)
            r_, g_, b_, al2 = lg2.split(); lg2.putalpha(al2.point(lambda v: int(v * a2)))
            img.paste(lg2, ((W - 380) // 2, 36), lg2)
    # lower panel
    sg = SEG[seg_i]
    if sg[2]:
        k = EASE(clamp((t - sg[0]) / 0.7))
        c = card(seg_i)
        zoom = 1 + 0.05 * (t - sg[0]) / (sg[1] - sg[0])
        cw, ch = c.size
        c2 = c.crop((int(cw * (1 - 1 / zoom) / 2), 0, int(cw - cw * (1 - 1 / zoom) / 2), int(ch / zoom))).resize((cw, ch))
        lay = Image.new("RGBA", (W, 720), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        ld.rounded_rectangle((38, 0, 38 + cw + 4, ch + 4), 18, fill=GOLD)
        lay.paste(c2, (40, 2))
        tx = 610
        ld.text((tx, 40), "จังหวัด", font=font(34, False), fill=GOLD)
        ld.text((tx, 90), sg[2], font=font(58), fill=CREAM)
        ld.line((tx, 190, tx + 390, 190), fill=GOLD, width=3)
        # monk name (wrap)
        name = sg[3]
        fs = 56 if len(name) <= 10 else 46
        ld.text((tx, 215), name, font=font(fs), fill=GOLD)
        ld.text((tx, 215 + fs + 28), sg[4], font=font(40, False), fill=CREAM)
        ld.text((tx, ch - 10), "ภาพประกอบ", font=font(26, False), fill=(190, 160, 100))
        al = lay.split()[3].point(lambda v: int(v * k)); lay.putalpha(al)
        img.paste(lay, (int((1 - k) * 60), 1110), lay)
    if not sg[2]:
        fa = EASE(clamp((t - sg[0] - 1) / 1.5))
        ly = Image.new("RGBA", (W, 700), (0, 0, 0, 0)); ld = ImageDraw.Draw(ly)
        if seg_i == 0:
            ld.text((W // 2, 40), "ตามรอยพระเกจิอีสาน", font=font(76), fill=GOLD, anchor="ma")
            ld.text((W // 2, 160), "๘ รูป ๗ จังหวัด", font=font(52, False), fill=CREAM, anchor="ma")
            ld.text((W // 2, 280), "นครราชสีมา · นครพนม · สุรินทร์ · ศรีสะเกษ", font=font(34, False), fill=GOLD, anchor="ma")
            ld.text((W // 2, 335), "ร้อยเอ็ด · ขอนแก่น · อุดรธานี", font=font(34, False), fill=GOLD, anchor="ma")
        else:
            names = [(s_[3], s_[4]) for s_ in SEG if s_[3]]
            ld.text((W // 2, 10), "พระเกจิอีสาน ๘ รูป", font=font(58), fill=GOLD, anchor="ma")
            for i, (m, tm) in enumerate(names):
                ld.text((90 + (i % 2) * 480, 120 + (i // 2) * 120), m, font=font(40), fill=CREAM)
                ld.text((90 + (i % 2) * 480, 120 + (i // 2) * 120 + 52), tm, font=font(28, False), fill=GOLD)
        ly.putalpha(ly.split()[3].point(lambda v: int(v * fa)))
        img.paste(ly, (0, 1130), ly)
    center_text(d, H - 70, "มหานิยม ๙ ทั่วไทย", font(26, False), (170, 140, 80))
    return img

def work(n):
    frame(n).save(f"{OUT}/f{n:05d}.jpg", quality=92)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        for t in sys.argv[2:]: frame(int(float(t) * FPS)).save(f"{OUT}/prev_{t}.png")
    else:
        N = int(DUR * FPS)
        with Pool(os.cpu_count()) as p: p.map(work, range(N), chunksize=20)
