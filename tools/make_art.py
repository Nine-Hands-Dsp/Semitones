"""
SEMITONES (NINE HANDS | DSP) - plugin artwork, simple pedal format (rule of nov. 2026, as Screamer):

  * background.png = 1235 x 1800 px RGBA, pedal full frame (the enclosure touches the top and the bottom of the
    picture, the jacks touch the left and right edges), transparent only around the jacks and in the rounded corners;
  * plugin frame = picture / 2 = 617.5 x 900 units (1 unit = 2 px) ; everything is drawn directly in that frame;
  * brushed steel from the user's texture (assets/kit/brushed_texture.jpg), one light for the whole pedal (top left);
  * parts from the user's photos: footswitch, IN / OUT jacks, corner screws (assets/kit, tools/photo_controls.py);
  * vector reliefs (height field + lighting + soft ray-marched shadows): embossed arrow on the shell, raised arrow
    buttons, raised MIX / FAV push buttons, preset slot, display window;
  * texts, name, IN / OUT and the NINE HANDS | DSP logo are drawn by the plugin (vector, Poppins / NineHandsLogo.svg),
    never printed in the picture.

Outputs (repo root unless noted):
  background.png, assets/seg_*.png (lit dot-matrix segments + MIX dot), assets/ringlight.png, assets/ring_lit.png,
  assets/ring_comet.png, assets/press_*.png, Layout.h, tools/layout.json, design/preview_art.png
Usage: python3 -I tools/make_art.py
"""
import os, sys, json, math, io
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import photo_controls as PC
import svg_raster

KIT = os.path.join(ROOT, 'assets', 'kit')
ASSETS = os.path.join(ROOT, 'assets')
AS = 2                                        # px per unit
FW, FH = 617.5, 900.0                         # plugin frame (units)
W, H = int(FW * AS), int(FH * AS)             # 1235 x 1800
RNG = np.random.default_rng(7)

L = np.array([-0.46, -0.62, 0.64]); L /= np.linalg.norm(L)          # towards the light (x right, y down, z to the eye)
V = np.array([0.0, 0.0, 1.0]); HV = (L + V) / np.linalg.norm(L + V)

# ------------------------------------------------------------------ geometry (units)
JACK_LEN = 105.0
_j = PC.jack_sprite(JACK_LEN, AS, 1)
JP = math.floor(_j[1] * AS) / AS                       # jack protrusion out of the enclosure
BX0, BX1 = JP, FW - JP                                 # enclosure
BY0, BY1 = 0.0, FH
BW = BX1 - BX0
CX = FW / 2
RAD = 30.0                                             # corner radius
EDGE = 9.0                                             # rounded edge of the enclosure
JACK_Y = 438.0                                         # IN / OUT height of the original artwork

# positions measured on the validated v1.4.1 artwork (source px -> units of the new frame)
def U(ox, oy):
    return (BX0 + (ox - 230.0) * BW / 1054.0, (oy - 26.0) * FH / 1805.0)

SCREW_R = 13.0
SCREWS = [(BX0 + 42, 28), (BX1 - 42, 28), (BX0 + 42, FH - 30), (BX1 - 42, FH - 30)]
PRESET = (CX - 150.0, 30.0, 300.0, 34.0)               # preset menu (drawn by the plugin) in a milled slot
GLASS = (CX - 156.0, 94.0, 312.0, 244.0)               # display window (x, y, w, h)
ARROW_W, ARROW_H = 54.0, 62.0
ARROW_L = (97.0, 216.0)
ARROW_R = (FW - ARROW_L[0], ARROW_L[1])
BTN_R = 20.0
MIX = (CX - 112.0, 389.0)
FAV = (CX + 112.0, 389.0)
LABEL_Y = 354.0
FS = (CX, 752.0)
FS_FLATS = 112.0
RING_R, RING_W = 88.0, 9.0                             # footswitch light ring (thinner than v1.4: 14)
LOGO_C, LOGO_W = (CX, 876.0), 168.0
NAME_C = (CX - 8.0, 536.0)                             # engraved name (drawn by the plugin)
SUB_Y = NAME_C[1] + 31.0
# embossed band of the shell (the stepped "arrow" of the original), centre lines in source px
BANDS = [[U(150, 520), U(1030, 1010), U(1030, 1240), U(575, 1240), U(1340, 1880)],
         [U(421, -40), U(421, 200), U(150, 400)],
         [U(990, 618), U(1360, 795)]]
BAND_W = 22.0

# dot-matrix display (glyph boxes in units)
DOT_PITCH = 2.9
SEG_T = 9.6
SEG_GAP = 1.3
DY0 = GLASS[1] + 70.0
DY1 = DY0 + 76.0
YT, YM, YB = DY0, (DY0 + DY1) / 2, DY1
DIG_W = 52.0
UNITS_X = (CX + 80.0 - DIG_W / 2, CX + 80.0 + DIG_W / 2)
TENS_X = (CX + 4.0 - DIG_W / 2, CX + 4.0 + DIG_W / 2)
SIGN_C = (CX - 82.0, YM)
SIGN_H, SIGN_V = 25.0, 23.0
MIX_DOT = (MIX[0], GLASS[1] + GLASS[3] - 20.0, 4.6)
STATUS = (CX - 90.0, GLASS[1] + GLASS[3] - 34.0, 180.0, 26.0)

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
UX, UY = (XX + 0.5) / AS, (YY + 0.5) / AS


# ------------------------------------------------------------------ helpers
def gblur(a, s):
    return cv2.GaussianBlur(a, (0, 0), s) if s > 0 else a


def shift(a, dx, dy):
    return cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (a.shape[1], a.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def rrect_sd(x, y, x0, y0, w, h, r):
    cx, cy = x0 + w / 2, y0 + h / 2
    qx = np.abs(x - cx) - (w / 2 - r); qy = np.abs(y - cy) - (h / 2 - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def poly_mask(poly, ss=4):
    im = Image.new('L', (W * ss, H * ss), 0)
    ImageDraw.Draw(im).polygon([(x * AS * ss, y * AS * ss) for x, y in poly], fill=255)
    return np.asarray(im.resize((W, H), Image.BOX)).astype(np.float32) / 255.0


def sd_from_mask(m):
    """signed distance (units, negative inside) of a binary-ish mask"""
    b = (m > 0.5).astype(np.uint8)
    din = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - b, cv2.DIST_L2, 5)
    return (dout - din) / AS


def srgb2lin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def over(dst, rgba, x0, y0):
    h, w = rgba.shape[:2]
    X0, Y0 = int(round(x0)), int(round(y0))
    sx0, sy0 = max(0, -X0), max(0, -Y0)
    ex, ey = min(w, W - X0), min(h, H - Y0)
    a = rgba[sy0:ey, sx0:ex, 3:4] / 255.0
    reg = dst[Y0 + sy0:Y0 + ey, X0 + sx0:X0 + ex]
    dst[Y0 + sy0:Y0 + ey, X0 + sx0:X0 + ex] = reg * (1 - a) + rgba[sy0:ey, sx0:ex, :3] * a


# ------------------------------------------------------------------ 1. height field (units)
sd_body = rrect_sd(UX, UY, BX0, BY0, BW, FH, RAD)
body = np.clip(0.5 - sd_body * AS, 0, 1)
d_in = np.clip(-sd_body, 0, None)
h = np.where(d_in < EDGE, np.sqrt(np.clip(EDGE ** 2 - (EDGE - d_in) ** 2, 0, None)), EDGE).astype(np.float32)
TOP = EDGE

# embossed band (raised 2.8 units, 2.2-unit bevel), mitred corners
def band_poly(p0, p1, w):
    d = np.array(p1) - np.array(p0); d /= np.linalg.norm(d); n = np.array([-d[1], d[0]]) * w / 2
    return [tuple(np.array(p0) + n), tuple(np.array(p1) + n), tuple(np.array(p1) - n), tuple(np.array(p0) - n)]

def seg_box_sd(p0, p1, w):
    """exact signed distance to the rectangle p0-p1 (width w), extended by w/2 at both ends (square joints)"""
    p0, p1 = np.array(p0), np.array(p1)
    d = p1 - p0; L_ = np.linalg.norm(d); d /= L_
    mx, my = (p0 + p1) / 2
    u = (UX - mx) * d[0] + (UY - my) * d[1]
    v = -(UX - mx) * d[1] + (UY - my) * d[0]
    qx = np.abs(u) - (L_ / 2 + w / 2); qy = np.abs(v) - w / 2
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)

def mitre_poly(path, w):
    P = [np.array(p, float) for p in path]
    nrm = []
    for a_, b_ in zip(P[:-1], P[1:]):
        d = (b_ - a_) / np.linalg.norm(b_ - a_); nrm.append(np.array([-d[1], d[0]]))
    off = [nrm[0] * w / 2]
    for n1, n2 in zip(nrm[:-1], nrm[1:]):
        off.append((n1 + n2) / (1 + n1 @ n2) * w / 2)
    off.append(nrm[-1] * w / 2)
    return [tuple(p + o) for p, o in zip(P, off)] + [tuple(p - o) for p, o in zip(P[::-1], off[::-1])]


def poly_sd(poly):
    """exact signed distance (units) to a polygon"""
    dmin = np.full((H, W), 1e9, np.float32)
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        ex, ey = x1 - x0, y1 - y0
        t = np.clip(((UX - x0) * ex + (UY - y0) * ey) / (ex * ex + ey * ey), 0, 1)
        dmin = np.minimum(dmin, np.hypot(UX - x0 - t * ex, UY - y0 - t * ey))
    inside = poly_mask(poly, 2) > 0.5
    return np.where(inside, -dmin, dmin)


sd_ar = np.full((H, W), 1e3, np.float32)
for path in BANDS:
    sd_ar = np.minimum(sd_ar, poly_sd(mitre_poly(path, BAND_W)))
arrow_m = np.clip(0.5 - sd_ar * AS, 0, 1)
EMB, BEV = 3.4, 3.2
emb = np.clip(-sd_ar / BEV, 0, 1)
emb = emb * emb * (3 - 2 * emb) * EMB
h = h + emb * np.clip((d_in - 1.0) / EDGE, 0, 1)

# preset slot (milled, -2.4) and display window (recessed -3.5, chamfered bezel)
sd_slot = rrect_sd(UX, UY, *PRESET, PRESET[3] / 2)
h -= np.clip(-sd_slot / 1.2, 0, 1) * 2.4
gx_, gy_, gw_, gh_ = GLASS
sd_gl = rrect_sd(UX, UY, gx_ - 6, gy_ - 6, gw_ + 12, gh_ + 12, 14)
h -= np.clip(-sd_gl / 3.0, 0, 1) * 3.5                           # chamfer down to the glass
glass_m = np.clip(0.5 - rrect_sd(UX, UY, gx_, gy_, gw_, gh_, 9) * AS, 0, 1)

# raised arrow buttons (chamfered triangles)
def tri(c, pointing):
    x, y = c; s = 1 if pointing > 0 else -1
    return [(x - s * ARROW_W / 2, y - ARROW_H / 2), (x + s * ARROW_W / 2, y), (x - s * ARROW_W / 2, y + ARROW_H / 2)]

def rtri_mask(c_, s_, r=6.0):
    sdm = sd_from_mask(poly_mask(tri(c_, s_)))
    return sd_from_mask((sdm < -r).astype(np.float32)) - r          # rounded corners

for c_, s_ in ((ARROW_L, -1), (ARROW_R, 1)):
    sdm = rtri_mask(c_, s_)
    t_ = np.clip(-sdm / 8.0, 0, 1)
    raise_ = (t_ * (2 - t_)) * 5.5                                  # 6-unit rounded chamfer, 4.5 high
    h = h + raise_
# raised push buttons MIX / FAV: collar recess + domed cap
for (bx, by) in (MIX, FAV):
    r = np.hypot(UX - bx, UY - by)
    collar = np.clip((BTN_R + 4.0 - r) / 1.2, 0, 1) * np.clip((r - BTN_R + 0.5) / 1.2, 0, 1)
    h -= collar * 1.6
    capr = np.clip(BTN_R - r, 0, None)
    cap = np.where(r < BTN_R, 4.6 + 1.6 * (1 - (r / BTN_R) ** 2) - np.clip(1 - capr / 1.6, 0, 1) * 2.0, 0)
    h = np.where(r < BTN_R, h + cap, h)

h = gblur(h.astype(np.float32), 0.6)

# ------------------------------------------------------------------ 2. shading
gy, gx = np.gradient(h, 1.0 / AS)
nrm = np.dstack([-gx, -gy, np.ones_like(h)]); nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
ndl = np.clip(nrm @ L, 0, 1); ndh = np.clip(nrm @ HV, 0, 1)

# soft shadows (ray march in the height field towards the light, small area light)
print('shadows...')
lxy = L[:2] / np.linalg.norm(L[:2]); slope = L[2] / np.linalg.norm(L[:2])
hm = h.astype(np.float32)
vis = np.zeros((H, W), np.float32)
for jit in (-0.12, 0.0, 0.12):
    ca, sa = math.cos(jit), math.sin(jit)
    dx_, dy_ = lxy[0] * ca - lxy[1] * sa, lxy[0] * sa + lxy[1] * ca
    occ = np.zeros((H, W), np.float32)
    for t in np.linspace(0.5, 22.0, 30):                       # units
        hs = cv2.remap(hm, (XX + dx_ * t * AS).astype(np.float32), (YY + dy_ * t * AS).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        occ = np.maximum(occ, np.clip((hs - (hm + t * slope * 1.0)) / 0.8, 0, 1))
    vis += 1 - occ
vis = gblur(vis / 3.0, 1.0)

# brushed steel albedo (user texture, horizontal brushing)
tex = np.asarray(Image.open(os.path.join(KIT, 'brushed_texture.jpg')).convert('L')).astype(np.float32)
tex = cv2.resize(tex, (int(tex.shape[1] * 0.62), int(tex.shape[0] * 0.62)), interpolation=cv2.INTER_AREA)
tex = tex[:H, :W]
tex = tex / tex.mean()
# the brushing reflects the light as a soft horizontal sheen (anisotropic): a bit brighter on the upper left
sheen = np.exp(-(((UX - 150) / 380) ** 2 + ((UY - 160) / 520) ** 2))
steel = 136.0 * tex
col = (steel * (0.34 + 0.86 * ndl) * (0.58 + 0.42 * vis))[..., None] * np.array([0.985, 0.995, 1.02])
col += ((95 * ndh ** 40 + 30 * ndh ** 8) * vis)[..., None]
col += (16.0 * sheen * tex)[..., None]
# fine grain + very low frequency variation (no flat CGI look)
lowf = gblur(RNG.normal(0, 1, (H // 16 + 1, W // 16 + 1)).astype(np.float32), 2)
lowf = cv2.resize(lowf, (W, H), interpolation=cv2.INTER_CUBIC)
col *= (1 + 0.025 * lowf)[..., None]

# MIX / FAV caps: turned metal (concentric lathe marks, bow-tie reflection towards the light)
thL = math.atan2(L[1], L[0])
for (bx, by) in (MIX, FAV):
    dxb, dyb = UX - bx, UY - by
    r = np.hypot(dxb, dyb)
    capm = np.clip((BTN_R - 0.3 - r) * AS, 0, 1)
    th = np.arctan2(dyb, dxb)
    lathe = 1 + 0.05 * np.sin(r * 9.0) + 0.04 * np.sin(r * 23.0 + 1.3)
    bow = np.clip(np.cos(2 * (th - thL)), 0, 1) ** 3
    base = 112.0 * lathe * (0.40 + 0.75 * ndl) * (0.55 + 0.45 * vis)
    capc = base[..., None] + ((70 * bow * np.clip(r / BTN_R, 0.2, 1) + 120 * ndh ** 60) * vis)[..., None]
    col = col * (1 - capm[..., None]) + capc * np.array([0.99, 1.0, 1.02]) * capm[..., None]

# milled surfaces are darker and matte (slot floor, arrow-button chamfers stay brushed)
slot_floor = np.clip(-sd_slot / 1.2, 0, 1)
col = col * (1 - 0.45 * slot_floor[..., None])

# display: black glass, faint diagonal reflection, empty dot grid (very faint)
glass = np.full((H, W, 3), 0.0, np.float32) + np.array([8.0, 10.0, 14.0])
refl = np.clip(1 - np.abs((UX - gx_) * 0.55 + (UY - gy_) - 70) / 46, 0, 1) * np.clip(1 - (UY - gy_) / gh_, 0, 1)
glass += (refl * 12)[..., None]
P = DOT_PITCH
fxg = np.mod(UX - gx_, P) - P / 2; fyg = np.mod(UY - gy_, P) - P / 2
dots = np.clip((0.40 * P - np.hypot(fxg, fyg)) * AS / 0.8 + 0.5, 0, 1)
glass += (dots * 3.2)[..., None] * np.array([0.6, 0.8, 1.0])
inner_sh = np.clip(1 - (rrect_sd(UX, UY, gx_, gy_, gw_, gh_, 9) * -1) / 7, 0, 1)    # inner shadow at the edges
glass *= (1 - 0.6 * inner_sh * glass_m)[..., None]
col = col * (1 - glass_m[..., None]) + glass * glass_m[..., None]
img = col * body[..., None]

# ------------------------------------------------------------------ 3. dot-matrix: ghost + lit sprites
GHOST = np.array([12.0, 30.0, 66.0])
LIT = np.array([40.0, 168.0, 255.0]) / 255.0
LIT_CORE = np.array([170.0, 235.0, 255.0]) / 255.0


def hseg(x1, x2, y, t=SEG_T, g=SEG_GAP):
    hh = t / 2
    return [(x1 + g, y), (x1 + g + hh, y - hh), (x2 - g - hh, y - hh), (x2 - g, y), (x2 - g - hh, y + hh), (x1 + g + hh, y + hh)]


def vseg(x, y1, y2, t=SEG_T, g=SEG_GAP):
    hh = t / 2
    return [(x, y1 + g), (x + hh, y1 + g + hh), (x + hh, y2 - g - hh), (x, y2 - g), (x - hh, y2 - g - hh), (x - hh, y1 + g + hh)]


def digit(xl, xr):
    return {'a': hseg(xl, xr, YT), 'b': vseg(xr, YT, YM), 'c': vseg(xr, YM, YB), 'd': hseg(xl, xr, YB),
            'e': vseg(xl, YM, YB), 'f': vseg(xl, YT, YM), 'g': hseg(xl, xr, YM)}


layers = []
cx_, cy_ = SIGN_C
for k, p in {'l': hseg(cx_ - SIGN_H, cx_ + SEG_T * 0.75, cy_), 'r': hseg(cx_ - SEG_T * 0.75, cx_ + SIGN_H, cy_),
             'u': vseg(cx_, cy_ - SIGN_V, cy_ + SEG_T * 0.75), 'd': vseg(cx_, cy_ - SEG_T * 0.75, cy_ + SIGN_V)}.items():
    layers.append(('s_' + k, p))
for k, p in digit(*TENS_X).items():
    layers.append(('t_' + k, p))
for k, p in digit(*UNITS_X).items():
    layers.append(('u_' + k, p))
seg_masks = {n: poly_mask(p) for n, p in layers}
ghost = np.zeros((H, W), np.float32)
for m_ in seg_masks.values():
    ghost = np.maximum(ghost, m_)
gd = ghost * dots * glass_m
img = img * (1 - gd[..., None] * 0.5) + GHOST * gd[..., None] * 0.5

sprites = {}


def save_rgba(name, rgb01, a, pad_px=0):
    ys, xs = np.nonzero(a > 0.004)
    x0, y0, x1, y1 = max(0, xs.min() - pad_px), max(0, ys.min() - pad_px), min(W, xs.max() + 1 + pad_px), min(H, ys.max() + 1 + pad_px)
    rgba = np.dstack([np.clip(rgb01[y0:y1, x0:x1], 0, 1), np.clip(a[y0:y1, x0:x1], 0, 1)])
    Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), 'RGBA').save(os.path.join(ASSETS, name + '.png'), optimize=True)
    sprites[name] = (x0 / AS, y0 / AS, (x1 - x0) / AS, (y1 - y0) / AS)


def lit_sprite(mask, name):
    md = mask * dots
    inner = gblur(cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1), 2.0)
    col_l = srgb2lin(LIT)[None, None, :] * md[..., None] * 1.25 + srgb2lin(LIT_CORE)[None, None, :] * (md * inner * 0.9)[..., None]
    glow = gblur(mask, 0.8 * AS) * 0.35 + gblur(mask, 2.5 * AS) * 0.75 + gblur(mask, 7.0 * AS) * 0.45 + gblur(mask, 16 * AS) * 0.18
    glow *= glass_m
    tot = col_l + srgb2lin(LIT)[None, None, :] * glow[..., None] * 0.75
    a_ = np.clip(md + glow * 0.85, 0, 1)
    out = lin2srgb(tot / np.maximum(a_[..., None], 1e-4))
    save_rgba(name, out, a_)


print('segments...')
for n, _ in layers:
    lit_sprite(seg_masks[n], 'seg_' + n)
lit_sprite(np.clip((MIX_DOT[2] - np.hypot(UX - MIX_DOT[0], UY - MIX_DOT[1])) * AS, 0, 1), 'seg_mix')

# ------------------------------------------------------------------ 4. photo parts: screws, footswitch, jacks
print('photo parts...')
def occlusion(mask, strength=1.0):
    contact = gblur(mask, 3.0 * AS); soft = gblur(mask, 12.0 * AS)
    o = np.clip(contact * 0.55 + soft * 0.8, 0, 0.5) * strength * (1.0 - mask)
    img[...] = img * (1.0 - o * body)[..., None]


scr = np.asarray(Image.open(os.path.join(KIT, 'screw_photo.png')).convert('RGBA')).astype(np.float32)
SCR_C, SCR_R = (138.5, 134.5), 130.0
for (sx, sy) in SCREWS:
    n = int(round(SCREW_R * 2 * 1.9 * AS))
    k = SCREW_R * AS / SCR_R
    rgb_, a_ = PC._place(scr, SCR_C, k, n)
    # counterbore around the head (dark ring) then the head with its shadow
    rr = np.hypot(UX - sx, UY - sy)
    bore = np.clip((SCREW_R + 2.2 - rr) * AS / 1.5, 0, 1)
    img *= (1 - 0.55 * bore * np.clip((rr - SCREW_R + 1.0) * AS / 2, 0, 1))[..., None]
    spr = PC._with_shadow(rgb_ * 0.95, a_, 2 * SCREW_R * AS, 0.55)
    over(img, spr, (sx - n / AS / 2) * AS, (sy - n / AS / 2) * AS)

# footswitch (user photo) + occlusion
FSPR, FSREFL, FS_SZ, FS_BTN = PC.footswitch_sprite(FS_FLATS, AS, refl_colour=(60.0, 160.0, 255.0))
fa = np.zeros((H, W), np.float32); n = FSPR.shape[0]
x0f, y0f = int(round((FS[0] - FS_SZ / 2) * AS)), int(round((FS[1] - FS_SZ / 2) * AS))
fa[y0f:y0f + n, x0f:x0f + n] = np.clip((FSPR[..., 3] / 255.0 - 0.86) / 0.14, 0, 1)
occlusion(fa, 0.9)
# light ring groove around the nut (unlit pipe: frosted, slightly cool), drawn before the nut
rr = np.hypot(UX - FS[0], UY - FS[1])
band = np.clip(RING_W / 2 - np.abs(rr - RING_R) + 0.5, 0, 1)
groove = np.clip(RING_W / 2 + 1.6 - np.abs(rr - RING_R) + 0.5, 0, 1)
img *= (1 - 0.55 * (groove - band))[..., None]
ang_l = ((-(UX - FS[0])) * 0.6 + (-(UY - FS[1])) * 0.8) / np.maximum(rr, 1e-3)
pipe_off = np.array([118.0, 124.0, 134.0]) * (0.82 + 0.18 * ang_l * np.sign(rr - RING_R + 1e-3))[..., None]
img = img * (1 - band[..., None]) + pipe_off * band[..., None]
over(img, FSPR, x0f, y0f)

alpha = body.copy()
for side in (-1, 1):
    j, jw_u, jh_u = PC.jack_sprite(JACK_LEN, AS, side)
    jh, jw = j.shape[:2]
    jw2 = int(round(JP * AS))
    if jw2 != jw:
        pm_ = j.copy(); pm_[..., :3] *= pm_[..., 3:4] / 255.0
        pm_ = cv2.resize(pm_, (jw2, jh), interpolation=cv2.INTER_AREA)
        a_ = np.maximum(pm_[..., 3:4], 1e-3); pm_[..., :3] = np.where(pm_[..., 3:4] > 0.5, pm_[..., :3] * 255.0 / a_, 0)
        j = pm_; jw = jw2
    x0j = 0 if side < 0 else int(round(BX1 * AS))
    y0j = int(round(JACK_Y * AS - jh / 2))
    occ = np.zeros((H, W), np.float32); occ[y0j:y0j + jh, x0j:x0j + jw] = j[..., 3] / 255.0
    o = np.clip(gblur(occ, 3 * AS) * 0.9 + gblur(occ, 16 * AS) * 1.4, 0, 0.55) * (0.25 if side > 0 else 0.12)
    img *= (1 - o * body)[..., None]
    over(img, j, x0j, y0j)
    ja = np.zeros((H, W), np.float32); ja[y0j:y0j + jh, x0j:x0j + jw] = j[..., 3] / 255.0
    alpha = np.maximum(alpha, ja)

FINAL = np.clip(img, 0, 255)
Image.fromarray(np.dstack([FINAL, alpha * 255]).astype(np.uint8), 'RGBA').save(os.path.join(ROOT, 'background.png'), optimize=True)

# ------------------------------------------------------------------ 5. light ring sprites (thinner ring + rotating light)
print('ring...')
COL = srgb2lin(LIT)
CORE = srgb2lin(np.array([70.0, 185.0, 255.0]) / 255.0)
F01 = FINAL / 255.0
dr = np.abs(rr - RING_R)
src_h = 5.5 * AS
E = (src_h ** 2 / ((dr * AS) ** 2 + src_h ** 2)) ** 1.5
illum = np.clip(E * 1.35 + np.exp(-dr / 26.0) * 0.32, 0, 3) * (1 - band) * body
lin = srgb2lin(F01)
target = lin2srgb(lin + lin * illum[..., None] * COL * 2.4 + illum[..., None] * COL * 0.025)
diff = np.clip(target - F01, 0, 1)
a_ = np.clip(np.maximum(diff.max(2) / np.maximum(1 - F01.max(2), 0.02), diff.max(2) * 3.0), 0, 1)
colr = np.clip(np.where(a_[..., None] > 1e-4, (target - F01 * (1 - a_[..., None])) / np.maximum(a_[..., None], 1e-4), 0), 0, 1)
box = int((RING_R + 60) * AS)
bx_, by_ = int(FS[0] * AS) - box, int(FS[1] * AS) - box
def save_box(name, rgb, a):
    rgba = np.dstack([rgb, a])[by_:by_ + 2 * box, bx_:bx_ + 2 * box]
    Image.fromarray((np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8), 'RGBA').save(os.path.join(ASSETS, name + '.png'), optimize=True)
    sprites[name] = (bx_ / AS, by_ / AS, 2 * box / AS, 2 * box / AS)
save_box('ringlight', colr, a_)
relc = 1 - np.clip(dr / (RING_W / 2), 0, 1)
pipe = lin2srgb(CORE[None, None, :] * (relc ** 1.5)[..., None] * 0.55 + COL[None, None, :] * (0.55 + 0.35 * (1 - relc))[..., None])
bloom = gblur(band, 2.2 * AS) * 0.5 + gblur(band, 6 * AS) * 0.26
a_h = np.clip(bloom * (1 - band), 0, 1)
o_rgb = pipe * band[..., None] + lin2srgb(COL)[None, None, :] * a_h[..., None]
o_a = np.clip(band + a_h, 0, 1)
o_rgb = np.where(o_a[..., None] > 1e-4, o_rgb / np.maximum(o_a[..., None], 1e-4), 0)
save_box('ring_lit', o_rgb, o_a)
# comet: square sprite centred on the ring centre, rotated by the editor
half = int(math.ceil((RING_R + 40) * AS)); cs = 2 * half
yy2, xx2 = np.mgrid[0:cs, 0:cs].astype(np.float32)
ddx, ddy = (xx2 + 0.5 - half) / AS, (yy2 + 0.5 - half) / AS
d2 = np.hypot(ddx, ddy)
phi = np.mod(-np.arctan2(ddy, ddx), 2 * np.pi)
prof = np.exp(-phi / 3.4) * np.clip(1 - phi / (1.9 * np.pi), 0, 1) ** 1.15
headp = np.exp(-(phi / 0.38) ** 2) + np.exp(-((2 * np.pi - phi) / 0.14) ** 2) * 0.8
band2 = np.clip(RING_W / 2 - np.abs(d2 - RING_R) + 0.5, 0, 1)
src_ = band2 * np.clip(prof + 0.8 * headp, 0, 1.6)
def nblur(a, sg):
    return gblur(a, sg) / (gblur(band2, sg).max() + 1e-6)
core = nblur(src_, 1.6 * AS)
halo = nblur(src_, 4.5 * AS) * 0.85 + nblur(src_, 10 * AS) * 0.6 + nblur(src_, 20 * AS) * 0.4 + nblur(band2 * headp, 14 * AS) * 0.55
hot = nblur(band2 * headp, 2.0 * AS)
blue_s, pipe_c, hot_s = lin2srgb(COL * 1.15), lin2srgb(COL * 1.6 + CORE * 0.35), lin2srgb(CORE * 1.6)
wc, wh, wt = np.clip(core * 1.05, 0, 1), np.clip(halo * 0.68, 0, 0.85), np.clip(hot * 0.9, 0, 1)
ca = np.clip(1 - (1 - wc) * (1 - wh), 0, 1)
crgb = (pipe_c * wc[..., None] + blue_s * (wh * (1 - wc))[..., None]) / np.maximum(ca[..., None], 1e-4)
crgb = np.clip(crgb * (1 - wt[..., None]) + hot_s * wt[..., None], 0, 1)
Image.fromarray((np.dstack([crgb, ca]) * 255 + 0.5).astype(np.uint8), 'RGBA').save(os.path.join(ASSETS, 'ring_comet.png'), optimize=True)
sprites['ring_comet'] = (FS[0] - half / AS, FS[1] - half / AS, cs / AS, cs / AS)

# ------------------------------------------------------------------ 6. pressed states
def save_pressed(name, cx, cy, rad_u, mask, push=1.0, dark=0.82):
    X0, Y0 = int((cx - rad_u) * AS), int((cy - rad_u) * AS)
    n_ = int(2 * rad_u * AS)
    reg = FINAL[Y0:Y0 + n_, X0:X0 + n_] / 255.0
    mk = mask[Y0:Y0 + n_, X0:X0 + n_]
    moved = cv2.warpAffine(reg, np.float32([[1, 0, push * AS * 0.6], [0, 1, push * AS]]), (n_, n_), borderMode=cv2.BORDER_REPLICATE)
    o = reg * (1 - mk[..., None]) + moved * dark * mk[..., None]
    a = np.clip(gblur(mk, 1.0) * 1.15, 0, 1)
    Image.fromarray((np.dstack([np.clip(o, 0, 1), a]) * 255 + 0.5).astype(np.uint8), 'RGBA').save(os.path.join(ASSETS, 'press_' + name + '.png'), optimize=True)
    sprites['press_' + name] = (X0 / AS, Y0 / AS, n_ / AS, n_ / AS)

for nm, c_, s_ in (('left', ARROW_L, -1), ('right', ARROW_R, 1)):
    m_ = np.clip((0.8 - rtri_mask(c_, s_)) * AS / 1.2, 0, 1)
    save_pressed(nm, c_[0], c_[1], 40.0, m_)
for nm, (bx, by) in (('mix', MIX), ('fav', FAV)):
    m_ = np.clip((BTN_R + 0.8 - np.hypot(UX - bx, UY - by)) * AS / 1.2, 0, 1)
    save_pressed(nm, bx, by, BTN_R + 8, m_, push=0.9, dark=0.80)
# footswitch: button sinks, darker while pressed (Echoflex behaviour)
rad_u = FS_FLATS * 0.62
X0, Y0, n_ = int((FS[0] - rad_u) * AS), int((FS[1] - rad_u) * AS), int(2 * rad_u * AS)
reg = FINAL[Y0:Y0 + n_, X0:X0 + n_] / 255.0
cxl, cyl = FS[0] * AS - X0, FS[1] * AS - Y0
M = cv2.getRotationMatrix2D((cxl, cyl), 0, 0.97); M[1, 2] += 1.2 * AS
sunk = cv2.warpAffine(reg, M, (n_, n_), borderMode=cv2.BORDER_REPLICATE)
yyp, xxp = np.mgrid[0:n_, 0:n_].astype(np.float32)
rp = np.hypot(xxp - cxl, yyp - cyl) / AS
chm = np.clip((FS_BTN + 6 - rp) * AS / 1.5, 0, 1)
o = reg * (1 - chm[..., None]) + sunk * chm[..., None]
o *= (1 - 0.25 * np.clip((FS_BTN * 1.02 - rp) * AS / 2, 0, 1))[..., None]
Image.fromarray((np.dstack([np.clip(o, 0, 1), chm]) * 255 + 0.5).astype(np.uint8), 'RGBA').save(os.path.join(ASSETS, 'press_foot.png'), optimize=True)
sprites['press_foot'] = (X0 / AS, Y0 / AS, n_ / AS, n_ / AS)

# ------------------------------------------------------------------ 7. Layout.h + layout.json
lay = {'frame': [FW, FH], 'box': [BX0, BY0, BW, FH], 'preset': list(PRESET), 'glass': list(GLASS), 'status': list(STATUS),
       'arrowL': [ARROW_L[0] - 34, ARROW_L[1] - 40, 68, 80], 'arrowR': [ARROW_R[0] - 34, ARROW_R[1] - 40, 68, 80],
       'mix': [MIX[0], MIX[1], BTN_R + 6], 'fav': [FAV[0], FAV[1], BTN_R + 6], 'foot': [FS[0], FS[1], RING_R + 4],
       'ring': [FS[0], FS[1]], 'jackY': JACK_Y, 'jackL': BX0 / 2, 'jackR': BX1 + JP / 2,
       'labelY': LABEL_Y, 'name': list(NAME_C), 'subY': SUB_Y, 'logo': [LOGO_C[0], LOGO_C[1], LOGO_W],
       'outX': BX0 + 26, 'inX': BX1 - 26, 'sprites': sprites}
json.dump(lay, open(os.path.join(HERE, 'layout.json'), 'w'), indent=1)
F = lambda v: f"{v:.2f}f"
with open(os.path.join(ROOT, 'Layout.h'), 'w') as fo:
    fo.write(f"// generated by tools/make_art.py - do not edit (frame {FW} x {FH} units = background.png {W} x {H} px / 2)\n")
    fo.write("#pragma once\nnamespace Layout {\nstruct R { float x, y, w, h; };\nstruct Sprite { const char* name; R r; };\n")
    fo.write(f"constexpr float W = {F(FW)}, H = {F(FH)};\n")
    fo.write(f"constexpr int defaultH = 493, defaultW = {int(round(493 * FW / FH))};   // \"60 %\"\n")
    fo.write("constexpr Sprite sprites[] = {\n" + ",\n".join(
        f'    {{ "{k}", {{ {F(v[0])}, {F(v[1])}, {F(v[2])}, {F(v[3])} }} }}' for k, v in sprites.items()) + "\n};\n")
    R4 = lambda r: "{ %s, %s, %s, %s }" % tuple(F(v) for v in r)
    fo.write(f"constexpr R hitLeft {R4(lay['arrowL'])};\nconstexpr R hitRight {R4(lay['arrowR'])};\n")
    fo.write(f"constexpr R hitDisplay {R4(GLASS)};\nconstexpr R statusBox {R4(STATUS)};\nconstexpr R presetBox {R4(PRESET)};\n")
    fo.write(f"constexpr R box {R4((BX0, BY0, BW, FH))};\n")
    for k in ('mix', 'fav', 'foot'):
        fo.write(f"constexpr float {k}X = {F(lay[k][0])}, {k}Y = {F(lay[k][1])}, {k}R = {F(lay[k][2])};\n")
    fo.write(f"constexpr float ringX = {F(FS[0])}, ringY = {F(FS[1])};\n")
    fo.write(f"constexpr float labelY = {F(LABEL_Y)}, nameX = {F(NAME_C[0])}, nameY = {F(NAME_C[1])}, subY = {F(SUB_Y)};\n")
    fo.write(f"constexpr float jackY = {F(JACK_Y)}, outX = {F(BX0 + 26)}, inX = {F(BX1 - 26)};\n")
    fo.write(f"constexpr float logoX = {F(LOGO_C[0])}, logoY = {F(LOGO_C[1])}, logoW = {F(LOGO_W)};\n")
    fo.write("}\n")
print('ok', W, H, 'JP', JP, 'BW', BW)
