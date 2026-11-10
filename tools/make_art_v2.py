"""
SEMITONES v2 (NINE HANDS | DSP) - plugin artwork from the user's validated picture (nov. 2026).

  source: design/v2src/source_pedal.png (user's picture, pedal in the middle of an 1800 x 1235 canvas), cleaned by
          tools/clean_source.py -> design/v2src/clean_pedal.png (no screws, no printed texts, no footswitch, no jacks,
          no preset text); the user's metal, embossed band, arrow buttons and MIX / FAV buttons are kept as they are.
  output: background.png = 1235 x 1800 px RGBA, simple-pedal rule: pedal full frame (enclosure touches top and bottom,
          jacks touch the left / right edges), transparent only around the jacks and in the rounded corners.
          Same scale in both directions for every part (buttons, arrows, footswitch, jacks); the enclosure is widened to
          the 1235 px frame by stretching only the flat metal / band / display horizontally (gaps follow the width).
  + photo kit: footswitch (user photo) with the Octopus light ring (drawn by the plugin), IN / OUT jacks (user photo);
  + dot-matrix display rebuilt (ghost digits + lit segment sprites), pressed states, Layout.h, tools/layout.json.
  Texts (Semitones, MIX, FAV, IN, OUT), the preset name and the NINE HANDS | DSP logo are vector, drawn by the plugin.
Usage: python3 -I tools/clean_source.py && python3 -I tools/make_art_v2.py
"""
import os, sys, json, math
import numpy as np, cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import photo_controls as PC

ASSETS = os.path.join(ROOT, 'assets')
AS = 2
FW, FH = 617.5, 900.0
W, H = int(FW * AS), int(FH * AS)

src = np.asarray(Image.open(os.path.join(ROOT, 'design', 'v2src', 'clean_pedal.png')).convert('RGBA')).astype(np.float32)
SB = (537.0, 4.5, 1263.0, 1230.5)                 # enclosure in the source picture (px)
KY = FH / (SB[3] - SB[1])                           # units per source px (iso scale)

JACK_LEN = 100.0
_j = PC.jack_sprite(JACK_LEN, AS, 1)
JP = _j[1]                                          # jack protrusion (units)
BX0, BX1 = JP, FW - JP
BW = BX1 - BX0
KX = BW / (SB[2] - SB[0])                           # horizontal units per source px (enclosure widened)
CX = FW / 2


def P(x, y):
    """source px -> plugin units"""
    return (BX0 + (x - SB[0]) * KX, (y - SB[1]) * KY)


print('JP', JP, 'KX/KY', KX / KY)

# ------------------------------------------------------------------ 1. enclosure, widened
crop = src[int(SB[1] - 1):int(SB[3] + 1), int(SB[0]):int(SB[2])]
y_off = SB[1] - int(SB[1] - 1)
bw_px, bh_px = int(round(BW * AS)), H
M = np.float32([[bw_px / (SB[2] - SB[0]), 0, 0], [0, bh_px / (SB[3] - SB[1]), -y_off * bh_px / (SB[3] - SB[1])]])
pm = crop.copy(); pm[..., :3] *= pm[..., 3:4] / 255.0
warped = cv2.warpAffine(pm, M, (bw_px, bh_px), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
a_ = np.clip(warped[..., 3:4], 0, 255)
body_rgb = np.where(a_ > 0.5, warped[..., :3] * 255.0 / np.maximum(a_, 1e-3), 0)
body_a = np.clip(a_[..., 0] / 255.0, 0, 1)
body_a = np.clip((body_a - 0.02) / 0.93, 0, 1)                  # opaque inside (source alpha was ~0.99)

img = np.zeros((H, W, 3), np.float32)
X0B = int(round(BX0 * AS))
img[:, X0B:X0B + bw_px] = body_rgb
body = np.zeros((H, W), np.float32); body[:, X0B:X0B + bw_px] = body_a

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
UX, UY = (XX + 0.5) / AS, (YY + 0.5) / AS


def gblur(a, s):
    return cv2.GaussianBlur(a, (0, 0), s) if s > 0 else a


def over(dst, rgba, x0, y0):
    h, w = rgba.shape[:2]
    X0, Y0 = int(round(x0)), int(round(y0))
    sx0, sy0 = max(0, -X0), max(0, -Y0)
    ex, ey = min(w, W - X0), min(h, H - Y0)
    a = rgba[sy0:ey, sx0:ex, 3:4] / 255.0
    reg = dst[Y0 + sy0:Y0 + ey, X0 + sx0:X0 + ex]
    dst[Y0 + sy0:Y0 + ey, X0 + sx0:X0 + ex] = reg * (1 - a) + rgba[sy0:ey, sx0:ex, :3] * a


def iso_patch(cx, cy, r_src, feather_src, shape='disc', tri=None):
    """unstretched (iso scale) copy of a round / triangular part of the source, pasted at its widened position"""
    k = KY * AS                                                    # px per source px
    n = int(math.ceil(2 * r_src * k)) + 2
    x0s, y0s = cx - r_src, cy - r_src
    Mi = np.float32([[1 / k, 0, x0s], [0, 1 / k, y0s]])            # dst -> src
    patch = cv2.warpAffine(src[..., :3], Mi, (n, n), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    d = np.hypot(xx / k + x0s - cx, yy / k + y0s - cy)
    m = np.clip((r_src - d) / feather_src, 0, 1)
    ux, uy = P(cx, cy)
    over(img, np.dstack([patch, m * 255]), ux * AS - n / 2, uy * AS - n / 2)
    return ux, uy


# parts measured on the source picture (px)
S_MIX, S_FAV, S_BTN_R = (765.0, 523.0), (1035.0, 523.0), 32.0
S_ARL, S_ARR = (648.0, 300.0), (1153.0, 300.0)
MIX = iso_patch(*S_MIX, 36, 3)                 # button + collar only: the band around it stays continuous
FAV = iso_patch(*S_FAV, 36, 3)
ARROW_L = iso_patch(*S_ARL, 50, 5)
ARROW_R = iso_patch(*S_ARR, 50, 5)
BTN_R = S_BTN_R * KY

# ------------------------------------------------------------------ 2. display: rebuilt glass + dot matrix
GL = P(712.0, 151.0) + P(1088.0, 457.0)
gx_, gy_ = GL[0], GL[1]; gw_, gh_ = GL[2] - GL[0], GL[3] - GL[1]
GLASS = (gx_, gy_, gw_, gh_)


def rrect_sd(x, y, x0, y0, w, h, r):
    cx, cy = x0 + w / 2, y0 + h / 2
    qx = np.abs(x - cx) - (w / 2 - r); qy = np.abs(y - cy) - (h / 2 - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


glass_m = np.clip(0.5 - rrect_sd(UX, UY, gx_ - 2, gy_ - 2, gw_ + 4, gh_ + 4, 10) * AS, 0, 1)
# base: the user's glass with the digits removed (inpainted at low resolution, then smoothed)
# base: the user's glass tone (dark, slightly lighter at the top left), digits and reflections ignored
g0 = img.copy()
NGX, NGY = 6, 6
cells = np.zeros((NGY, NGX, 3), np.float32)
for iy in range(NGY):
    for ix in range(NGX):
        x0c = int((gx_ + 6 + (gw_ - 12) * ix / NGX) * AS); x1c = int((gx_ + 6 + (gw_ - 12) * (ix + 1) / NGX) * AS)
        y0c = int((gy_ + 6 + (gh_ - 12) * iy / NGY) * AS); y1c = int((gy_ + 6 + (gh_ - 12) * (iy + 1) / NGY) * AS)
        c = g0[y0c:y1c, x0c:x1c].reshape(-1, 3)
        lum = c.mean(1); blu = c[:, 2] - c[:, 0]
        ok = (blu < 5) & (lum < np.percentile(lum, 40) + 1)
        cells[iy, ix] = np.median(c[ok], 0) if ok.sum() > 20 else np.median(c, 0)
gw_px, gh_px = int(gw_ * AS), int(gh_ * AS)
gl = cv2.resize(cells, (gw_px, gh_px), interpolation=cv2.INTER_CUBIC)
glass = g0.copy()
glass[int(gy_ * AS):int(gy_ * AS) + gh_px, int(gx_ * AS):int(gx_ * AS) + gw_px] = gl
glass = gblur(glass, 3)
DOT_PITCH = 3.4
fxg = np.mod(UX - gx_, DOT_PITCH) - DOT_PITCH / 2; fyg = np.mod(UY - gy_, DOT_PITCH) - DOT_PITCH / 2
dots = np.clip((0.40 * DOT_PITCH - np.hypot(fxg, fyg)) * AS / 0.8 + 0.5, 0, 1)
inner = np.clip(0.5 - rrect_sd(UX, UY, gx_ + 1, gy_ + 1, gw_ - 2, gh_ - 2, 9) * AS, 0, 1)
img = img * (1 - inner[..., None]) + glass * inner[..., None]

# segments (centre lines in units); digit boxes from the source "+84" (horizontal gaps follow the width)
SEG_T, SEG_GAP = 11.0, 1.3
DCY = P(0, 296.0)[1]
DH = 76.0
YT, YM, YB = DCY - DH / 2, DCY, DCY + DH / 2
DIG_W = 52.0
SIGN_C = (P(800.0, 0)[0], DCY)
TENS_C = P(915.0, 0)[0]
UNITS_C = P(1000.0, 0)[0]
SIGN_H, SIGN_V = 29.0, 33.0


def hseg(x1, x2, y, t=SEG_T, g=SEG_GAP):
    hh = t / 2
    return [(x1 + g, y), (x1 + g + hh, y - hh), (x2 - g - hh, y - hh), (x2 - g, y), (x2 - g - hh, y + hh), (x1 + g + hh, y + hh)]


def vseg(x, y1, y2, t=SEG_T, g=SEG_GAP):
    hh = t / 2
    return [(x, y1 + g), (x + hh, y1 + g + hh), (x + hh, y2 - g - hh), (x, y2 - g), (x - hh, y2 - g - hh), (x - hh, y1 + g + hh)]


def digit(xc):
    xl, xr = xc - DIG_W / 2, xc + DIG_W / 2
    return {'a': hseg(xl, xr, YT), 'b': vseg(xr, YT, YM), 'c': vseg(xr, YM, YB), 'd': hseg(xl, xr, YB),
            'e': vseg(xl, YM, YB), 'f': vseg(xl, YT, YM), 'g': hseg(xl, xr, YM)}


def poly_mask(poly, ss=4):
    im = Image.new('L', (W * ss, H * ss), 0)
    ImageDraw.Draw(im).polygon([(x * AS * ss, y * AS * ss) for x, y in poly], fill=255)
    return np.asarray(im.resize((W, H), Image.BOX)).astype(np.float32) / 255.0


cx_, cy_ = SIGN_C
layers = [('s_' + k, p) for k, p in {
    'l': hseg(cx_ - SIGN_H, cx_ + SEG_T * 0.75, cy_), 'r': hseg(cx_ - SEG_T * 0.75, cx_ + SIGN_H, cy_),
    'u': vseg(cx_, cy_ - SIGN_V, cy_ + SEG_T * 0.75), 'd': vseg(cx_, cy_ - SEG_T * 0.75, cy_ + SIGN_V)}.items()]
layers += [('t_' + k, p) for k, p in digit(TENS_C).items()]
layers += [('u_' + k, p) for k, p in digit(UNITS_C).items()]
seg_masks = {n: poly_mask(p) for n, p in layers}
ghost = np.zeros((H, W), np.float32)
for m_ in seg_masks.values():
    ghost = np.maximum(ghost, m_)
GHOST = np.array([10.0, 26.0, 60.0])
gd = ghost * dots * inner
img = img * (1 - gd[..., None] * 0.55) + GHOST * gd[..., None] * 0.55
img += (dots * inner * 2.2)[..., None] * np.array([0.6, 0.8, 1.0])          # unlit dot grid, barely visible


def srgb2lin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


LIT = np.array([40.0, 168.0, 255.0]) / 255.0
LIT_CORE = np.array([170.0, 235.0, 255.0]) / 255.0
sprites = {}


def save_rgba(name, rgb01, a):
    ys, xs = np.nonzero(a > 0.004)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    rgba = np.dstack([np.clip(rgb01[y0:y1, x0:x1], 0, 1), np.clip(a[y0:y1, x0:x1], 0, 1)])
    Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), 'RGBA').save(os.path.join(ASSETS, name + '.png'), optimize=True)
    sprites[name] = (x0 / AS, y0 / AS, (x1 - x0) / AS, (y1 - y0) / AS)


def lit_sprite(mask, name):
    md = mask * dots
    core = gblur(cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1), 2.0)
    col_l = srgb2lin(LIT)[None, None, :] * md[..., None] * 1.25 + srgb2lin(LIT_CORE)[None, None, :] * (md * core * 0.9)[..., None]
    glow = gblur(mask, 0.8 * AS) * 0.35 + gblur(mask, 2.5 * AS) * 0.75 + gblur(mask, 7.0 * AS) * 0.45 + gblur(mask, 16 * AS) * 0.18
    glow *= inner
    tot = col_l + srgb2lin(LIT)[None, None, :] * glow[..., None] * 0.75
    a_ = np.clip(md + glow * 0.85, 0, 1)
    save_rgba(name, lin2srgb(tot / np.maximum(a_[..., None], 1e-4)), a_)


print('segments...')
for n, _ in layers:
    lit_sprite(seg_masks[n], 'seg_' + n)
MIX_DOT = (MIX[0], gy_ + gh_ - 18.0, 4.6)
lit_sprite(np.clip((MIX_DOT[2] - np.hypot(UX - MIX_DOT[0], UY - MIX_DOT[1])) * AS, 0, 1), 'seg_mix')
STATUS = (CX - 115.0, gy_ + gh_ - 36.0, 230.0, 30.0)

# ------------------------------------------------------------------ 3. footswitch (photo) + jacks (photo)
FS = (CX, P(0, 1007.5)[1] + 27.5)              # lowered 30 + 10 + 15 px (user, 10 nov.)
FS_FLATS = 122.0
FSPR, FSREFL, FS_SZ, FS_BTN = PC.footswitch_sprite(FS_FLATS, AS, refl_colour=(60.0, 160.0, 255.0))
n = FSPR.shape[0]
x0f, y0f = int(round((FS[0] - FS_SZ / 2) * AS)), int(round((FS[1] - FS_SZ / 2) * AS))
fa = np.zeros((H, W), np.float32)
fa[y0f:y0f + n, x0f:x0f + n] = np.clip((FSPR[..., 3] / 255.0 - 0.86) / 0.14, 0, 1)
contact = gblur(fa, 3.0 * AS); soft = gblur(fa, 12.0 * AS)
o = np.clip(contact * 0.55 + soft * 0.8, 0, 0.5) * 0.9 * (1.0 - fa)
img *= (1.0 - o * body)[..., None]
def black_tint(rgba):
    """light black tint (dark gunmetal like the MIX / FAV buttons and the arrows of the user's picture)"""
    out = rgba.astype(np.float32).copy()
    c = out[..., :3] * np.array([0.9675, 1.0087, 1.047], np.float32)       # neutral (the photos are warm)
    out[..., :3] = 255.0 * np.power(np.clip(c / 255.0, 0, 1), 1.25) * 0.85
    return out


FSPR = black_tint(FSPR)
FSPR[..., :3] = 255.0 * np.power(np.clip(FSPR[..., :3] / 255.0, 0, 1), 1.15) * 0.80      # darker still (user, 10 nov.)
over(img, FSPR, x0f, y0f)
RING_R = FS_BTN * 33.2 / 31.5                       # Octopus: ring in the groove between the button and the nut
RING_W = 2.1 * FS_BTN / 31.5
RING_K = FS_BTN / 31.5

JACK_Y = P(0, 584.0)[1]
alpha = body.copy()
for side in (-1, 1):
    j, jw_u, jh_u = PC.jack_sprite(JACK_LEN, AS, side)
    jh, jw = j.shape[:2]
    x0j = 0 if side < 0 else int(round(BX1 * AS))
    y0j = int(round(JACK_Y * AS - jh / 2))
    occ = np.zeros((H, W), np.float32); occ[y0j:y0j + jh, x0j:x0j + jw] = j[..., 3] / 255.0
    oo = np.clip(gblur(occ, 3 * AS) * 0.9 + gblur(occ, 16 * AS) * 1.4, 0, 0.55) * (0.25 if side > 0 else 0.12)
    img *= (1 - oo * body)[..., None]
    j = black_tint(j)
    over(img, j, x0j, y0j)
    ja = np.zeros((H, W), np.float32); ja[y0j:y0j + jh, x0j:x0j + jw] = j[..., 3] / 255.0
    alpha = np.maximum(alpha, ja)

# arrows: brighter faces (like the centre of the MIX / FAV caps) with a reflection that follows the triangle:
# a sheen along the top edge (the one facing the light), fading across the face, and a glint in the top corner
def arrow_light(centre_u, s_src, side):
    cx, cy = centre_u
    k = KY
    face = [(-31 * side, -38), (-31 * side, 38), (39 * side, 0)]                 # face corners (source px, from centre)
    F = [(cx + fx * k, cy + fy * k) for fx, fy in face]
    m = gblur(poly_mask(F), 0.5 * AS)
    m = np.clip((cv2.erode(m, np.ones((5, 5), np.uint8)) - 0.05) / 0.9, 0, 1)
    m = gblur(m, 1.2 * AS)
    # top edge: from the top corner of the face to the tip
    (x0, y0), (x1, y1) = F[0], F[2]
    ex, ey = x1 - x0, y1 - y0; L_ = math.hypot(ex, ey)
    nx_, ny_ = -ey / L_, ex / L_
    if ny_ < 0:
        nx_, ny_ = -nx_, -ny_                                                  # normal pointing into the face (down)
    d_top = (UX - x0) * nx_ + (UY - y0) * ny_                                  # distance from the top edge (units)
    t_along = np.clip(((UX - x0) * ex + (UY - y0) * ey) / L_ ** 2, 0, 1)      # 0 at the top corner, 1 at the tip
    sheen = np.exp(-np.clip(d_top, 0, None) / 9.0) * (0.55 + 0.45 * np.sin(np.pi * np.clip(t_along * 1.1, 0, 1)))
    grad = np.clip(1.0 - ((UX - F[0][0]) * 0.46 + (UY - F[0][1]) * 0.62) / (60.0 * k), 0.25, 1.0)   # brighter towards the light
    glint = np.exp(-(((UX - (F[0][0] + 5 * side * k)) / 7.0) ** 2 + ((UY - (F[0][1] + 7 * k)) / 7.0) ** 2))
    base = img * (1.0 + 1.35 * m)[..., None]                                   # lifts the brushed face
    add = (m * (70.0 * sheen + 38.0 * grad + 60.0 * glint))[..., None] * np.array([0.97, 0.99, 1.02])
    return base + add


img = arrow_light(ARROW_L, S_ARL, -1)
img = arrow_light(ARROW_R, S_ARR, 1)
FINAL = np.clip(img, 0, 255)
Image.fromarray(np.dstack([FINAL, alpha * 255]).astype(np.uint8), 'RGBA').save(os.path.join(ROOT, 'background.png'), optimize=True)

# ------------------------------------------------------------------ 4. pressed states
def save_pressed(name, cx, cy, rad_u, mask, push=1.0, dark=0.82):
    X0, Y0 = int((cx - rad_u) * AS), int((cy - rad_u) * AS)
    n_ = int(2 * rad_u * AS)
    reg = FINAL[Y0:Y0 + n_, X0:X0 + n_] / 255.0
    mk = mask[Y0:Y0 + n_, X0:X0 + n_]
    moved = cv2.warpAffine(reg, np.float32([[1, 0, push * AS * 0.6], [0, 1, push * AS]]), (n_, n_), borderMode=cv2.BORDER_REPLICATE)
    o_ = reg * (1 - mk[..., None]) + moved * dark * mk[..., None]
    a = np.clip(gblur(mk, 1.0) * 1.15, 0, 1)
    Image.fromarray((np.dstack([np.clip(o_, 0, 1), a]) * 255 + 0.5).astype(np.uint8), 'RGBA').save(os.path.join(ASSETS, 'press_' + name + '.png'), optimize=True)
    sprites['press_' + name] = (X0 / AS, Y0 / AS, n_ / AS, n_ / AS)


def tri_mask(c, s):
    # arrow triangles of the source: 68 x 76 px, tip outwards
    w_, h_ = 78.0 * KY, 88.0 * KY
    x, y = c
    return poly_mask([(x - s * w_ / 2 + s * 2, y - h_ / 2), (x + s * w_ / 2, y), (x - s * w_ / 2 + s * 2, y + h_ / 2)])


save_pressed('left', ARROW_L[0], ARROW_L[1], 40.0, gblur(tri_mask(ARROW_L, -1), 1.2))
save_pressed('right', ARROW_R[0], ARROW_R[1], 40.0, gblur(tri_mask(ARROW_R, 1), 1.2))
for nm, (bx, by) in (('mix', MIX), ('fav', FAV)):
    m_ = np.clip((BTN_R + 0.5 - np.hypot(UX - bx, UY - by)) * AS / 1.2, 0, 1)
    save_pressed(nm, bx, by, BTN_R + 8, m_, push=0.9, dark=0.80)
rad_u = FS_FLATS * 0.62
X0, Y0, n_ = int((FS[0] - rad_u) * AS), int((FS[1] - rad_u) * AS), int(2 * rad_u * AS)
reg = FINAL[Y0:Y0 + n_, X0:X0 + n_] / 255.0
cxl, cyl = FS[0] * AS - X0, FS[1] * AS - Y0
Mr = cv2.getRotationMatrix2D((cxl, cyl), 0, 0.97); Mr[1, 2] += 1.2 * AS
sunk = cv2.warpAffine(reg, Mr, (n_, n_), borderMode=cv2.BORDER_REPLICATE)
yyp, xxp = np.mgrid[0:n_, 0:n_].astype(np.float32)
rp = np.hypot(xxp - cxl, yyp - cyl) / AS
chm = np.clip((FS_BTN + 6 - rp) * AS / 1.5, 0, 1)
o_ = reg * (1 - chm[..., None]) + sunk * chm[..., None]
o_ *= (1 - 0.25 * np.clip((FS_BTN * 1.02 - rp) * AS / 2, 0, 1))[..., None]
Image.fromarray((np.dstack([np.clip(o_, 0, 1), chm]) * 255 + 0.5).astype(np.uint8), 'RGBA').save(os.path.join(ASSETS, 'press_foot.png'), optimize=True)
sprites['press_foot'] = (X0 / AS, Y0 / AS, n_ / AS, n_ / AS)
for old in ('ringlight', 'ring_lit', 'ring_comet'):
    p_ = os.path.join(ASSETS, old + '.png')
    if os.path.exists(p_):
        os.remove(p_)

# ------------------------------------------------------------------ 5. Layout.h + layout.json
PRESET = P(716.0, 57.0) + P(1090.0, 101.0)
PRESET = (PRESET[0], PRESET[1], PRESET[2] - PRESET[0], PRESET[3] - PRESET[1])
LABEL_Y = P(0, 478.0)[1]
NAME_C = P(890.0, 715.0)
LOGO_C, LOGO_W = (CX, P(0, 1179.0)[1]), 195.0
OUT_X, IN_X = P(580.0, 0)[0], P(1220.0, 0)[0]
TEXT_Y = P(0, 590.0)[1]
lay = {'frame': [FW, FH], 'box': [BX0, 0, BW, FH], 'preset': list(PRESET), 'glass': list(GLASS), 'status': list(STATUS),
       'arrowL': [ARROW_L[0] - 34, ARROW_L[1] - 40, 68, 80], 'arrowR': [ARROW_R[0] - 34, ARROW_R[1] - 40, 68, 80],
       'mix': [MIX[0], MIX[1], BTN_R + 6], 'fav': [FAV[0], FAV[1], BTN_R + 6], 'foot': [FS[0], FS[1], FS_FLATS * 0.56],
       'ring': [FS[0], FS[1], RING_R, RING_W, RING_K], 'jackY': JACK_Y, 'textY': TEXT_Y,
       'labelY': LABEL_Y, 'name': list(NAME_C), 'subY': NAME_C[1] + 30, 'logo': [LOGO_C[0], LOGO_C[1], LOGO_W],
       'outX': OUT_X, 'inX': IN_X, 'sprites': sprites}
json.dump(lay, open(os.path.join(HERE, 'layout.json'), 'w'), indent=1)
F = lambda v: f"{v:.2f}f"
R4 = lambda r: "{ %s, %s, %s, %s }" % tuple(F(v) for v in r)
with open(os.path.join(ROOT, 'Layout.h'), 'w') as fo:
    fo.write(f"// generated by tools/make_art_v2.py - do not edit (frame {FW} x {FH} units = background.png {W} x {H} px / 2)\n")
    fo.write("#pragma once\nnamespace Layout {\nstruct R { float x, y, w, h; };\nstruct Sprite { const char* name; R r; };\n")
    fo.write(f"constexpr float W = {F(FW)}, H = {F(FH)};\n")
    fo.write(f"constexpr int defaultH = 493, defaultW = {int(round(493 * FW / FH))};   // \"60 %\"\n")
    fo.write("constexpr Sprite sprites[] = {\n" + ",\n".join(
        f'    {{ "{k}", {{ {F(v[0])}, {F(v[1])}, {F(v[2])}, {F(v[3])} }} }}' for k, v in sprites.items()) + "\n};\n")
    fo.write(f"constexpr R hitLeft {R4(lay['arrowL'])};\nconstexpr R hitRight {R4(lay['arrowR'])};\n")
    fo.write(f"constexpr R hitDisplay {R4(GLASS)};\nconstexpr R statusBox {R4(STATUS)};\nconstexpr R presetBox {R4(PRESET)};\n")
    fo.write(f"constexpr R box {R4((BX0, 0, BW, FH))};\n")
    for k in ('mix', 'fav', 'foot'):
        fo.write(f"constexpr float {k}X = {F(lay[k][0])}, {k}Y = {F(lay[k][1])}, {k}R = {F(lay[k][2])};\n")
    fo.write(f"constexpr float ringX = {F(FS[0])}, ringY = {F(FS[1])}, ringR = {F(RING_R)}, ringW = {F(RING_W)}, ringK = {F(RING_K)};\n")
    fo.write(f"constexpr float labelY = {F(LABEL_Y)}, nameX = {F(NAME_C[0])}, nameY = {F(NAME_C[1])};\n")
    fo.write(f"constexpr float jackY = {F(JACK_Y)}, textY = {F(TEXT_Y)}, outX = {F(OUT_X)}, inX = {F(IN_X)};\n")
    fo.write(f"constexpr float logoX = {F(LOGO_C[0])}, logoY = {F(LOGO_C[1])}, logoW = {F(LOGO_W)};\n")
    fo.write("}\n")
print('ok', W, H, 'BW', BW, 'ring', RING_R, RING_W)
