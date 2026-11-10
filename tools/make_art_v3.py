"""
SEMITONES v2 artwork, rebuilt on the user's mockup design/Semitones_Design_v002.svg (nov. 2026).

Frame = the mockup: 1235 x 1800 px (plugin units = px / 2), pedal full frame, transparent only around the jacks.
Light: one source, top left corner, 45 degrees (in plane), 40 degrees above the faceplate.

  plate      brushed black aluminium = the user's texture design/v3src/brushed_black_4096.png (x 0.45, one tile, no
             repetition), lit by a smooth top-left -> bottom-right falloff
  0AA3EF     round bevel of the enclosure edge (quarter round, 17 px)
  ED0CED     flat inner bevels (45 deg chamfers) of the display and preset slot holes
  000000     display glass (dot matrix) and preset slot (black), drawn in their holes
  FF00F9     flat bevel of the MIX / FAV holes; 20FF00 = the buttons (photo caps), the arrow buttons (vector)
  FF0000     bands: three edges each (outer edge of the matte band, band / channel, inner edge of the channel); their
             cross-sections (lips, creases, floors) are the mean cross-sections of the photographed bands of the same
             orientation, re-lit by the new plate and carrying its grain (sharp, no photo blur)
  145B1D     IN / OUT jacks (photo kit), BABCBE = footswitch (photo kit) + Octopus ring (drawn by the plugin)
  texts      vector (design/Semitones_Texts.svg, paths of the mockup), debossed by the plugin
Usage: python3 -I tools/make_art_v3.py
"""
import os, sys, json, math, re
import numpy as np, cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import photo_controls as PC
import regroove as RG

ASSETS = os.path.join(ROOT, 'assets')
AS = 2
W, H = 1235, 1800
FW, FH = W / AS, H / AS
CX = FW / 2
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
PXc, PYc = XX + 0.5, YY + 0.5                     # pixel centres (px)
UX, UY = PXc / AS, PYc / AS                      # units

# ------------------------------------------------------------------ mockup geometry (px)
OUTER = (41.5, 0.0, 1193.5, 1800.0, 32.59)      # enclosure (0AA3EF outer edge)
INNER = (58.5, 17.0, 1176.5, 1783.0, 15.59)     # top of the round bevel = faceplate
DISP_O = (312.5, 204.32, 922.5, 672.32, 31.27)   # display hole, top of the chamfer
DISP_I = (316.5, 208.32, 918.5, 668.32, 27.27)   # glass
SLOT_O = (312.5, 66.0, 922.5, 154.0, 44.0)       # preset slot hole
SLOT_I = (316.5, 70.0, 918.5, 150.0, 40.0)
BTNS = {'mix': (405.0, 790.0), 'fav': (830.0, 790.0)}
CAP_R, FLOOR_R, HOLE_R = 42.5, 45.0, 50.0         # button cap, recess floor, top of the chamfer
FS_C = (618.97, 1530.19); FS_FLATS_PX = 257.93
JACK_Y_PX, JACK_LEN_PX = (748.21 + 950.76) / 2, 950.76 - 748.21
ARROW_TRI = {-1: [(268.0, 369.6), (146.0, 434.0), (268.0, 498.4)],      # sharp triangles of the rounded guides
             1: [(967.0, 369.6), (1089.0, 434.0), (967.0, 498.4)]}
ARROW_RAD = 10.5

Lv = np.array([-1.0, -1.0, 1.19]); Lv /= np.linalg.norm(Lv)          # top left, 45 deg
Hv = Lv + np.array([0, 0, 1.0]); Hv /= np.linalg.norm(Hv)


def gblur(a, s):
    return cv2.GaussianBlur(a, (0, 0), s) if s > 0 else a


def rrect_sd(x, y, r_):
    x0, y0, x1, y1, r = r_
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    qx = np.abs(x - cx) - (hw - r); qy = np.abs(y - cy) - (hh - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def cov(sd):
    """anti-aliased coverage of sd < 0 (px)"""
    return np.clip(0.5 - sd, 0, 1)


# ------------------------------------------------------------------ 1. plate (texture + light)
tex = np.asarray(Image.open(os.path.join(ROOT, 'design', 'v3src', 'brushed_black_4096.png')).convert('RGB')).astype(np.float32)
tex = cv2.resize(tex, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)[:H, :W]
TEX_MEAN = tex.mean()
grain = tex / np.maximum(gblur(tex.mean(2), 40)[..., None], 1)            # multiplicative grain, mean 1
u = (PXc + PYc) / (W + H)                                                 # 0 top left -> 1 bottom right
light = 22.5 + 50.0 * (1 - u) ** 2.3
light += 6.0 * np.exp(-((PXc - 330) ** 2 / (2 * 420 ** 2) + (PYc - 150) ** 2 / (2 * 520 ** 2)))   # soft sheen
plate = grain * light[..., None]

# ------------------------------------------------------------------ 2. bands (FF0000) : zones + heights
# One height field for the whole band (no per-run photo profile): the bevels, lips and creases come from the same
# geometry and the same light everywhere, so the mitred corners join exactly (normals turn with the edges).
#   face    = 00FFFF zones of design/Semitones_Design_Mask_v002.svg (between A and M), 3 px below the faceplate
#   channel = between M and C, 8 px below the faceplate
#   chamfers: 2.5 px from the faceplate down to the face, 2 px from the face / faceplate down to the channel
bands = RG.read_path(os.path.join(ROOT, 'design', 'Semitones_Design_v002.svg'))
bands_photo = np.asarray(Image.open(os.path.join(ROOT, 'design', 'v3src', 'bands_photo.png')).convert('RGB')).astype(np.float32)   # MIX / FAV caps
SS = 2


def poly_bin(polys):
    im_ = Image.new('L', (W * SS, H * SS), 0)
    for v in polys:
        ImageDraw.Draw(im_).polygon([(x * SS, y * SS) for x, y in v], fill=255)
    return np.asarray(im_) > 127


def inner_dist(binm):
    """distance (px) from the inside of a zone to its edge, computed at 2x, back to 1x"""
    d = cv2.distanceTransform(binm.astype(np.uint8), cv2.DIST_L2, 5) / SS
    return cv2.resize(d, (W, H), interpolation=cv2.INTER_AREA)


_ms = open(os.path.join(ROOT, 'design', 'Semitones_Design_Mask_v002.svg')).read()
face_polys = [np.array(list(map(float, p_.split())), float).reshape(-1, 2) for p_ in re.findall(r'points="([^"]+)"', _ms)]
chan_polys = [np.vstack([M_, C_[::-1]]) for _, A_, M_, C_, _ in bands]
YYs, XXs = np.mgrid[0:H * SS, 0:W * SS].astype(np.float32)
inner_ss = rrect_sd((XXs + 0.5) / SS, (YYs + 0.5) / SS, INNER) < 0          # bands stop at the round bevel
face_b = [poly_bin([p_]) & inner_ss for p_ in face_polys]
chan_b = poly_bin(chan_polys) & inner_ss
groove_b = chan_b.copy()
for fb in face_b:
    groove_b |= fb
dG, dK = inner_dist(groove_b), inner_dist(chan_b)
sm = lambda x: x * x * (3 - 2 * x)
h_band = -3.0 * sm(np.clip(dG / 2.2, 0, 1)) - 5.0 * sm(np.clip(dK / 2.2, 0, 1))
del YYs, XXs

# albedo: faceplate grain; faces = ONE continuous texture each (another part of the same sheet), overall tint of
# the former faces kept; channel floor a little darker
band_tex = np.roll(tex, (611, 977), (0, 1))
band_grain = np.power(band_tex / np.maximum(gblur(band_tex.mean(2), 40)[..., None], 1), 0.6)
img = plate.copy()
K_FACE = [1.60, 2.09]                                     # face / faceplate brightness of the validated faces
TINT = np.array([0.992, 0.992, 1.015])
for fb, kf in zip(face_b, K_FACE):
    fm = cv2.resize(fb.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)[..., None]
    img = img * (1 - fm) + band_grain * (kf * light)[..., None] * TINT * fm
cm = cv2.resize(chan_b.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)[..., None]
img = img * (1 - cm) + plate * 0.72 * cm

# ------------------------------------------------------------------ 3. relief: rim, holes (height field, one light)
d_out = -rrect_sd(PXc, PYc, OUTER)                         # > 0 inside the enclosure
RIM = 17.0
t = np.clip(d_out / RIM, 0, 1)
h = -RIM + RIM * np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1))    # quarter round, 0 on the faceplate
h += h_band
DEPTH = 4.0
holes = {}
def chamfer_hole(sd_o, width, depth, floor):
    """flat 45 deg chamfer from the hole edge (sd_o = 0) down to the floor"""
    hh = -np.clip(-sd_o / width, 0, 1) * depth
    return np.where(sd_o < 0, np.minimum(hh, 0), 0) if floor is None else hh
sd_disp = rrect_sd(PXc, PYc, DISP_O); sd_slot = rrect_sd(PXc, PYc, SLOT_O)
h += chamfer_hole(sd_disp, 4.0, 4.0, None)
h += chamfer_hole(sd_slot, 4.0, 4.0, None)
for k, (bx, by) in BTNS.items():
    rr = np.hypot(PXc - bx, PYc - by)
    h += -np.clip((HOLE_R - rr) / (HOLE_R - FLOOR_R), 0, 1) * 5.0
hs = gblur(h.astype(np.float32), 0.55)
gy, gx = np.gradient(hs)
nrm = np.dstack([-gx, -gy, np.ones_like(hs)]); nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
ndl = np.clip(nrm @ Lv, 0, 1); ndh = np.clip(nrm @ Hv, 0, 1)
nl0, nh0 = Lv[2], Hv[2]
A_, D_ = 0.22, 1.0
# cast shadows (height field ray march towards the light)
shadow = np.zeros((H, W), np.float32)
lxy = -np.array([Lv[0], Lv[1]]) / np.hypot(Lv[0], Lv[1]); tanE = Lv[2] / np.hypot(Lv[0], Lv[1])
for k_ in np.arange(1.0, 24.0, 1.0):
    sh = cv2.warpAffine(hs, np.float32([[1, 0, -lxy[0] * k_], [0, 1, -lxy[1] * k_]]), (W, H),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    shadow = np.maximum(shadow, np.clip((sh - hs - k_ * tanE) / 1.5, 0, 1))
shadow = gblur(shadow, 1.0)
ao = 1 - 0.5 * np.clip(-(hs - gblur(hs, 6.0)) / 3.0, 0, 1)
relief = (A_ + D_ * ndl) / (A_ + D_ * nl0) * (1 - 0.75 * shadow) * ao
spec = 330.0 * (ndh ** 24 - nh0 ** 24).clip(0, None)
# convex edges (plate -> chamfer lips, plate -> rim) catch a thin line of light, brightest where they face the light
lap = cv2.Laplacian(gblur(h.astype(np.float32), 0.7), cv2.CV_32F)
gmag = np.hypot(gx, gy) + 1e-6
facing = np.clip(((-gx / gmag) * Lv[0] + (-gy / gmag) * Lv[1]) / np.hypot(Lv[0], Lv[1]), -1, 1)
edge_hl = np.clip(-lap / 0.5, 0, 1) * (0.3 + 0.7 * np.clip(facing + 0.35, 0, 1))   # area light: a lip a little off the light still glints
spec += 150.0 * gblur(edge_hl, 0.4)
inside_body = cov(rrect_sd(PXc, PYc, OUTER))
# chamfers of the holes and the round bevel cut the plain metal (not the band faces)
cut = np.clip((-(h - h_band) - 0.05) * 4, 0, 1)[..., None]
alb = img * (1 - cut) + plate * cut
img = alb * relief[..., None] + spec[..., None] * np.array([0.97, 0.99, 1.02])
# rim: outer edge falls off into the dark
img *= (0.35 + 0.65 * np.clip(d_out / 6.0, 0, 1))[..., None]

# ------------------------------------------------------------------ 4. display glass + dot matrix, preset slot
glass_m = cov(rrect_sd(PXc, PYc, DISP_I))
slot_m = cov(rrect_sd(PXc, PYc, SLOT_I))
gx_, gy_ = DISP_I[0] / AS, DISP_I[1] / AS
gw_, gh_ = (DISP_I[2] - DISP_I[0]) / AS, (DISP_I[3] - DISP_I[1]) / AS
GLASS = (gx_, gy_, gw_, gh_)
gtone = 6.0 + 7.0 * np.clip(1 - ((UX - gx_) / gw_ * 0.6 + (UY - gy_) / gh_ * 0.4), 0, 1)
glass_rgb = np.dstack([gtone * 0.55, gtone * 0.62, gtone * 0.85])
ish = gblur(np.clip(-rrect_sd(PXc, PYc, DISP_I) / 10.0, 0, 1), 3.0)               # inner shadow under the lip
glass_rgb *= (0.45 + 0.55 * ish)[..., None]
img = img * (1 - glass_m[..., None]) + glass_rgb * glass_m[..., None]
slot_rgb = np.full((H, W, 3), 3.0, np.float32)
img = img * (1 - slot_m[..., None]) + slot_rgb * slot_m[..., None]

DOT_PITCH = 3.4
fxg = np.mod(UX - gx_, DOT_PITCH) - DOT_PITCH / 2; fyg = np.mod(UY - gy_, DOT_PITCH) - DOT_PITCH / 2
dots = np.clip((0.40 * DOT_PITCH - np.hypot(fxg, fyg)) * AS / 0.8 + 0.5, 0, 1)
inner = glass_m

# digit boxes: same place in the glass as on the validated v2 display (relative to the glass)
G2 = (159.72, 107.54, 298.06, 224.63)                    # v2 glass (units)
def gmap(x, y):
    return (gx_ + (x - G2[0]) * gw_ / G2[2], gy_ + (y - G2[1]) * gh_ / G2[3])
SEG_T, SEG_GAP = 11.0, 1.3
DCY = gmap(0, 213.99)[1]
DH = 76.0
YT, YM, YB = DCY - DH / 2, DCY, DCY + DH / 2
DIG_W = 52.0
SIGN_C = (gmap(229.48, 0)[0], DCY)
TENS_C = gmap(320.64, 0)[0]
UNITS_C = gmap(388.02, 0)[0]
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
img += (dots * inner * 2.2)[..., None] * np.array([0.6, 0.8, 1.0])


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
for n_, _ in layers:
    lit_sprite(seg_masks[n_], 'seg_' + n_)
MIX = (BTNS['mix'][0] / AS, BTNS['mix'][1] / AS)
FAV = (BTNS['fav'][0] / AS, BTNS['fav'][1] / AS)
MIX_DOT = (gx_ + 22.0, gy_ + gh_ - 18.0, 4.6)
lit_sprite(np.clip((MIX_DOT[2] - np.hypot(UX - MIX_DOT[0], UY - MIX_DOT[1])) * AS, 0, 1), 'seg_mix')
STATUS = (CX - 115.0, gy_ + gh_ - 36.0, 230.0, 30.0)


# ------------------------------------------------------------------ 5. MIX / FAV caps (the user's photo caps)
def over(dst, rgba, x0, y0):
    hh_, ww_ = rgba.shape[:2]
    X0, Y0 = int(round(x0)), int(round(y0))
    sx0, sy0 = max(0, -X0), max(0, -Y0)
    ex, ey = min(ww_, W - X0), min(hh_, H - Y0)
    a = rgba[sy0:ey, sx0:ex, 3:4] / 255.0
    reg = dst[Y0 + sy0:Y0 + ey, X0 + sx0:X0 + ex]
    dst[Y0 + sy0:Y0 + ey, X0 + sx0:X0 + ex] = reg * (1 - a) + rgba[sy0:ey, sx0:ex, :3] * a


CAP_SRC = {'mix': (403.5, 761.3), 'fav': (831.5, 761.3)}      # caps on the validated v2 artwork (px)
CAP_SRC_R = 43.0
for k, (bx, by) in BTNS.items():
    sx, sy = CAP_SRC[k]
    sc = CAP_SRC_R / CAP_R
    n = int(2 * (CAP_R + 3)) + 2
    Mi = np.float32([[sc, 0, sx - sc * (n / 2)], [0, sc, sy - sc * (n / 2)]])
    patch = cv2.warpAffine(bands_photo, Mi, (n, n), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    d = np.hypot(xx + 0.5 - n / 2, yy + 0.5 - n / 2)
    m = np.clip(CAP_R + 0.5 - d, 0, 1)
    # cap shadow on the recess floor (down right)
    rr = np.hypot(PXc - bx - 2.2, PYc - by - 2.6)
    capsh = gblur(np.clip(CAP_R - rr, 0, 1), 2.2)
    rr0 = np.hypot(PXc - bx, PYc - by)
    floor = (rr0 < FLOOR_R + 0.5)
    img = np.where(floor[..., None], img * (1 - 0.7 * capsh)[..., None] * 0.55, img)
    over(img, np.dstack([patch, m * 255]), bx - n / 2, by - n / 2)

# ------------------------------------------------------------------ 6. arrows (vector reliefs, 20FF00 guides)
def poly_sd_px(poly):
    dmin = np.full((H, W), 1e9, np.float32)
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        ex, ey = x1 - x0, y1 - y0
        tt = np.clip(((PXc - x0) * ex + (PYc - y0) * ey) / (ex * ex + ey * ey), 0, 1)
        dmin = np.minimum(dmin, np.hypot(PXc - x0 - tt * ex, PYc - y0 - tt * ey))
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(poly, fill=255)
    inside = np.asarray(im) > 127
    return np.where(inside, -dmin, dmin)


def shrink(poly, r):
    P_ = np.array(poly, float); n = len(P_)
    lines = []
    for i in range(n):
        a_, b_ = P_[i], P_[(i + 1) % n]
        d = (b_ - a_) / np.linalg.norm(b_ - a_)
        nn = np.array([d[1], -d[0]])
        if np.dot(nn, P_.mean(0) - a_) < 0:
            nn = -nn
        lines.append((a_ + nn * r, d))
    out = []
    for i in range(n):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.array([d1, -d2]).T
        tt = np.linalg.solve(A, p2 - p1)
        out.append(tuple(p1 + d1 * tt[0]))
    return out


def shift_px(a, dx, dy):
    return cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (a.shape[1], a.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


arrow_masks = {}
def vector_arrow(side):
    global img
    tri = ARROW_TRI[side]
    sd = poly_sd_px(shrink(tri, ARROW_RAD)) - ARROW_RAD
    BEV, HT, POCK, DEP = 6.0, 6.8, 6.0, 5.2
    t_ = np.clip(-sd / BEV, 0, 1)
    raise_ = HT * (1 - (1 - t_) ** 2)
    pocket = np.clip((POCK - sd) / 2.2, 0, 1)
    hh = gblur((raise_ - DEP * pocket).astype(np.float32), 0.7)
    gy_, gx_ = np.gradient(hh)
    nr = np.dstack([-gx_, -gy_, np.ones_like(hh)]); nr /= np.linalg.norm(nr, axis=2, keepdims=True)
    nd = np.clip(nr @ Lv, 0, 1); nh = np.clip(nr @ Hv, 0, 1)
    tri_m = np.clip(0.5 - sd, 0, 1)
    arrow_masks[side] = tri_m
    shadow_ = gblur(shift_px(tri_m, 2.8, 2.8), 2.8) * (1 - tri_m)
    g_ = (grain.mean(2) - 1.0) * 40.0
    face = 100.0 * (0.18 + 0.95 * nd) + 95.0 * nh ** 50 + 1.6 * np.clip(g_, -5, 5)
    (x0, y0), (x1, y1) = tri[0], tri[1]
    ex, ey = x1 - x0, y1 - y0; L_ = math.hypot(ex, ey); nx_, ny_ = -ey / L_, ex / L_
    if ny_ < 0:
        nx_, ny_ = -nx_, -ny_
    d_top = (PXc - x0) * nx_ + (PYc - y0) * ny_
    face += 14.0 * np.exp(-np.clip(d_top - BEV, 0, None) / 16.0) * (sd < -BEV * 0.6)
    cxa, cya = np.mean([p[0] for p in tri]), np.mean([p[1] for p in tri])
    gd_ = ((PXc - cxa) * -0.7 + (PYc - cya) * -0.7) / 60.0
    face += 16.0 * np.clip(gd_, -1, 1)
    face_rgb = face[..., None] * np.array([0.98, 0.99, 1.02])
    ao_ = 1 - 0.7 * np.clip(1 - np.abs(sd - POCK * 0.45) / (POCK * 0.8), 0, 1)
    floor_ = (img * 0.55 + 6) * (0.15 + 1.25 * nd)[..., None] * ao_[..., None] * (1 - 0.6 * shadow_)[..., None]
    wall_m = pocket * (1 - tri_m)
    out = img * (1 - wall_m[..., None]) + floor_ * wall_m[..., None]
    img = out * (1 - tri_m[..., None]) + face_rgb * tri_m[..., None]


vector_arrow(-1)
vector_arrow(1)

# ------------------------------------------------------------------ 7. footswitch (photo) + jacks (photo)
def black_tint(rgba):
    out = rgba.astype(np.float32).copy()
    c = out[..., :3] * np.array([0.9675, 1.0087, 1.047], np.float32)
    out[..., :3] = 255.0 * np.power(np.clip(c / 255.0, 0, 1), 1.25) * 0.85
    return out


FS = (FS_C[0] / AS, FS_C[1] / AS)
FS_FLATS = 122.0 * FS_FLATS_PX / 250.0                    # sprite: 250 px between the flats at 122
FSPR, FSREFL, FS_SZ, FS_BTN = PC.footswitch_sprite(FS_FLATS, AS, refl_colour=(60.0, 160.0, 255.0))
n = FSPR.shape[0]
x0f, y0f = int(round((FS[0] - FS_SZ / 2) * AS)), int(round((FS[1] - FS_SZ / 2) * AS))
fa = np.zeros((H, W), np.float32)
fa[y0f:y0f + n, x0f:x0f + n] = np.clip((FSPR[..., 3] / 255.0 - 0.86) / 0.14, 0, 1)
contact = gblur(fa, 3.0 * AS); soft = gblur(shift_px(fa, 6, 6), 12.0 * AS)
o = np.clip(contact * 0.55 + soft * 0.8, 0, 0.5) * 0.9 * (1.0 - fa)
img *= (1.0 - o * inside_body)[..., None]
FSPR = black_tint(FSPR)
FSPR[..., :3] = 255.0 * np.power(np.clip(FSPR[..., :3] / 255.0, 0, 1), 1.15) * 0.80
over(img, FSPR, x0f, y0f)
RING_R = FS_BTN * 33.2 / 31.5
RING_W = 2.1 * FS_BTN / 31.5
RING_K = FS_BTN / 31.5

JACK_LEN = JACK_LEN_PX / AS
JACK_Y = JACK_Y_PX / AS
alpha = inside_body.copy()
for side in (-1, 1):
    j, jw_u, jh_u = PC.jack_sprite(JACK_LEN, AS, side)
    jh, jw = j.shape[:2]
    x0j = 0 if side < 0 else W - jw                       # face of the cut 1.5 px under the edge of the enclosure
    y0j = int(round(JACK_Y_PX - jh / 2))
    occ = np.zeros((H, W), np.float32); occ[y0j:y0j + jh, x0j:x0j + jw] = j[..., 3] / 255.0
    oo = np.clip(gblur(occ, 3 * AS) * 0.9 + gblur(occ, 16 * AS) * 1.4, 0, 0.55) * (0.25 if side > 0 else 0.12)
    img *= (1 - oo * inside_body)[..., None]
    j = black_tint(j)
    over(img, j, x0j, y0j)
    ja = np.zeros((H, W), np.float32); ja[y0j:y0j + jh, x0j:x0j + jw] = j[..., 3] / 255.0
    alpha = np.maximum(alpha, ja)

FINAL = np.clip(img, 0, 255)
Image.fromarray(np.dstack([FINAL, alpha * 255]).astype(np.uint8), 'RGBA').save(os.path.join(ROOT, 'background.png'), optimize=True)


# ------------------------------------------------------------------ 8. pressed states
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


ARROW_L = (np.mean([p[0] for p in ARROW_TRI[-1]]) / AS, 434.0 / AS)
ARROW_R = (np.mean([p[0] for p in ARROW_TRI[1]]) / AS, 434.0 / AS)
save_pressed('left', ARROW_L[0], ARROW_L[1], 40.0, gblur(arrow_masks[-1], 1.2))
save_pressed('right', ARROW_R[0], ARROW_R[1], 40.0, gblur(arrow_masks[1], 1.2))
BTN_R = CAP_R / AS
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

# ------------------------------------------------------------------ 9. Layout.h + layout.json
PRESET = (SLOT_I[0] / AS, SLOT_I[1] / AS, (SLOT_I[2] - SLOT_I[0]) / AS, (SLOT_I[3] - SLOT_I[1]) / AS)
BX0, BW = OUTER[0] / AS, (OUTER[2] - OUTER[0]) / AS
lay = {'frame': [FW, FH], 'box': [BX0, 0, BW, FH], 'preset': list(PRESET), 'glass': list(GLASS), 'status': list(STATUS),
       'arrowL': [ARROW_L[0] - 34, ARROW_L[1] - 40, 68, 80], 'arrowR': [ARROW_R[0] - 34, ARROW_R[1] - 40, 68, 80],
       'mix': [MIX[0], MIX[1], BTN_R + 2], 'fav': [FAV[0], FAV[1], BTN_R + 2], 'foot': [FS[0], FS[1], FS_FLATS * 0.56],
       'ring': [FS[0], FS[1], RING_R, RING_W, RING_K], 'jackY': JACK_Y,
       'texts': 'design/Semitones_Texts.svg', 'sprites': sprites}
json.dump(lay, open(os.path.join(HERE, 'layout.json'), 'w'), indent=1)
F = lambda v: f"{v:.2f}f"
R4 = lambda r: "{ %s, %s, %s, %s }" % tuple(F(v) for v in r)
with open(os.path.join(ROOT, 'Layout.h'), 'w') as fo:
    fo.write(f"// generated by tools/make_art_v3.py - do not edit (frame {FW} x {FH} units = background.png {W} x {H} px / 2)\n")
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
    fo.write(f"constexpr float jackY = {F(JACK_Y)};\n")
    fo.write("}\n")
print('ok', W, H, 'ring', RING_R, RING_W)
