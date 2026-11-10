"""Cleans the user's validated pedal picture (design/v2src/source_pedal.png, 1800 x 1235, pedal in the middle):
removes what the plugin redraws or what comes from the photo kit:
  screws (removed by request), printed texts (Semitones, MIX, FAV, IN, OUT, STUDIO WIGA logo -> vector in the plugin),
  preset text + chevron (drawn by the plugin), the footswitch and its blue ring (photo footswitch + Octopus ring),
  the jacks (photo jacks).
Vertical brushing => holes are filled column by column (the streak continues), or along the band direction where an
embossed band crosses the hole (footswitch, lower right screw), with synthetic vertical streaks added.
Output: design/v2src/clean_pedal.png (same size)."""
import os, sys
import numpy as np, cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, 'design', 'v2src', 'source_pedal_black.png')   # user's picture, black version (10 nov.)
im = np.asarray(Image.open(SRC).convert('RGBA')).astype(np.float32)
rgb, alpha = im[..., :3].copy(), im[..., 3].copy()
H, W = alpha.shape
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
L = rgb.mean(2)
rng = np.random.default_rng(3)

# streak texture statistics from a clean patch (vertical brushing)
patch = L[620:680, 640:740]
hp = patch - cv2.GaussianBlur(patch, (0, 0), 6)
STREAK = float(hp.std())


TEX_SRC = rgb[780:1040, 612:752].mean(2)
TEX_HP = TEX_SRC - cv2.GaussianBlur(TEX_SRC, (0, 0), sigmaX=3, sigmaY=40)


def real_tex():
    """real brushed grain tiled over the picture (mirrored horizontally so the tiles join)"""
    t = np.concatenate([TEX_HP, TEX_HP[:, ::-1]], 1)
    t = np.concatenate([t, t[::-1]], 0)
    return np.tile(t, (H // t.shape[0] + 1, W // t.shape[1] + 1))[:H, :W].astype(np.float32)


def streaks(shape_mask):
    """synthetic vertical brushing (columns of noise, slowly varying along y)"""
    n = rng.normal(0, 1, (H // 6 + 2, W)).astype(np.float32)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    n = cv2.GaussianBlur(n, (0, 0), sigmaX=0.6, sigmaY=3)
    n = n / (n.std() + 1e-6) * STREAK * 0.9
    return n * shape_mask


def vfill(mask):
    """column interpolation between the nearest clean pixels above and below"""
    global rgb
    m = mask > 0.5
    ys, xs = np.nonzero(m)
    for x in np.unique(xs):
        col = m[:, x]
        y = 0
        while y < H:
            if col[y]:
                y0 = y
                while y < H and col[y]:
                    y += 1
                y1 = y
                a, b = max(y0 - 1, 0), min(y1, H - 1)
                ca = np.median(rgb[max(a - 5, 0):a + 1, x], 0); cb = np.median(rgb[b:min(b + 6, H), x], 0)
                t = (np.arange(y0, y1) - a) / max(b - a, 1)
                rgb[y0:y1, x] = ca[None] * (1 - t[:, None]) + cb[None] * t[:, None]
            else:
                y += 1
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
    rgb += (real_tex() * soft * 0.55)[..., None]


def dfill(mask, d, maxd=400):
    """interpolation along direction d (unit vector) between the nearest clean pixels on both sides"""
    global rgb
    m = mask > 0.5
    ys, xs = np.nonzero(m)
    px, py = xs.astype(np.float32), ys.astype(np.float32)
    k = np.zeros((81, 81), np.float32)
    for t in np.linspace(-38, 38, 153):
        k[int(round(40 + d[1] * t)), int(round(40 + d[0] * t))] = 1
    k /= k.sum()
    keep = (~m).astype(np.float32)
    src = cv2.filter2D(rgb * keep[..., None], -1, k, borderType=cv2.BORDER_REPLICATE) / \
        np.maximum(cv2.filter2D(keep, -1, k, borderType=cv2.BORDER_REPLICATE), 1e-3)[..., None]
    res = {}
    for sgn in (1, -1):
        dist = np.full(len(xs), -1.0, np.float32)
        for t in range(1, maxd):
            qx, qy = px + sgn * d[0] * t, py + sgn * d[1] * t
            qi, qj = np.clip(np.round(qy).astype(int), 0, H - 1), np.clip(np.round(qx).astype(int), 0, W - 1)
            hit = (~m[qi, qj]) & (dist < 0)
            dist[hit] = t
            if (dist >= 0).all():
                break
        dist[dist < 0] = maxd
        qi_, qj_ = np.clip(np.round(py + sgn * d[1] * (dist + 2)).astype(int), 0, H - 1), np.clip(np.round(px + sgn * d[0] * (dist + 2)).astype(int), 0, W - 1)
        dist[(alpha[qi_, qj_] < 200) | (qi_ > 1212) | (qj_ > 1248) | (qj_ < 552)] = 1e4      # ray left the flat top face: ignore that side
        qx, qy = (px + sgn * d[0] * (dist + 1.0)).astype(np.float32), (py + sgn * d[1] * (dist + 1.0)).astype(np.float32)
        x0i = np.clip(np.floor(qx).astype(int), 0, W - 2); y0i = np.clip(np.floor(qy).astype(int), 0, H - 2)
        fx = np.clip(qx - x0i, 0, 1)[:, None]; fy = np.clip(qy - y0i, 0, 1)[:, None]
        val = (src[y0i, x0i] * (1 - fx) * (1 - fy) + src[y0i, x0i + 1] * fx * (1 - fy)
               + src[y0i + 1, x0i] * (1 - fx) * fy + src[y0i + 1, x0i + 1] * fx * fy)
        res[sgn] = (dist, val)
    (d1, v1), (d2, v2) = res[1], res[-1]
    t = (d2 / (d1 + d2))[:, None]
    new = rgb.copy()
    new[ys, xs] = v2 * (1 - t) + v1 * t
    f = np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 4) * 2 - 0.15, 0, 1)[..., None] * m[..., None]
    rgb = rgb * (1 - f) + new * f
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)
    rgb += (real_tex() * soft * 0.9)[..., None]


def disc(cx, cy, r):
    return (np.hypot(XX - cx, YY - cy) < r).astype(np.float32)


def box(x0, y0, x1, y1):
    return ((XX >= x0) & (XX < x1) & (YY >= y0) & (YY < y1)).astype(np.float32)


def ink(region, thr=10, grow=3):
    """printed / engraved marks inside a box: deviation from the column-smoothed metal"""
    sm = cv2.GaussianBlur(L, (0, 0), sigmaX=0.5, sigmaY=12)
    dev = np.abs(L - sm) * region
    m = (dev > thr).astype(np.uint8)
    m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return m.astype(np.float32) * region


BAND_DIR = np.array([1.0, 0.8659]); BAND_DIR /= np.linalg.norm(BAND_DIR)

def vcopy(mask, dy):
    global rgb
    sh = np.roll(rgb, -dy, axis=0)                      # pixel (y) <- (y + dy)
    ring = np.clip(cv2.GaussianBlur(mask, (0, 0), 6) - mask, 0, 1)
    w = ring[..., None] + 1e-6
    g = ((rgb - sh) * w).sum((0, 1)) / w.sum((0, 1))
    num = cv2.GaussianBlur((rgb - sh) * ring[..., None], (0, 0), 28)
    den = cv2.GaussianBlur(ring, (0, 0), 28)[..., None]
    corr = (num + g * 0.03) / (den + 0.03)              # local (low-frequency) tone match, global where unsure
    m = cv2.GaussianBlur(mask, (0, 0), 4.0)[..., None]
    rgb = rgb * (1 - m) + (sh + corr) * m


# --- printed texts
TEXTS = {'mix': (744, 468, 792, 492), 'fav': (1010, 468, 1056, 492),
         'out': (566, 556, 596, 626), 'in': (1205, 572, 1236, 624), 'logo': (774, 1160, 1024, 1198)}
vcopy(box(742, 682, 902, 750), -70)
def hcopy(mask, dx):
    global rgb
    sh = np.roll(rgb, dx, axis=1)                       # pixel (x) <- (x - dx)
    ring = np.clip(cv2.GaussianBlur(mask, (0, 0), 6) - mask, 0, 1)
    w = ring[..., None] + 1e-6
    g = ((rgb - sh) * w).sum((0, 1)) / w.sum((0, 1))
    num = cv2.GaussianBlur((rgb - sh) * ring[..., None], (0, 0), 28)
    den = cv2.GaussianBlur(ring, (0, 0), 28)[..., None]
    corr = (num + g * 0.03) / (den + 0.03)              # local (low-frequency) tone match, global where unsure
    m = cv2.GaussianBlur(mask, (0, 0), 4.0)[..., None]
    rgb = rgb * (1 - m) + (sh + corr) * m


hcopy(box(884, 682, 1040, 750), 262)
# arrow buttons: removed, redrawn as vector reliefs by make_art_v2 (with their milled sockets)
vcopy(box(1104, 244, 1212, 358), -112)
vfill(box(588, 244, 694, 358))
for k in ('mix', 'fav', 'out', 'in', 'logo'):
    x0, y0, x1, y1 = TEXTS[k]
    reg = box(x0, y0, x1, y1)
    sm = cv2.GaussianBlur(L, (0, 0), 4)
    m = ((np.abs(L - sm) > 8) * reg).astype(np.uint8)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8)) * reg.astype(np.uint8)
    img8 = np.clip(rgb, 0, 255).astype(np.uint8)
    inp = cv2.inpaint(img8, m, 4, cv2.INPAINT_TELEA).astype(np.float32)
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)[..., None]
    rgb = rgb * (1 - soft) + (inp + real_tex()[..., None] * 0.5) * soft
L = rgb.mean(2)

# --- footswitch + blue ring + its glow (the band crosses it)
FSC = (895.5, 1007.5)
ref = rgb[620:680, 640:740].reshape(-1, 3).mean(0)
spill = np.clip((235 - np.hypot(XX - FSC[0], YY - FSC[1])) / 40, 0, 1) * (alpha > 0)
lum = rgb @ (ref / ref.sum())
neutral = lum[..., None] * (ref / ref.mean())[None, None, :]
rgb = rgb * (1 - spill[..., None]) + neutral * spill[..., None]
# remove the glow (light added to the metal around the ring): radial gain from the median brightness per radius
rr = np.hypot(XX - FSC[0], YY - FSC[1])
Lm = rgb.mean(2)
inb = (alpha > 250)
base = np.median(Lm[(rr > 250) & (rr < 300) & inb])
# gain per (angle, radius): removes the blue glow (too bright) and the footswitch drop shadow (too dark)
NA, R0, R1, DR = 24, 140, 256, 6
th = np.mod(np.arctan2(YY - FSC[1], XX - FSC[0]), 2 * np.pi)
radii = np.arange(R0, R1, DR) + DR / 2
G = np.ones((NA, len(radii)), np.float32)
for ia in range(NA):
    asel = (th >= ia * 2 * np.pi / NA) & (th < (ia + 1) * 2 * np.pi / NA) & inb
    for ir, r0 in enumerate(range(R0, R1, DR)):
        sel = asel & (rr >= r0) & (rr < r0 + DR)
        if sel.sum() > 30:
            G[ia, ir] = base / max(np.median(Lm[sel]), 1e-3)
G = np.clip(G, 0.55, 1.8)
G[:, -3:] = 1.0
G = cv2.GaussianBlur(np.concatenate([G, G, G], 0), (0, 0), sigmaX=0.8, sigmaY=1.0)[NA:2 * NA]
fa = th / (2 * np.pi) * NA - 0.5
fr = np.clip((rr - R0) / DR - 0.5, 0, len(radii) - 1)
gmap = cv2.remap(np.concatenate([G, G[:1]], 0).astype(np.float32), fr.astype(np.float32), np.mod(fa, NA).astype(np.float32), cv2.INTER_LINEAR)
gmap = np.where((rr < R1) & (rr >= R0 - 20), gmap, 1.0)
rgb *= gmap[..., None]
pr = np.stack([radii, np.median(G, 0)], 1)
print('glow gain at 150/200/240', np.interp([150, 200, 240], pr[:, 0], pr[:, 1]))
dfill(disc(FSC[0], FSC[1] + 4, 156), BAND_DIR)
L = rgb.mean(2)


# --- preset slot: remove "Custom" + chevron (rows of the slot floor are uniform)
sx0, sy0, sx1, sy1 = 726, 61, 1084, 97
for y in range(sy0, sy1):
    row = rgb[y, sx0:sx1]
    lum = row.mean(1)
    med = np.median(row[lum < np.percentile(lum, 50)], 0)
    rgb[y, sx0:sx1] = med
# --- jacks: transparent (photo jacks placed by make_art)
alpha[:, :537] = 0; alpha[:, 1263:] = 0

out = np.dstack([np.clip(rgb, 0, 255), alpha]).astype(np.uint8)
Image.fromarray(out, 'RGBA').save(os.path.join(ROOT, 'design', 'v2src', 'clean_pedal.png'))
print('ok, streak std', STREAK)
