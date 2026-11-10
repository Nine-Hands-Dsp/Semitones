"""Bands of the faceplate redrawn on the user's vector path (design/Semitones_Design_v001.svg, red FF0000 lines,
frame 1235 x 1800 = background.png).

The look of the bands is NOT re-invented: every straight run of the new path takes its cross-section (bevels, lips,
creases, matte floor, channel, grain) from a run of the photographed bands with the same orientation and the same
order (band / channel), sampled on background.png's first version (photo). Only the path changes:
  1. the old bands are erased (design/v2src/old_groove_mask.png): vertical copies of the brushed metal just above /
     below (the brushing is vertical, so the streaks run on), horizontal copies for the vertical runs, tone ramped
     to both sides;
  2. each run of the new path gets the photo cross-section of its class, mapped 1:1 near the three edges (lips and
     creases keep their width) and stretched only on the flat floors, lit by the local tone of the plate;
  3. the corners are mitred on the bisectors (the three lines of a band meet there), the display, the MIX / FAV
     recesses and the rim of the enclosure stay on top, untouched."""
import os, re, math
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------------------------------------- geometry (px)
BEZEL = (310.5, 202.5, 924.0, 674.5, 36.0)          # display bezel, outer edge (x0, y0, x1, y1, r)
PLATE = (59.5, 17.0, 1175.0, 1781.5, 20.0)          # inside of the enclosure rim
BTN = [((403.5, 761.3), 52.5), ((831.5, 761.3), 52.5)]   # MIX / FAV recesses (outer lip)
PRESET = (325.0, 55.0, 935.0, 165.0)
ARROWS = [(130, 345, 290, 525), (945, 345, 1105, 525)]     # vector arrows, drawn later (never a copy source)

# classes of runs, measured on the photo bands: three edges A (outer edge of the matte band), M (band / channel),
# C (inner edge of the channel), each as two points; the sampled stretch (s0, s1) on x (or y for vertical runs)
OLD = {
    'a': (((251, 0), (251, 100)), ((279, 0), (279, 100)), ((298, 0), (298, 100)), (30, 195)),
    'b': (((908, 662), (1176, 824)), ((919, 623), (1175, 777)), ((920, 601), (1171, 755)), (935, 1130)),
    'c': (((54, 481), (310, 636)), ((54, 525), (345, 700)), ((61, 559), (314, 711)), (110, 280)),
    'd': (((919, 0), (919, 100)), ((880.5, 0), (880.5, 100)), ((859, 0), (859, 100)), (1035, 1125)),
    'e': (((0, 1196), (100, 1196)), ((0, 1166.5), (100, 1166.5)), ((0, 1146), (100, 1146)), (600, 840)),
    'f': (((559.65, 1196.56), (1179.59, 1690.35)), ((462.34, 1167.81), (1179.47, 1742.71)),
          ((403.29, 1145.85), (1179.52, 1765.96)), (870, 1140)),
}
TM = 12.0          # strip margin beyond A and C (px)
KEEP = 5.0         # full weight up to KEEP px outside A / C, then fade to TM
EDGE = 5.0         # 1:1 zone around each edge


def read_path(svg):
    """the 8 red elements of the user's SVG -> two bands, each (A, M, C) polylines with the same vertices count"""
    s = open(svg).read()
    pl = [np.array(list(map(float, m.split())), float).reshape(-1, 2) for m in re.findall(r'points="([^"]+)"', s)]
    ln = [np.array(list(map(float, m)), float).reshape(-1, 2)
          for m in re.findall(r'x1="([\d.]+)" y1="([\d.]+)" x2="([\d.]+)" y2="([\d.]+)"', s)]
    def orient(p, first):                      # first(p) True when p already starts at the right end
        return p if first(p) else p[::-1]
    up = [orient(p, lambda q: q[0, 1] < 20) for p in pl if len(p) == 3 and min(p[0, 1], p[-1, 1]) < 20]
    up.sort(key=lambda p: p[0, 0])                       # A = x 251, M = x 278.6, C = x 298.7
    A_u, M_u, C_u = up
    lo = [orient(p, lambda q: q[0, 0] < 100) for p in pl if len(p) == 5]
    lo.sort(key=lambda p: p[0, 1])                       # A starts at y 491.9, M at y 531.5
    A_l, M_l = lo
    c3 = [orient(p, lambda q: q[0, 0] > 800) for p in pl if len(p) == 3 and min(p[0, 1], p[-1, 1]) > 1000][0]
    diag = [l for l in ln if abs(l[0, 1] - l[1, 1]) > 200][0]
    diag = diag if diag[0, 0] < diag[1, 0] else diag[::-1]     # (55.5,555.2) -> (859.6,1042.4)
    C_l = np.array([diag[0], diag[1], c3[0], c3[1], c3[2]])
    return [('upper', A_u, M_u, C_u, ['a', 'b']), ('lower', A_l, M_l, C_l, ['c', 'd', 'e', 'f'])]


def rrect_sd(x, y, x0, y0, x1, y1, r):
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    qx = np.abs(x - cx) - (hw - r); qy = np.abs(y - cy) - (hh - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def masks(H, W):
    YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
    PX, PY = XX + 0.5, YY + 0.5
    prot = np.clip(0.5 - rrect_sd(PX, PY, *BEZEL), 0, 1)
    for (cx, cy), r in BTN:
        prot = np.maximum(prot, np.clip(r + 0.5 - np.hypot(PX - cx, PY - cy), 0, 1))
    x0, y0, x1, y1 = PRESET
    prot[int(y0):int(y1), int(x0):int(x1)] = 1.0
    plate = np.clip(0.5 - rrect_sd(PX, PY, *PLATE), 0, 1)
    return PX, PY, prot, plate


# ---------------------------------------------------------------------------------------------- 1. erase
def erase(img, hole, bad, maxrun=150):
    """fill `hole` with the brushed metal around it: vertical shifted copies (streaks run on), tone ramped from the
    edge above to the edge below; runs taller than maxrun are left to a horizontal pass"""
    out = img.copy()
    H, W = hole.shape
    done = np.zeros_like(hole)

    def fill_line(arr, hl, bd, dn, along_rows, fixed=None):
        n_ = len(hl)
        idx = np.flatnonzero(np.diff(np.concatenate([[0], hl.astype(np.int8), [0]])))
        for y0, y1 in zip(idx[::2], idx[1::2]):
            n = y1 - y0
            def clean_count(start, step):
                k = 0; y = start
                while 0 <= y < n_ and not bd[y] and k < n:
                    k += 1; y += step
                return k
            ka, kb = clean_count(y0 - 1, -1), clean_count(y1, 1)
            if ka == 0 and kb == 0:
                continue
            i = np.arange(n)
            src = None
            for S in (fixed or ()):                     # same shift on every line: the streaks stay straight
                for sg_ in (1, -1):
                    a0, a1 = y0 + sg_ * S, y1 + sg_ * S
                    if 0 <= a0 and a1 <= n_ and not bd[a0:a1].any():
                        src = a0 + i; break
                if src is not None:
                    break
            if src is None and n > maxrun and max(ka, kb) < n:   # long run: only a straight shifted copy
                continue
            if src is not None:
                pass
            elif ka >= kb:
                k = ka
                if k >= n:
                    src = y0 - n + i
                else:                                                    # ping-pong over the k clean rows
                    j = i % (2 * k); j = np.where(j < k, j, 2 * k - 1 - j); src = y0 - 1 - j[::-1]
            else:
                k = kb
                if k >= n:
                    src = y1 + i
                else:
                    j = i % (2 * k); j = np.where(j < k, j, 2 * k - 1 - j); src = y1 + j
            vals = arr[src].copy()
            m = 6
            corr = np.zeros((n, 3), np.float32)
            t = ((i + 0.5) / n)[:, None]
            ta = tb = None
            if ka > 0:
                ta = arr[max(0, y0 - min(m, ka)):y0].mean(0) - vals[:m].mean(0)
            if kb > 0:
                tb = arr[y1:y1 + min(m, kb)].mean(0) - vals[-m:].mean(0)
            if ta is not None and tb is not None:
                corr = (1 - t) * ta + t * tb
            else:
                corr = np.repeat(((ta if ta is not None else tb))[None], n, 0)
            arr_out = vals + corr
            yield y0, y1, arr_out

    for x in range(W):
        col = img[:, x]
        for y0, y1, v in fill_line(col, hole[:, x], bad[:, x], None, True):
            out[y0:y1, x] = v; done[y0:y1, x] = True
    rest = hole & ~done
    bad2 = bad & ~done
    img2 = out.copy()
    for y in range(H):
        if not rest[y].any():
            continue
        for x0, x1, v in fill_line(img2[y], rest[y], bad2[y] | rest[y], None, False, fixed=(150, 230, 310)):
            out[y, x0:x1] = v; done[y, x0:x1] = True
    left = hole & ~done
    if left.any():                                                       # last resort (tiny bits)
        u8 = np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.inpaint(u8, left.astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32)
    return out


# ---------------------------------------------------------------------------------------------- 2. strips
def _line(p, q):
    p = np.array(p, float); q = np.array(q, float); d = (q - p) / np.linalg.norm(q - p)
    return p, d


def strip(old, tone, cls):
    A, M, C, (s0, s1) = OLD[cls]
    pA, d = _line(*A)
    n = np.array([-d[1], d[0]])
    pC, _ = _line(*C); pM, _ = _line(*M)
    if (pC - pA) @ n < 0:
        n = -n
    oM, oC = (pM - pA) @ n, (pC - pA) @ n
    vert = abs(d[1]) > abs(d[0])
    start = pA + d * (((s0 - pA[1]) / d[1]) if vert else ((s0 - pA[0]) / d[0]))
    end = pA + d * (((s1 - pA[1]) / d[1]) if vert else ((s1 - pA[0]) / d[0]))
    Ls = int(np.linalg.norm(end - start))
    ts = np.arange(-TM - 2, oC + TM + 3, 1.0)
    sg = np.arange(Ls, dtype=float)
    def sample(shift):
        tt = ts[None, :] + shift[:, None]
        px = start[0] + sg[:, None] * d[0] + tt * n[0]
        py = start[1] + sg[:, None] * d[1] + tt * n[1]
        mx, my = (px - 0.5).astype(np.float32), (py - 0.5).astype(np.float32)
        rgb = np.dstack([cv2.remap(old[..., c], mx, my, cv2.INTER_LINEAR) for c in range(3)])
        tn = cv2.remap(tone, mx, my, cv2.INTER_LINEAR)
        return rgb / np.maximum(tn, 1.0)[..., None]
    # straighten: the photo edges are not exactly on the measured lines, so every row is re-centred on the mean
    # cross-section (sub-pixel), with a smooth (quadratic) drift along the run -> sharp lips, no doubled edges
    shift = np.zeros(Ls)
    win = (ts > -6) & (ts < oC + 6)
    cand = np.arange(-3.0, 3.01, 0.25)
    for _ in range(3):
        N = sample(shift)
        Lm = N.mean(2)
        pr = Lm.mean(0)
        best = np.zeros(Ls)
        for i in range(Ls):
            errs = [np.sum((np.interp(ts[win] + c, ts, Lm[i]) - pr[win]) ** 2) for c in cand]
            best[i] = cand[int(np.argmin(errs))]
        co = np.polyfit(sg, best, 2)
        shift = shift + np.polyval(co, sg)
    N = sample(shift)
    prof = N.mean(0)
    det = N - prof[None]
    return dict(prof=prof.astype(np.float32), det=det.astype(np.float32), t0=ts[0], oM=oM, oC=oC, Ls=Ls)


def bilin(a, r, c):
    """bilinear sample of a (rows x cols x ch) at float (r, c), clamped"""
    r = np.clip(r, 0, a.shape[0] - 1.001); c = np.clip(c, 0, a.shape[1] - 1.001)
    r0 = np.floor(r).astype(int); c0 = np.floor(c).astype(int); fr = (r - r0)[:, None]; fc = (c - c0)[:, None]
    return (a[r0, c0] * (1 - fr) * (1 - fc) + a[r0 + 1, c0] * fr * (1 - fc) + a[r0, c0 + 1] * (1 - fr) * fc
            + a[r0 + 1, c0 + 1] * fr * fc)


def plate_tone(img, valid, s=22.0):
    L = img.mean(2)
    num = cv2.GaussianBlur(L * valid, (0, 0), s); den = cv2.GaussianBlur(valid, (0, 0), s)
    return num / np.maximum(den, 1e-3)


# ---------------------------------------------------------------------------------------------- 3. draw
def draw(clean, tone, PX, PY, bands, strips):
    out = clean.copy()
    acc_w = np.zeros(clean.shape[:2], np.float32)
    rng = np.random.default_rng(7)
    for name, A, M, C, classes in bands:
        nseg = len(A) - 1
        for k in range(nseg):
            st = strips[classes[k]]
            A0, A1, M0, M1, C0, C1 = A[k], A[k + 1], M[k], M[k + 1], C[k], C[k + 1]
            d = (A1 - A0) / np.linalg.norm(A1 - A0)
            n = np.array([-d[1], d[0]])
            if (C0 - A0) @ n < 0:
                n = -n
            qM = 0.5 * ((M0 - A0) @ n + (M1 - A1) @ n)
            qC = 0.5 * ((C0 - A0) @ n + (C1 - A1) @ n)
            q = (PX - A0[0]) * n[0] + (PY - A0[1]) * n[1]
            sig = (PX - A0[0]) * d[0] + (PY - A0[1]) * d[1]
            sel = (q > -TM) & (q < qC + TM)
            # mitre: between the bisectors through (A_k, C_k) and (A_k+1, C_k+1)
            def side(P0, P1, ref):
                e = P1 - P0
                s_ = (PX - P0[0]) * e[1] - (PY - P0[1]) * e[0]
                r_ = (ref[0] - P0[0]) * e[1] - (ref[1] - P0[1]) * e[0]
                return s_ * np.sign(r_) >= 0
            mid = 0.5 * (A0 + A1) * 0.5 + 0.5 * (C0 + C1) * 0.5
            if k > 0:
                sel &= side(A0, C0, mid)
            if k < nseg - 1:
                sel &= side(A1, C1, mid)
            ys, xs = np.nonzero(sel)
            qq = q[ys, xs]
            # 1:1 near the edges, stretch on the floors
            kn_new = [-TM - 2, -EDGE, EDGE, qM - EDGE, qM + EDGE, qC - EDGE, qC + EDGE, qC + TM + 2]
            kn_old = [-TM - 2, -EDGE, EDGE, st['oM'] - EDGE, st['oM'] + EDGE, st['oC'] - EDGE, st['oC'] + EDGE,
                      st['oC'] + TM + 2]
            t = np.interp(qq, kn_new, kn_old)
            ti = (t - st['t0']).astype(np.float32)
            # along the run: tiles of the photo stretch, cross-faded
            Ls = st['Ls']; X = min(40, Ls // 3); Pd = Ls - X
            s_ = sig[ys, xs] + 400.0 + rng.uniform(0, Pd)
            i_ = np.floor(s_ / Pd); f_ = s_ - i_ * Pd
            prof = np.stack([np.interp(ti, np.arange(len(st['prof'])), st['prof'][:, c]) for c in range(3)], 1)
            det = st['det']
            def samp(si):
                return bilin(det, si, ti)
            d1 = samp(f_)
            wb = np.clip(f_ / X, 0, 1)[:, None]
            d2 = samp(np.minimum(f_ + Pd, Ls - 1))
            dv = d1 * wb + d2 * (1 - wb)
            val = (prof + dv) * tone[ys, xs][:, None]
            w = np.clip(np.minimum(qq + TM, qC + TM - qq) / (TM - KEEP), 0, 1)
            w = w * w * (3 - 2 * w)
            out[ys, xs] = out[ys, xs] * (1 - w[:, None]) + val * w[:, None]
            acc_w[ys, xs] = np.maximum(acc_w[ys, xs], w)
    return out, acc_w


def apply(old, body):
    """old: warped photo (H x W x 3 float, 0-255) -> same with the bands on the new path"""
    H, W = old.shape[:2]
    PX, PY, prot, plate = masks(H, W)
    oldm = cv2.imread(os.path.join(ROOT, 'design', 'v2src', 'old_groove_mask.png'), 0) > 0
    inside = (plate > 0.5) & (prot < 0.5) & (body > 0.5)
    hole = oldm & inside
    bad = oldm | ~inside
    for x0, y0, x1, y1 in ARROWS:
        bad[y0:y1, x0:x1] = True
    for (cx, cy), r in BTN:                                   # recess shading reaches a little beyond its lip
        bad |= np.hypot(PX - cx, PY - cy) < r + 14
    bad |= rrect_sd(PX, PY, *BEZEL) < 5
    clean = erase(old, hole, bad)
    valid = (inside & ~oldm).astype(np.float32)
    tone_old = plate_tone(old, valid)                                       # tone of the plate (photo)
    tone_new = plate_tone(clean, inside.astype(np.float32))
    strips = {c: strip(old, tone_old, c) for c in OLD}
    bands = read_path(os.path.join(ROOT, 'design', 'Semitones_Design_v001.svg'))
    drawn, w = draw(clean, tone_new, PX, PY, bands, strips)
    if os.environ.get('RG_DEBUG'):
        from PIL import Image
        Image.fromarray(np.clip(clean, 0, 255).astype(np.uint8)).save('/tmp/claude-0/dbg_clean.png')
        Image.fromarray(np.clip(tone_new * 3, 0, 255).astype(np.uint8)).save('/tmp/claude-0/dbg_tone.png')
    keep = np.maximum(prot, 1 - plate)[..., None]
    return old * keep + drawn * (1 - keep), clean
