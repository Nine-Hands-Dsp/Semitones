"""Preview of the v2 artwork with what the plugin draws on top (vector texts, logo, lit display, ring).
python3 -I tools/preview_art.py [semitones] [mix 0/1] [comet angle deg] -> design/preview_v2.png"""
import os, sys, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import svg_raster
AS = 2
lay = json.load(open(os.path.join(HERE, 'layout.json')))
SP = lay['sprites']
st = int(sys.argv[1]) if len(sys.argv) > 1 else 12
mix_on = (sys.argv[2] != '0') if len(sys.argv) > 2 else True
ang = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
ring_on = (sys.argv[4] != '0') if len(sys.argv) > 4 else True
OUT = sys.argv[5] if len(sys.argv) > 5 else 'preview_v2.png'
F = lambda n: os.path.join(ROOT, 'assets', n)

bg = Image.open(os.path.join(ROOT, 'background.png')).convert('RGBA')
W, H = bg.size
canvas = Image.new('RGBA', (W, H), (18, 18, 20, 255))
canvas.alpha_composite(bg)


def put(name, opacity=1.0, rot=None):
    x, y, w, h = SP[name]
    im = Image.open(F(name + '.png')).convert('RGBA')
    if rot is not None:
        im = im.rotate(-rot, resample=Image.BICUBIC)
    if opacity < 1:
        a = np.asarray(im).copy(); a[..., 3] = (a[..., 3] * opacity).astype(np.uint8); im = Image.fromarray(a)
    canvas.alpha_composite(im, (int(round(x * AS)), int(round(y * AS))))


# ring: faceplate light + lit pipe + comet
def octo_ring(lvl=1.0, col=(40, 168, 255)):
    """Octopus drawRing(): halo strokes 13 -> 3 (alpha .045), diffuser (alpha .85), lighter core (alpha .9)"""
    cx, cy, R, Wd, K = lay['ring']
    SS = 4
    def stroke(width_u, rgb, a):
        m = Image.new('L', (W * SS, H * SS), 0)
        r0, r1 = (R - width_u / 2) * AS * SS, (R + width_u / 2) * AS * SS
        dd = ImageDraw.Draw(m)
        dd.ellipse((cx * AS * SS - r1, cy * AS * SS - r1, cx * AS * SS + r1, cy * AS * SS + r1), fill=255)
        dd.ellipse((cx * AS * SS - r0, cy * AS * SS - r0, cx * AS * SS + r0, cy * AS * SS + r0), fill=0)
        m = m.resize((W, H), Image.BOX).point(lambda p: int(p * a * lvl))
        canvas.paste(Image.new('RGBA', (W, H), rgb + (255,)), (0, 0), m)
    hw = 13.0
    while hw > 3.0:
        stroke(hw * K, col, 0.045 * 1.15); hw -= 1.5          # glow +15 % (user, 10 nov.)
    stroke(Wd, col, 0.85)
    core = tuple(int(c + (255 - c) * 0.55) for c in col)
    stroke(Wd * 0.42, core, 0.9)

if ring_on:
    octo_ring()
# display value
segs = {0: 'abcdef', 1: 'bc', 2: 'abged', 3: 'abgcd', 4: 'fgbc', 5: 'afgcd', 6: 'afgedc', 7: 'abc', 8: 'abcdefg', 9: 'abcdfg'}
v = abs(st)
if st > 0: [put('seg_s_' + k) for k in 'lrud']
elif st < 0: [put('seg_s_' + k) for k in 'lr']
if v >= 10: [put('seg_t_' + k) for k in segs[v // 10]]
[put('seg_u_' + k) for k in segs[v % 10]]
if mix_on: put('seg_mix')

d = ImageDraw.Draw(canvas)
font = lambda n, s: ImageFont.truetype(F(n), int(round(s * AS)))
BLUE = (64, 175, 255)


def text_c(xy, t, f, fill, anchor='mm'):
    d.text((xy[0] * AS, xy[1] * AS), t, font=f, fill=fill, anchor=anchor)


def engraved(xy, t, f, depth=1.0, anchor='mm', base=(196, 199, 204), rot=0):
    m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).text((xy[0] * AS, xy[1] * AS), t, font=f, fill=255, anchor=anchor)
    if rot:
        m = m.rotate(rot, center=(xy[0] * AS, xy[1] * AS), resample=Image.BICUBIC)
    sh = ImageChops.offset(m, int(round(1.2 * depth * AS)), int(round(1.6 * depth * AS)))
    hi = ImageChops.subtract(sh, m)          # light catches the lower-right lip of the cut
    lo = ImageChops.subtract(m, ImageChops.offset(m, -int(round(1.0 * depth * AS)), -int(round(1.3 * depth * AS))))
    canvas.paste(Image.new('RGBA', (W, H), (235, 238, 242, 255)), (0, 0), hi.point(lambda p: p * 0.35))
    canvas.paste(Image.new('RGBA', (W, H), base + (255,)), (0, 0), m.point(lambda p: p * 0.92))
    canvas.paste(Image.new('RGBA', (W, H), (5, 6, 8, 255)), (0, 0), lo.point(lambda p: p * 0.75))


def deboss_mask(m, fill=(142, 145, 149), depth=1.0):            # floor -20 % (user, 10 nov.)
    """light, natural deboss: lettering pressed into the metal, lit from the top left (walls at the top left in
    shadow, lip at the bottom right catching the light, soft inner shadow on the floor), no shadow outside"""
    import numpy as np
    mf = np.asarray(m).astype(np.float32) / 255.0
    c = np.asarray(canvas).astype(np.float32)
    rgb, al = c[..., :3], c[..., 3:]
    blur = lambda a, s_: np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(s_))).astype(np.float32) / 255.0
    dx, dy = int(round(0.9 * depth * AS)), int(round(1.2 * depth * AS))
    rgb[:] = rgb * (1 - mf[..., None] * 0.9) + np.array(fill, np.float32) * (mf[..., None] * 0.9)
    # soft inner shadow on the floor, along the top-left walls
    occ = blur(np.roll(np.roll(1 - mf, dx, 1), dy, 0), 0.8 * AS) * mf
    rgb *= (1 - 0.45 * occ)[..., None]
    # walls: top-left walls dark, bottom-right lip lit
    hgt = blur(mf, 0.55 * AS)
    gy, gx = np.gradient(hgt)
    face = (gx * -0.6 + gy * -0.8) * AS * 2.4                 # >0 : wall facing the light (bottom-right side)
    wall = np.clip(hgt * (1 - hgt) * 4, 0, 1)
    rgb += (np.clip(face, 0, 1) * wall * 55)[..., None]
    rgb *= (1 - np.clip(-face, 0, 1) * wall * 0.5)[..., None]
    # edge highlight inside the cut: along the bottom edges (horizontal) and up the lower half of the right edges,
    # strongest in the bottom-right corner (user, 10 nov.)
    import cv2
    mb = (mf > 0.5).astype(np.uint8)
    rim = mf - cv2.erode(mf, np.ones((3, 3), np.uint8), iterations=1)          # ~1.5 px band inside the edge
    rim = np.clip(rim, 0, 1)
    hs = blur(mf, 0.8 * AS); ny_, nx_ = np.gradient(hs)
    nn = np.hypot(nx_, ny_) + 1e-6
    ox, oy = -nx_ / nn, -ny_ / nn                                               # outward normal
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(mb, 8)
    yy_ = np.mgrid[0:H, 0:W][0].astype(np.float32)
    top = stats[lab, cv2.CC_STAT_TOP].astype(np.float32); hh = np.maximum(stats[lab, cv2.CC_STAT_HEIGHT], 1).astype(np.float32)
    tdown = np.clip((yy_ - top) / hh, 0, 1)                                     # 0 at the top of the glyph, 1 at the bottom
    w_bottom = np.clip(oy, 0, 1) ** 1.5
    w_right = np.clip(ox, 0, 1) ** 1.5 * np.clip((tdown - 0.5) / 0.25, 0, 1)   # lower half of the right side
    hl = np.clip(w_bottom + w_right, 0, 1) * rim
    hl = blur(hl, 0.35 * AS) * 1.6
    rgb += (np.clip(hl, 0, 1) * float(os.environ.get("EDGE_HL", 38)))[..., None]                                # subtle (user, 10 nov.)
    canvas.paste(Image.fromarray(np.clip(np.dstack([rgb, al]), 0, 255).astype(np.uint8), 'RGBA'), (0, 0))


def text_mask(xy, t, f, rot=0):
    m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).text((xy[0] * AS, xy[1] * AS), t, font=f, fill=255, anchor='mm')
    if rot:
        m = m.rotate(rot, center=(xy[0] * AS, xy[1] * AS), resample=Image.BICUBIC)
    return m


# preset menu text (in its milled slot)
px, py, pw, ph = lay['preset']
text_c((px + pw / 2, py + ph / 2 + 0.5), 'Custom', font('Poppins-Medium.ttf', 19), (225, 230, 236))
d.line([((px + pw - 22) * AS, (py + ph / 2 - 2.5) * AS), ((px + pw - 17) * AS, (py + ph / 2 + 2.5) * AS),
        ((px + pw - 12) * AS, (py + ph / 2 - 2.5) * AS)], fill=(225, 230, 236), width=int(2.2 * AS), joint='curve')
# status line inside the display
sx, sy, sw, sh = lay['status']
text_c((sx + sw / 2 + 12, sy + sh / 2), ('MIX 100 %   ·   ' if mix_on else 'DRY OFF   ·   ') + ('%+d ST' % st),
       font('Poppins-Medium.ttf', 15.5), (70, 165, 240))
# MIX / FAV labels, name, subtitle, IN / OUT
lf = font('Poppins-Medium.ttf', 15)
deboss_mask(text_mask((lay['mix'][0], lay['labelY']), 'MIX', lf), depth=0.6)
deboss_mask(text_mask((lay['fav'][0], lay['labelY']), 'FAV', lf), depth=0.6)
deboss_mask(text_mask(tuple(lay['name']), 'Semitones', font('Poppins-Medium.ttf', 38)), depth=1.0)
deboss_mask(text_mask((lay['outX'], lay['textY']), 'OUT', font('Poppins-Bold.ttf', 20), rot=-90), depth=0.8)
deboss_mask(text_mask((lay['inX'], lay['textY']), 'IN', font('Poppins-Bold.ttf', 20), rot=90), depth=0.8)
# logo (white on the black pedal, vector)
lx, ly, lw = lay['logo']
tmp = '/tmp/claude-0/logo_prev.png'
svg_raster.raster(os.path.join(ROOT, 'design', 'NineHandsLogo_white.svg'), int(lw * AS * 3000 / 2670), tmp)
lg = Image.open(tmp).convert('RGBA'); bb = lg.getbbox(); lg = lg.crop(bb)
lm = Image.new('L', (W, H), 0); lm.paste(lg.split()[3], (int(lx * AS - lg.width / 2), int(ly * AS - lg.height / 2)))
deboss_mask(lm, depth=0.7)

out = os.path.join(ROOT, 'design', OUT)
canvas.save(out)
# same with the real transparency (what the PNG holds)
print(out, canvas.size)
