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
if ring_on:
    put('ringlight', 0.85); put('ring_lit', 0.9); put('ring_comet', 1.0, rot=ang)
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


def engraved(xy, t, f, depth=1.0, anchor='mm', base=(28, 30, 33), rot=0):
    m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).text((xy[0] * AS, xy[1] * AS), t, font=f, fill=255, anchor=anchor)
    if rot:
        m = m.rotate(rot, center=(xy[0] * AS, xy[1] * AS), resample=Image.BICUBIC)
    sh = ImageChops.offset(m, int(round(1.2 * depth * AS)), int(round(1.6 * depth * AS)))
    hi = ImageChops.subtract(sh, m)          # light catches the lower-right lip of the cut
    lo = ImageChops.subtract(m, ImageChops.offset(m, -int(round(1.0 * depth * AS)), -int(round(1.3 * depth * AS))))
    canvas.paste(Image.new('RGBA', (W, H), (235, 238, 242, 255)), (0, 0), hi.point(lambda p: p * 0.55))
    canvas.paste(Image.new('RGBA', (W, H), base + (255,)), (0, 0), m.point(lambda p: p * 0.80))
    canvas.paste(Image.new('RGBA', (W, H), (5, 6, 8, 255)), (0, 0), lo.point(lambda p: p * 0.6))


# preset menu text (in its milled slot)
px, py, pw, ph = lay['preset']
text_c((px + pw / 2, py + ph / 2 + 0.5), 'Capo  +2', font('Poppins-Medium.ttf', 14.5), (225, 230, 236))
d.polygon([((px + pw - 20) * AS, (py + ph / 2 - 2.5) * AS), ((px + pw - 12) * AS, (py + ph / 2 - 2.5) * AS),
           ((px + pw - 16) * AS, (py + ph / 2 + 2.5) * AS)], fill=(160, 168, 178))
# status line inside the display
sx, sy, sw, sh = lay['status']
text_c((sx + sw / 2 + 12, sy + sh / 2), ('MIX 100 %   ·   ' if mix_on else 'DRY OFF   ·   ') + ('%+d ST' % st),
       font('Poppins-Medium.ttf', 11.5), (70, 165, 240))
# MIX / FAV labels, name, subtitle, IN / OUT
lf = font('Poppins-Medium.ttf', 14)
engraved((lay['mix'][0], lay['labelY']), 'MIX', lf, 0.6)
engraved((lay['fav'][0], lay['labelY']), 'FAV', lf, 0.6)
engraved(tuple(lay['name']), 'Semitones', font('Poppins-Medium.ttf', 46), 1.0)
engraved((lay['name'][0], lay['subY']), 'POLYPHONIC  PITCH  SHIFTER', font('Poppins-Medium.ttf', 9.5), 0.5)
engraved((lay['outX'], lay['jackY']), 'OUT', font('Poppins-Medium.ttf', 11), 0.5, rot=90)
engraved((lay['inX'], lay['jackY']), 'IN', font('Poppins-Medium.ttf', 11), 0.5, rot=-90)
# logo (black, vector)
lx, ly, lw = lay['logo']
tmp = '/tmp/claude-0/logo_prev.png'
svg_raster.raster(os.path.join(ROOT, 'NineHandsLogo.svg'), int(lw * AS * 3000 / 2670), tmp)
lg = Image.open(tmp).convert('RGBA'); bb = lg.getbbox(); lg = lg.crop(bb)
canvas.alpha_composite(lg, (int(lx * AS - lg.width / 2), int(ly * AS - lg.height / 2)))

out = os.path.join(ROOT, 'design', OUT)
canvas.save(out)
# same with the real transparency (what the PNG holds)
print(out, canvas.size)
