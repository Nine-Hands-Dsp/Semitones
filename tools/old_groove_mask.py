"""One-off: mask of the grooves of the first v2 artwork (photo bands), measured on background.png before the
redraw of the bands (Hough lines of the groove edges + closing + a few polygons where the bands meet the parts).
-> design/v2src/old_groove_mask.png (1235 x 1800), read by regroove.py to erase the old bands."""
import os, cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
im = cv2.imread(os.path.join(ROOT, 'background.png'))
g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
e = cv2.Canny(g, 40, 110)
L = cv2.HoughLinesP(e, 1, np.pi / 720, 50, minLineLength=40, maxLineGap=6).reshape(-1, 4)
H, W = g.shape
m = np.zeros((H, W), np.uint8)
PROT = [(302, 196, 928, 678), (320, 55, 935, 165), (130, 345, 290, 525), (945, 345, 1105, 525), (460, 1370, 780, 1700)]
for x1, y1, x2, y2 in L:
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    if any(b[0] <= mx <= b[2] and b[1] <= my <= b[3] for b in PROT):
        continue
    ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
    if (ang < 2 or ang > 178) and (my < 24 or my > 1772):
        continue
    if 88 < ang < 92 and (mx < 64 or mx > 1170 or not (230 < mx < 330 or 840 < mx < 940)):
        continue
    cv2.line(m, (int(x1), int(y1)), (int(x2), int(y2)), 255, 7)
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
# where the bands meet the display bezel and run past MIX (bent by the old warp)
polys = [[(282, 190), (320, 190), (320, 310), (282, 270)],
         [(270, 440 + 0.6055 * 270), (560, 440 + 0.6055 * 560), (560, 610 + 0.6055 * 560), (270, 610 + 0.6055 * 270)],
         [(480, 705), (925, 950), (925, 1045), (480, 800)],                       # old lower band, right part
         [(470, 1150), (560, 1150), (840, 1380), (840, 1500), (760, 1440), (470, 1250)]]   # past the footswitch
for p in polys:
    cv2.fillPoly(m, [np.round(np.array(p)).astype(np.int32)], 255)
m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
cv2.imwrite(os.path.join(ROOT, 'design', 'v2src', 'old_groove_mask.png'), m)
print('ok', (m > 0).mean())
