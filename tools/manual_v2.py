# SEMITONES (NINE HANDS | DSP) - notice PDF en francais, illustree avec le rendu reel du plugin (tools/render_main.cpp).
# Usage : python3 tools/manual_v2.py <dossier des captures de SemitonesRender> <fichier PDF de sortie>
# Couverture « minimale » (modele 6, gabarit fixe commun a toutes les notices, voir strong-pickle tools/manual.py).
import os, sys, json, re, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, PageTemplate, NextPageTemplate, Frame, Paragraph, Spacer, Image as RLImage,
                                Table, TableStyle, PageBreak, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS, OUT = sys.argv[1], sys.argv[2]
VERSION = re.search(r'project\(Semitones VERSION ([0-9.]+)', open(os.path.join(ROOT, 'CMakeLists.txt')).read()).group(1)
TMP = os.path.dirname(os.path.abspath(OUT))
FONTS = os.path.join(ROOT, 'assets', 'fonts')
for n, f in (('Pop', 'Poppins-Regular'), ('PopL', 'Poppins-Light'), ('PopM', 'Poppins-Medium'), ('PopB', 'Poppins-Bold'),
             ('Title', 'BarlowCondensed-ExtraBold')):
    pdfmetrics.registerFont(TTFont(n, os.path.join(FONTS, f + '.ttf')))

ACC = colors.HexColor('#2F8FE8'); ACC_L = colors.HexColor('#8CCBFF')            # bleu de l'afficheur
DARK = colors.HexColor('#17191D'); GREY = colors.HexColor('#5F646C'); LIGHT = colors.HexColor('#EAF2FA')
LAY = json.load(open(os.path.join(ROOT, 'tools', 'layout.json')))
LOGO_PNG = os.path.join(TMP, 'nine_hands_logo_blanc.png')
sys.path.insert(0, os.path.join(ROOT, 'tools')); import svg_raster
svg_raster.raster(os.path.join(ROOT, 'design', 'NineHandsLogo_white.svg'), 1200, LOGO_PNG)
Image.open(LOGO_PNG).crop(Image.open(LOGO_PNG).getbbox()).save(LOGO_PNG)


def shot(name):
    return Image.open(os.path.join(SHOTS, name)).convert('RGB')


src = shot('ui_minus12_mix.png')
k = src.height / 900.0                                    # repere du plugin : 617,5 x 900 unites

# ---------- capture annotee ----------
gx, gy, gw, gh = LAY['glass']
pr = LAY['preset']
marks = [(1, gx + gw - 18, gy + 24), (2, LAY['arrowL'][0] + 8, LAY['arrowL'][1] - 6), (3, LAY['arrowR'][0] + 60, LAY['arrowR'][1] - 6),
         (4, LAY['mix'][0] - 40, LAY['mix'][1] + 26), (5, LAY['fav'][0] + 40, LAY['fav'][1] + 26),
         (6, LAY['foot'][0] + 92, LAY['foot'][1] - 30), (7, LAY['foot'][0] - 92, LAY['foot'][1] + 30),
         (8, pr[0] + 14, pr[1] + pr[3] / 2), (9, gx + 20, gy + gh - 22), (10, 588, LAY['jackY'] - 62), (11, 30, LAY['jackY'] - 62)]
ann = src.copy(); dr = ImageDraw.Draw(ann)
fnt = ImageFont.truetype(os.path.join(FONTS, 'Poppins-Bold.ttf'), 30)
for n, x, y in marks:
    cx, cy, r = x * k, y * k, 24
    dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255), outline=(47, 143, 232), width=5)
    dr.text((cx, cy + 1), str(n), font=fnt, fill=(20, 24, 30), anchor='mm')
annotated = os.path.join(TMP, 'sem_annotated.png'); ann.save(annotated)


def crop_u(im, box):
    return im.crop(tuple(int(v * k) for v in box))


def strip(parts, path, gap=18):
    s = Image.new('RGB', (sum(p.width for p in parts) + gap * (len(parts) - 1), max(p.height for p in parts)), (255, 255, 255))
    x = 0
    for p in parts:
        s.paste(p, (x, 0)); x += p.width + gap
    s.save(path); return path


fx, fy = LAY['foot'][0], LAY['foot'][1]
box_fs = (fx - 95, fy - 95, fx + 95, fy + 95)
fs_img = strip([crop_u(shot('ui_plus4.png'), box_fs), crop_u(shot('ui_bypass_press.png'), box_fs)], os.path.join(TMP, 'sem_fs.png'))
box_d = (gx - 6, gy - 6, gx + gw + 6, gy + gh + 6)
disp_img = strip([crop_u(shot('ui_plus4.png'), box_d), crop_u(src, box_d), crop_u(shot('ui_mix_adjust.png'), box_d)],
                 os.path.join(TMP, 'sem_display.png'))
menus = [os.path.join(SHOTS, n) for n in ('menu_main.png', 'menu_foot.png', 'menu_glow.png')]
menu_img = strip([Image.open(m).convert('RGB') for m in menus], os.path.join(TMP, 'sem_menus.png'), gap=30)

# ---------- couverture : pedale detouree (canal alpha du fond), reflet au sol, halo bleu tres doux ----------
bg_a = Image.open(os.path.join(ROOT, 'assets', 'background.png')).split()[3].resize(src.size, Image.LANCZOS)
cut = Image.new('RGBA', src.size, (0, 0, 0, 0)); cut.paste(shot('ui_plus4.png'), (0, 0), bg_a)
cut = cut.crop(cut.getbbox())
cw, ch = cut.size
pad = int(cw * 0.16)
canvas = Image.new('RGBA', (cw + 2 * pad, int(ch * 1.30) + pad), (0, 0, 0, 255))
yy, xx = np.mgrid[0:canvas.height, 0:canvas.width].astype(np.float64)
gcx, gcy = canvas.width / 2, pad + ch * 0.45
glow = np.exp(-(((xx - gcx) / (cw * 0.62)) ** 2 + ((yy - gcy) / (ch * 0.48)) ** 2)) * 30
edge = np.clip(np.minimum.reduce([xx, canvas.width - 1 - xx, yy, canvas.height - 1 - yy]) / (pad * 0.9), 0, 1)
glow *= edge ** 2
halo = np.dstack([glow * 0.25, glow * 0.6, glow * 1.0, np.full_like(glow, 255)]).astype(np.uint8)
canvas = Image.alpha_composite(canvas, Image.fromarray(halo, 'RGBA'))
refl = cut.transpose(Image.FLIP_TOP_BOTTOM)
ra = np.array(refl).astype(np.float64)
fade = np.clip(1 - np.arange(refl.height) / (refl.height * 0.22), 0, 1) ** 1.6 * 0.22
ra[..., 3] *= fade[:, None]
canvas.alpha_composite(Image.fromarray(ra.astype(np.uint8), 'RGBA'), (pad, pad + ch + 6))
canvas.alpha_composite(cut, (pad, pad))
canvas = canvas.crop((0, 0, canvas.width, pad + int(ch * 1.20)))
cover = os.path.join(TMP, 'sem_cover_cutout.png'); canvas.convert('RGB').save(cover)


def img(path, width):
    im = Image.open(path); h = width * im.height / im.width
    return RLImage(path, width=width, height=h)


st = {
    'h1': ParagraphStyle('h1', fontName='PopB', fontSize=20, leading=24, textColor=ACC, spaceAfter=8, spaceBefore=4, keepWithNext=1),
    'h2': ParagraphStyle('h2', fontName='PopM', fontSize=13, leading=17, textColor=DARK, spaceAfter=4, spaceBefore=10, keepWithNext=1),
    'p': ParagraphStyle('p', fontName='Pop', fontSize=9.6, leading=14, textColor=DARK, spaceAfter=5),
    'tip': ParagraphStyle('tip', fontName='Pop', fontSize=9.0, leading=12.4, textColor=DARK, spaceAfter=3),
    'small': ParagraphStyle('small', fontName='Pop', fontSize=8.2, leading=11.5, textColor=GREY),
    'cell': ParagraphStyle('cell', fontName='Pop', fontSize=8.4, leading=11, textColor=DARK),
    'cellb': ParagraphStyle('cellb', fontName='PopM', fontSize=8.4, leading=11, textColor=DARK),
}


def P(t, s='p'): return Paragraph(t, st[s])
def bullets(items, s='p'): return [P('&bull;&nbsp;&nbsp;' + i, s) for i in items]


def table(rows, widths, header=True):
    data = [[P(c, 'cellb' if (header and r == 0) else 'cell') for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, -1), 0.3, colors.HexColor('#D3DDE8')),
             ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
    if header:
        style += [('BACKGROUND', (0, 0), (-1, 0), LIGHT), ('LINEBELOW', (0, 0), (-1, 0), 0.8, ACC)]
    t.setStyle(TableStyle(style)); return t


PRESETS = [
    ('Standard (0)', '0', 'off', 'Signal direct, pédale prête à être déplacée.'),
    ('Eb Standard (-1)', '−1', 'off', 'Accordage Hendrix / SRV / Van Halen depuis une guitare en Mi.'),
    ('D Standard (-2)', '−2', 'off', 'Rythmiques plus lourdes, tonalités plus faciles à chanter.'),
    ('C# Standard (-3)', '−3', 'off', 'Territoire années 90 / néo-métal.'),
    ('C Standard (-4)', '−4', 'off', 'Métal moderne, riffs en drop C depuis une guitare en drop D.'),
    ('B Standard (-5)', '−5', 'off', 'Sensation baryton sur une six-cordes.'),
    ('Bb Standard (-6)', '−6', 'off', 'Accordage très grave pour six-cordes.'),
    ('Baritone A (-7)', '−7', 'off', 'Une quinte plus bas : Mi devient La.'),
    ('Octave down (-12)', '−12', 'off', 'Ligne de basse jouée à la guitare.'),
    ('Octave doubler (-12)', '−12', '50 %', 'Octaver rapide : guitare + octave inférieure.'),
    ('Sub octave (-12)', '−12', '30 %', 'Léger poids sub sous le riff.'),
    ('Power fifth (-7)', '−7', '45 %', 'Ajoute la quinte inférieure : power chord instantané.'),
    ('Capo 1 / 2 / 3 / 5 / 7', '+1 … +7', 'off', 'Capodastre électrique sans toucher la guitare.'),
    ('Fourth harmony (+5)', '+5', '40 %', 'Harmonie parallèle à la quarte.'),
    ('Fifth harmony (+7)', '+7', '40 %', 'Harmonie parallèle à la quinte.'),
    ('Octave up (+12)', '+12', 'off', 'Une octave plus haut, lignes de solo.'),
    ('12-string (+12)', '+12', '30 %', 'Brillance d\'octave façon 12 cordes.'),
]

W = A4[0] - 36 * mm
f = [NextPageTemplate('p'), Spacer(1, 1), PageBreak()]
f.append(P('Présentation', 'h1'))
f += [P("Semitones est une réplique VST3 de la <b>Solar Guitars CHUG Capo</b>, la pédale de transposition à faible latence conçue par "
        "l'équipe Solar d'Ola Englund comme un « capodastre numérique bidirectionnel ». Elle transpose guitare ou basse d'une octave vers "
        "le bas à une octave vers le haut, par demi-tons, avec les flèches de part et d'autre de l'afficheur, comme un capodastre qu'on "
        "déplace sur le manche."),
      P("Comme la pédale, le moteur privilégie <b>la latence</b> (les chugs en palm-mute rapides ne doivent pas traîner) et un <b>son "
        "transparent</b>, identique quelle que soit la transposition. Le mode MIX mélange le signal transposé à la guitare directe "
        "(octaver rapide, harmonies) et une mémoire FAV rappelle votre réglage favori en un clic ou depuis un footswitch MIDI."),
      P('Le son', 'h2'),
      P("Le transpositeur travaille dans le domaine temporel et suit la période de la note jouée (jusqu'à environ 31 Hz : les accordages "
        "graves de 7 et 8 cordes et la basse sont suivis). Les raccords sont placés là où les deux têtes de lecture voient la même phase de "
        "l'onde : notes seules et power chords restent nets et pleins ; chaque nouvelle attaque ramène le transpositeur sur l'audio le plus "
        "récent pour garder des attaques serrées. Retard mesuré : environ 2 ms sur les notes aiguës, 6 à 10 ms sur les cordes graves. Comme "
        "sur la vraie pédale, les accords très riches et les grands intervalles laissent entendre quelques artefacts numériques, surtout aux "
        "extrêmes (±12).")]
f.append(P('Installation', 'h1'))
f += bullets(["<b>Windows</b> : copiez le dossier <b>Semitones.vst3</b> dans <b>C:\\Program Files\\Common Files\\VST3</b>.",
              "<b>macOS</b> : copiez <b>Semitones.vst3</b> dans <b>~/Library/Audio/Plug-Ins/VST3</b> (ou /Library/Audio/Plug-Ins/VST3). "
              "Si macOS le bloque, lancez <i>xattr -dr com.apple.quarantine</i> sur le paquet.",
              "<b>Reaper</b> : Options &gt; Préférences &gt; Plug-ins &gt; VST &gt; <i>Re-scan</i>. Le plugin apparaît sous le nom "
              "<b>Semitones (NINE HANDS | DSP)</b>."])
f.append(PageBreak())
f.append(P("L'interface", 'h1'))
ov_rows = [['N°', 'Commande', 'Fonction'],
           ['1', 'Afficheur', 'Transposition en demi-tons (−12 … +12) ; en réglage du mix, le mix en % (0 … 100). Glisser vers le haut / bas ou molette dessus pour changer la valeur.'],
           ['2', 'Flèche gauche', '−1 demi-ton (maintenir pour répéter) ; en réglage du mix : baisse le mix.'],
           ['3', 'Flèche droite', '+1 demi-ton (maintenir pour répéter) ; en réglage du mix : monte le mix.'],
           ['4', 'MIX', 'Clic : mode MIX marche / arrêt. Maintien 0,6 s : réglage du mix aux flèches.'],
           ['5', 'FAV', 'Clic : rappelle / quitte le réglage FAV. Maintien 0,9 s : mémorise le son actuel.'],
           ['6', 'Anneau lumineux', 'Allumé = effet actif, éteint = bypass, clignotant = réglage du mix. Intensité du halo : Glow.'],
           ['7', 'Footswitch', 'Effet actif / bypass (bascule dès l\'appui, sans clic).'],
           ['8', 'Barre de presets', 'Presets d\'usine groupés par usage ; « Custom » quand le son ne correspond à aucun.'],
           ['9', "Ligne d'état", 'Dans l\'afficheur : MIX (ou DRY OFF), transposition, FAV, BYPASS ; aide au survol, MIDI Learn.'],
           ['10 / 11', 'IN / OUT', 'Entrée sur le côté droit, sortie à gauche, comme sur la pédale.']]
ov = Table([[img(annotated, 84 * mm), table(ov_rows, [11 * mm, 25 * mm, W - 88 * mm - 36 * mm])]], colWidths=[88 * mm, W - 88 * mm])
ov.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 0)]))
f.append(ov)
f.append(P("Façade en aluminium brossé noir, bandes usinées, afficheur bleu à matrice de points, boutons MIX et FAV dans leurs "
           "défonces, nom Semitones et inscriptions creusés dans le métal (dessinés en vectoriel, nets à toutes les tailles).", 'small'))
f.append(PageBreak())
f.append(P('Les commandes', 'h1'))
f.append(P('Transposition (flèches et afficheur)', 'h2'))
f.append(P("Les flèches changent la transposition d'un demi-ton, de −12 à +12 (maintenir pour répéter) ; on peut aussi glisser vers le "
           "haut / bas ou utiliser la molette sur l'afficheur. Mode MIX éteint, seule la guitare transposée est entendue, comme avec un "
           "capodastre ou un instrument réaccordé. À 0, le signal passe directement (aucun retard)."))
f.append(P('Mode MIX', 'h2'))
f.append(P("Un clic sur MIX active / coupe le mode MIX (point bleu en bas à gauche de l'afficheur, « MIX n % » sur la ligne d'état). Le mode "
           "MIX mélange signal transposé et signal direct de 0 à 100 % (loi à puissance constante : à 50 % les deux sont à −3 dB). "
           "Maintenez MIX 0,6 s pour régler le mix aux flèches : l'anneau lumineux et le point MIX clignotent et l'afficheur indique le mix "
           "en % ; recliquez sur MIX ou attendez 4 s pour sortir. Les réglages bas ajoutent une octave ou une quinte sous ou sur le riff ; "
           "100 % = le signal transposé seul. La partie directe n'est jamais retardée : le mélange reste serré."))
f.append(P('FAV', 'h2'))
f.append(P("Une mémoire favorite, comme sur la pédale. Un clic sur FAV passe à la transposition, au mode MIX et au mix mémorisés (« FAV » "
           "sur la ligne d'état) ; un nouveau clic revient au réglage précédent. Maintenez FAV 0,9 s pour mémoriser le son actuel "
           "(l'afficheur clignote). Toucher une flèche en FAV repart du son FAV et quitte FAV. La mémoire FAV est enregistrée avec le projet."))
f.append(KeepTogether([Spacer(1, 2 * mm), img(disp_img, W), P("Afficheur : +4 demi-tons, mode MIX éteint (DRY OFF) ; −12 en mode MIX à 50 % ; "
                                                               "réglage du mix (100 %).", 'small')]))
f.append(P("Footswitch et anneau lumineux", 'h1'))
ft = Table([[[P(x) for x in (
    "<b>Footswitch</b> : effet actif / bypass, bascule dès l'appui. Bypass bufférisé avec fondu de 6 ms, sans clic.",
    "<b>Anneau lumineux</b> (style Octopus) : anneau bleu dans la gorge entre le bouton et l'écrou, avec son halo. Allumé = effet actif, "
    "éteint = bypass, clignotant = réglage du mix.",
    "<b>Glow</b> : clic droit &gt; <b>Glow</b> &gt; 25 %, 50 %, 75 % ou 100 % règle l'intensité du halo de l'anneau (100 % au départ, "
    "enregistré avec le projet).")],
    [img(fs_img, 62 * mm), P('Effet actif / bypass (footswitch enfoncé).', 'small')]]], colWidths=[W - 66 * mm, 66 * mm])
ft.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (0, 0), 0)]))
f.append(ft)
f.append(P('MIDI', 'h1'))
f.append(P("Toutes les fonctions sont des paramètres de l'hôte (Effect On, Pitch, Mix Mode, Mix, FAV) automatisables. Pour un contrôleur "
           "MIDI comme la M-Vave Chocolate Plus, chaque commande a son propre <b>MIDI Learn</b>. Semitones écoute tous les canaux MIDI ; "
           "<b>rien n'est assigné par défaut</b> : vous choisissez vos CC."))
f += bullets(["<b>Clic droit sur une commande</b> : le menu s'ouvre là où vous cliquez, avec en tête le nom de la commande et son "
              "assignation (« MIDI: CC n » ou « MIDI: not assigned »). Choisissez <b>MIDI Learn: &lt;commande&gt;</b> : la ligne d'état "
              "affiche « MIDI LEARN: &lt;commande&gt; » ; le premier CC reçu est assigné et confirmé 2 secondes (« &lt;commande&gt; = CC n »).",
              "MIX propose le mode MIX et le Mix (expression) ; l'afficheur propose Pitch (expression) et les deux flèches.",
              "<b>Clear MIDI: &lt;commande&gt;</b> retire l'assignation ; <b>Cancel MIDI learn</b> (ou rechoisir le même Learn) arrête l'attente.",
              "Switchs : <i>Latching switch (CC &gt;= 64 = on)</i> ou <i>Momentary switch (each press toggles)</i>. Pédales d'expression : "
              "<i>Reset expression pedal travel</i>.",
              "Clic droit sur le fond &gt; <b>MIDI</b> : Learn de chaque commande avec son CC actuel et <b>Reset MIDI to defaults (nothing assigned)</b>.",
              "Assignations, types de switch et course de pédale apprise sont enregistrés avec le projet."])
rows = [['Commande (nom du MIDI Learn)', 'Type'],
        ['Footswitch (on / bypass)', 'Footswitch, à accrochage par défaut (CC &gt;= 64 = marche) ou momentané (chaque appui bascule).'],
        ['FAV', 'Footswitch, momentané par défaut (comme un switch non verrouillable sur le jack FAV) ou à accrochage.'],
        ['MIX mode', 'Bascule, à accrochage par défaut ou momentanée.'],
        ['Right arrow (+1 st)', 'Déclencheur : +1 demi-ton à chaque passage du CC au-dessus de 64.'],
        ['Left arrow (-1 st)', 'Déclencheur : −1 demi-ton à chaque passage du CC au-dessus de 64.'],
        ['Pitch (expression)', 'Continu, 25 pas : talon = −12, pointe = +12 (14 bits accepté).'],
        ['Mix (expression)', 'Continu 0-100 % ; le bouger active aussi le mode MIX (14 bits accepté).']]
f.append(KeepTogether([Spacer(1, 2 * mm), table(rows, [46 * mm, W - 46 * mm])]))
f.append(P("Les pédales d'expression sont sans saccades (rampe puis filtre à 2 pôles) ; les contrôleurs 14 bits (CC n + CC n+32) sont "
           "acceptés et la course de la pédale est apprise automatiquement : 0 et 100 % sont atteints même si elle n'envoie que 2 à 116.", 'small'))
f.append(P('Fenêtre et menus', 'h1'))
f.append(P("Tous les menus clic droit ont le design d'Octopus et s'ouvrent à l'endroit du clic. Clic droit n'importe où &gt; <b>Plugin "
           "size</b> : <b>60 %</b>, <b>75 %</b>, <b>100 %</b> redimensionne la fenêtre ; <b>(default)</b> signale la taille par défaut. "
           "<b>Default size (new windows)</b> choisit la taille d'ouverture de toutes les nouvelles fenêtres de Semitones (réglage commun à "
           "toutes les instances, enregistré sur l'ordinateur). La fenêtre ne peut jamais être réduite sous 60 % ; l'hôte peut l'agrandir "
           "jusqu'à 200 % (proportions conservées)."))
f.append(KeepTogether([img(menu_img, W), P('Menu principal, menu du footswitch et sous-menu Glow.', 'small')]))
f.append(PageBreak())
f.append(P('Presets', 'h1'))
f.append(P("La barre au-dessus de l'afficheur rappelle les presets d'usine, groupés par usage. Elle affiche « Custom » quand le son ne "
           "correspond à aucun. Les presets sont aussi les programmes de l'hôte (Program Change)."))
f.append(table([['Preset', 'Transp.', 'MIX', 'Usage']] + [list(p) for p in PRESETS], [52 * mm, 18 * mm, 14 * mm, W - 84 * mm]))
f.append(PageBreak())
f.append(P('Conseils', 'h1'))
f += bullets(["Placez Semitones <b>en premier</b> dans la chaîne Reaper, avant les saturations et avant Ampire : le transpositeur suit "
              "mieux une guitare claire, et l'ampli réagit naturellement au signal transposé.",
              "Pour descendre une 7 cordes en Sol : −1 à −5 restent très solides sur les riffs ; l'octave basse est énorme sur les lignes en "
              "notes seules et les power chords, gardez les grands accords ouverts pour les réglages modérés.",
              "Jouez en direct à travers le plugin : aucune latence n'est déclarée à l'hôte, le signal transposé est entendu immédiatement.",
              "Un mix vers 30 % à −12 donne l'octaver rapide du mode MIX de la pédale ; +12 à 30 % ajoute une brillance de 12 cordes."], 'tip')
f.append(KeepTogether([P('Caractéristiques techniques', 'h1'), table([
    ['Format', 'VST3 - Windows 10 / 11 (x64), macOS 10.15+ (Apple Silicon + Intel)'],
    ["Pédale d'origine", 'Solar Guitars CHUG Capo'],
    ['Plage', '−12 … +12 demi-tons, pas d\'un demi-ton'],
    ['Mix', '0 … 100 % (pas de 1 %, puissance constante), mode MIX marche / arrêt'],
    ['Mémoire', '1 mémoire FAV (transposition, mode MIX, mix), enregistrée avec le projet'],
    ['Latence', '~2 ms (aigus) à ~6-10 ms (cordes graves), 0 déclarée à l\'hôte'],
    ['Suivi de hauteur', 'environ 31 Hz à 1,1 kHz (accordages graves 7/8 cordes et basse)'],
    ['Canaux', 'entrée mono ou stéréo, sortie mono ou stéréo'],
    ['Bypass', 'bufférisé, fondu de 6 ms'],
    ['MIDI', 'MIDI Learn sur chaque commande (rien d\'assigné par défaut), omni, switchs à accrochage / momentanés, CC 14 bits, course de pédale apprise'],
    ['Fenêtre', 'Plugin size 60 % (minimum), 75 %, 100 %, taille par défaut des nouvelles fenêtres au choix ; Glow 25 / 50 / 75 / 100 % ; anti-gel v2 sous Windows'],
], [32 * mm, W - 32 * mm], header=False)]))


def deco(c, doc):
    c.saveState()
    c.setFillColor(DARK); c.rect(0, A4[1] - 6 * mm, A4[0], 6 * mm, stroke=0, fill=1)
    c.setFillColor(ACC); c.rect(0, A4[1] - 6.8 * mm, A4[0], 0.8 * mm, stroke=0, fill=1)
    c.setFont('Pop', 7.5); c.setFillColor(GREY)
    c.drawString(18 * mm, 10 * mm, 'Semitones (NINE HANDS | DSP) · v' + VERSION)
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))
    c.restoreState()


doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                      title='SEMITONES - Notice v' + VERSION, author='NINE HANDS | DSP')
COVER_NAME, COVER_SUB = 'SEMITONES', 'Transposition / capodastre numérique'
PEDAL_H, PEDAL_MAXW, PEDAL_TOP = 82 * mm, 115 * mm, 84 * mm
LANG_TOP = 18 * mm
NAME_TOP, NAME_PT = 194 * mm, 30
SUB_TOP, SUB_PT = 208 * mm, 10
LOGO_W, LOGO_BOTTOM = 38 * mm, 22 * mm
BAND_H = 3 * mm


def cover_bg(c, doc):
    c.saveState(); c.setFillColor(colors.black); c.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)
    c.setFillColor(ACC); c.rect(0, 0, A4[0], BAND_H, stroke=0, fill=1)

    def line(text, top, font, size, cs, col):
        w = pdfmetrics.stringWidth(text, font, size) + cs * (len(text) - 1)
        t = c.beginText((A4[0] - w) / 2, A4[1] - top - size * 0.8); t.setFont(font, size); t.setCharSpace(cs)
        t.setFillColor(col); t.textOut(text); c.drawText(t)
    line('FRANÇAIS', LANG_TOP, 'PopM', 8.5, 0, ACC_L)
    ci = Image.open(cover); iw, ih = ci.size
    cw_ = iw / 1.32; ch_ = (ih - int(cw_ * 0.16)) / 1.2; pad_ = int(cw_ * 0.16)
    kk = min(PEDAL_H / ch_, PEDAL_MAXW / cw_)
    dw, dh = iw * kk, ih * kk
    y0 = A4[1] - (PEDAL_TOP + (PEDAL_H - ch_ * kk) / 2) + pad_ * kk
    c.drawImage(cover, (A4[0] - dw) / 2, y0 - dh, dw, dh)
    line(COVER_NAME, NAME_TOP - NAME_PT * 0.02, 'Title', NAME_PT, NAME_PT * 0.5, colors.white)
    line(COVER_SUB, SUB_TOP, 'Pop', SUB_PT, 1.5, colors.HexColor('#BDC4B8'))
    lw, lh = Image.open(LOGO_PNG).size
    c.drawImage(LOGO_PNG, (A4[0] - LOGO_W) / 2, LOGO_BOTTOM, LOGO_W, LOGO_W * lh / lw, mask='auto')
    c.restoreState()


doc.addPageTemplates([PageTemplate(id='cover', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='fc')], onPage=cover_bg),
                      PageTemplate(id='p', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='f')], onPage=deco)])
doc.build(f)
print(OUT)
