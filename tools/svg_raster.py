"""Rasterise an SVG to a transparent PNG (Chromium via Playwright; no rsvg on this machine)."""
import sys, os, base64
from playwright.sync_api import sync_playwright


def raster(svg_path, width_px, out_png):
    svg = open(svg_path, 'rb').read()
    b64 = base64.b64encode(svg).decode()
    html = f"""<html><body style='margin:0;background:transparent'>
<img id=i src='data:image/svg+xml;base64,{b64}' style='width:{width_px}px;display:block'></body></html>"""
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={'width': width_px, 'height': 10})
        pg.set_content(html)
        pg.wait_for_function("document.getElementById('i').complete")
        el = pg.query_selector('#i')
        el.screenshot(path=out_png, omit_background=True)
        b.close()


if __name__ == '__main__':
    raster(sys.argv[1], int(sys.argv[2]), sys.argv[3])
