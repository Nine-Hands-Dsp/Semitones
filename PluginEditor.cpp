#include "PluginEditor.h"
#include "BinaryData.h"
#include "WigaWindowSize.h"

namespace
{
    double nowMs() { return juce::Time::getMillisecondCounterHiRes(); }

    // tailles du menu Plugin size (pédale simple : hauteurs 493 / 719 / 958 px, 493 = « 60 % »)
    constexpr int kNumSizes = 3;
    const float kSizeFactors[kNumSizes] = { 1.0f, 719.0f / 493.0f, 958.0f / 493.0f };   // 60 %, 75 %, 100 %
    const char* const kSizeLabels[kNumSizes] = { "60 %", "75 %", "100 %" };

    const juce::Colour kInk     { 0xffd9dadc };
    const juce::Colour kInkDim  { 0xff8d8f94 };
    const juce::Colour kAccent  { 0xff39a0ff };    // display blue
    const juce::Colour kBar     { 0xff121214 };
    const juce::Colour kStatus  { 0xff46a5f0 };    // status line in the display
    const juce::Colour kRing    { 0xff28a8ff };    // footswitch light ring (Octopus)

    // ---------------------------------------------------------------- deboss (cache only, never in paint)
    void blur (std::vector<float>& a, int w, int h, float sigma)
    {
        if (sigma < 0.3f) return;
        const int r = juce::jmax (1, (int) std::ceil (sigma * 3.0f));
        std::vector<float> k ((size_t) (2 * r + 1));
        float sum = 0;
        for (int i = -r; i <= r; ++i) { k[(size_t) (i + r)] = std::exp (-(float) (i * i) / (2.0f * sigma * sigma)); sum += k[(size_t) (i + r)]; }
        for (auto& v : k) v /= sum;
        std::vector<float> tmp (a.size());
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                float acc = 0;
                for (int i = -r; i <= r; ++i) acc += k[(size_t) (i + r)] * a[(size_t) (y * w + juce::jlimit (0, w - 1, x + i))];
                tmp[(size_t) (y * w + x)] = acc;
            }
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                float acc = 0;
                for (int i = -r; i <= r; ++i) acc += k[(size_t) (i + r)] * tmp[(size_t) (juce::jlimit (0, h - 1, y + i) * w + x)];
                a[(size_t) (y * w + x)] = acc;
            }
    }

    /** lettering pressed into the metal, lit from the top left at 45 deg (same maths as tools/preview_art.py):
        light grey floor, soft inner shadow along the top-left walls, bottom-right walls catching the light */
    void deboss (juce::Image& img, const juce::Path& unitsPath, float k, float depth)
    {
        depth *= 1.9f;
        const int pad = (int) std::ceil ((4.0f * depth + 4.0f) * k) + 4;
        auto bb = (unitsPath.getBounds() * k).getSmallestIntegerContainer().expanded (pad)
                      .getIntersection (img.getBounds());
        if (bb.isEmpty()) return;
        const int w = bb.getWidth(), h = bb.getHeight();
        juce::Image mask (juce::Image::SingleChannel, w, h, true);
        {
            juce::Graphics g (mask);
            g.addTransform (juce::AffineTransform::scale (k).translated ((float) -bb.getX(), (float) -bb.getY()));
            g.setColour (juce::Colours::white);
            g.fillPath (unitsPath);
        }
        std::vector<float> m ((size_t) (w * h)), occ ((size_t) (w * h)), hg;
        {
            juce::Image::BitmapData md (mask, juce::Image::BitmapData::readOnly);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    m[(size_t) (y * w + x)] = *md.getPixelPointer (x, y) / 255.0f;
        }
        const int d = juce::roundToInt (1.05f * depth * k);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const int sx = x - d, sy = y - d;
                occ[(size_t) (y * w + x)] = (sx < 0 || sy < 0) ? 1.0f : 1.0f - m[(size_t) (sy * w + sx)];
            }
        blur (occ, w, h, 0.9f * k);
        hg = m;
        blur (hg, w, h, 0.75f * k);
        const float fill[3] = { 142.0f, 145.0f, 149.0f };
        juce::Image::BitmapData bd (img, bb.getX(), bb.getY(), w, h, juce::Image::BitmapData::readWrite);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const size_t i = (size_t) (y * w + x);
                const float mf = m[i];
                const float gx = 0.5f * (hg[(size_t) (y * w + juce::jmin (w - 1, x + 1))] - hg[(size_t) (y * w + juce::jmax (0, x - 1))]);
                const float gy = 0.5f * (hg[(size_t) (juce::jmin (h - 1, y + 1) * w + x)] - hg[(size_t) (juce::jmax (0, y - 1) * w + x)]);
                const float face = (gx * -0.7071f + gy * -0.7071f) * k * 2.4f;
                const float hv = hg[i];
                const float wall = juce::jlimit (0.0f, 1.0f, hv * (1.0f - hv) * 4.0f);
                if (mf <= 0.0f && wall <= 0.0f) continue;
                auto* px = (juce::PixelARGB*) bd.getPixelPointer (x, y);
                if (px->getAlpha() < 250) continue;
                float c[3] = { (float) px->getRed(), (float) px->getGreen(), (float) px->getBlue() };
                const float o = 1.0f - 0.72f * occ[i] * mf;
                for (int ch = 0; ch < 3; ++ch)
                {
                    float v = c[ch] * (1.0f - mf * 0.9f) + fill[ch] * mf * 0.9f;
                    v *= o;
                    v += juce::jlimit (0.0f, 1.0f, face) * wall * 75.0f;
                    v *= 1.0f - juce::jlimit (0.0f, 1.0f, -face) * wall * 0.75f;
                    c[ch] = juce::jlimit (0.0f, 255.0f, v);
                }
                px->setARGB (255, (juce::uint8) c[0], (juce::uint8) c[1], (juce::uint8) c[2]);
            }
    }

    juce::Image loadPng (const char* resource)
    {
        int size = 0;
        if (auto* data = BinaryData::getNamedResource (resource, size))
            return juce::ImageCache::getFromMemory (data, size);
        return {};
    }

    constexpr const char* kDigits[10] = { "abcdef", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg", "abc", "abcdefg", "abcdfg" };
    constexpr float kMixHoldMs = 600.0f, kFavHoldMs = 900.0f;
}

// ============================================================================
// Look and feel (preset menu / pop-up menus)
SemitonesLook::SemitonesLook()
{
    setColour (juce::ComboBox::textColourId, juce::Colour (0xffe1e6ec));
    setColour (juce::ComboBox::backgroundColourId, juce::Colours::transparentBlack);
    setColour (juce::ComboBox::outlineColourId, juce::Colours::transparentBlack);
    setColour (juce::ComboBox::arrowColourId, kInkDim);
}

juce::Font SemitonesLook::uiFont (float h) const
{
    // h = em size (point height), like the artwork previews (PIL sizes)
    if (face != nullptr) return juce::Font (juce::FontOptions (face).withPointHeight (h));
    return juce::Font (juce::FontOptions().withPointHeight (h));
}

void SemitonesLook::drawComboBox (juce::Graphics& g, int w, int h, bool down, int, int, int, int, juce::ComboBox& box)
{
    // the black slot is milled in the pedal (background): only the chevron is drawn here
    const float u = (float) h / 40.0f;                                   // slot = 40 units high
    const float cx = (float) w - 17.0f * u, cy = (float) h * 0.5f;
    juce::Path p;
    p.startNewSubPath (cx - 5.0f * u, cy - 2.5f * u);
    p.lineTo (cx, cy + 2.5f * u);
    p.lineTo (cx + 5.0f * u, cy - 2.5f * u);
    const bool hot = box.isMouseOver (true) || down;
    g.setColour (hot ? juce::Colours::white : juce::Colour (0xffe1e6ec));
    g.strokePath (p, juce::PathStrokeType (2.2f * u, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
}

juce::Font SemitonesLook::getComboBoxFont (juce::ComboBox& b) { return uiFont ((float) b.getHeight() * 0.475f); }   // 19 units in the 40-unit slot

void SemitonesLook::positionComboBoxText (juce::ComboBox& box, juce::Label& label)
{
    const int side = juce::roundToInt ((float) box.getHeight() * 0.75f);      // keeps the text centred, clear of the chevron
    label.setBounds (side, 0, box.getWidth() - 2 * side, box.getHeight());
    label.setInterceptsMouseClicks (false, false);
    label.setEditable (false, false, false);
    label.setFont (getComboBoxFont (box));
    label.setJustificationType (juce::Justification::centred);
    label.setMinimumHorizontalScale (0.72f);
}

// ============================================================================
void PresetBox::showPopup()
{
    if (! isEnabled() || menuLnf == nullptr)
        return;
    const int sel = getSelectedId();
    juce::PopupMenu menu;
    menu.setLookAndFeel (menuLnf);
    NineHandsMenu::addHeader (menu, menuLnf->f, NineHandsMenu::icoPreset, kAccent, "Presets",
                              sel > 0 ? getText() : juce::String ("Custom"));
    for (juce::PopupMenu::MenuItemIterator it (*getRootMenu(), false); it.next();)
    {
        const auto& item = it.getItem();
        if (item.isSectionHeader) menu.addSectionHeader (item.text);
        else if (item.isSeparator) menu.addSeparator();
        else if (item.itemID != 0) menu.addItem (item.itemID, item.text, true, item.itemID == sel);
    }
    // the preset menu stays under its bar (rule of the skills); the other menus open at the click
    auto options = juce::PopupMenu::Options().withTargetComponent (this)
                                             .withMinimumWidth (getWidth())
                                             .withMaximumNumColumns (1)
                                             .withPreferredPopupDirection (juce::PopupMenu::Options::PopupDirection::downwards)
                                             .withStandardItemHeight (30);
    juce::Component::SafePointer<PresetBox> safe (this);
    menu.showMenuAsync (options, [safe] (int result)
    {
        if (safe == nullptr)
            return;
        safe->hidePopup();
        if (result > 0 && result != 9001)
            safe->pick (result);
    });
}

void PresetBox::pick (int itemId)
{
    setSelectedId (itemId, juce::dontSendNotification);
    if (onPick != nullptr)
        onPick (itemId);
}

// ============================================================================
SemitonesEditor::SemitonesEditor (SemitonesProcessor& p)
    : AudioProcessorEditor (p), proc (p)
{
    setOpaque (true);
    face = juce::Typeface::createSystemTypefaceFor (BinaryData::PoppinsMedium_ttf, BinaryData::PoppinsMedium_ttfSize);
    look.setTypeface (face);
    setLookAndFeel (&look);
    menuLnf.f = { juce::Typeface::createSystemTypefaceFor (BinaryData::PoppinsRegular_ttf, BinaryData::PoppinsRegular_ttfSize), face,
                  juce::Typeface::createSystemTypefaceFor (BinaryData::PoppinsBold_ttf, BinaryData::PoppinsBold_ttfSize) };
    presetBox.menuLnf = &menuLnf;

    bgImage = loadPng ("background_png");
    buildTexts();
    for (const auto& s : Layout::sprites)
    {
        Sprite sp;
        sp.img = loadPng ((juce::String (s.name) + "_png").toRawUTF8());
        sp.r = { s.r.x, s.r.y, s.r.w, s.r.h };
        sprites[s.name] = sp;
    }
    for (auto n : { "s_l", "s_r", "s_u", "s_d" }) segNames.push_back (std::string ("seg_") + n);
    for (char c = 'a'; c <= 'g'; ++c) segNames.push_back (std::string ("seg_t_") + c);
    for (char c = 'a'; c <= 'g'; ++c) segNames.push_back (std::string ("seg_u_") + c);
    segNames.push_back ("seg_mix");   // MIX indicator dot, in the display

    // preset menu
    // grouped list: section headers are titles only (not selectable), every
    // preset item is selectable; the box itself is never an editable text field
    const auto& presets = SemitonesProcessor::factoryPresets();
    for (int i = 0; i < (int) presets.size(); ++i)
    {
        const auto& p = presets[(size_t) i];
        const auto& prev = presets[(size_t) juce::jmax (0, i - 1)];
        auto group = [] (const FactoryPreset& fp)
        {
            if (fp.semis == 0) return 0;
            if (fp.semis < 0 && ! fp.mixOn) return 1;
            if (fp.semis < 0) return 2;
            if (! fp.mixOn && fp.semis < 12) return 3;
            return 4;
        };
        static const char* titles[] = { "Standard", "Tune down", "Blend below (MIX)", "Capo up", "Harmonies / octave up" };
        if (i == 0 || group (p) != group (prev))
            presetBox.addSectionHeading (titles[group (p)]);
        presetBox.addItem (p.name, i + 1);
    }
    presetBox.setEditableText (false);
    presetBox.setTextWhenNothingSelected ("Custom");
    presetBox.setTextWhenNoChoicesAvailable ("Custom");
    presetBox.setWantsKeyboardFocus (false);
    presetBox.setMouseClickGrabsKeyboardFocus (false);
    presetBox.setScrollWheelEnabled (false);
    presetBox.setTooltip ("Factory presets");
    presetBox.onPick = [this] (int id)
    {
        lastPresetPickMs = nowMs();
        proc.applyPreset (id - 1);         // immediate: parameters are set right now
        syncPresetBox();
        timerCallback();                   // display / status / ring follow at once
    };
    addAndMakeVisible (presetBox);
    syncPresetBox();

    setResizable (true, false);                 // the host resizes the window (no corner grip on the pedal)
    // default size = minimum size (simple pedal format, like Echoflex / Holyverb)
    const int defW = Layout::defaultW;   // 338 x 493 (pedal 1235 x 1800)
    const int defH = Layout::defaultH;
    setResizeLimits (defW, defH, defW * 2, defH * 2);
    if (auto* c = getConstrainer())
        c->setFixedAspectRatio ((double) defW / (double) defH);
    setSize (defW, defH);
    // taille des nouvelles fenêtres : celle choisie dans le menu « Default size (new windows) »
    {
        const float f = kSizeFactors[WigaWindowSize::defaultIndex (JucePlugin_Name, kNumSizes)];
        setSize (juce::roundToInt (defW * f), juce::roundToInt (defH * f));
    }

    const auto e = proc.getEffective();
    if (e.on) led[0] = 1.0f;
    segMask = computeSegments();
    rebuildCacheAt (juce::Component::getApproximateScaleFactorForComponent (this));
    startTimerHz (30);
}

SemitonesEditor::~SemitonesEditor()
{
    stopTimer();
    juce::PopupMenu::dismissAllActiveMenus();
    setLookAndFeel (nullptr);
}

// ---------------------------------------------------------------- helpers
juce::Rectangle<int> SemitonesEditor::toScreen (juce::Rectangle<float> r, float grow) const
{
    const float s = scale();
    return (r * s).expanded (grow * s).getSmallestIntegerContainer();
}

const SemitonesEditor::Sprite* SemitonesEditor::sprite (const std::string& name) const
{
    auto it = sprites.find (name);
    return it == sprites.end() ? nullptr : &it->second;
}

void SemitonesEditor::drawSprite (juce::Graphics& g, const std::string& name, float opacity) const
{
    if (opacity <= 0.002f) return;
    if (auto* sp = sprite (name))
    {
        if (! sp->img.isValid()) return;
        g.setOpacity (juce::jlimit (0.0f, 1.0f, opacity));
        g.drawImage (sp->img, sp->r * scale(), juce::RectanglePlacement::stretchToFit);
    }
}

SemitonesEditor::Hit SemitonesEditor::hitTest (juce::Point<float> p) const
{
    auto inCircle = [&] (float x, float y, float r) { return p.getDistanceFrom ({ x, y }) <= r; };
    auto inRect = [&] (const Layout::R& r) { return juce::Rectangle<float> (r.x, r.y, r.w, r.h).contains (p); };
    if (inCircle (Layout::footX, Layout::footY, Layout::footR)) return Hit::foot;
    if (inCircle (Layout::mixX, Layout::mixY, Layout::mixR)) return Hit::mix;
    if (inCircle (Layout::favX, Layout::favY, Layout::favR)) return Hit::fav;
    if (inRect (Layout::hitLeft)) return Hit::left;
    if (inRect (Layout::hitRight)) return Hit::right;
    if (inRect (Layout::hitDisplay)) return Hit::display;
    return Hit::background;
}

uint32_t SemitonesEditor::computeSegments() const
{
    const auto e = proc.getEffective();
    const double t = nowMs();
    uint32_t m = 0;
    auto set = [&] (const std::string& n)
    {
        for (size_t i = 0; i < segNames.size(); ++i)
            if (segNames[i] == n) { m |= (1u << i); return; }
    };
    auto digit = [&] (char pos, int d)
    {
        if (d < 0 || d > 9) return;
        for (const char* c = kDigits[d]; *c; ++c) set (std::string ("seg_") + pos + "_" + *c);
    };
    const bool blinkOn = std::fmod (blinkPhase, 1.0) < 0.5;

    bool showValue = true;
    if (t < favBlinkUntil) showValue = std::fmod ((favBlinkUntil - t) / 180.0, 1.0) > 0.45;

    if (showValue)
    {
        if (mixAdjust)
        {
            const int n = juce::jlimit (0, 100, (int) std::lround (e.mix));
            if (n == 100) { set ("seg_s_u"); set ("seg_s_d"); digit ('t', 0); digit ('u', 0); }
            else
            {
                if (n >= 10) digit ('t', n / 10);
                digit ('u', n % 10);
            }
        }
        else
        {
            const int s = e.semis;
            if (s != 0) { set ("seg_s_l"); set ("seg_s_r"); }
            if (s > 0) { set ("seg_s_u"); set ("seg_s_d"); }
            const int a = std::abs (s);
            if (a >= 10) digit ('t', a / 10);
            digit ('u', a % 10);
        }
    }
    // MIX indicator: blue dash above the MIX button while MIX mode is on
    // (flashes while the mix is being adjusted)
    if ((e.mixOn && ! mixAdjust) || (mixAdjust && blinkOn))
        set ("seg_mix");
    return m;
}

// ---------------------------------------------------------------- painting
void SemitonesEditor::paint (juce::Graphics& g)
{
    lastPaintMs = nowMs();
    pendingSince = 0;
    g.fillAll (kBar);
    const auto bounds = getLocalBounds().toFloat();
    g.setImageResamplingQuality (juce::Graphics::highResamplingQuality);
    if (bgCache.isValid())
        g.drawImage (bgCache, bounds, juce::RectanglePlacement::stretchToFit);
    else if (bgImage.isValid())
        g.drawImage (bgImage, bounds, juce::RectanglePlacement::stretchToFit);

    // footswitch light: Octopus ring (halo, diffuser, core)
    if (led[0] > 0.003f)
        drawRing (g, led[0]);

    // pressed controls
    if (downLeft)  drawSprite (g, "press_left");
    if (downRight) drawSprite (g, "press_right");
    if (downMix)   drawSprite (g, "press_mix");
    if (downFav)   drawSprite (g, "press_fav");
    if (downFoot)  drawSprite (g, "press_foot");

    // display
    for (size_t i = 0; i < segNames.size(); ++i)
        if (segMask & (1u << i))
            drawSprite (g, segNames[i]);
    g.setOpacity (1.0f);

    // status line, at the bottom of the display (Poppins Medium 15.5 units, display blue)
    const float s = scale();
    const auto status = statusText();
    const auto st = Layout::statusBox;
    auto sr = juce::Rectangle<float> ((st.x + 12.0f) * s, st.y * s, st.w * s, st.h * s);
    g.setFont (look.uiFont (15.5f * s));
    g.setColour (kStatus);
    g.drawFittedText (status, sr.toNearestInt(), juce::Justification::centred, 1, 0.55f);
}

void SemitonesEditor::drawRing (juce::Graphics& g, float lvl) const
{
    // Octopus drawRing(): halo strokes 13 -> 3 (alpha .045 x Glow), diffuser (alpha .85), lighter core (alpha .9)
    const float s = scale();
    const float cx = Layout::ringX * s, cy = Layout::ringY * s, R = Layout::ringR * s, K = Layout::ringK * s;
    const float glow = (float) juce::jlimit (25, 100, proc.glowPct.load()) / 100.0f;
    auto stroke = [&] (float width, juce::Colour c, float a)
    {
        g.setColour (c.withAlpha (juce::jlimit (0.0f, 1.0f, a)));
        g.drawEllipse (cx - R, cy - R, 2 * R, 2 * R, width);
    };
    for (float hw = 13.0f; hw > 3.0f; hw -= 1.5f)
        stroke (hw * K, kRing, 0.045f * 1.15f * lvl * glow);
    stroke (Layout::ringW * s, kRing, 0.85f * lvl);
    stroke (Layout::ringW * 0.42f * s, kRing.interpolatedWith (juce::Colours::white, 0.55f), 0.9f * lvl);
}

juce::String SemitonesEditor::statusText() const
{
    const int lt = proc.learnTarget.load();
    if (lt >= 0)
        return "MIDI LEARN: " + SemitonesProcessor::targetName ((MidiTarget) lt);
    if (learnLineUntil > 0 && nowMs() < learnLineUntil)
        return learnLine;
    if (hover != Hit::none && hover != Hit::background && nowMs() - hoverSince < 2500.0)
        return hintFor (hover);
    const auto e = proc.getEffective();
    const juce::String dot ("   " + juce::String::fromUTF8 ("\xc2\xb7") + "   ");
    juce::String st = (e.semis >= 0 ? "+" : juce::String::fromUTF8 ("\xe2\x88\x92")) + juce::String (std::abs (e.semis)) + " ST";
    juce::String txt = e.mixOn ? "MIX " + juce::String ((int) std::lround (e.mix)) + " %" : juce::String ("DRY OFF");
    txt << dot << st;
    if (e.fav) txt << dot << "FAV";
    if (! e.on) txt << dot << "BYPASS";
    return txt;
}

juce::String SemitonesEditor::hintFor (Hit h) const
{
    switch (h)
    {
        case Hit::left:    return mixAdjust ? "MIX  -1 %" : juce::String::fromUTF8 ("\xe2\x88\x92") + "1 ST  (hold = repeat)";
        case Hit::right:   return mixAdjust ? "MIX  +1 %" : juce::String ("+1 ST  (hold = repeat)");
        case Hit::mix:     return "MIX on / off  -  hold: set the mix";
        case Hit::fav:     return "FAV recall  -  hold: save";
        case Hit::foot:    return "Effect on / bypass";
        case Hit::display: return "Drag or scroll: pitch";
        default: break;
    }
    return {};
}

void SemitonesEditor::resized()
{
    // never setSize() here (Windows scaling loops) - the constrainer does the job
    const float s = scale();
    presetBox.setBounds (juce::Rectangle<float> (Layout::presetBox.x * s, Layout::presetBox.y * s, Layout::presetBox.w * s, Layout::presetBox.h * s).toNearestInt());
    resizeSettle = 4;
}

void SemitonesEditor::buildTexts()
{
    // vector texts of the mockup (design/Semitones_Texts.svg, px of the 1235 x 1800 frame -> units)
    texts.clear();
    int size = 0;
    auto* data = BinaryData::getNamedResource ("Semitones_Texts_svg", size);
    if (data == nullptr) return;
    auto xml = juce::XmlDocument::parse (juce::String::fromUTF8 (data, size));
    if (xml == nullptr) return;
    const std::map<juce::String, float> depth { { "mix", 0.6f }, { "fav", 0.6f }, { "name", 1.0f },
                                                { "out", 0.8f }, { "in", 0.8f }, { "logo", 0.7f } };
    for (auto* grp : xml->getChildWithTagNameIterator ("g"))
    {
        TextPath tp;
        for (auto* pe : grp->getChildWithTagNameIterator ("path"))
            tp.path.addPath (juce::Drawable::parseSVGPath (pe->getStringAttribute ("d")));
        tp.path.applyTransform (juce::AffineTransform::scale (Layout::W / 1235.5f));
        const auto it = depth.find (grp->getStringAttribute ("id"));
        tp.depth = it != depth.end() ? it->second : 0.7f;
        texts.push_back (std::move (tp));
    }
}

void SemitonesEditor::rebuildCacheAt (float ds)
{
    if (! bgImage.isValid() || getWidth() <= 0) return;
    const int pw = juce::jmax (1, juce::roundToInt ((float) getWidth() * ds));
    const int ph = juce::jmax (1, juce::roundToInt ((float) getHeight() * ds));
    if (bgCache.isValid() && bgCache.getWidth() == pw && bgCache.getHeight() == ph) return;
    auto img = (pw == bgImage.getWidth() && ph == bgImage.getHeight()) ? bgImage.createCopy()
                                                                        : bgImage.rescaled (pw, ph, juce::Graphics::highResamplingQuality);
    img = img.convertedToFormat (juce::Image::ARGB);
    const float k = (float) pw / Layout::W;                      // physical px per unit
    for (const auto& t : texts)
        deboss (img, t.path, k, t.depth);
    bgCache = img;
    cacheScale = ds;
    repaint();
}

void SemitonesEditor::rebuildCache()
{
    rebuildCacheAt (juce::Component::getApproximateScaleFactorForComponent (this));
}

// ---------------------------------------------------------------- anti-freeze
void SemitonesEditor::useSoftwareRenderer()
{
   #if JUCE_WINDOWS
    if (auto* peer = getPeer())
    {
        const auto engines = peer->getAvailableRenderingEngines();
        int idx = engines.indexOf ("Software Renderer");
        if (idx < 0) idx = 0;
        if (peer->getCurrentRenderingEngine() != idx)
            peer->setCurrentRenderingEngine (idx);
        if ((void*) peer != lastPeer)
        {
            lastPeer = (void*) peer;
            repaint();
        }
    }
   #else
    lastPeer = (void*) getPeer();
   #endif
}

void SemitonesEditor::parentHierarchyChanged() { useSoftwareRenderer(); }

void SemitonesEditor::visibilityChanged()
{
    useSoftwareRenderer();
    if (isShowing()) repaint();
}

void SemitonesEditor::askRepaint (juce::Rectangle<int> r)
{
    if (pendingSince == 0) pendingSince = nowMs();
    repaint (r);
}

void SemitonesEditor::syncPresetBox()
{
    if (presetBox.isPopupActive())
        return;                            // never touch the box while its menu is open
    const auto e = proc.getEffective();
    const auto& list = SemitonesProcessor::factoryPresets();
    int id = 0;
    const int cur = proc.currentPreset;
    auto matches = [&] (const FactoryPreset& p)
    {
        return ! e.fav && p.semis == e.semis && p.mixOn == e.mixOn && (! p.mixOn || std::abs (p.mix - e.mix) < 0.5f);
    };
    if (cur >= 0 && cur < (int) list.size() && matches (list[(size_t) cur])) id = cur + 1;
    else
        for (int i = 0; i < (int) list.size(); ++i)
            if (matches (list[(size_t) i])) { id = i + 1; break; }
    if (id == 0)
    {
        // "Custom" at full brightness (the stock "nothing selected" text is drawn at half alpha)
        if (presetBox.getSelectedId() != 0 || presetBox.getText() != "Custom")
            presetBox.setText ("Custom", juce::dontSendNotification);
    }
    else if (presetBox.getSelectedId() != id)
        presetBox.setSelectedId (id, juce::dontSendNotification);
}

// ---------------------------------------------------------------- timer (30 Hz)
void SemitonesEditor::timerCallback()
{
    const double t = nowMs();
    blinkPhase = t / 250.0;          // 4 Hz flashing

    // anti-freeze: renderer + watchdog
    if (getPeer() != lastPeer || t - lastRendererCheck > 1000.0)
    {
        lastRendererCheck = t;
        useSoftwareRenderer();
    }
    if (pendingSince > 0 && t - pendingSince > 1500.0 && isShowing())
    {
        lastPeer = nullptr;
        useSoftwareRenderer();
        repaint();
        pendingSince = t;
    }
    // caches: rebuilt only from here, after a resize has settled or the scale changed
    if (resizeSettle > 0 && --resizeSettle == 0) rebuildCache();
    if (t - lastScaleCheck > 250.0)
    {
        lastScaleCheck = t;
        const float ds = juce::Component::getApproximateScaleFactorForComponent (this);
        if (std::abs (ds - lastDisplayScale) > 0.001f) { lastDisplayScale = ds; rebuildCache(); }
    }

    // long presses
    if (downMix && ! longDone && t - tDown >= kMixHoldMs)
    {
        longDone = true;
        mixAdjust = true;
        mixAdjustLast = t;
        if (proc.getEffective().fav) proc.exitFavKeeping();
        proc.setParam (ParamID::mixOn, 1.0f);
    }
    if (downFav && ! longDone && t - tDown >= kFavHoldMs)
    {
        longDone = true;
        proc.saveFav();
        favBlinkUntil = t + 900.0;
    }
    // arrow auto-repeat
    if ((downLeft || downRight) && t >= nextRepeat)
    {
        stepAction (downRight ? 1 : -1);
        nextRepeat = t + (mixAdjust ? 45.0 : 150.0);
    }
    if (mixAdjust && ! downMix && ! downLeft && ! downRight && t - mixAdjustLast > 4000.0)
        mixAdjust = false;

    // footswitch ring = effect on (Echoflex fades: 0.45 on / 0.28 off per frame),
    // flashing while the mix is being adjusted
    const auto e = proc.getEffective();
    bool lit = e.on;
    if (mixAdjust) lit = e.on && std::fmod (blinkPhase, 1.0) < 0.5;
    {
        const float target = lit ? 1.0f : 0.0f;
        led[0] += (target - led[0]) * (target > led[0] ? 0.45f : 0.28f);
        if (std::abs (led[0] - target) < 0.004f) led[0] = target;
    }
    // the ring is steady: repaint only while it fades (or when the Glow setting changes)
    const float glowNow = (float) proc.glowPct.load();
    if (std::abs (led[0] - drawnLed[0]) > 0.002f || glowNow != drawnGlow)
    {
        drawnLed[0] = led[0];
        drawnGlow = glowNow;
        const float rr = Layout::ringR + 16.0f * Layout::ringK;
        askRepaint (toScreen ({ Layout::ringX - rr, Layout::ringY - rr, 2 * rr, 2 * rr }, 4.0f));
    }

    segMask = computeSegments();
    if (segMask != drawnSegMask)
    {
        drawnSegMask = segMask;
        askRepaint (toScreen ({ Layout::hitDisplay.x, Layout::hitDisplay.y, Layout::hitDisplay.w, Layout::hitDisplay.h }, 4.0f));
    }
    const int pressed = (downLeft ? 1 : 0) | (downRight ? 2 : 0) | (downMix ? 4 : 0) | (downFav ? 8 : 0) | (downFoot ? 16 : 0);
    if (pressed != drawnPressed)
    {
        const int changed = pressed ^ (drawnPressed < 0 ? 31 : drawnPressed);
        drawnPressed = pressed;
        const char* names[5] = { "press_left", "press_right", "press_mix", "press_fav", "press_foot" };
        for (int b = 0; b < 5; ++b)
            if (changed & (1 << b))
                if (auto* sp = sprite (names[b])) askRepaint (toScreen (sp->r, 3.0f));
    }
    const int learned = proc.lastLearned.exchange (-1);
    if (learned >= 0 && learned < (int) MidiTarget::count)
    {
        learnLine = SemitonesProcessor::targetName ((MidiTarget) learned) + " = CC " + juce::String (proc.midiCC[(size_t) learned].load());
        learnLineUntil = t + 2000.0;
    }
    const auto status = statusText();
    if (status != drawnStatus)
    {
        drawnStatus = status;
        askRepaint (toScreen ({ Layout::statusBox.x, Layout::statusBox.y, Layout::statusBox.w, Layout::statusBox.h }, 2.0f));
    }
    syncPresetBox();
}

// ---------------------------------------------------------------- actions
void SemitonesEditor::stepAction (int dir)
{
    if (mixAdjust)
    {
        proc.stepMix ((float) dir);
        mixAdjustLast = nowMs();
    }
    else
        proc.stepSemitone (dir);
}

void SemitonesEditor::mouseDown (const juce::MouseEvent& ev)
{
    const auto p = toLogical (ev.position);
    const Hit h = hitTest (p);
    if (ev.mods.isPopupMenu())
    {
        showMenu (h);
        return;
    }
    const double t = nowMs();
    tDown = t;
    longDone = false;
    dragAccum = 0;
    switch (h)
    {
        case Hit::left:  downLeft = true;  stepAction (-1); nextRepeat = t + 450.0; break;
        case Hit::right: downRight = true; stepAction (+1); nextRepeat = t + 450.0; break;
        case Hit::mix:   downMix = true; break;
        case Hit::fav:   downFav = true; break;
        case Hit::foot:  downFoot = true; proc.toggleParam (ParamID::on); break;   // toggles on press, like Echoflex
        default: break;
    }
    timerCallback();
}

void SemitonesEditor::mouseUp (const juce::MouseEvent&)
{
    const double held = nowMs() - tDown;
    if (downMix && ! longDone)
    {
        if (mixAdjust) mixAdjust = false;                         // leave mix adjust
        else
        {
            if (proc.getEffective().fav) proc.exitFavKeeping();
            proc.toggleParam (ParamID::mixOn);
        }
    }
    if (downFav && ! longDone && held < kFavHoldMs)
    {
        mixAdjust = false;
        proc.toggleParam (ParamID::fav);
    }
    downLeft = downRight = downMix = downFav = downFoot = false;
    longDone = false;
    timerCallback();
}

void SemitonesEditor::mouseDrag (const juce::MouseEvent& ev)
{
    if (hitTest (toLogical (ev.mouseDownPosition)) != Hit::display) return;
    const float dy = (float) ev.getDistanceFromDragStartY() / scale();
    const float stepPx = mixAdjust ? 3.0f : 16.0f;
    const int target = (int) (-dy / stepPx);
    const int done = (int) dragAccum;
    if (target != done)
    {
        stepAction (target > done ? 1 : -1);
        dragAccum = (float) (done + (target > done ? 1 : -1));
    }
}

void SemitonesEditor::mouseMove (const juce::MouseEvent& ev)
{
    const Hit h = hitTest (toLogical (ev.position));
    if (h != hover)
    {
        hover = h;
        hoverSince = nowMs();
    }
    setMouseCursor (h == Hit::background || h == Hit::none ? juce::MouseCursor::NormalCursor
                                                                             : juce::MouseCursor::PointingHandCursor);
}

void SemitonesEditor::mouseExit (const juce::MouseEvent&) { hover = Hit::none; }

void SemitonesEditor::mouseWheelMove (const juce::MouseEvent& ev, const juce::MouseWheelDetails& w)
{
    const Hit h = hitTest (toLogical (ev.position));
    if (h == Hit::display || h == Hit::left || h == Hit::right)
    {
        const float d = w.deltaY != 0 ? w.deltaY : -w.deltaX;
        if (d != 0) stepAction (d > 0 ? 1 : -1);
    }
}

void SemitonesEditor::showMenu (Hit h)
{
    juce::PopupMenu m;
    m.setLookAndFeel (&menuLnf);
    std::vector<MidiTarget> targets;
    switch (h)
    {
        case Hit::foot:    targets = { MidiTarget::footswitch }; break;
        case Hit::left:    targets = { MidiTarget::down }; break;
        case Hit::right:   targets = { MidiTarget::up }; break;
        case Hit::mix:     targets = { MidiTarget::mixOn, MidiTarget::mix }; break;
        case Hit::fav:     targets = { MidiTarget::fav }; break;
        case Hit::display: targets = { MidiTarget::semitones, MidiTarget::up, MidiTarget::down }; break;
        default: break;
    }
    // header (Octopus): the pedal, or the control with its MIDI assignment
    {
        juce::String title ("Semitones"), sub (juce::String ("Pitch shifter  v") + JucePlugin_VersionString);
        int ico = NineHandsMenu::icoPedal;
        if (! targets.empty())
        {
            const auto t0 = targets.front();
            const int cc = proc.midiCC[(size_t) t0].load();
            title = h == Hit::foot ? juce::String ("Footswitch") : h == Hit::mix ? juce::String ("MIX")
                  : h == Hit::fav ? juce::String ("FAV") : h == Hit::left ? juce::String ("Left arrow")
                  : h == Hit::right ? juce::String ("Right arrow") : juce::String ("Display");
            sub = cc >= 0 ? "MIDI: CC " + juce::String (cc) : juce::String ("MIDI: not assigned");
            ico = h == Hit::display ? NineHandsMenu::icoKnob : NineHandsMenu::icoSwitch;
        }
        NineHandsMenu::addHeader (m, menuLnf.f, ico, kAccent, title, sub);
    }
    int id = 1;
    struct Act { int kind; MidiTarget t; float size; };
    std::map<int, Act> acts;
    auto ccText = [this] (MidiTarget t)
    {
        const int cc = proc.midiCC[(size_t) t].load();
        return cc >= 0 ? "CC " + juce::String (cc) : juce::String ("none");
    };
    for (auto t : targets)
    {
        const auto name = SemitonesProcessor::targetName (t);
        m.addSectionHeader ("MIDI  -  " + name);
        const int cc = proc.midiCC[(size_t) t].load();
        acts[id] = { 0, t, 0 };
        m.addItem (id++, "MIDI Learn: " + name + " (" + ccText (t) + ")", true,
                   proc.learnTarget.load() == (int) t);
        if (SemitonesProcessor::isToggle (t))
        {
            const bool mom = proc.momentary[(size_t) t].load();
            acts[id] = { 1, t, 0 };
            m.addItem (id++, "Latching switch (CC >= 64 = on)", true, ! mom);
            acts[id] = { 2, t, 0 };
            m.addItem (id++, "Momentary switch (each press toggles)", true, mom);
        }
        if (SemitonesProcessor::isContinuous (t))
        {
            acts[id] = { 4, t, 0 };
            m.addItem (id++, "Reset expression pedal travel", cc >= 0);
        }
        acts[id] = { 3, t, 0 };
        m.addItem (id++, "Clear MIDI: " + name, cc >= 0);
    }
    if (proc.learnTarget.load() >= 0)
    {
        m.addSeparator();
        acts[id] = { 5, MidiTarget::count, 0 };
        m.addItem (id++, "Cancel MIDI learn");
    }
    // right click outside the controls: MIDI sub-menu with every control
    if (targets.empty())
    {
        juce::PopupMenu midi;
        for (int i = 0; i < (int) MidiTarget::count; ++i)
        {
            const auto t = (MidiTarget) i;
            acts[id] = { 0, t, 0 };
            midi.addItem (id++, "Learn: " + SemitonesProcessor::targetName (t) + " (" + ccText (t) + ")", true,
                          proc.learnTarget.load() == i);
        }
        midi.addSeparator();
        acts[id] = { 7, MidiTarget::count, 0 };
        midi.addItem (id++, "Reset MIDI to defaults (nothing assigned)");
        m.addSubMenu ("MIDI", midi);
    }
    {
        // Plugin size : taille de cette fenêtre + taille par défaut des nouvelles fenêtres (comme Octopus)
        // même liste que les autres pédales : 60 % (taille par défaut d'origine), 75 %, 100 %
        const int def = WigaWindowSize::defaultIndex (JucePlugin_Name, kNumSizes);
        juce::PopupMenu sizes, defSizes, glow;
        for (int k = 0; k < kNumSizes; ++k)
        {
            const float sz = kSizeFactors[k];
            const int w = juce::roundToInt (Layout::defaultW * sz);
            const bool cur = std::abs (getWidth() - w) < 2;
            acts[id] = { 6, MidiTarget::count, sz };
            sizes.addItem (id++, juce::String (kSizeLabels[k]) + (k == def ? "  (default)" : ""), true, cur);
            acts[id] = { 8, MidiTarget::count, (float) k };
            defSizes.addItem (id++, kSizeLabels[k], true, k == def);
        }
        sizes.addSeparator();
        sizes.addSubMenu ("Default size (new windows)", defSizes);
        m.addSeparator();
        m.addSubMenu ("Plugin size", sizes);
        // Glow: intensity of the light halos (footswitch ring)
        for (int pct : { 25, 50, 75, 100 })
        {
            acts[id] = { 9, MidiTarget::count, (float) pct };
            glow.addItem (id++, "Glow " + juce::String (pct) + " %", true, proc.glowPct.load() == pct);
        }
        m.addSubMenu ("Glow", glow);
    }

    juce::Component::SafePointer<SemitonesEditor> safe (this);
    m.showMenuAsync (juce::PopupMenu::Options().withMousePosition(),          // opens where the user clicks
        [safe, acts] (int r)
        {
            if (safe == nullptr || r <= 0) return;
            auto it = acts.find (r);
            if (it == acts.end()) return;
            auto& pr = safe->proc;
            const auto a = it->second;
            const size_t ti = (size_t) a.t;
            switch (a.kind)
            {
                case 0: pr.learnTarget.store (pr.learnTarget.load() == (int) a.t ? -1 : (int) a.t); break;   // clicking again cancels
                case 1: pr.momentary[ti].store (false); break;
                case 2: pr.momentary[ti].store (true); break;
                case 3: pr.clearMidi (a.t); break;
                case 4: pr.courseMin[ti].store (1.0f); pr.courseMax[ti].store (0.0f); break;
                case 5: pr.learnTarget.store (-1); break;
                case 7: pr.resetMidiMap(); break;
                case 6: safe->setSize (juce::roundToInt (Layout::defaultW * a.size), juce::roundToInt (Layout::defaultH * a.size)); break;
                case 9: pr.glowPct.store ((int) a.size); break;
                case 8:
                {
                    const int k = juce::jlimit (0, kNumSizes - 1, (int) a.size);
                    WigaWindowSize::setDefaultIndex (JucePlugin_Name, k);   // toutes les nouvelles fenêtres de ce plugin
                    const float sz = kSizeFactors[k];                       // et cette fenêtre tout de suite
                    safe->setSize (juce::roundToInt (Layout::defaultW * sz), juce::roundToInt (Layout::defaultH * sz));
                    break;
                }
                default: break;
            }
        });
}

// ---------------------------------------------------------------- offline rendering
void SemitonesEditor::setPreviewState (bool l, bool r, bool mx, bool fv, bool ft, bool adjustMix)
{
    downLeft = l; downRight = r; downMix = mx; downFav = fv; downFoot = ft;
    mixAdjust = adjustMix;
    longDone = true;
    tDown = nowMs();
    nextRepeat = 1e300;
}

void SemitonesEditor::settleForSnapshot()
{
    resizeSettle = 0;
    for (int i = 0; i < 40; ++i) timerCallback();
    bgCache = {};
    rebuildCacheAt (4.0f);        // full resolution art (with the debossed texts) for 4x snapshots
}
