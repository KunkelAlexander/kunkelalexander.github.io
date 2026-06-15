/*!
 * chordscale.js — tiny, dependency-free guitar chord & scale diagram renderer.
 *
 * Renders SVG fretboard diagrams from plain-text blocks in your rendered HTML.
 * Designed for Jekyll / GitHub Pages: write fenced code blocks (```chord / ```scale)
 * in Markdown, drop this script in, done. No build step, no dependencies.
 *
 * Colors inherit `currentColor`, so diagrams adapt to light/dark themes.
 * Root notes use --chordscale-root (default #d6452c); override in your CSS.
 *
 * MIT-style: use it however you like.
 */
(function (global) {
  'use strict';

  var DEFAULTS = {
    chordSelector: '.language-chord, .chord-diagram',
    scaleSelector: '.language-scale, .scale-diagram',
    rootColorVar: 'var(--chordscale-root, #d6452c)'
  };

  // ---------- small helpers ----------
  var r = function (n) { return Math.round(n * 100) / 100; };

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (ch) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch];
    });
  }

  function clampInt(v, def, lo, hi) {
    var n = parseInt(v, 10);
    if (isNaN(n)) return def;
    return Math.max(lo, Math.min(hi, n));
  }

  // Parse a block of "key: value" lines into an object.
  // A first line without a colon is treated as the chord/scale name.
  function parseConfig(text) {
    var cfg = {};
    var lines = String(text).replace(/\r/g, '').split('\n');
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (!line) continue;
      var m = line.match(/^([A-Za-z][\w-]*)\s*:\s*(.*)$/);
      if (m) {
        cfg[m[1].toLowerCase()] = m[2].trim();
      } else if (!cfg.name && !cfg._sawKey) {
        cfg.name = line; // bare leading line = name
      }
      if (m) cfg._sawKey = true;
    }
    return cfg;
  }

  // "x 3 2 0 1 0" -> [null, 3, 2, 0, 1, 0]  (null == muted)
  function parseFrets(str) {
    if (!str) return [];
    return str.trim().split(/\s+/).map(function (t) {
      if (/^[xX.\-]$/.test(t)) return null;
      var n = parseInt(t, 10);
      return isNaN(n) ? null : n;
    });
  }

  // "x 3 2 0 1 0" -> [0,3,2,0,1,0]  (0 == no finger)
  function parseFingers(str) {
    if (!str) return [];
    return str.trim().split(/\s+/).map(function (t) {
      if (/^[xX.\-0]$/.test(t)) return 0;
      var n = parseInt(t, 10);
      return isNaN(n) ? 0 : n;
    });
  }

  // "6:5 4:7:A 1:5" -> [{s:6,f:5,label:''},{s:4,f:7,label:'A'},...]
  function parseDots(str) {
    if (!str) return [];
    var out = [];
    str.trim().split(/\s+/).forEach(function (tok) {
      var p = tok.split(':');
      if (p.length < 2) return;
      var s = parseInt(p[0], 10), f = parseInt(p[1], 10);
      if (isNaN(s) || isNaN(f)) return;
      out.push({ s: s, f: f, label: p[2] || '' });
    });
    return out;
  }

  // "5-8" -> {start:5, end:8}
  function parseRange(str) {
    if (!str) return null;
    var m = str.match(/(\d+)\s*[-–]\s*(\d+)/);
    if (!m) return null;
    return { start: parseInt(m[1], 10), end: parseInt(m[2], 10) };
  }

  // ---------- chord diagram ----------
  function renderChord(cfg) {
    var ROOT = DEFAULTS.rootColorVar;
    var nStr = clampInt(cfg.strings, 6, 3, 12);

    var frets = parseFrets(cfg.frets);
    while (frets.length < nStr) frets.push(null);
    frets = frets.slice(0, nStr);

    var fingers = parseFingers(cfg.fingers);
    while (fingers.length < nStr) fingers.push(0);

    var used = frets.filter(function (v) { return typeof v === 'number' && v > 0; });
    var maxFret = used.length ? Math.max.apply(null, used) : 0;
    var minFret = used.length ? Math.min.apply(null, used) : 0;

    var baseFret = parseInt(cfg.basefret, 10);
    if (isNaN(baseFret)) baseFret = (maxFret > 5) ? minFret : 1;

    var rows = parseInt(cfg.rows, 10);
    if (isNaN(rows)) {
      rows = (baseFret === 1) ? Math.max(4, maxFret) : Math.max(4, maxFret - baseFret + 1);
    }
    if (!isFinite(rows) || rows < 1) rows = 4;

    // geometry
    var sSp = 26, fSp = 30;
    var mL = 24, mR = 14, mT = 44, mB = 8;
    var gridW = (nStr - 1) * sSp;
    var gridH = rows * fSp;
    var gT = mT, gL = mL;
    var W = mL + gridW + mR, H = mT + gridH + mB;
    var xOf = function (i) { return gL + i * sSp; };

    var p = [];

    // chord name
    if (cfg.name) {
      p.push('<text x="' + r(W / 2) + '" y="20" text-anchor="middle" ' +
        'font-size="17" font-weight="600" fill="currentColor">' + esc(cfg.name) + '</text>');
    }

    // base-fret label (when starting up the neck)
    if (baseFret > 1) {
      p.push('<text x="' + r(gL - 8) + '" y="' + r(gT + fSp * 0.66) + '" ' +
        'text-anchor="end" font-size="12" fill="currentColor" opacity="0.75">' +
        baseFret + 'fr</text>');
    }

    // fret lines
    for (var f = 0; f <= rows; f++) {
      var y = gT + f * fSp;
      var thick = (baseFret === 1 && f === 0) ? 4 : 1.4;
      p.push('<line x1="' + r(gL) + '" y1="' + r(y) + '" x2="' + r(gL + gridW) +
        '" y2="' + r(y) + '" stroke="currentColor" stroke-width="' + thick + '"/>');
    }
    // string lines (thicker toward low E on the left)
    for (var i = 0; i < nStr; i++) {
      var sw = 1.4 - (i / nStr) * 0.7; // subtle taper
      p.push('<line x1="' + r(xOf(i)) + '" y1="' + r(gT) + '" x2="' + r(xOf(i)) +
        '" y2="' + r(gT + gridH) + '" stroke="currentColor" stroke-width="' + r(sw) + '"/>');
    }

    // barre detection: same fret + same finger across 2+ strings
    var barres = {};
    for (i = 0; i < nStr; i++) {
      var fr = frets[i], fg = fingers[i];
      if (typeof fr === 'number' && fr > 0 && fg > 0) {
        var key = fr + ':' + fg;
        (barres[key] = barres[key] || []).push(i);
      }
    }
    Object.keys(barres).forEach(function (key) {
      var idx = barres[key];
      if (idx.length < 2) return;
      var fr = parseInt(key.split(':')[0], 10);
      var rel = fr - baseFret + 1;
      var cy = gT + (rel - 0.5) * fSp;
      var x1 = xOf(Math.min.apply(null, idx)), x2 = xOf(Math.max.apply(null, idx));
      p.push('<rect x="' + r(x1 - 9) + '" y="' + r(cy - 9) + '" width="' + r(x2 - x1 + 18) +
        '" height="18" rx="9" fill="currentColor"/>');
    });

    // notes + open/mute markers
    for (i = 0; i < nStr; i++) {
      var v = frets[i], x = xOf(i);
      if (v === null) { // muted
        var my = gT - 11;
        p.push('<line x1="' + r(x - 4.5) + '" y1="' + r(my - 4.5) + '" x2="' + r(x + 4.5) +
          '" y2="' + r(my + 4.5) + '" stroke="currentColor" stroke-width="1.6"/>');
        p.push('<line x1="' + r(x - 4.5) + '" y1="' + r(my + 4.5) + '" x2="' + r(x + 4.5) +
          '" y2="' + r(my - 4.5) + '" stroke="currentColor" stroke-width="1.6"/>');
      } else if (v === 0) { // open
        p.push('<circle cx="' + r(x) + '" cy="' + r(gT - 11) + '" r="4.5" fill="none" ' +
          'stroke="currentColor" stroke-width="1.6"/>');
      } else { // fretted
        var rel2 = v - baseFret + 1;
        var cy2 = gT + (rel2 - 0.5) * fSp;
        p.push('<circle cx="' + r(x) + '" cy="' + r(cy2) + '" r="9.5" fill="currentColor"/>');
        if (fingers[i] > 0) {
          p.push('<text x="' + r(x) + '" y="' + r(cy2 + 4) + '" text-anchor="middle" ' +
            'font-size="11" font-weight="600" fill="#fff">' + fingers[i] + '</text>');
        }
      }
    }

    return svgWrap(W, H, p.join(''), cfg.name || 'chord diagram');
  }

  // ---------- scale / fretboard diagram ----------
  function renderScale(cfg) {
    var ROOT = DEFAULTS.rootColorVar;
    var nStr = clampInt(cfg.strings, 6, 3, 12);

    var dots = parseDots(cfg.dots);
    var rootSet = {};
    parseDots(cfg.roots).forEach(function (d) { rootSet[d.s + ':' + d.f] = true; });
    dots.forEach(function (d) { if (d.label.toUpperCase() === 'R') rootSet[d.s + ':' + d.f] = true; });

    // fret window
    var range = parseRange(cfg.frets);
    var usedF = dots.filter(function (d) { return d.f > 0; }).map(function (d) { return d.f; });
    var startFret, endFret;
    if (range) {
      startFret = range.start; endFret = range.end;
    } else if (usedF.length) {
      startFret = Math.min.apply(null, usedF);
      endFret = Math.max(Math.max.apply(null, usedF), startFret + 3);
    } else {
      startFret = 1; endFret = 5;
    }
    if (endFret < startFret) endFret = startFret;

    var spaces = endFret - startFret + 1;

    // geometry (horizontal neck): string 1 (high E) on top, string n (low E) bottom
    var sSp = 26, fSp = 42;
    var mL = 30, mR = 16, mT = 36, mB = 24;
    var gridW = spaces * fSp;
    var gridH = (nStr - 1) * sSp;
    var gT = mT, gL = mL;
    var W = mL + gridW + mR, H = mT + gridH + mB;

    var yOf = function (s) { return gT + (s - 1) * sSp; };          // string 1..n
    var xOf = function (fr) { return gL + (fr - startFret + 0.5) * fSp; }; // note center
    var lineX = function (k) { return gL + k * fSp; };             // fret line k (0..spaces)

    var p = [];

    if (cfg.name) {
      p.push('<text x="' + r(W / 2) + '" y="20" text-anchor="middle" ' +
        'font-size="16" font-weight="600" fill="currentColor">' + esc(cfg.name) + '</text>');
    }

    // inlay markers (single at 3,5,7,9,15,17,19,21 ; double at 12,24)
    var singles = { 3: 1, 5: 1, 7: 1, 9: 1, 15: 1, 17: 1, 19: 1, 21: 1 };
    var doubles = { 12: 1, 24: 1 };
    for (var fr = startFret; fr <= endFret; fr++) {
      var cx = xOf(fr), midY = gT + gridH / 2;
      if (doubles[fr]) {
        p.push('<circle cx="' + r(cx) + '" cy="' + r(midY - sSp) + '" r="4" fill="currentColor" opacity="0.18"/>');
        p.push('<circle cx="' + r(cx) + '" cy="' + r(midY + sSp) + '" r="4" fill="currentColor" opacity="0.18"/>');
      } else if (singles[fr]) {
        p.push('<circle cx="' + r(cx) + '" cy="' + r(midY) + '" r="4" fill="currentColor" opacity="0.18"/>');
      }
    }

    // fret lines (vertical)
    for (var k = 0; k <= spaces; k++) {
      var x = lineX(k);
      var thick = (startFret === 1 && k === 0) ? 4 : 1.4;
      p.push('<line x1="' + r(x) + '" y1="' + r(gT) + '" x2="' + r(x) + '" y2="' + r(gT + gridH) +
        '" stroke="currentColor" stroke-width="' + thick + '"/>');
    }
    // string lines (horizontal); low strings slightly thicker
    for (var s = 1; s <= nStr; s++) {
      var sw = 0.8 + (s / nStr) * 0.9;
      p.push('<line x1="' + r(gL) + '" y1="' + r(yOf(s)) + '" x2="' + r(gL + gridW) + '" y2="' + r(yOf(s)) +
        '" stroke="currentColor" stroke-width="' + r(sw) + '"/>');
    }

    // fret numbers
    for (k = 0; k < spaces; k++) {
      p.push('<text x="' + r(gL + (k + 0.5) * fSp) + '" y="' + r(gT + gridH + 16) +
        '" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.7">' +
        (startFret + k) + '</text>');
    }

    // dots
    dots.forEach(function (d) {
      if (d.s < 1 || d.s > nStr) return;
      var isRoot = !!rootSet[d.s + ':' + d.f] || d.label.toUpperCase() === 'R';
      var fill = isRoot ? ROOT : 'currentColor';
      var label = (d.label && d.label.toUpperCase() !== 'R') ? d.label : '';
      var y = yOf(d.s);
      var cx;
      if (d.f === 0) { // open-string note, left of nut
        cx = gL - 14;
      } else if (d.f < startFret || d.f > endFret) {
        return; // outside window
      } else {
        cx = xOf(d.f);
      }
      p.push('<circle cx="' + r(cx) + '" cy="' + r(y) + '" r="10" fill="' + fill +
        '" stroke="currentColor" stroke-width="' + (d.f === 0 ? 1.2 : 0) + '"/>');
      if (label) {
        p.push('<text x="' + r(cx) + '" y="' + r(y + 3.5) + '" text-anchor="middle" ' +
          'font-size="9.5" font-weight="600" fill="#fff">' + esc(label) + '</text>');
      }
    });

    return svgWrap(W, H, p.join(''), cfg.name || 'scale diagram');
  }

  function svgWrap(w, h, inner, label) {
    return '<svg class="chordscale-svg" xmlns="http://www.w3.org/2000/svg" ' +
      'viewBox="0 0 ' + r(w) + ' ' + r(h) + '" width="' + r(w) + '" height="' + r(h) + '" ' +
      'role="img" aria-label="' + esc(label) + '" ' +
      'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif">' +
      inner + '</svg>';
  }

  // ---------- DOM wiring ----------
  function injectStyles() {
    if (document.getElementById('chordscale-styles')) return;
    var st = document.createElement('style');
    st.id = 'chordscale-styles';
    st.textContent =
      '.chordscale-figure{display:inline-block;margin:0.6em 0.8em 0.6em 0;vertical-align:top;text-align:center}' +
      '.chordscale-svg{display:block;overflow:visible}';
    document.head.appendChild(st);
  }

  function targetsFor(selector) {
    var found = [];
    var seen = [];
    document.querySelectorAll(selector).forEach(function (el) {
      if (el.getAttribute('data-cs-done')) return;
      // Replace the outermost rendered wrapper so no stray <pre> remains.
      var node = el.closest('div[class*="language-"]') ||
        el.closest('figure.highlight') ||
        el.closest('pre') || el;
      if (seen.indexOf(node) !== -1) return;
      seen.push(node);
      found.push({ node: node, text: el.textContent });
    });
    return found;
  }

  function replaceWith(node, svg) {
    var fig = document.createElement('figure');
    fig.className = 'chordscale-figure';
    fig.setAttribute('data-cs-done', '1');
    fig.innerHTML = svg;
    node.parentNode.replaceChild(fig, node);
  }

  function renderAll(opts) {
    opts = opts || {};
    if (typeof document === 'undefined') return;
    injectStyles();
    var chordSel = opts.chordSelector || DEFAULTS.chordSelector;
    var scaleSel = opts.scaleSelector || DEFAULTS.scaleSelector;

    targetsFor(chordSel).forEach(function (t) {
      try { replaceWith(t.node, renderChord(parseConfig(t.text))); }
      catch (e) { if (global.console) console.error('chordscale (chord):', e); }
    });
    targetsFor(scaleSel).forEach(function (t) {
      try { replaceWith(t.node, renderScale(parseConfig(t.text))); }
      catch (e) { if (global.console) console.error('chordscale (scale):', e); }
    });
  }

  var ChordScale = {
    renderAll: renderAll,
    renderChord: renderChord,
    renderScale: renderScale,
    parseConfig: parseConfig
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = ChordScale;
  global.ChordScale = ChordScale;

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', function () { renderAll(); });
    } else {
      renderAll();
    }
  }
})(typeof window !== 'undefined' ? window : this);
