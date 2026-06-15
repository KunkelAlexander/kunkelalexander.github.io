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

  // ---------- pitch / tuning ----------
  var STD6 = [40, 45, 50, 55, 59, 64]; // E2 A2 D3 G3 B3 E4 (low -> high)
  var SEMI = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

  // "E2" -> 40, "Bb3" -> 58, "45" -> 45 (raw MIDI passthrough)
  function noteToMidi(tok) {
    if (/^\d+$/.test(tok)) return parseInt(tok, 10);
    var m = tok.match(/^([A-Ga-g])([#b]*)(-?\d+)$/);
    if (!m) return null;
    var acc = 0;
    for (var i = 0; i < m[2].length; i++) acc += (m[2][i] === '#') ? 1 : -1;
    return 12 * (parseInt(m[3], 10) + 1) + SEMI[m[1].toUpperCase()] + acc;
  }

  function midiToFreq(m) { return 440 * Math.pow(2, (m - 69) / 12); }

  // Open-string MIDI per string (index 0 = lowest string). Defaults to standard
  // 6-string guitar; otherwise all-fourths from E2. Override with `tuning:`.
  function tuningFor(cfg, n) {
    if (cfg && cfg.tuning) {
      var toks = cfg.tuning.trim().split(/\s+/).map(noteToMidi).filter(function (x) { return x != null; });
      if (toks.length) {
        while (toks.length < n) toks.push(toks[toks.length - 1] + 5);
        return toks.slice(0, n);
      }
    }
    if (n === 6) return STD6.slice();
    var t = [];
    for (var i = 0; i < n; i++) t.push(40 + 5 * i);
    return t;
  }

  // Frequencies for a chord, low string -> high (natural downstrum order).
  function chordNotes(cfg) {
    var n = clampInt(cfg.strings, 6, 3, 12);
    var frets = parseFrets(cfg.frets);
    var tun = tuningFor(cfg, n);
    var out = [];
    for (var i = 0; i < n; i++) {
      if (frets[i] === null || frets[i] === undefined) continue; // muted
      out.push(midiToFreq(tun[i] + frets[i]));
    }
    return out;
  }

  // Frequencies for a scale's dots, sorted low -> high.
  function scaleNotes(cfg) {
    var n = clampInt(cfg.strings, 6, 3, 12);
    var tun = tuningFor(cfg, n);
    var out = parseDots(cfg.dots)
      .filter(function (d) { return d.s >= 1 && d.s <= n; })
      .map(function (d) { return midiToFreq(tun[n - d.s] + d.f); }); // string 1 = highest
    out.sort(function (a, b) { return a - b; });
    return out;
  }

  // ---------- audio engine (Karplus-Strong plucked string) ----------
  var AC = null;
  function ensureCtx() {
    if (typeof window === 'undefined') return null;
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    if (!AC) AC = new Ctx();
    if (AC.state === 'suspended' && AC.resume) AC.resume();
    return AC;
  }

  function pluck(ctx, freq, when, dur, gain) {
    var sr = ctx.sampleRate;
    var N = Math.max(2, Math.round(sr / freq));
    var len = Math.ceil(sr * dur);
    var buf = ctx.createBuffer(1, len, sr);
    var out = buf.getChannelData(0);
    var ring = new Float32Array(N);
    var i;
    for (i = 0; i < N; i++) ring[i] = Math.random() * 2 - 1;
    for (i = 0; i < N; i++) ring[i] = 0.5 * (ring[i] + ring[(i + 1) % N]); // warm the pluck
    var idx = 0, decay = 0.996;
    for (i = 0; i < len; i++) {
      var cur = ring[idx];
      out[i] = cur;
      ring[idx] = 0.5 * (cur + ring[(idx + 1) % N]) * decay;
      idx = (idx + 1) % N;
    }
    var src = ctx.createBufferSource(); src.buffer = buf;
    var g = ctx.createGain();
    g.gain.setValueAtTime(gain, when);
    g.gain.setTargetAtTime(0.0001, when + dur * 0.65, 0.25); // smooth tail
    src.connect(g).connect(ctx.destination);
    src.start(when);
    src.stop(when + dur);
  }

  // Strum a chord (slight per-string delay = downstroke).
  function play(freqs, opts) {
    opts = opts || {};
    var ctx = ensureCtx();
    if (!ctx || !freqs || !freqs.length) return;
    var t0 = ctx.currentTime + 0.02;
    if (opts.arpeggio) {
      var step = opts.step || 0.18;
      var g = 0.22;
      freqs.forEach(function (f, i) { pluck(ctx, f, t0 + i * step, 1.1, g); });
    } else {
      var spread = (opts.spread != null) ? opts.spread : 0.028;
      var gc = Math.min(0.25, 0.9 / Math.max(1, freqs.length));
      freqs.forEach(function (f, i) { pluck(ctx, f, t0 + i * spread, 2.6, gc); });
    }
  }

  function audioEnabled(cfg, opts) {
    if (opts && opts.audio === false) return false;
    if (cfg && /^(off|false|no|0)$/i.test(cfg.audio || '')) return false;
    return typeof window !== 'undefined' && !!(window.AudioContext || window.webkitAudioContext);
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
      '.chordscale-figure{display:inline-block;margin:0.6em 0.8em 0.6em 0;vertical-align:top;text-align:center;transition:transform .1s ease}' +
      '.chordscale-svg{display:block;overflow:visible}' +
      '.chordscale-figure.cs-playable{cursor:pointer;-webkit-user-select:none;user-select:none}' +
      '.chordscale-figure.cs-playable:hover{opacity:.85}' +
      '.chordscale-figure.cs-playable:focus{outline:2px solid var(--chordscale-root,#d6452c);outline-offset:4px;border-radius:4px}' +
      '.chordscale-figure.cs-active{transform:scale(.97)}' +
      '.cs-play{display:block;font-size:11px;line-height:1;opacity:.5;margin-top:1px}' +
      '.chordscale-figure.cs-playable:hover .cs-play{opacity:.85}';
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
    return fig;
  }

  function makePlayable(fig, name, onPlay) {
    fig.classList.add('cs-playable');
    fig.setAttribute('role', 'button');
    fig.setAttribute('tabindex', '0');
    fig.setAttribute('aria-label', 'Play ' + (name || 'diagram'));
    var badge = document.createElement('span');
    badge.className = 'cs-play';
    badge.setAttribute('aria-hidden', 'true');
    badge.textContent = '\u25B6'; // ▶
    fig.appendChild(badge);
    var go = function () {
      onPlay();
      fig.classList.add('cs-active');
      setTimeout(function () { fig.classList.remove('cs-active'); }, 180);
    };
    fig.addEventListener('click', go);
    fig.addEventListener('keydown', function (e) {
      if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); go(); }
    });
  }

  function renderAll(opts) {
    opts = opts || {};
    if (typeof document === 'undefined') return;
    injectStyles();
    var chordSel = opts.chordSelector || DEFAULTS.chordSelector;
    var scaleSel = opts.scaleSelector || DEFAULTS.scaleSelector;

    targetsFor(chordSel).forEach(function (t) {
      try {
        var cfg = parseConfig(t.text);
        var fig = replaceWith(t.node, renderChord(cfg));
        if (audioEnabled(cfg, opts)) {
          var notes = chordNotes(cfg);
          if (notes.length) makePlayable(fig, cfg.name, function () { play(notes); });
        }
      } catch (e) { if (global.console) console.error('chordscale (chord):', e); }
    });
    targetsFor(scaleSel).forEach(function (t) {
      try {
        var cfg2 = parseConfig(t.text);
        var fig2 = replaceWith(t.node, renderScale(cfg2));
        if (audioEnabled(cfg2, opts)) {
          var notes2 = scaleNotes(cfg2);
          if (notes2.length) makePlayable(fig2, cfg2.name, function () { play(notes2, { arpeggio: true }); });
        }
      } catch (e) { if (global.console) console.error('chordscale (scale):', e); }
    });
  }

  var ChordScale = {
    renderAll: renderAll,
    renderChord: renderChord,
    renderScale: renderScale,
    parseConfig: parseConfig,
    chordNotes: chordNotes,
    scaleNotes: scaleNotes,
    noteToMidi: noteToMidi,
    play: play
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
