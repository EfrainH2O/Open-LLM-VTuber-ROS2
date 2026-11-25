// [robot fork] Inochi2D avatars on the VTuber page. robot/frontend.py appends this file to the
// frontend bundle. A model_dict.json entry with "inochi2d": {"puppet": "<.inx/.inp url>", ...} is
// still loaded by the frontend as a Live2D model (its "url", e.g. mao_pro): the frontend plays a
// sentence's audio only through a Live2D model, which also gives the lip-sync level and receives
// the expressions. That Live2D canvas is made invisible and the puppet is drawn over it with inox2d
// (WebAssembly, robot-inochi2d/, served at /robot-web/inochi2d/), driven by:
//  - the mouth: the hidden model's lip-sync level (its WAV handler's RMS, 0 when silent), smoothed;
//  - the emotion: each expression the frontend sets on the hidden model (an [emotion] tag in a
//    reply, the operator's wheel, the reset after a turn) is mapped back to its emotionMap name
//    and shown as that name's pose ("poses": {"joy": {"<param>": [x, y]}}), eased in;
//  - idle life: blinks, breathing and a slow head sway.
// Entries without "inochi2d" are not touched. If the runtime or the puppet fails to load, the
// hidden model is shown again, so the robot never loses its face.
(function () {
  "use strict";
  var RUNTIME = "/robot-web/inochi2d/inochi2d.js";
  var runtime = null;          // promise of the wasm module
  var cur = null;              // the shown puppet: {key, cfg, emo, avatar, canvas, ...}
  var pending = null;          // key of the puppet being loaded
  var failed = {};             // keys that did not load: not retried until the page reloads
  var emotion = "neutral";     // the emotion the frontend asked for last
  var hookedProto = null;

  function modelInfo() {
    try {
      return JSON.parse(localStorage.getItem("modelInfo") || "null");
    } catch (e) {
      return null;
    }
  }

  function live2dCanvas() {
    return document.getElementById("canvas");
  }

  function hideLive2D(hidden) {
    var c = live2dCanvas();
    if (c && c.style.opacity !== (hidden ? "0" : "")) c.style.opacity = hidden ? "0" : "";
  }

  // The frontend sets expressions through window.getLAppAdapter().setExpression(<name>), names
  // from the hidden model's list. Remember the emotion each one stands for.
  function hookExpressions() {
    var adapter = window.getLAppAdapter && window.getLAppAdapter();
    var proto = adapter && Object.getPrototypeOf(adapter);
    if (!proto || proto === hookedProto || typeof proto.setExpression !== "function") return;
    var original = proto.setExpression;
    proto.setExpression = function (name) {
      if (cur) emotion = emotionOf(this, name);
      return original.apply(this, arguments);
    };
    hookedProto = proto;
  }

  function emotionOf(adapter, name) {
    var map = (cur && cur.emotionMap) || {};
    for (var key in map) {
      try {
        if (adapter.getExpressionName(map[key]) === name) return key;
      } catch (e) { /* model not ready */ }
    }
    return "neutral";
  }

  function lipLevel() {
    var mgr = window.getLive2DManager && window.getLive2DManager();
    var m = mgr && mgr.getModel(0);
    var h = m && m._wavFileHandler;
    return h && typeof h._lastRms === "number" && isFinite(h._lastRms) ? h._lastRms : 0;
  }

  function vec(v) {
    return Array.isArray(v) ? [Number(v[0]) || 0, Number(v[1]) || 0] : [Number(v) || 0, 0];
  }

  function load(info) {
    var cfg = info.inochi2d;
    var key = String(cfg.puppet) + "|" + JSON.stringify(cfg);
    pending = key;
    runtime = runtime || import(RUNTIME).then(function (mod) {
      return mod.default().then(function () { return mod; });
    });
    var canvas = document.createElement("canvas");
    canvas.id = "inochi2d-canvas";
    canvas.style.cssText = "position:fixed;pointer-events:none;left:0;top:0;width:0;height:0";
    var bytes = fetch(cfg.puppet).then(function (r) {
      if (!r.ok) throw new Error("puppet " + cfg.puppet + ": HTTP " + r.status);
      return r.arrayBuffer();
    });
    Promise.all([runtime, bytes]).then(function (got) {
      if (pending !== key) return;   // another model was picked meanwhile
      var host = live2dCanvas();
      (host && host.parentNode ? host.parentNode : document.body).appendChild(canvas);
      fit(canvas);
      var avatar = new got[0].Avatar(canvas, new Uint8Array(got[1]));
      unload();
      cur = setup(key, cfg, info, avatar, canvas);
      pending = null;
      console.log("[inochi2d] shown:", cfg.puppet);
    }).catch(function (e) {
      if (pending !== key) return;
      pending = null;
      failed[key] = true;
      if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
      console.warn("[inochi2d] not shown, the Live2D model stays:", e);
      unload();
    });
  }

  function setup(key, cfg, info, avatar, canvas) {
    var poses = cfg.poses || {};
    var names = {};   // every parameter a pose or the base sets
    [cfg.base || {}].concat(Object.keys(poses).map(function (k) { return poses[k]; })).forEach(function (p) {
      for (var n in p) names[n] = true;
    });
    var mouth = cfg.mouth || {};
    var defaults = {};   // the puppet's own rest values: what a parameter no pose sets goes back to
    try {
      JSON.parse(avatar.params()).forEach(function (q) { defaults[q.name] = q.defaults; });
    } catch (e) { /* older runtime: 0 */ }
    var p = {
      key: key, cfg: cfg, avatar: avatar, canvas: canvas, emotionMap: info.emotionMap || {}, defaults: defaults,
      names: Object.keys(names), values: {}, w: 0, h: 0, t: 0, last: 0, mouth: 0,
      mouthParam: mouth.param || "Mouth:: Shape", mouthAxis: mouth.axis === 0 ? 0 : 1,
      mouthGain: typeof mouth.gain === "number" ? mouth.gain : 2.5,
      blink: cfg.blink || ["Eye:: Left:: Blink", "Eye:: Right:: Blink"],
      breath: cfg.breath === undefined ? "Breath" : cfg.breath,
      head: cfg.head === undefined ? "Head:: Yaw-Pitch" : cfg.head,
      nextBlink: 2, warned: false,
    };
    [p.mouthParam].concat(p.blink, p.breath ? [p.breath] : [], p.head ? [p.head] : []).forEach(function (n) {
      if (p.names.indexOf(n) < 0) p.names.push(n);
    });
    p.names.forEach(function (n) { p.values[n] = target(p, n); });
    emotion = "neutral";
    return p;
  }

  // What the current emotion's pose (else the base, else the puppet's default) gives a parameter.
  function target(p, name) {
    var pose = (p.cfg.poses || {})[emotion] || {};
    if (name in pose) return vec(pose[name]);
    var base = p.cfg.base || {};
    if (name in base) return vec(base[name]);
    return name in p.defaults ? vec(p.defaults[name]) : [0, 0];
  }

  function unload() {
    if (cur) {
      try { cur.avatar.free(); } catch (e) { /* already gone */ }
      if (cur.canvas.parentNode) cur.canvas.parentNode.removeChild(cur.canvas);
      cur = null;
    }
    hideLive2D(false);
  }

  // Our canvas covers the Live2D canvas exactly; its drawing buffer follows the screen's pixels.
  function fit(canvas) {
    var host = live2dCanvas();
    var r = host ? host.getBoundingClientRect() : { left: 0, top: 0, width: innerWidth, height: innerHeight };
    var s = canvas.style;
    var css = [r.left + "px", r.top + "px", r.width + "px", r.height + "px"];
    if (s.left !== css[0] || s.top !== css[1] || s.width !== css[2] || s.height !== css[3]) {
      s.left = css[0]; s.top = css[1]; s.width = css[2]; s.height = css[3];
    }
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    return [w, h];
  }

  function frame(p, now) {
    var dt = p.last ? Math.min((now - p.last) / 1000, 0.1) : 0;
    p.last = now;
    p.t += dt;
    var size = fit(p.canvas);
    if (size[0] !== p.w || size[1] !== p.h) {
      p.w = size[0];
      p.h = size[1];
      p.avatar.resize(p.w, p.h);
      var cam = p.cfg.camera || {};
      // `height`: puppet units from the top to the bottom of the screen
      p.avatar.set_camera(Number(cam.x) || 0, Number(cam.y) || 0, p.h / (Number(cam.height) || 3000));
    }
    // ease every parameter toward the emotion's pose
    var k = 1 - Math.exp(-dt * 8);
    p.names.forEach(function (n) {
      var t = target(p, n), v = p.values[n];
      v[0] += (t[0] - v[0]) * k;
      v[1] += (t[1] - v[1]) * k;
    });
    // mouth: fast open, slower close
    var open = Math.min(1, lipLevel() * p.mouthGain);
    p.mouth += (open - p.mouth) * (1 - Math.exp(-dt * (open > p.mouth ? 30 : 12)));
    var out = {};
    p.names.forEach(function (n) { out[n] = p.values[n].slice(); });
    var m = out[p.mouthParam];
    m[p.mouthAxis] = Math.max(m[p.mouthAxis], p.mouth);
    // blink every 2-6 s for 0.15 s, over whatever the pose does with the eyes
    if (p.t >= p.nextBlink) {
      var b = 1 - Math.abs((p.t - p.nextBlink) / 0.075 - 1);
      if (b < 0) p.nextBlink = p.t + 2 + Math.random() * 4;
      else p.blink.forEach(function (n) { out[n][0] = Math.max(out[n][0], b); });
    }
    if (p.breath) out[p.breath][0] = (1 - Math.cos(p.t * 2 * Math.PI / 3.5)) / 2;
    if (p.head) {
      out[p.head][0] += 0.12 * Math.sin(p.t * 0.53);
      out[p.head][1] += 0.06 * Math.sin(p.t * 0.31 + 1);
    }
    var values = new Float32Array(2 * p.names.length);
    p.names.forEach(function (n, i) { values[2 * i] = out[n][0]; values[2 * i + 1] = out[n][1]; });
    var unknown = p.avatar.draw(dt, p.names.join("\n"), values);
    if (unknown && !p.warned) {
      p.warned = true;
      console.warn("[inochi2d] the puppet has no parameter:", unknown.split("\n").join(", "));
    }
  }

  function tick(now) {
    requestAnimationFrame(tick);
    var info = modelInfo();
    var cfg = info && info.inochi2d;
    if (!cfg || !cfg.puppet) {
      pending = null;
      if (cur) unload();
      return;
    }
    hookExpressions();
    var key = String(cfg.puppet) + "|" + JSON.stringify(cfg);
    if ((!cur || cur.key !== key) && pending !== key && !failed[key]) load(info);
    hideLive2D(!failed[key]);   // the stand-in model stays hidden while the puppet loads
    if (!cur || cur.key !== key) return;
    try {
      frame(cur, now);
    } catch (e) {
      console.warn("[inochi2d] drawing failed, the Live2D model is shown again:", e);
      failed[key] = true;
      unload();
    }
  }
  requestAnimationFrame(tick);

  // for checks from the browser console (and the tests): what is shown now
  window.robotInochi2d = {
    state: function () {
      return cur ? { shown: true, puppet: cur.cfg.puppet, emotion: emotion, mouth: cur.mouth }
                 : { shown: false, loading: pending !== null };
    },
  };
})();
