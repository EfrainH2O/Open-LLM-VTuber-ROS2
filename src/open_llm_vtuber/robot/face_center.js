// [robot fork] Starts the avatar centered on its face. robot/frontend.py appends this file to the
// frontend bundle. With "faceCenter": {"x", "y"} in the model's model_dict.json entry, each model the
// page loads is moved once, as a drag would move it (the frontend's setModelPosition), so the center
// of its head hit area (e.g. mao_pro's HitAreaHead: hat brim to chin) is at (x, y) of the canvas,
// as fractions from the top left. The zoom is the entry's kScale, where the wheel zoom starts.
// Afterwards the frontend's wheel zoom and drag work as usual. Without the field: upstream.
(function () {
  "use strict";
  var handled = null;   // the last model moved (or left as it is)

  function num(v, d) {
    return typeof v === "number" && isFinite(v) ? v : d;
  }

  // The model's entry as the frontend keeps it (localStorage "modelInfo", written from the server's
  // set-model-and-conf before it loads the model).
  function target() {
    try {
      var c = JSON.parse(localStorage.getItem("modelInfo") || "null");
      c = c && c.faceCenter;
      return c && typeof c === "object" ? { x: num(c.x, 0.5), y: num(c.y, 0.45) } : null;
    } catch (e) {
      return null;
    }
  }

  // Center of the head hit area's drawable in model units; null if the model has none, undefined
  // while its vertices are not computed yet.
  function headCenter(m) {
    var set = m._modelSetting, model = m._model;
    for (var i = 0; i < set.getHitAreasCount(); i++) {
      var id = set.getHitAreaId(i), s = id.getString();
      if (!/head/i.test((typeof s === "string" ? s : (s && s.s) || "") + " " + set.getHitAreaName(i))) continue;
      var d = model.getDrawableIndex(id);
      if (d < 0) continue;
      var v = model.getDrawableVertices(d), n = model.getDrawableVertexCount(d);
      var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (var k = 0; k < n; k++) {
        x0 = Math.min(x0, v[2 * k]); x1 = Math.max(x1, v[2 * k]);
        y0 = Math.min(y0, v[2 * k + 1]); y1 = Math.max(y1, v[2 * k + 1]);
      }
      return y1 > y0 ? { x: (x0 + x1) / 2, y: (y0 + y1) / 2 } : undefined;
    }
    return null;
  }

  function place(m) {
    var t = target();
    if (!t) return true;
    var h = headCenter(m);
    if (h === undefined) return false;
    if (h === null) {
      console.warn("[faceCenter] the model has no head hit area, it is not moved");
      return true;
    }
    var canvas = document.getElementById("canvas");
    var W = canvas ? canvas.width : 0, H = canvas ? canvas.height : 0;
    if (!W || !H) return false;
    // The frontend's projection: model units span the canvas height, or its width when a model
    // wider than one unit is on a portrait canvas (that case also forces its scale every frame).
    var byWidth = m._model.getCanvasWidth() > 1 && W < H;
    var k = byWidth ? 2 / m._model.getCanvasWidth() : m.getModelMatrix().getArray()[0];
    var halfW = byWidth ? 1 : W / H, halfH = byWidth ? H / W : 1;
    window.getLAppAdapter().setModelPosition((2 * t.x - 1) * halfW - k * h.x, (1 - 2 * t.y) * halfH - k * h.y);
    return true;
  }

  // Watches for a newly loaded model (page start, another model, a kScale change), once the frontend
  // has set up Live2D (window.getLive2DManager; asking for the manager earlier would create it).
  function tick() {
    requestAnimationFrame(tick);
    var mgr = window.getLive2DManager && window.getLive2DManager();
    var m = mgr && mgr.getModel(0);
    if (!m || m === handled || !m.isInitialized()) return;
    try {
      if (place(m)) handled = m;
    } catch (e) {
      handled = m;
      console.warn("[faceCenter] not moved:", e);
    }
  }
  tick();
})();
