//! [robot fork] One Inochi2D puppet drawn on a WebGL2 canvas, driven from JavaScript
//! (src/open_llm_vtuber/robot/inochi2d.js): the page sets parameter values every frame (mouth from
//! the lip sync, blinks, breathing, the emotion's pose) and calls `draw`.
//!
//! inox2d skips a parameter whose value is exactly (0, 0) (`ParamCtx::apply`, a TODO there), while
//! Inochi2D applies every parameter at any value. Rigs keep offsets at a parameter's zero point
//! (Aka's eyelids, Midori's arm and body), so those parts were drawn out of place at rest and
//! jumped when the parameter left zero (a blink). Here every parameter is set every frame, and an
//! exact (0, 0) is nudged by ZERO_NUDGE, far below anything visible.

use glam::Vec2;
use inox2d::formats::inp::parse_inp;
use inox2d::puppet::Puppet;
use inox2d::render::InoxRendererExt;
use inox2d_opengl::OpenglRenderer;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

#[wasm_bindgen(start)]
fn start() {
	console_error_panic_hook::set_once();
}

const ZERO_NUDGE: f32 = 1e-5;

#[wasm_bindgen]
pub struct Avatar {
	puppet: Puppet,
	renderer: OpenglRenderer,
	first: bool,
	params: Vec<(String, Vec2)>, // every parameter of the puppet and its default
}

/// The puppet's parameters (name, default) from its file's JSON payload (TRNSRTS\0, u32 length, JSON).
fn read_params(bytes: &[u8]) -> Vec<(String, Vec2)> {
	let len = bytes.get(8..12).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize).unwrap_or(0);
	let doc = bytes
		.get(12..12 + len)
		.and_then(|b| std::str::from_utf8(b).ok())
		.and_then(|s| json::parse(s).ok());
	let Some(doc) = doc else { return Vec::new() };
	doc["param"]
		.members()
		.filter_map(|p| {
			let name = p["name"].as_str()?.to_string();
			let d = &p["defaults"];
			Some((name, Vec2::new(d[0].as_f32().unwrap_or(0.0), d[1].as_f32().unwrap_or(0.0))))
		})
		.collect()
}

#[wasm_bindgen]
impl Avatar {
	/// Loads a puppet (`.inp` or `.inx` file bytes) onto `canvas`, which gets a WebGL2 context
	/// with a stencil buffer (inox2d's masks need it) and a transparent background.
	#[wasm_bindgen(constructor)]
	pub fn new(canvas: HtmlCanvasElement, bytes: &[u8]) -> Result<Avatar, JsError> {
		let options = js_sys::Object::new();
		for (key, value) in [("stencil", true), ("alpha", true), ("premultipliedAlpha", true), ("antialias", true)] {
			js_sys::Reflect::set(&options, &key.into(), &value.into()).map_err(|_| JsError::new("context options"))?;
		}
		let context = canvas
			.get_context_with_context_options("webgl2", &options)
			.ok()
			.flatten()
			.and_then(|c| c.dyn_into::<WebGl2RenderingContext>().ok())
			.ok_or_else(|| JsError::new("WebGL2 is not available"))?;
		let gl = glow::Context::from_webgl2_context(context);

		let mut model = parse_inp(bytes).map_err(|e| JsError::new(&e.to_string()))?;
		model.puppet.init_transforms();
		model.puppet.init_rendering();
		model.puppet.init_params();
		model.puppet.init_physics();
		let mut renderer = OpenglRenderer::new(gl, &model).map_err(|e| JsError::new(&e.to_string()))?;
		renderer.resize(canvas.width().max(1), canvas.height().max(1));
		Ok(Avatar { puppet: model.puppet, renderer, first: true, params: read_params(bytes) })
	}

	/// The puppet's parameters as JSON: [{"name", "defaults": [x, y]}].
	pub fn params(&self) -> String {
		let list: Vec<json::JsonValue> = self
			.params
			.iter()
			.map(|(name, d)| json::object! { "name": name.as_str(), "defaults": [d.x, d.y] })
			.collect();
		json::stringify(list)
	}

	/// The canvas' drawing buffer changed size (pixels).
	pub fn resize(&mut self, width: u32, height: u32) {
		self.renderer.resize(width.max(1), height.max(1));
	}

	/// Puppet units shown: `scale` pixels per unit, the puppet point (`x`, `y`) at the canvas
	/// center (puppet units, y down; inox2d's camera position is the opposite of that point).
	pub fn set_camera(&mut self, x: f32, y: f32, scale: f32) {
		self.renderer.camera.position = Vec2::new(-x, -y);
		self.renderer.camera.scale = Vec2::splat(scale);
	}

	/// One frame. `names` are parameter names separated by "\n"; parameter i gets
	/// (`values[2i]`, `values[2i + 1]`), every other parameter its default. `dt` (seconds) runs the
	/// physics (hair, skirt). Returns the names the puppet does not have, "\n"-separated.
	pub fn draw(&mut self, dt: f32, names: &str, values: &[f32]) -> String {
		let mut unknown = Vec::new();
		self.puppet.begin_frame();
		if let Some(params) = self.puppet.param_ctx.as_mut() {
			let mut set = |name: &str, value: Vec2| {
				let value = if value == Vec2::ZERO { Vec2::new(ZERO_NUDGE, 0.0) } else { value };
				params.set(name, value).is_ok()
			};
			for (name, default) in &self.params {
				set(name, *default);
			}
			for (i, name) in names.split('\n').filter(|n| !n.is_empty()).enumerate() {
				let value = Vec2::new(
					values.get(2 * i).copied().unwrap_or(0.0),
					values.get(2 * i + 1).copied().unwrap_or(0.0),
				);
				if !set(name, value) {
					unknown.push(name);
				}
			}
		}
		// the first frame runs no physics (a long first step would swing everything at start)
		self.puppet.end_frame(if self.first { 0.0 } else { dt.clamp(0.0, 0.1) });
		self.first = false;

		self.renderer.clear();
		self.renderer.on_begin_draw(&self.puppet);
		self.renderer.draw(&self.puppet);
		self.renderer.on_end_draw(&self.puppet);
		unknown.join("\n")
	}
}
