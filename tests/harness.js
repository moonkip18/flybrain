// Headless harness for flymind.html: stub DOM + THREE, run the sim loop, report state transitions.
const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.join(__dirname, '..', 'flymind.html'), 'utf8');
const blocks = [...src.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const circuitBlock = blocks[2];
const m = [null, blocks[3]];
if (!m) { console.error('no script'); process.exit(1); }

const stateHist = [];
let frameCount = 0;

const ctxProxy = new Proxy({}, {
  get: (t, p) => (p === 'canvas' ? {} : (..._a) => {}),
  set: () => true,
});

function makeElem(id) {
  const listeners = {};
  const el = {
    id, style: {}, dataset: {}, textContent: '', innerHTML: '',
    classList: { toggle() {}, add() {}, remove() {} },
    prepend() {}, append() {},
    addEventListener(t, f) { (listeners[t] = listeners[t] || []).push(f); },
    appendChild() {},
    clientWidth: 360, clientHeight: 680, width: 0, height: 0,
    getContext: () => ctxProxy,
    _listeners: listeners,
  };
  return el;
}

const elems = {};
global.document = {
  createElement(tag) { return makeElem('created-' + tag); },
  getElementById(id) {
    if (!elems[id]) {
      elems[id] = makeElem(id);
      if (id === 'stateLabel') {
        Object.defineProperty(elems[id], 'textContent', {
          get() { return this._tc || ''; },
          set(v) { if (v !== this._tc) stateHist.push([frameCount, v]); this._tc = v; },
        });
      }
    }
    return elems[id];
  },
  querySelectorAll: () => [],
};

class V3 {
  constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
  copy(v) { this.x = v.x; this.y = v.y; this.z = v.z; return this; }
  set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
  normalize() { const l = Math.hypot(this.x, this.y, this.z) || 1; this.x /= l; this.y /= l; this.z /= l; return this; }
  multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; return this; }
  lengthSq() { return this.x * this.x + this.y * this.y + this.z * this.z; }
}
class V2 {
  constructor(x = 0, y = 0) { this.x = x; this.y = y; }
  set(x, y) { this.x = x; this.y = y; return this; }
  multiplyScalar(s) { this.x *= s; this.y *= s; return this; }
}
class Color { constructor() {} copy() { return this; } lerp() { return this; } set() { return this; } }

class Obj {
  constructor(g, mt) {
    this.geometry = g; this.material = mt;
    this.position = new V3(); this.rotation = { x: 0, y: 0, z: 0, set(x, y, z) { this.x = x; this.y = y; this.z = z; } };
    this.scale = { set() {}, setScalar() {} };
    this.frustumCulled = true; this.intensity = 1;
  }
  add() {}
}
class Geo { setAttribute(n, a) { this.attributes = this.attributes || {}; this.attributes[n] = a; } setDrawRange() {} setFromPoints() { return this; } dispose() {} }

global.THREE = {
  Vector3: V3, Vector2: V2, Color,
  Scene: class { constructor() { this.background = null; } add() {} remove() {} },
  Group: class extends Obj { constructor() { super(); } },
  PerspectiveCamera: class { constructor() { this.position = new V3(); this.aspect = 1; } updateProjectionMatrix() {} },
  WebGLRenderer: class { constructor() { this.domElement = makeElem('glcanvas'); } setPixelRatio() {} setSize() {} render() {} },
  OrbitControls: class { constructor() { this.target = new V3(); } update() {} },
  AmbientLight: class extends Obj { constructor() { super(); } },
  DirectionalLight: class extends Obj { constructor() { super(); } },
  PointLight: class extends Obj { constructor() { super(); } },
  Mesh: Obj, Line: Obj, LineLoop: Obj, Points: Obj, LineSegments: Obj, GridHelper: class extends Obj { constructor() { super(); } },
  Sprite: class extends Obj { constructor() { super(); } },
  SpriteMaterial: class {}, CanvasTexture: class {},
  SphereGeometry: Geo, ConeGeometry: Geo, PlaneGeometry: Geo, CircleGeometry: Geo, TorusGeometry: Geo, CylinderGeometry: Geo, BufferGeometry: Geo,
  FogExp2: class {}, MeshPhysicalMaterial: class {},
  BufferAttribute: class { constructor(array) { this.array = array; this.itemSize = 3; this.needsUpdate = false; } },
  MeshStandardMaterial: class {}, MeshBasicMaterial: class {}, LineBasicMaterial: class {}, PointsMaterial: class {},
  Plane: class { constructor() { this.normal = new V3(0, 1, 0); this.constant = 0; } },
  Raycaster: class { setFromCamera() {} ray = { intersectPlane: () => null }; },
};

global.devicePixelRatio = 1;
global.window = { addEventListener() {} };

const rafQ = [];
global.requestAnimationFrame = (cb) => { rafQ.push(cb); return rafQ.length; };

try {
  eval(circuitBlock);
  eval(m[1]);
} catch (e) {
  console.error('LOAD ERROR:', e.stack);
  process.exit(1);
}

// pump 8 minutes of sim at 60fps
const FRAMES = 60 * 240;
let t = performance.now();
for (let i = 0; i < FRAMES; i++) {
  frameCount = i;
  const q = rafQ.splice(0);
  t += 16.6667;
  for (const cb of q) {
    try { cb(t); } catch (e) { console.error('FRAME ERR at', i, e.stack); process.exit(1); }
  }
  // poke the scare button at t=100s like a user would
  if (i === Math.floor(100 * 60)) {
    const btn = elems['btnScare'];
    if (btn && btn._listeners.click) btn._listeners.click.forEach(f => f({ target: { textContent: '' } }));
  }
  // deterministic exposure: food every 25s keeps hunger satisfiable; danger
  // placed EVERY 20s (one source lives 2.5+ min, sources cap at 6) — with the
  // AVOIDING branch fixed to beat RESTING, a persistent cloud next to the fly
  // guarantees the avoidance state regardless of drift direction.
  if (i % (25 * 60) === 10 * 60) {
    const FM0 = global.__FM || window.__FM;
    if (FM0 && FM0.placeFood) FM0.placeFood();
  }
  if (i % (20 * 60) === 5 * 60) {
    const FM0 = global.__FM || window.__FM;
    if (FM0 && FM0.placeDanger) FM0.placeDanger();
  }
}

// run-length encode state history
console.log('--- state transitions ---');
if (!stateHist.length) console.log('(no transitions ever ï¿½ stuck)');
for (const [f, s] of stateHist) console.log((f / 60).toFixed(1) + 's  ' + s);

// ---------- assertions ----------
const seen = new Set(stateHist.map(x => x[1]));
const results = [];
const test = (name, ok) => results.push([ok ? 'PASS' : 'FAIL', name]);
test('boots without errors', true);
test('feeding works (EATING seen)', seen.has('EATING'));
test('satiety works (RESTING seen)', seen.has('RESTING'));
test('forage/search cycle (FORAGING seen)', seen.has('FORAGING'));
test('sleep cycle (SLEEPING seen)', seen.has('SLEEPING'));
test('escape circuit fires on scare (ESCAPE seen)', seen.has('ESCAPE'));
test('avoidance of danger (AVOIDING seen)', seen.has('AVOIDING'));
test('state actually changes (>=5 distinct states)', seen.size >= 5);
const FM = global.__FM || window.__FM;
if (FM) {
  const s = FM.stats();
  test('KC layer sparse (<25%)', s.kcFrac < 0.25);
  test('KC layer not dead (>0.5%)', s.kcFrac > 0.005);
  test('no NaN in network state', !Number.isNaN(s.kcFrac) && !Number.isNaN(s.learned));
  test('all 139,255 real neurons loaded', FM.counts.all === 139255);
  test('web built over the brain (>=3000 links)', FM.counts.web >= 3000);
} else {
  test('debug hooks exposed (__FM)', false);
}
console.log('--- tests ---');
let failed = 0;
for (const [st, name] of results) {
  if (st === 'FAIL') failed++;
  console.log(st + '  ' + name);
}
console.log(failed === 0 ? 'ALL TESTS PASSED' : failed + ' TESTS FAILED');
process.exit(failed === 0 ? 0 : 1);
