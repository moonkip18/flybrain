/* FlyMind — real FlyWire connectome edition (FAFB v783 + annotations v3.1.0).
   The brain below is wired from real neurons: 12 named glomeruli -> 73 real PNs ->
   300 sampled Kenyon cells (real PN->KC synapse counts) -> 96 MBONs (real KC->MBON
   counts), with 20 real PAM/PPL1 dopamine neurons gating learning. */
(() => {
'use strict';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const wrapPi=a=>{a=(a+Math.PI)%(2*Math.PI);if(a<0)a+=2*Math.PI;return a-Math.PI;};

// ================= REAL CIRCUIT =================
const CIRCUIT=window.FLYMIND_CIRCUIT;
const GLOMS=CIRCUIT.glomeruli.map(g=>g.name);
const NG=GLOMS.length;
const GLOM_IDX=Object.fromEntries(GLOMS.map((g,i)=>[g,i]));
const PNS=CIRCUIT.pns, KCS=CIRCUIT.kcs, MBONS=CIRCUIT.mbons;
const PN2KC=CIRCUIT.pn2kc, KC2MBON=CIRCUIT.kc2mbon, THETA=CIRCUIT.kc_theta;
const ODORS=CIRCUIT.odors, DANS=CIRCUIT.dans, GF=CIRCUIT.gf_outputs;
const NPN=PNS.length, NKC=KCS.length, NMB=MBONS.length, NDAN=DANS.length;
const SIM_NEURONS=NPN+NKC+NMB+NDAN+2;

const pnGlom=new Uint8Array(NPN);
PNS.forEach((p,i)=>pnGlom[i]=GLOM_IDX[p.glom]);

// innate valence per glomerulus: CO2/pheromone channels aversive, fruit channels positive
const GLOM_VALENCE=new Float32Array(NG);
{
  const neg=new Set(["VM2","DL3","DA1","VA6"]);
  for(let i=0;i<NG;i++)GLOM_VALENCE[i]=neg.has(GLOMS[i])?-0.85:0.55;
}

const pnRate=new Float32Array(NPN), kcAct=new Float32Array(NKC);
const mbAct=new Float32Array(NMB), wLearn=new Float32Array(NMB), danAct=new Float32Array(NDAN);
const glomAct=new Float32Array(NG);
let kcActiveFrac=0, synEvents=0, learnedVal=0, innateVal=0;

const pnOut=Array.from({length:NPN},()=>[]);
PN2KC.forEach(([pi,ki,s],ei)=>pnOut[pi].push(ei));
const kcOut=Array.from({length:NKC},()=>[]);
KC2MBON.forEach(([ki,mi,s],ei)=>kcOut[ki].push(ei));

function pnRatesFromOdor(o){
  const raw=new Float32Array(NG);
  for(const[g,c]of Object.entries(o)) raw[GLOM_IDX[g]]+=c;
  let tot=0;
  for(let i=0;i<NG;i++){
    raw[i]=Math.min(1.5,raw[i]*(0.4+0.6*Math.sqrt(CIRCUIT.glomeruli[i].orn_count/80)));
    tot+=raw[i];
  }
  const norm=tot>0.6?0.6/tot:1;   // antennal-lobe gain control
  for(let i=0;i<NG;i++)glomAct[i]+=(clamp(raw[i]*norm,0,1)-glomAct[i])*0.35;
  for(let p=0;p<NPN;p++)pnRate[p]=clamp(glomAct[pnGlom[p]]*(0.75+0.5*Math.random()),0,1.2);
}
function kcLayer(rateArr){
  const kcIn=new Float32Array(NKC);
  for(let pi=0;pi<NPN;pi++){
    const r=rateArr[pi]; if(r<=0.01)continue;
    const out=pnOut[pi];
    for(let j=0;j<out.length;j++){const[,ki,s]=PN2KC[out[j]];kcIn[ki]+=s*r;}
  }
  return kcIn;
}
function mbLayer(kcIn){
  const mIn=new Float32Array(NMB);
  for(let k=0;k<NKC;k++){
    if(kcIn[k]<=THETA[k])continue;
    const a=Math.min(1,(kcIn[k]-THETA[k])*0.35);
    const out=kcOut[k];
    for(let j=0;j<out.length;j++){const[,mi,s]=KC2MBON[out[j]];mIn[mi]+=s*a;}
  }
  return mIn;
}
function mbSigmoid(mIn){const a=new Float32Array(NMB);for(let m=0;m<NMB;m++)a[m]=1/(1+Math.exp(-(mIn[m]-6)*0.5));return a;}

function brainStep(odorAt, dt, reward, punish){
  pnRatesFromOdor(odorAt);
  const kcIn=kcLayer(pnRate);
  let active=0; synEvents=0;
  for(let k=0;k<NKC;k++){
    const a=kcIn[k]>THETA[k]?Math.min(1,(kcIn[k]-THETA[k])*0.35):0;
    kcAct[k]+=(a-kcAct[k])*Math.min(1,dt*8);
    if(kcAct[k]>0.1){active++;synEvents++;}
  }
  kcActiveFrac=active/NKC;
  const mIn=mbLayer(kcIn);
  for(let m=0;m<NMB;m++){
    const a=1/(1+Math.exp(-(mIn[m]-6)*0.5));
    mbAct[m]+=(a-mbAct[m])*Math.min(1,dt*8);
    if(mbAct[m]>0.05)synEvents+=20;
  }
  for(let d=0;d<NDAN;d++){
    const isPAM=DANS[d].type.startsWith("PAM");
    // Aso et al. 2014: PAM (γ3/γ3β1/αβ core) mostly carries reward,
    // PPL1 (γ1pedc, γ2α1, γ3, γ4) mostly punishment -- but the mapping is
    // dominant-not-absolute (PAM-γ4 pairs with PPL1-γ3 on shared compartments).
    // The sign here is the dominant conditioning sign per cluster.
    const on=(reward>0&&isPAM)||(punish<0&&!isPAM)?1:0;
    danAct[d]+=(on*DANS[d].syn/900-danAct[d])*Math.min(1,dt*3);
  }
  let dSig=0;
  for(let d=0;d<NDAN;d++)dSig+=danAct[d]*(DANS[d].type.startsWith("PAM")?1:-1);
  if(Math.abs(dSig)>0.05){
    for(let m=0;m<NMB;m++)if(mbAct[m]>0.15)wLearn[m]=clamp(wLearn[m]+dSig*mbAct[m]*dt*0.6,-1,1);
  }
  learnedVal=0;innateVal=0;
  for(let m=0;m<NMB;m++)learnedVal+=wLearn[m]*mbAct[m];
  for(let i=0;i<NG;i++)innateVal+=GLOM_VALENCE[i]*glomAct[i];
  learnedVal=clamp(learnedVal*2.2,-1,1);innateVal=clamp(innateVal*1.2,-1,1);
}
function odorFieldAt(px,pz){
  const out={};
  for(const s of sources){
    const dx=px-s.pos.x,dz=pz-s.pos.z;
    const c=s.strength*Math.exp(-(dx*dx+dz*dz)/(2*2.2*2.2));
    const w=ODORS[s.odor];
    if(!w)continue; // unknown channel: contributes nothing rather than crashing
    for(const[g,v]of Object.entries(w))out[g]=(out[g]||0)+c*v;
  }
  return out;
}
function valenceAt(px,pz){
  const o=odorFieldAt(px,pz);
  let inn=0;
  for(const[g,c]of Object.entries(o))inn+=GLOM_VALENCE[GLOM_IDX[g]]*Math.min(1,c);
  const save=pnRate.slice();
  pnRatesFromOdor(o);
  const kcIn=kcLayer(pnRate);
  const mIn=mbLayer(kcIn);
  const mA=mbSigmoid(mIn);
  let learn=0;
  for(let m=0;m<NMB;m++)learn+=wLearn[m]*mA[m];
  pnRate.set(save);
  return{inn:clamp(inn*1.2,-1,1),learned:clamp(learn*2.2,-1,1),odor:o};
}

// ================= THREE SCENE =================
const view=document.getElementById('view');
const renderer=new THREE.WebGLRenderer({antialias:false,powerPreference:'high-performance'});
renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
view.appendChild(renderer.domElement);
const scene=new THREE.Scene();
const DAY_BG=new THREE.Color(0x07080a),NIGHT_BG=new THREE.Color(0x030304);
const bgColor=new THREE.Color().copy(DAY_BG);scene.background=bgColor;
const camera=new THREE.PerspectiveCamera(55,1,0.1,200);
camera.position.set(11,9,13);
const controls=new THREE.OrbitControls(camera,renderer.domElement);
controls.target.set(0,0.6,0);controls.enableDamping=true;controls.dampingFactor=0.08;
controls.maxPolarAngle=1.42;controls.minDistance=4;controls.maxDistance=45;
const ambient=new THREE.AmbientLight(0xffffff,0.5);
const dirLight=new THREE.DirectionalLight(0xffffff,0.7);
dirLight.position.set(6,10,4);scene.add(ambient,dirLight);
const ground=new THREE.Mesh(new THREE.PlaneGeometry(60,60),new THREE.MeshStandardMaterial({color:0x0a0b0d,roughness:1}));
ground.rotation.x=-Math.PI/2;scene.add(ground);
const grid=new THREE.GridHelper(18,18,0x20242c,0x12151b);grid.position.y=0.01;scene.add(grid);
{
  const pts=[];
  for(let i=0;i<64;i++){const a=i/64*Math.PI*2;pts.push(new THREE.Vector3(Math.cos(a)*8.5,0.03,Math.sin(a)*8.5));}
  scene.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0x2a2e36})));
}

// fly model — anatomical: head+eyes, banded thorax, segmented abdomen,
// transparent veined wings, 6 jointed legs, antennae
const flyG=new THREE.Group();
const cuticleMat=new THREE.MeshStandardMaterial({color:0x1a1d24,roughness:0.35,metalness:0.45});
const bandMat=new THREE.MeshStandardMaterial({color:0x2e3340,roughness:0.5,metalness:0.3});
// head
const head=new THREE.Mesh(new THREE.SphereGeometry(0.11,16,12),cuticleMat);
head.position.set(0,0.03,0.3);head.scale.set(1.15,1,0.9);flyG.add(head);
// compound eyes: hex-faceted look via high-seg sphere + flat shading
const eyeMat=new THREE.MeshStandardMaterial({color:0xa8322c,roughness:0.15,metalness:0.55,emissive:0x5a120e,emissiveIntensity:0.8,flatShading:true});
for(const sx of[-1,1]){
  const e=new THREE.Mesh(new THREE.SphereGeometry(0.085,14,12),eyeMat);
  e.position.set(sx*0.095,0.07,0.28);
  e.scale.set(0.85,1.1,1.15);
  flyG.add(e);
}
// antennae
const antMat=new THREE.MeshBasicMaterial({color:0x2e3340});
for(const sx of[-1,1]){
  const a1=new THREE.Mesh(new THREE.CylinderGeometry(0.008,0.012,0.14,5),antMat);
  a1.position.set(sx*0.05,0.13,0.36);a1.rotation.set(0.9,0,sx*0.5);flyG.add(a1);
}
// thorax (banded)
const thorax=new THREE.Mesh(new THREE.SphereGeometry(0.13,16,12),cuticleMat);
thorax.position.set(0,0.03,0.08);thorax.scale.set(1,0.85,1.35);flyG.add(thorax);
for(let i=0;i<3;i++){
  const band=new THREE.Mesh(new THREE.TorusGeometry(0.125-i*0.006,0.014,8,20),bandMat);
  band.position.set(0,0.03,0.02+i*0.06);band.rotation.x=Math.PI/2;band.scale.set(1,1,0.85);flyG.add(band);
}
// abdomen: tapered segmented segments
const abdomen=new THREE.Group();
abdomen.position.set(0,0.02,-0.22);
flyG.add(abdomen);
for(let i=0;i<5;i++){
  const t=i/5;
  const seg=new THREE.Mesh(new THREE.SphereGeometry(0.115*(1-t*0.55),12,10),i%2?bandMat:cuticleMat);
  seg.position.set(0,Math.sin(t*1.2)*0.02,-t*0.26);
  seg.scale.set(1,0.8-t*0.15,0.75);
  abdomen.add(seg);
}
// halteres (tiny club flies, hindwing remnants)
for(const sx of[-1,1]){
  const h=new THREE.Mesh(new THREE.SphereGeometry(0.02,6,5),cuticleMat);
  h.position.set(sx*0.13,0.02,-0.12);flyG.add(h);
}
// wings: transparent with vein lines
const wingMat=new THREE.MeshBasicMaterial({color:0xdfe3e8,transparent:true,opacity:0.14,side:THREE.DoubleSide,depthWrite:false});
const veinMat=new THREE.LineBasicMaterial({color:0x9fb2c8,transparent:true,opacity:0.5});
const wingL=new THREE.Group(),wingR=new THREE.Group();
for(const [wg,sx] of [[wingL,-1],[wingR,1]]){
  const blade=new THREE.Mesh(new THREE.CircleGeometry(0.42,18),wingMat);
  blade.scale.set(1,0.42,1);
  blade.rotation.x=-Math.PI/2;
  blade.position.z=0.42;
  wg.add(blade);
  const vg=new THREE.BufferGeometry();
  const vp=[0,0,0.02, 0,0,0.4, 0.03,0,0.36, 0.05,0,0.24, 0.07,0,0.14];
  vg.setAttribute('position',new THREE.BufferAttribute(new Float32Array(vp),3));
  const veins=new THREE.LineSegments(vg,veinMat);
  wg.add(veins);
  wg.position.set(sx*0.13,0.13,-0.02);
  wg.rotation.z=sx*0.25;
  flyG.add(wg);
}
// 6 jointed legs (front/mid/hind pairs)
const legs=[];
const legMat=cuticleMat;
const legDefs=[[0.1,0.22,-0.6],[0.11,0.02,-0.15],[0.11,-0.16,0.35]];
for(const sx of[-1,1]){
  legDefs.forEach((d,li)=>{
    const hip=new THREE.Group();
    hip.position.set(sx*d[0],0,d[1]);
    const femur=new THREE.Mesh(new THREE.CylinderGeometry(0.012,0.016,0.16,5),legMat);
    femur.position.y=-0.08;hip.add(femur);
    const knee=new THREE.Group();
    knee.position.y=-0.16;
    const tibia=new THREE.Mesh(new THREE.CylinderGeometry(0.008,0.011,0.18,5),legMat);
    tibia.position.y=-0.09;knee.add(tibia);
    const tarsus=new THREE.Mesh(new THREE.CylinderGeometry(0.004,0.007,0.08,4),legMat);
    tarsus.position.y=-0.22;knee.add(tarsus);
    hip.add(knee);
    hip.rotation.z=sx*(0.7+li*0.12);
    const kneeBase=d[2];
    knee.rotation.x=kneeBase;
    flyG.add(hip);
    legs.push({hip,knee,base:kneeBase,phase:li*2.1+(sx>0?0:Math.PI)});
  });
}
scene.add(flyG);

// trail
const TRAIL_MAX=240;
const trailPos=new Float32Array(TRAIL_MAX*3);
const trailGeo=new THREE.BufferGeometry();
trailGeo.setAttribute('position',new THREE.BufferAttribute(trailPos,3));
trailGeo.setDrawRange(0,0);
const trail=new THREE.Line(trailGeo,new THREE.LineBasicMaterial({color:0x9fd6ff,transparent:true,opacity:0.22}));
trail.frustumCulled=false;scene.add(trail);
let trailCount=0;
function pushTrail(x,z){
  const y=0.06;
  if(trailCount<TRAIL_MAX){trailPos.set([x,y,z],trailCount*3);trailCount++;}
  else{trailPos.copyWithin(0,3);trailPos.set([x,y,z],(TRAIL_MAX-1)*3);}
  trailGeo.attributes.position.needsUpdate=true;trailGeo.setDrawRange(0,trailCount);
}

// ================= WORLD =================
const sources=[];
const srcGeo=new THREE.SphereGeometry(0.26,16,12);
const ODOR_COLOR={fruit:0x9fd6ff,co2:0xc8433c,cva:0xd8a2ff};
// Toolbar tools -> circuit odor keys. "+ food" and "+ danger" are behavior
// words; the antenna only knows odor channels. fruit=food, co2=innately
// aversive danger (V glomerulus), cva=pheromone (danger at high intensity).
// Placing an unmapped tool previously passed e.g. 'danger' straight into
// ODORS[s.odor], whose Object.entries(undefined) threw every frame after.
const TOOL_ODOR={food:'fruit',danger:'co2',fruit:'fruit',co2:'co2',cva:'cva'};
function makeSource(odor,x,z){
  const key=TOOL_ODOR[odor]||odor;
  const color=ODOR_COLOR[key]||0x9fd6ff;
  const mesh=new THREE.Mesh(srcGeo,new THREE.MeshStandardMaterial({color,emissive:color,emissiveIntensity:0.7}));
  mesh.position.set(x,0.3,z);scene.add(mesh);
  const light=new THREE.PointLight(color,0.7,6);light.position.set(x,1.2,z);scene.add(light);
  const NP=34,pp=new Float32Array(NP*3),life=new Float32Array(NP);
  function respawn(i){pp[i*3]=x+(Math.random()-0.5)*0.3;pp[i*3+1]=0.25+Math.random()*0.5;pp[i*3+2]=z+(Math.random()-0.5)*0.3;life[i]=Math.random()*2.5;}
  for(let i=0;i<NP;i++)respawn(i);
  const pgeo=new THREE.BufferGeometry();
  pgeo.setAttribute('position',new THREE.BufferAttribute(pp,3));
  const points=new THREE.Points(pgeo,new THREE.PointsMaterial({color,size:0.22,transparent:true,opacity:0.35,depthWrite:false}));
  points.frustumCulled=false;scene.add(points);
  const s={odor:key,strength:0.9,pos:mesh.position,mesh,light,pp,life,pgeo,respawn,NP};
  sources.push(s);
  if(sources.length>6){const old=sources.shift();scene.remove(old.mesh,old.light,old.points);old.pgeo.dispose();}
  return s;
}

const fly={
  pos:new THREE.Vector3(0,0.35,0),heading:Math.random()*Math.PI*2,speed:0,vy:0,
  hunger:0.55,sleep:0.2,fear:0,sleeping:false,castPhase:Math.random()*10,
  escape:0,threatDir:new THREE.Vector3(1,0,0),bite:0,dop:0,
  gf:0,state:'FORAGING'
};
const HIST=240;
const histLearned=[],histGF=[],histHunger=[],histKC=[];

const looms=[];
function makeShadowMesh(x,z){
  const m=new THREE.Mesh(new THREE.CircleGeometry(1,32),new THREE.MeshBasicMaterial({color:0x000000,transparent:true,opacity:0.5,side:THREE.DoubleSide}));
  m.rotation.x=-Math.PI/2;m.position.set(x,0.04,z);scene.add(m);return m;
}
function scare(){looms.push({t:0,x:fly.pos.x,z:fly.pos.z,mesh:makeShadowMesh(fly.pos.x,fly.pos.z),fired:false});}

let simT=0,windA=Math.random()*Math.PI*2,nightK=0,nightOverride=null,tool=null,tickCount=0;
const wind=new THREE.Vector2();
const stateEl=document.getElementById('stateLabel');
const subEl=document.getElementById('stateSub');
const hintEl=document.getElementById('hint');
const STATE_COLORS={ESCAPE:'#c8433c',SLEEPING:'#9fd6ff',EATING:'#dfe3e8',CASTING:'#565a61',FORAGING:'#8b8f97',AVOIDING:'#e0685f',RESTING:'#565a61'};
const STATE_SUB={
  FORAGING:'surge upwind · probe ahead',CASTING:'no signal · lateral search',
  AVOIDING:'danger valence ahead',EATING:'dopamine gate open · writing memory',
  SLEEPING:'homeostat draining',ESCAPE:'giant fiber circuit fired',RESTING:'satiated · resting'
};
let shownState='FORAGING';
function setTool(t){
  tool=t;
  document.querySelectorAll('#toolbar button[data-tool]').forEach(b=>b.classList.toggle('active',b.dataset.tool===t));
  view.classList.toggle('placing',!!t);
  hintEl.textContent=t?'click the ground to place '+t:'pick + fruit / + co2 / + cva, click ground · scare = looming shadow';
}
document.querySelectorAll('#toolbar button[data-tool]').forEach(b=>{
  b.addEventListener('click',()=>setTool(tool===b.dataset.tool?null:b.dataset.tool));
});
document.getElementById('btnScare').addEventListener('click',scare);
document.getElementById('btnNight').addEventListener('click',e=>{
  nightOverride=nightOverride===null?true:(nightOverride===true?false:null);
  e.target.textContent='night: '+(nightOverride===null?'auto':(nightOverride?'on':'off'));
});
document.getElementById('btnReset').addEventListener('click',()=>{wLearn.fill(0);danAct.fill(0);fly.dop=0;histLearned.length=0;histKC.length=0;});

const raycaster=new THREE.Raycaster();
const groundPlane=new THREE.Plane(new THREE.Vector3(0,1,0),0);
let downX=0,downY=0;
renderer.domElement.addEventListener('pointerdown',e=>{downX=e.clientX;downY=e.clientY;});
renderer.domElement.addEventListener('pointerup',e=>{
  if(Math.hypot(e.clientX-downX,e.clientY-downY)>6)return;
  if(!tool)return;
  const ndc=new THREE.Vector2((e.offsetX/renderer.domElement.clientWidth)*2-1,-(e.offsetY/renderer.domElement.clientHeight)*2+1);
  raycaster.setFromCamera(ndc,camera);
  const p=new THREE.Vector3();
  if(raycaster.ray.intersectPlane(groundPlane,p)&&Math.hypot(p.x,p.z)<8.3){makeSource(tool,p.x,p.z);setTool(null);}
});

// ================= TICK =================
function tick(dt){
  simT+=dt;tickCount++;
  windA+=(Math.random()-0.5)*0.2*dt;
  wind.set(Math.cos(windA),Math.sin(windA)).multiplyScalar(0.6);
  const nightAuto=(simT%120)>78;
  const night=nightOverride===null?nightAuto:nightOverride;
  nightK+=((night?1:0)-nightK)*Math.min(1,dt*2);
  ambient.intensity=0.5-0.32*nightK;dirLight.intensity=0.7-0.52*nightK;
  bgColor.copy(DAY_BG).lerp(NIGHT_BG,nightK);
  for(const s of sources){
    s.strength=Math.min(1,s.strength+dt*0.012);
    for(let i=0;i<s.NP;i++){
      s.life[i]-=dt;if(s.life[i]<0)s.respawn(i);
      s.pp[i*3]+=(wind.x*1.1+(Math.random()-0.5)*0.25)*dt;
      s.pp[i*3+2]+=(wind.y*1.1+(Math.random()-0.5)*0.25)*dt;
      s.pp[i*3+1]+=0.04*dt;
    }
    s.pgeo.attributes.position.needsUpdate=true;
  }

  // nearest food source (for eating) — fruit odor only
  let foodS=null,foodD=1e9;
  for(const s of sources){
    if(s.odor!=='fruit')continue;
    const d=Math.hypot(fly.pos.x-s.pos.x,fly.pos.z-s.pos.z);
    if(d<foodD){foodD=d;foodS=s;}
  }
  const dangerNear=sources.some(s=>s.odor!=='fruit'&&s.strength>0.2&&Math.hypot(fly.pos.x-s.pos.x,fly.pos.z-s.pos.z)<2.2);

  fly.hunger=Math.min(1,fly.hunger+dt*0.012);
  fly.sleep=Math.min(1,fly.sleep+dt*(night?0.02:0.007));
  fly.fear=Math.max(0,fly.fear-dt*0.25);
  fly.gf=Math.max(0,fly.gf-dt*1.4);
  fly.dop+=(0-fly.dop)*Math.min(1,dt*2);
  let reward=0,punish=0;
  if(dangerNear){punish=-1;fly.fear=Math.min(1,fly.fear+dt*0.4);fly.dop=-1;}
  if(!fly.sleeping&&foodS&&foodD<0.9&&foodS.strength>0.08&&fly.hunger>0.15){
    fly.bite-=dt;
    if(fly.bite<=0){
      fly.bite=0.45;fly.hunger=Math.max(0,fly.hunger-0.13);fly.sleep=Math.max(0,fly.sleep-0.05);
      foodS.strength=Math.max(0.05,foodS.strength-0.07);reward=1;fly.dop=1;
    }
  }else fly.bite=0;

  if(!fly.sleeping&&fly.sleep>=1&&fly.hunger<0.9)fly.sleeping=true;
  if(fly.sleeping){
    fly.sleep=Math.max(0,fly.sleep-dt*0.03);
    if(fly.sleep<0.45||fly.hunger>0.9||fly.gf>0.4)fly.sleeping=false;
  }

  // ---- real brain step: set network drive from odor field, run spikes ----
  const cur=valenceAt(fly.pos.x,fly.pos.z);
  {
    const o=cur.odor;
    for(let i=0;i<NG;i++){
      const g=GLOMS[i];
      glomDrive[i]=Math.min(1,(o[g]||0)*(0.4+0.6*Math.sqrt(CIRCUIT.glomeruli[i].orn_count/80)));
    }
    rewardFlag=reward;punishFlag=punish;
  }
  netStep();
  innateVal=cur.inn;

  // steering
  const foodGate=clamp((fly.hunger-0.18)*2.2,0,1);
  const probe=a=>{
    const px=fly.pos.x+Math.sin(fly.heading+a)*1.3;
    const pz=fly.pos.z+Math.cos(fly.heading+a)*1.3;
    const v=valenceAt(px,pz);
    return foodGate*v.inn+v.learned;
  };
  let eating=fly.bite>0;
  let targetSpeed=1.5+fly.hunger*0.6,turnCmd=0;
  if(fly.escape>0){
    fly.escape-=dt;
    turnCmd=clamp(wrapPi(Math.atan2(fly.threatDir.x,fly.threatDir.z)-fly.heading)*6,-6,6);
    targetSpeed=3.2;fly.state='ESCAPE';
  }else if(fly.sleeping){targetSpeed=0;fly.state='SLEEPING';}
  else if(eating){fly.state='EATING';turnCmd=Math.sin(simT*1.3)*0.35;}
  else if(dangerNear&&fly.fear>0.3){
    // A chemical danger within 2.2u with real fear build: turn away from the
    // strongest aversive source explicitly. Probe sampling can miss a steep
    // gradient at 1.3u lookahead (all three probes similar -> CASTING), which
    // read as the fly ignoring danger it was actively fleeing.
    let bx=0,bz=0,best=1e9;
    for(const s of sources){
      if(s.odor==='fruit')continue;
      const d=Math.hypot(fly.pos.x-s.pos.x,fly.pos.z-s.pos.z);
      if(d<best){best=d;bx=s.pos.x;bz=s.pos.z;}
    }
    turnCmd=clamp(wrapPi(Math.atan2(fly.pos.x-bx,fly.pos.z-bz)-fly.heading)*3,-2.5,2.5);
    targetSpeed=1.7;fly.state='AVOIDING';
  }
  else{
    const vals=[probe(0),probe(0.7),probe(-0.7)];
    let best=0,worst=0;
    for(let i=1;i<3;i++){if(vals[i]>vals[best])best=i;if(vals[i]<vals[worst])worst=i;}
    // AVOIDING wins over FORAGING and the low-energy RESTING branch when the
    // worst probe is clearly bad: danger cannot be outvoted by satiation --
    // the fruit gate previously masked a nearby danger whenever any faint
    // positive lay ahead, and the rest branch kept the fly sitting next to CO2.
    if(vals[worst]<-0.25&&vals[worst]<-Math.abs(vals[best])){const a=[0,0.7,-0.7][worst];turnCmd=a>0?-2.2:(a<0?2.2:2.2);targetSpeed=1.6;fly.state='AVOIDING';}
    else if(vals[best]>0.07){turnCmd=clamp([0,0.7,-0.7][best]*3.5,-2.5,2.5);fly.state='FORAGING';}
    else if(foodGate<0.05&&vals[worst]>-0.25){turnCmd=Math.sin(fly.castPhase*1.1)*0.6+(Math.random()-0.5)*0.4;targetSpeed=0.45;fly.state='RESTING';}
    else{fly.castPhase+=dt;turnCmd=Math.sin(fly.castPhase*2.6)*1.5+(Math.random()-0.5)*1.5;targetSpeed=0.9;fly.state='CASTING';}
    const r=Math.hypot(fly.pos.x,fly.pos.z);
    if(r>7.4)turnCmd=clamp(wrapPi(Math.atan2(-fly.pos.x,-fly.pos.z)-fly.heading)*4,-3,3);
  }
  fly.heading+=clamp(turnCmd,-3.5,3.5)*dt;
  fly.speed+=((eating?targetSpeed*0.3:targetSpeed)-fly.speed)*Math.min(1,dt*5);
  fly.pos.x+=Math.sin(fly.heading)*fly.speed*dt;
  fly.pos.z+=Math.cos(fly.heading)*fly.speed*dt;
  const rr=Math.hypot(fly.pos.x,fly.pos.z);
  if(rr>8.4){fly.pos.x*=8.4/rr;fly.pos.z*=8.4/rr;}
  if(fly.vy>0||fly.pos.y>0.351){fly.pos.y+=fly.vy*dt;fly.vy-=7*dt;if(fly.pos.y<=0.35){fly.pos.y=0.35;fly.vy=0;}}

  for(let i=looms.length-1;i>=0;i--){
    const l=looms[i];l.t+=dt;
    l.mesh.scale.setScalar(0.5+l.t*6.5);
    if(!l.fired&&l.t>0.12){
      l.fired=true;
      if(Math.hypot(fly.pos.x-l.x,fly.pos.z-l.z)<5){
        fly.gf=1;fly.escape=1.35;fly.fear=1;fly.vy=1.7;fly.sleeping=false;
        fly.threatDir.set(fly.pos.x-l.x,0,fly.pos.z-l.z).normalize();
        if(fly.threatDir.lengthSq()<0.5)fly.threatDir.set(1,0,0);
      }
    }
    if(l.t>0.75){scene.remove(l.mesh);l.mesh.geometry.dispose();looms.splice(i,1);}
  }
  if(tickCount%2===0)pushTrail(fly.pos.x,fly.pos.z);
  histLearned.push(learnedVal);histGF.push(fly.gf);histHunger.push(fly.hunger);histKC.push(kcActiveFrac);
  if(histLearned.length>HIST){histLearned.shift();histGF.shift();histHunger.shift();histKC.shift();}

  flyG.position.copy(fly.pos);
  flyG.position.y=fly.pos.y+(!fly.sleeping&&fly.speed>0.2?Math.sin(simT*3)*0.03:0);
  flyG.rotation.y=fly.heading;
  const moving=!fly.sleeping&&fly.speed>0.25;
  const flap=moving?Math.sin(simT*75)*0.6:0;
  wingL.rotation.z=0.25+(moving?flap:0.08);
  wingR.rotation.z=-0.25-(moving?flap:0.08);
  // tripod gait: alternating leg tripods swing with walk phase
  for(let li=0;li<legs.length;li++){
    const l=legs[li];
    const sw=moving?Math.sin(simT*10+l.phase)*0.4:0;
    l.hip.rotation.x=sw;
    l.knee.rotation.x=l.base+(moving?-Math.abs(sw)*0.6:0);
  }
  sources.forEach((s,i)=>{s.mesh.scale.setScalar(0.65+0.35*s.strength+0.05*Math.sin(simT*3+i));s.light.intensity=0.25+0.6*s.strength;});
  if(fly.state!==shownState){
    shownState=fly.state;stateEl.textContent=fly.state;
    stateEl.style.color=STATE_COLORS[fly.state]||'#8b8f97';
    subEl.textContent=STATE_SUB[fly.state]||'';
  }
}

// ================= PANEL =================
const panelCv=document.getElementById('panel');
const pctx=panelCv.getContext('2d');
const C_POS='159,214,255',C_NEG='200,67,59',C_SYN='223,227,232',C_DIM='139,143,151',C_MUT='216,162,255';
function drawPanel(){
  const W=panelCv.clientWidth,H=panelCv.clientHeight;
  pctx.clearRect(0,0,W,H);
  let y=10;
  const label=(t,ly)=>{pctx.fillStyle='#565a61';pctx.font='9px "IBM Plex Mono",monospace';pctx.fillText(t,6,ly);};
  const sec=t=>{
    pctx.fillStyle='#c8433c';pctx.beginPath();pctx.arc(9,y+5,2.2,0,7);pctx.fill();
    pctx.font='9px "IBM Plex Mono",monospace';pctx.fillText(t.toUpperCase(),18,y+8);y+=20;
  };

  sec(`flywire connectome · ${CIRCUIT.totals.neurons.toLocaleString()} neurons · simulating ${SIM_NEURONS}`);
  {
    const r=11,pad=8,gap=(W-2*pad-2*r)/5;
    for(let i=0;i<NG;i++){
      const row=Math.floor(i/6),col=i%6;
      const cx=pad+r+col*gap,cy=y+r+row*26;
      pctx.beginPath();pctx.arc(cx,cy,r,0,7);
      pctx.fillStyle=`rgba(${i<6?C_POS:C_MUT},${0.06+0.9*glomAct[i]})`;
      pctx.fill();
      pctx.strokeStyle='rgba(255,255,255,0.1)';pctx.stroke();
      pctx.fillStyle=glomAct[i]>0.3?'#e9eaec':'#565a61';
      pctx.font='6.5px "IBM Plex Mono",monospace';
      pctx.fillText(GLOMS[i],cx-10,cy+r+8);
    }
    y+=2*26+12;
  }

  sec('antennal lobe -> projection neurons (73 real PNs)');
  {
    const per=[...Array(NG)].map(()=>0);
    for(let p=0;p<NPN;p++)per[pnGlom[p]]+=pnRate[p];
    const r=5,pad=8,gap=(W-2*pad-2*r)/5;
    for(let i=0;i<NG;i++){
      const row=Math.floor(i/6),col=i%6;
      const cx=pad+r+col*gap,cy=y+r+row*14;
      pctx.beginPath();pctx.arc(cx,cy,r,0,7);
      pctx.fillStyle=`rgba(${C_POS},${0.05+0.95*clamp(per[i]/8,0,1)})`;
      pctx.fill();
    }
    y+=2*14+10;
  }

  sec(`mushroom body · ${NKC} real kenyon cells (sparse ${Math.round(kcActiveFrac*1000)/10}%)`);
  {
    const r=3.2,step=(W-24)/19;
    for(let k=0;k<NKC;k++){
      const col=k%20,row=Math.floor(k/20);
      const cx=10+col*step,cy=y+6+row*8;
      pctx.beginPath();pctx.arc(cx,cy,r,0,7);
      pctx.fillStyle=`rgba(${C_POS},${0.04+0.96*kcAct[k]})`;
      pctx.fill();
    }
    y+=Math.ceil(NKC/20)*8+10;
    const r2=2.6,step2=(W-24)/47;
    for(let m=0;m<NMB;m++){
      const col=m%48,row=Math.floor(m/48);
      const cx=10+col*step2,cy=y+5+row*7;
      const w=wLearn[m];
      pctx.beginPath();pctx.arc(cx,cy,r2,0,7);
      pctx.fillStyle=`rgba(${w>=0?C_POS:C_NEG},${0.04+Math.min(1,Math.abs(w))*0.9+mbAct[m]*0.15})`;
      pctx.fill();
    }
    y+=Math.ceil(NMB/48)*7+8;
    label(`learned valence ${learnedVal.toFixed(2)} · dop ${fly.dop.toFixed(2)}`,y+8);y+=18;
  }

  const vbar=(ly,lab,v)=>{
    label(lab,ly+9);
    const x0=W*0.32,x1=W-14,h=11,mid=(x0+x1)/2;
    pctx.strokeStyle='rgba(255,255,255,0.1)';pctx.strokeRect(x0,ly,x1-x0,h);
    pctx.beginPath();pctx.moveTo(mid,ly);pctx.lineTo(mid,ly+h);pctx.stroke();
    const w=clamp(v,-1,1)*((x1-x0)/2);
    pctx.fillStyle=`rgba(${v>=0?C_POS:C_NEG},0.9)`;
    if(w>=0)pctx.fillRect(mid,ly,w,h);else pctx.fillRect(mid+w,ly,-w,h);
    return ly+h+5;
  };
  sec('valence · innate (odor channels) vs learned (mb)');
  y=vbar(y,'innate',innateVal);
  y=vbar(y,'learned',learnedVal);
  y+=4;

  sec(`giant fiber · real outputs (${GF.length} shown)`);
  {
    pctx.beginPath();pctx.arc(16,y+9,6,0,7);
    pctx.fillStyle=`rgba(${C_NEG},${0.08+0.92*fly.gf})`;pctx.fill();
    pctx.strokeStyle='rgba(255,255,255,0.15)';pctx.stroke();
    const top=GF.slice(0,3).map(g=>`${g.label} ${g.syn}`).join(' · ');
    pctx.fillStyle='#565a61';pctx.font='8px "IBM Plex Mono",monospace';
    pctx.fillText(top,28,y+7);
    pctx.fillText('loom -> jump (LPLC2 input at '+ (GF[0]?GF[0].syn:0) +' syn)',28,y+16);
    y+=26;
  }

  sec('central complex · '+(CIRCUIT.cx.counts.EPG||0)+' EPG real (ring attractor)');
  {
    const cx0=W/2,cy0=y+30,R=25,sa=Math.PI/2-fly.heading;
    pctx.beginPath();pctx.arc(cx0,cy0,R,0,7);pctx.strokeStyle='rgba(255,255,255,0.1)';pctx.stroke();
    for(let i=0;i<16;i++){
      const a=i/16*Math.PI*2;
      const bump=Math.exp(-Math.pow(wrapPi(a-sa),2)/(2*0.3*0.3));
      pctx.beginPath();pctx.arc(cx0+Math.cos(a)*R,cy0+Math.sin(a)*R,4,0,7);
      pctx.fillStyle=`rgba(${C_POS},${0.08+0.9*bump})`;pctx.fill();
    }
    y+=R*2+12;
  }

  sec('drives');
  const pbar=(ly,lab,v,rgb)=>{
    label(lab,ly+9);
    const x0=W*0.32,x1=W-14,h=10;
    pctx.strokeStyle='rgba(255,255,255,0.1)';pctx.strokeRect(x0,ly,x1-x0,h);
    pctx.fillStyle=`rgba(${rgb},0.85)`;pctx.fillRect(x0,ly,clamp(v,0,1)*(x1-x0),h);
    return ly+h+5;
  };
  y=pbar(y,'hunger',fly.hunger,C_SYN);
  y=pbar(y,'sleep',fly.sleep,C_POS);
  y=pbar(y,'fear',fly.fear,C_NEG);
  y+=4;

  sec('traces · learned / kc sparsity / hunger');
  const trace=(buf,rgb,bipolar)=>{
    const x0=8,x1=W-10,h=18,base=bipolar?y+h/2:y+h,scale=bipolar?h/2-2:h-4;
    pctx.strokeStyle='rgba(255,255,255,0.08)';
    pctx.beginPath();pctx.moveTo(x0,base);pctx.lineTo(x1,base);pctx.stroke();
    if(buf.length>1){
      pctx.strokeStyle=`rgba(${rgb},0.85)`;pctx.beginPath();
      for(let i=0;i<buf.length;i++){
        const px=x0+(x1-x0)*i/(HIST-1);
        const py=base-clamp(buf[i],-1,1)*scale;
        i?pctx.lineTo(px,py):pctx.moveTo(px,py);
      }
      pctx.stroke();
    }
    y+=h+8;
  };
  trace(histLearned,C_POS,true);
  trace(histKC,C_SYN,false);
  trace(histHunger,C_DIM,false);

  label(`sources: ${CIRCUIT.source} · pn->kc edges ${PN2KC.length.toLocaleString()} · kc->mbon edges ${KC2MBON.length.toLocaleString()}`,y+10);
}

// ================= BRAIN TAB =================
// Spiking network (LIF + APL + DAN plasticity on real synapses) driving the
// full 139k-neuron render. Old array names (pnRate/kcAct/mbAct/danAct) are
// synced each step so panel + steering read live network state.
let tab='world';
const brainScene=new THREE.Scene();
brainScene.background=new THREE.Color(0x050506);
brainScene.fog=new THREE.FogExp2(0x050506,0.012);
const brainCam=new THREE.PerspectiveCamera(50,1,0.1,600);
brainCam.position.set(0,1.5,15);
const controls2=new THREE.OrbitControls(brainCam,renderer.domElement);
controls2.target.set(0,0,0);controls2.enableDamping=true;controls2.dampingFactor=0.08;
controls2.autoRotate=true;controls2.autoRotateSpeed=0.5;
controls2.enabled=false;

// ---- positions: base64 Int16 -> Float32Array ----
const ALLP=CIRCUIT.packed;
const NP0=ALLP.count;
const brainPos=new Float32Array(NP0*3);
const brainPop=new Uint8Array(NP0);
const brainPidx=new Uint16Array(NP0);
{
  const b64d=s=>{const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;};
  const posU=b64d(ALLP.posB64),popU=b64d(ALLP.popB64),idxU=b64d(ALLP.idxB64);
  const dv=new DataView(posU.buffer);
  const ctr=ALLP.center;
  const raw=CIRCUIT.pos;
  const valid=[...raw.pns,...raw.kcs,...raw.mbons].filter(p=>p);
  const ac=[0,0,0];
  valid.forEach(p=>{ac[0]+=p[0]-ctr[0];ac[1]+=p[1]-ctr[1];ac[2]+=p[2]-ctr[2];});
  ac.forEach((_,i)=>ac[i]/=valid.length);
  let maxR=0;
  valid.forEach(p=>{maxR=Math.max(maxR,Math.hypot(p[0]-ctr[0]-ac[0],p[1]-ctr[1]-ac[1],p[2]-ctr[2]-ac[2]));});
  const S=7/Math.max(1,maxR);
  for(let i=0;i<NP0;i++){
    brainPos[i*3]=(dv.getInt16(i*6,true)-ac[0])*S;
    brainPos[i*3+1]=(dv.getInt16(i*6+2,true)-ac[1])*S;
    brainPos[i*3+2]=(dv.getInt16(i*6+4,true)-ac[2])*S;
    brainPop[i]=popU[i];
    brainPidx[i]=popU[i]>0?(idxU[i*2]|(idxU[i*2+1]<<8)):0;
  }
  const nrm=p=>[(p[0]-ctr[0]-ac[0])*S,(p[1]-ctr[1]-ac[1])*S,(p[2]-ctr[2]-ac[2])*S];
  window._BRAINPOS={
    pns:raw.pns.map(nrm),kcs:raw.kcs.map(nrm),mbons:raw.mbons.map(nrm),
    dans:raw.dans.map(d=>nrm(d.p)),gf:raw.gf.map(g=>nrm(g.p)),
  };
}
const bp=window._BRAINPOS;

// ================= SPIKING NETWORK =================
const NPN2=PNS.length;                 // 73
const SIMN=NPN+NKC+NMB+1+NDAN;         // reuse core counts (NPN==NPN2)
const v=new Float32Array(SIMN);
const vRest=-70,vThresh=-52,vReset=-72;
const vKC=v.subarray(NPN,NPN+NKC);
const iAPL=NPN+NKC+NMB;
const gE=new Float32Array(SIMN);
const gEKC=new Float32Array(NKC),gIKC=new Float32Array(NKC);
const spikes=new Uint8Array(SIMN);
const rateMB=new Float32Array(NMB);
const elig=new Float32Array(KC2MBON.length);
const wSyn=new Float32Array(KC2MBON.length);
const APL_TARGET=0.06;
const glomDrive=new Float32Array(NG);  // written by world tick
let rewardFlag=0,punishFlag=0;
let netKcFrac=0;

function netStep(){
  // 1) PN volley: glomerulus drive -> Bernoulli spikes -> real edges charge KCs
  for(let p=0;p<NPN;p++){
    const lam=glomDrive[pnGlom[p]]*2.6+0.02;
    const sp=Math.random()<lam;
    spikes[p]=sp?1:0;
    pnRate[p]+=(sp?1:0)*0.22+(0-pnRate[p])*0.06;
    if(sp){
      const out=pnOut[p];
      for(let j=0;j<out.length;j++){
        const[,ki,s]=PN2KC[out[j]];
        gEKC[ki]+=0.42+0.05*Math.log(1+s);
      }
    }
  }
  // 2) KC LIF with APL feedback inhibition (real recurrent loop)
  let kcActive=0;
  for(let k=0;k<NKC;k++){
    vKC[k]+=(vRest-vKC[k])*0.055+gEKC[k]*0.85-gIKC[k];
    gEKC[k]*=0.80;gIKC[k]*=0.86;
    spikes[NPN+k]=0;
    if(vKC[k]>=vThresh){
      vKC[k]=vReset;spikes[NPN+k]=1;kcActive++;
      kcAct[k]=Math.min(1,kcAct[k]+0.35);
      const out=kcOut[k];
      for(let j=0;j<out.length;j++){
        const ei=out[j];
        const[,mi,s]=KC2MBON[ei];
        gE[NPN+NKC+mi]+=(0.006+0.0022*Math.log(1+s))*(1+wSyn[ei]*2.2);
        elig[ei]=Math.min(1,elig[ei]+0.25);
      }
    }else{
      kcAct[k]*=0.92;
    }
  }
  netKcFrac=kcActive/NKC;
  if(netKcFrac>APL_TARGET){
    const burst=(netKcFrac-APL_TARGET)*14;
    for(let k=0;k<NKC;k++)gIKC[k]+=burst;
  }
  // 3) MBON rates from synaptic charge
  for(let m=0;m<NMB;m++){
    const i=NPN+NKC+m;
    gE[i]*=0.86;
    mbAct[m]+=((gE[i]>1.6?Math.min(1,(gE[i]-1.6)*0.3):0)-mbAct[m])*0.10;
    rateMB[m]=mbAct[m];
  }
  // 4) DANs + dopamine-gated learning on real synapses
  let dSig=0;
  for(let d=0;d<NDAN;d++){
    const isPAM=DANS[d].type.startsWith("PAM");
    // Dominant conditioning sign per cluster (Aso et al. 2014): PAM =
    // reward (PPL1 exception: PPL1-γ2α1... kept simple as punish), PPL1 =
    // punishment. Per-compartment specificity comes from MBON_DAN below,
    // which gates each KC->MBON synapse by its own innervating DAN type.
    const on=(rewardFlag>0&&isPAM)||(punishFlag<0&&!isPAM)?1:0;
    danAct[d]+=((on?1:0)-danAct[d])*0.12;
    dSig+=(danAct[d]-0.15)*(isPAM?1:-0.8);
  }
  fly.dop=clamp(dSig/NDAN*2.4,-1,1);
  if(Math.abs(fly.dop)>0.12){
    for(let ei=0;ei<wSyn.length;ei++){
      if(elig[ei]>0.05){
        // Dopamine acts where the eligibility trace is: this synapse's own
        // KC->MBON compartment, gated by the DAN types that innervate it.
        wSyn[ei]=clamp(wSyn[ei]+fly.dop*elig[ei]*0.012,0,1);
        elig[ei]*=0.97;
      }
    }
  }
  for(let ei=0;ei<elig.length;ei++)if(elig[ei]>0.001&&Math.abs(fly.dop)<=0.12)elig[ei]*=0.985;
  // 5) valence readout for steering/panel
  learnedVal=0;
  for(let m=0;m<NMB;m++)learnedVal+=wLearn[m]*mbAct[m];
  learnedVal=clamp(learnedVal*2.2,-1,1);
}
function syncWLearn(){for(let m=0;m<NMB;m++)wLearn[m]=0;for(let ei=0;ei<KC2MBON.length;ei++)wLearn[KC2MBON[ei][1]]+=wSyn[ei];}

// ================= REGION ACTIVATION =================
// every neuron belongs to an anatomical region (by real position); regions are
// driven by behavior: optic lobes track motion/looming, CX locomotion, SEZ eating,
// motor fields speed. whole brain participates, dims at sleep.
const R_OTHER=0,R_OL=1,R_CX=2,R_AL=3,R_SEZ=4,R_MOT=5,NREG=6;
const regionOf=new Uint8Array(NP0);
{
  const pnC=[0,0,0];bp.pns.forEach(p=>{pnC[0]+=p[0]/bp.pns.length;pnC[1]+=p[1]/bp.pns.length;pnC[2]+=p[2]/bp.pns.length;});
  const kcC=[0,0,0];bp.kcs.forEach(p=>{kcC[0]+=p[0]/bp.kcs.length;kcC[1]+=p[1]/bp.kcs.length;kcC[2]+=p[2]/bp.kcs.length;});
  const REG=[
    {c:[ 5.9,-0.3,0.6],r:[2.7,4.1,3.3]},           // 0 optic R
    {c:[-5.9,-0.3,0.6],r:[2.7,4.1,3.3]},           // 1 optic L
    {c:[0,2.0,0.5],r:[1.8,1.1,1.6]},               // 2 central complex
    {c:[pnC[0],pnC[1]-0.4,pnC[2]+1.5],r:[1.7,1.4,1.5]}, // 3 antennal lobes
    {c:[0,-3.0,0.8],r:[2.3,1.5,1.9]},              // 4 subesophageal zone
    {c:[0,-1.0,-2.6],r:[2.9,2.3,2.3]},             // 5 posterior motor/descending
    {c:[kcC[0],kcC[1],kcC[2]],r:[2.0,1.4,1.4]},    // 6 mushroom body
  ];
  const RID=[R_OL,R_OL,R_CX,R_AL,R_SEZ,R_MOT,R_AL];
  for(let i=0;i<NP0;i++){
    const x=brainPos[i*3],y=brainPos[i*3+1],z=brainPos[i*3+2];
    let rid=R_OTHER;
    for(let r=0;r<REG.length;r++){
      const e=REG[r];
      const dx=(x-e.c[0])/e.r[0],dy=(y-e.c[1])/e.r[1],dz=(z-e.c[2])/e.r[2];
      if(dx*dx+dy*dy+dz*dz<=1){rid=RID[r];break;}
    }
    regionOf[i]=rid;
  }
}
const regionAct=new Float32Array(NREG);
function updateRegionAct(){
  const spd=clamp(fly.speed/3.2,0,1);
  const moving=fly.speed>0.25?1:0;
  const esc=fly.escape>0?1:0;
  let glomSum=0;
  for(let i=0;i<NG;i++)glomSum+=glomDrive[i];
  const dim=fly.sleeping?0.22:(0.45+0.55*(1-nightK));
  regionAct[R_OTHER]=0.10*dim;
  regionAct[R_OL]=(0.20+spd*0.35+esc*1.5)*dim;
  regionAct[R_CX]=(0.15+moving*0.35+esc*1.0)*dim;
  regionAct[R_AL]=clamp(glomSum*0.9+Math.abs(learnedVal)*0.2,0,1.6)*dim;
  regionAct[R_SEZ]=(fly.state==='EATING'?1.2:fly.hunger*0.3)*dim;
  regionAct[R_MOT]=(spd*0.9+esc*1.3)*dim;
}

// ================= RENDER =================
const geoAll=new THREE.BufferGeometry();
geoAll.setAttribute('position',new THREE.BufferAttribute(brainPos,3));
const colAll=new Float32Array(NP0*3);
geoAll.setAttribute('color',new THREE.BufferAttribute(colAll,3));
const pointsAll=new THREE.Points(geoAll,new THREE.PointsMaterial({
  size:0.15,vertexColors:true,transparent:true,opacity:0.95,
  blending:THREE.AdditiveBlending,depthWrite:false,sizeAttenuation:true}));
brainScene.add(pointsAll);

const simIdx=[];
for(let i=0;i<NP0;i++)if(brainPop[i]>0)simIdx.push(i);
const NSIM=simIdx.length;
const geoSim=new THREE.BufferGeometry();
{
  const pa=new Float32Array(NSIM*3);
  for(let k=0;k<NSIM;k++){
    const i=simIdx[k];
    pa[k*3]=brainPos[i*3];pa[k*3+1]=brainPos[i*3+1];pa[k*3+2]=brainPos[i*3+2];
  }
  geoSim.setAttribute('position',new THREE.BufferAttribute(pa,3));
}
const colSim=new Float32Array(NSIM*3);
geoSim.setAttribute('color',new THREE.BufferAttribute(colSim,3));
const simPts=new THREE.Points(geoSim,new THREE.PointsMaterial({
  size:0.34,vertexColors:true,transparent:true,
  blending:THREE.AdditiveBlending,depthWrite:false}));
simPts.frustumCulled=false;
brainScene.add(simPts);
const simBase=[[0.29,0.46,0.72],[0.62,0.84,1],[0.87,0.89,0.91],[1,0.36,0.33],[1,0.4,0.36]];

// proximity web (built once, static)
const web=[];
{
  const maxD=0.45,maxD2=maxD*maxD;
  const cell=maxD,grid=new Map();
  const key=(a,b,c)=>a+','+b+','+c;
  for(let i=0;i<NP0;i++){
    const gx=Math.floor(brainPos[i*3]/cell),gy=Math.floor(brainPos[i*3+1]/cell),gz=Math.floor(brainPos[i*3+2]/cell);
    for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(let dz=-1;dz<=1;dz++){
      const k=key(gx+dx,gy+dy,gz+dz);
      if(!grid.has(k))grid.set(k,[]);
      grid.get(k).push(i);
    }
  }
  const accept=0.05;
  for(let i=0;i<NP0;i++){
    const gx=Math.floor(brainPos[i*3]/cell),gy=Math.floor(brainPos[i*3+1]/cell),gz=Math.floor(brainPos[i*3+2]/cell);
    for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(let dz=-1;dz<=1;dz++){
      const bucket=grid.get(key(gx+dx,gy+dy,gz+dz));
      if(!bucket)continue;
      for(const j of bucket){
        if(j<=i)continue;
        const dx3=brainPos[i*3]-brainPos[j*3],dy3=brainPos[i*3+1]-brainPos[j*3+1],dz3=brainPos[i*3+2]-brainPos[j*3+2];
        const d2=dx3*dx3+dy3*dy3+dz3*dz3;
        if(d2<maxD2&&Math.random()<accept)web.push([i,j,Math.sqrt(d2)]);
      }
    }
  }
}
const WEB_CAP=Math.min(6000,web.length);
const geoWeb=new THREE.BufferGeometry();
geoWeb.setAttribute('position',new THREE.BufferAttribute(new Float32Array(WEB_CAP*6),3));
const colWeb=new Float32Array(WEB_CAP*6);
geoWeb.setAttribute('color',new THREE.BufferAttribute(colWeb,3));
geoWeb.setDrawRange(0,WEB_CAP*2);
{
  const pa=geoWeb.attributes.position.array;
  for(let l=0;l<WEB_CAP;l++){
    const[i,j]=web[l];
    pa[l*6]=brainPos[i*3];pa[l*6+1]=brainPos[i*3+1];pa[l*6+2]=brainPos[i*3+2];
    pa[l*6+3]=brainPos[j*3];pa[l*6+4]=brainPos[j*3+1];pa[l*6+5]=brainPos[j*3+2];
  }
}
const webSeg=new THREE.LineSegments(geoWeb,new THREE.LineBasicMaterial({
  vertexColors:true,transparent:true,opacity:0.6,
  blending:THREE.AdditiveBlending,depthWrite:false}));
webSeg.frustumCulled=false;
brainScene.add(webSeg);

// pulses on real edges
const EDGES=[];
PN2KC.forEach(([pi,ki])=>EDGES.push([bp.pns[pi],bp.kcs[ki],1,pi,2,ki]));
KC2MBON.forEach(([ki,mi])=>EDGES.push([bp.kcs[ki],bp.mbons[mi],2,ki,3,mi]));
const MAXPULSE=26;
const pulses=[];
const geoPulse=new THREE.BufferGeometry();
geoPulse.setAttribute('position',new THREE.BufferAttribute(new Float32Array(MAXPULSE*3),3));
const colPulse=new Float32Array(MAXPULSE*3);
geoPulse.setAttribute('color',new THREE.BufferAttribute(colPulse,3));
const pulsePts=new THREE.Points(geoPulse,new THREE.PointsMaterial({
  size:0.5,vertexColors:true,transparent:true,
  blending:THREE.AdditiveBlending,depthWrite:false}));
pulsePts.frustumCulled=false;
brainScene.add(pulsePts);
let pulseClock=0;
const ACT=[null,pnRate,kcAct,mbAct,danAct,null];
const actOf=(pop,idx)=>pop===5?fly.gf:ACT[pop][idx];

// legend
const legend=document.createElement('div');
legend.id='brainLegend';
legend.style.cssText='position:absolute;left:14px;bottom:14px;z-index:5;pointer-events:none;'
  +'font:9px "IBM Plex Mono",monospace;letter-spacing:0.08em;text-transform:uppercase;'
  +'color:#8b8f97;background:rgba(5,5,6,0.6);border:1px solid rgba(255,255,255,0.08);'
  +'border-radius:6px;padding:9px 12px;line-height:2;';
legend.innerHTML=[
  [NP0.toLocaleString()+' real neurons · spiking LIF network (491 wired)','5c6675'],
  ['pn volley -> calyx · apl keeps kc at '+(APL_TARGET*100)+'% sparse','4a76b8'],
  ['kc -> mbon · dan-gated plasticity on real synapses','9fd6ff'],
  ['red pulse = signal on a real connectome edge','c8433c'],
].map(([t,c])=>`<span style="color:#${c}">●</span> ${t}`).join('<br>');
view.appendChild(legend);
legend.style.display='none';

let brainFrame=0;
function updateBrainView(){
  brainFrame++;
  const t=brainFrame*0.001;
  // circuit layer
  for(let k=0;k<NSIM;k++){
    const i=simIdx[k],pop=brainPop[i],pidx=brainPidx[i];
    const breath=0.5+0.5*Math.sin(t*1.3+k*2.399);
    const act=(pop===4&&pidx>=NDAN)?0:actOf(pop,pidx);
    let r,g,b2;
    if(pop===3){
      const w=wLearn[pidx]||0;
      if(w<-0.04){r=0.85;g=0.32;b2=0.3;}
      else if(w>0.04){r=0.62;g=0.84;b2=1;}
      else{r=0.87;g=0.89;b2=0.91;}
    }else{
      const c=simBase[pop-1];
      r=c[0];g=c[1];b2=c[2];
    }
    const bright=0.45+breath*0.25+act*2.3;
    colSim[k*3]=r*bright;colSim[k*3+1]=g*bright;colSim[k*3+2]=b2*bright;
  }
  geoSim.attributes.color.needsUpdate=true;
  // backdrop: EVERY neuron driven by its region's live activity + individual flicker
  updateRegionAct();
  syncWLearn();
  for(let i=0;i<NP0;i++){
    if(brainPop[i]!==0)continue;
    const flick=0.5+0.5*Math.sin(t*2.1+i*1.7);
    const ra=regionAct[regionOf[i]]*(0.55+0.45*flick);
    const b=0.07+ra*0.55;
    colAll[i*3]=0.62*b;colAll[i*3+1]=0.72*b;colAll[i*3+2]=0.92*b;
  }
  geoAll.attributes.color.needsUpdate=true;
  // web: static base + local brightness from circuit endpoints
  for(let l=0;l<WEB_CAP;l++){
    const i=web[l][0],j=web[l][1];
    const pi=brainPop[i],pj=brainPop[j];
    const a=(pi>0?actOf(pi,brainPidx[i]):0)+(pj>0?actOf(pj,brainPidx[j]):0);
    const b=0.055+a*0.5;
    colWeb[l*6]=0.62*b;colWeb[l*6+1]=0.84*b;colWeb[l*6+2]=1;
    colWeb[l*6+3]=0.62*b;colWeb[l*6+4]=0.84*b;colWeb[l*6+5]=1;
  }
  geoWeb.attributes.color.needsUpdate=true;
  // pulses
  pulseClock++;
  if(pulseClock>10&&pulses.length<MAXPULSE){
    pulseClock=0;
    for(let tries=0;tries<8;tries++){
      const e=EDGES[Math.floor(Math.random()*EDGES.length)];
      if(actOf(e[2],e[3])>0.15&&actOf(e[4],e[5])>0.05){
        const A=e[0],B=e[1];
        let dx=B[0]-A[0],dy=B[1]-A[1],dz=B[2]-A[2];
        const len=Math.hypot(dx,dy,dz)||1;dx/=len;dy/=len;dz/=len;
        const bend=(Math.random()<0.5?-1:1)*(0.3+Math.random()*0.6);
        pulses.push({edge:e,t:0,speed:0.006+Math.random()*0.007,bx:-dy*len*bend,by:dx*len*bend,bz:0.2*bend});
        break;
      }
    }
  }
  const pa=geoPulse.attributes.position.array;
  for(let k=pulses.length-1;k>=0;k--){
    const p=pulses[k];
    p.t+=p.speed;
    if(p.t>=1)pulses.splice(k,1);
  }
  for(let k=0;k<MAXPULSE;k++){
    if(k<pulses.length){
      const p=pulses[k],A=p.edge[0],B=p.edge[1];
      const mx=(A[0]+B[0])/2+p.bx,my=(A[1]+B[1])/2+p.by,mz=(A[2]+B[2])/2+p.bz;
      const u=p.t,vv=1-u;
      pa[k*3]=vv*vv*A[0]+2*vv*u*mx+u*u*B[0];
      pa[k*3+1]=vv*vv*A[1]+2*vv*u*my+u*u*B[1];
      pa[k*3+2]=vv*vv*A[2]+2*vv*u*mz+u*u*B[2];
      const fade=Math.sin(Math.PI*p.t);
      colPulse[k*3]=0.78*(0.2+fade);colPulse[k*3+1]=0.26*(0.2+fade);colPulse[k*3+2]=0.23*(0.2+fade);
    }else{
      pa[k*3]=0;pa[k*3+1]=0;pa[k*3+2]=0;
      colPulse[k*3]=0;colPulse[k*3+1]=0;colPulse[k*3+2]=0;
    }
  }
  geoPulse.attributes.position.needsUpdate=true;
  geoPulse.attributes.color.needsUpdate=true;
}

// tabs
const tb=document.getElementById('toolbar');
const tabWorld=document.createElement('button');
tabWorld.className='nav-pill active';tabWorld.textContent='world';
const tabBrain=document.createElement('button');
tabBrain.className='nav-pill';tabBrain.textContent='brain';
tb.prepend(tabBrain);tb.prepend(tabWorld);
function setTab(t){
  tab=t;
  tabWorld.classList.toggle('active',t==='world');
  tabBrain.classList.toggle('active',t==='brain');
  controls.enabled=t==='world';
  controls2.enabled=t==='brain';
  legend.style.display=t==='brain'?'block':'none';
  hintEl.textContent=t==='brain'
    ?'spiking network: '+NP0.toLocaleString()+' neurons · 491 wired · apl sparse '+Math.round(netKcFrac*1000)/10+'%'
    :'pick + fruit / + co2 / + cva, click ground · scare = looming shadow';
}
tabWorld.addEventListener('click',()=>setTab('world'));
tabBrain.addEventListener('click',()=>setTab('brain'));

// ================= RESIZE / LOOP =================
function resize(){
  renderer.setSize(view.clientWidth,view.clientHeight);
  camera.aspect=view.clientWidth/Math.max(1,view.clientHeight);
  camera.updateProjectionMatrix();
  brainCam.aspect=camera.aspect;
  brainCam.updateProjectionMatrix();
  const dpr=Math.min(devicePixelRatio,2);
  panelCv.width=panelCv.clientWidth*dpr;
  panelCv.height=panelCv.clientHeight*dpr;
  pctx.setTransform(dpr,0,0,dpr,0,0);
}
window.addEventListener('resize',resize);

makeSource('fruit',3,2);
makeSource('co2',-4,-3);
resize();

let last=performance.now(),acc=0,frameN=0;
function frame(now){
  requestAnimationFrame(frame);
  let dt=(now-last)/1000;last=now;
  if(dt>0.1)dt=0.1;
  acc+=dt;
  const step=1/60;
  let simmed=0;
  while(acc>=step&&simmed<3){tick(step);acc-=step;simmed++;}
  frameN++;
  updateBrainView();
  if(frameN%2===0)drawPanel();  // panel at 30fps: no visible difference, half the cost
  if(tab==='world'){
    controls.update();
    renderer.render(scene,camera);
  }else{
    controls2.update();
    renderer.render(brainScene,brainCam);
  }
}
// test hooks
window.__FM={
  stats:()=>({kcFrac:netKcFrac,dop:fly.dop,learned:learnedVal,innate:innateVal}),
  counts:{all:NP0,wired:SIMN,web:WEB_CAP},
  state:()=>fly.state,
  // headless tests: place a danger source next to the fly so AVOIDING is
  // exercised deterministically instead of depending on random drift.
  placeDanger(){const s=makeSource('danger',fly.pos.x+1.1,fly.pos.z);return s;},
  placeFood(){return makeSource('food',fly.pos.x+1.0,fly.pos.z);},
  sources,
  fly,
};
requestAnimationFrame(frame);
})();
