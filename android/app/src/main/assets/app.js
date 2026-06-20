import { FaceLandmarker, PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const fmt = (v, d = 1) => Number.isFinite(v) ? v.toFixed(d) : "—";
const rad = (deg) => deg * Math.PI / 180;
const deg = (r) => r * 180 / Math.PI;
const mean = (arr) => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
const median = (arr) => {
  const x = arr.filter(Number.isFinite).slice().sort((a,b)=>a-b);
  if (!x.length) return NaN;
  const m = Math.floor(x.length/2);
  return x.length % 2 ? x[m] : (x[m-1]+x[m])/2;
};
const normalCdf = (x) => {
  // Abramowitz-Stegun approximation, good enough for visual scoring.
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * a);
  const erf = 1 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*Math.exp(-a*a);
  return 0.5 * (1 + sign * erf);
};
const dist = (a,b) => Math.hypot(a.x-b.x, a.y-b.y);
const dot = (a,b) => a.x*b.x + a.y*b.y;
const normalize = (v) => {
  const l = Math.hypot(v.x,v.y) || 1;
  return {x:v.x/l,y:v.y/l};
};
const avgPoint = (pts) => ({x:mean(pts.map(p=>p.x)), y:mean(pts.map(p=>p.y)), z:mean(pts.map(p=>p.z ?? 0))});
const scoreBand = (value, lo, hi, softness = (hi-lo)*0.65 || 1) => {
  if (!Number.isFinite(value)) return 5;
  if (value >= lo && value <= hi) return 9.2 - Math.abs(value - (lo+hi)/2) / ((hi-lo)/2 || 1) * 0.35;
  const d = value < lo ? lo - value : value - hi;
  return clamp(9.0 - (d / softness) * 6.2, 0.6, 9.2);
};
const scoreTarget = (value, target, spread) => clamp(9.5 - Math.abs(value-target)/spread*4.2, 0.6, 9.6);
const weightedMean = (items) => {
  const valid = items.filter(x => Number.isFinite(x.value) && x.weight > 0);
  const w = valid.reduce((a,b)=>a+b.weight,0) || 1;
  return valid.reduce((a,b)=>a + b.value*b.weight,0)/w;
};

const IDX = {
  faceTop:10, glabella:9, noseBridge:168, noseBridge2:6, noseTip:1, noseBottom:2, chin:152,
  leftEyeOuter:33, leftEyeInner:133, rightEyeInner:362, rightEyeOuter:263,
  leftEyeTop:159, leftEyeBottom:145, rightEyeTop:386, rightEyeBottom:374,
  leftIris:468, rightIris:473,
  mouthLeft:61, mouthRight:291, upperLip:13, lowerLip:14, upperLip2:0, lowerLip2:17,
  leftBrowInner:107, leftBrowMid:105, leftBrowOuter:70, rightBrowInner:336, rightBrowMid:334, rightBrowOuter:300,
  leftCheek:234, rightCheek:454, leftCheekBone:127, rightCheekBone:356,
  leftJaw:172, rightJaw:397, leftJawLow:150, rightJawLow:379,
  leftTemple:162, rightTemple:389, leftNostril:98, rightNostril:327,
  leftLipOuter:78, rightLipOuter:308, leftMouthUpper:40, rightMouthUpper:270, leftMouthLower:88, rightMouthLower:318
};
const PAIRS = [
  [33,263],[133,362],[159,386],[145,374],[70,300],[105,334],[107,336],
  [61,291],[78,308],[40,270],[88,318],[234,454],[127,356],[172,397],[150,379],[162,389],[98,327]
];
const MESH_LINES = [
  [10,9],[9,168],[168,6],[6,1],[1,2],[2,13],[13,14],[14,152],
  [33,133],[133,362],[362,263],[70,105],[105,107],[336,334],[334,300],
  [61,13],[13,291],[291,14],[14,61],[234,127],[127,162],[454,356],[356,389],
  [234,172],[172,150],[150,152],[152,379],[379,397],[397,454],
  [98,1],[1,327],[98,2],[2,327]
];

const els = {
  frontInput: $("frontInput"), profileInput: $("profileInput"), bodyFrontInput: $("bodyFrontInput"), bodySideInput: $("bodySideInput"),
  analyzeBtn: $("analyzeBtn"), resetImagesBtn: $("resetImagesBtn"), fullResetBtn: $("fullResetBtn"),
  modeSelect: $("modeSelect"), strictnessSelect: $("strictnessSelect"), canvas: $("imageCanvas"), wrap: $("canvasWrap"), dropHint: $("dropHint"), poseBadge: $("poseBadge"),
  viewerSub: $("viewerSub"), mainScore: $("mainScore"), scoreRange: $("scoreRange"), qualityRing: $("qualityRing"), verdictText: $("verdictText"),
  percentileStat: $("percentileStat"), validityStat: $("validityStat"), poseStat: $("poseStat"), coverageStat: $("coverageStat"),
  gapList: $("gapList"), strengthList: $("strengthList"), auditGrid: $("auditGrid"), featureGrid: $("featureGrid"),
  curveCanvas: $("curveCanvas"), radarCanvas: $("radarCanvas"), percentileLabel: $("percentileLabel"), radarLabel: $("radarLabel"),
  metricSearch: $("metricSearch"), categoryFilter: $("categoryFilter"), playbookList: $("playbookList"),
  modelCanvas: $("modelCanvas"), modelNotes: $("modelNotes"), toast: $("toast"), installBtn: $("installBtn"), mobileAnalyzeBtn: $("mobileAnalyzeBtn"), mobileResetBtn: $("mobileResetBtn"),
  uploadSummary: $("uploadSummary"), frontPreview: $("frontPreview"), profilePreview: $("profilePreview"), bodyFrontPreview: $("bodyFrontPreview"), bodySidePreview: $("bodySidePreview"),
  appStatus: $("appStatus"), cameraModal: $("cameraModal"), cameraVideo: $("cameraVideo"), cameraCanvas: $("cameraCanvas"), cameraTitle: $("cameraTitle"), cameraHint: $("cameraHint"), cameraCloseBtn: $("cameraCloseBtn"), cameraCaptureBtn: $("cameraCaptureBtn"), cameraSwitchBtn: $("cameraSwitchBtn"), cameraFallbackBtn: $("cameraFallbackBtn")
};
const ctx = els.canvas.getContext("2d");
const curveCtx = els.curveCanvas.getContext("2d");
const radarCtx = els.radarCanvas.getContext("2d");
const modelCtx = els.modelCanvas.getContext("2d");

const state = {
  slots:{ front:null, profile:null, bodyFront:null, bodySide:null },
  faceLandmarker:null, poseLandmarker:null, ready:false, busy:false,
  overlay:"proportions", fit:null, analysis:null, deferredInstallPrompt:null,
  activeScreen:"home", camera:{stream:null, slot:null, facingMode:"user"}, autoAnalyzeTimer:null
};


function registerPwaSupport(){
  if('serviceWorker' in navigator){
    window.addEventListener('load', ()=> navigator.serviceWorker.register('service-worker.js').catch(()=>{}));
  }
  window.addEventListener('beforeinstallprompt', (e)=>{
    e.preventDefault();
    state.deferredInstallPrompt = e;
    if(els.installBtn) els.installBtn.classList.remove('hidden');
  });
  els.installBtn?.addEventListener('click', async ()=>{
    if(!state.deferredInstallPrompt) return;
    state.deferredInstallPrompt.prompt();
    try { await state.deferredInstallPrompt.userChoice; } catch(_) {}
    state.deferredInstallPrompt = null;
    els.installBtn.classList.add('hidden');
  });
}

function toast(msg, ms=2800){
  els.toast.textContent = msg; els.toast.classList.remove("hidden");
  clearTimeout(toast._t); toast._t = setTimeout(()=>els.toast.classList.add("hidden"), ms);
}

async function loadImageFile(file){
  return new Promise((resolve, reject)=>{
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve({file, url, img, name:file.name});
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Image failed to load")); };
    img.src = url;
  });
}

function inputForSlot(slot){ return ({front:els.frontInput, profile:els.profileInput, bodyFront:els.bodyFrontInput, bodySide:els.bodySideInput})[slot]; }
function previewForSlot(slot){ return ({front:els.frontPreview, profile:els.profilePreview, bodyFront:els.bodyFrontPreview, bodySide:els.bodySidePreview})[slot]; }
function slotLabel(slot){ return ({front:"Front face", profile:"Side profile", bodyFront:"Body front", bodySide:"Body side"})[slot] || "Image"; }
function setStatus(text){ if(els.appStatus) els.appStatus.textContent = text; }
function updateUploadSummary(){
  const count = Object.values(state.slots).filter(Boolean).length;
  if(els.uploadSummary) els.uploadSummary.textContent = `${count} / 4 loaded`;
  setStatus(state.busy ? "Analyzing" : state.analysis ? "Analyzed" : state.slots.front ? "Ready" : "Capture");
}
function updateSlotPreview(slot, data){
  const img = previewForSlot(slot);
  if(!img) return;
  if(data?.url){ img.src = data.url; img.style.display='block'; }
  else { img.removeAttribute('src'); img.style.display=''; }
}
function imageFromDataUrl(url, name="camera-capture.jpg"){
  return new Promise((resolve, reject)=>{
    const img = new Image();
    img.onload = () => resolve({file:null, url, img, name});
    img.onerror = () => reject(new Error("Captured image failed to load"));
    img.src = url;
  });
}
function fallbackFilePicker(slot, source="gallery"){
  const input = inputForSlot(slot); if(!input) return;
  input.setAttribute('accept','image/*');
  if(source === 'camera') input.setAttribute('capture', slot === 'bodyFront' || slot === 'bodySide' ? 'environment' : 'user');
  else input.removeAttribute('capture');
  input.click();
}
async function triggerSlotInput(slot, source){
  if(source === 'camera') return openCamera(slot);
  return fallbackFilePicker(slot, 'gallery');
}
function setSlot(slot, data){
  if (state.slots[slot]?.url && state.slots[slot].url.startsWith('blob:')) URL.revokeObjectURL(state.slots[slot].url);
  state.slots[slot] = data;
  document.querySelector(`.upload-card[data-slot="${slot}"]`)?.classList.toggle("loaded", !!data);
  updateSlotPreview(slot, data);
  updateUploadSummary();
  updateAnalyzeEnabled();
  clearAnalysis(false);
  if (slot === "front") drawViewer();
  if (state.slots.front) {
    clearTimeout(state.autoAnalyzeTimer);
    state.autoAnalyzeTimer = setTimeout(()=>{ if(state.slots.front && !state.busy) analyze(); }, 520);
  }
}
async function openCamera(slot){
  state.camera.slot = slot;
  state.camera.facingMode = (slot === 'bodyFront' || slot === 'bodySide') ? 'environment' : 'user';
  els.cameraTitle.textContent = `Capture ${slotLabel(slot)}`;
  els.cameraHint.textContent = slot === 'front' ? 'Neutral expression, eye-level camera, fill the face oval.' : 'Keep the phone steady and use clean light.';
  els.cameraModal.classList.remove('hidden');
  els.cameraModal.setAttribute('aria-hidden','false');
  await startCameraStream();
}
async function startCameraStream(){
  try{
    stopCameraStream(false);
    if(!navigator.mediaDevices?.getUserMedia) throw new Error('Browser camera API unavailable');
    const constraints = {video:{facingMode:state.camera.facingMode, width:{ideal:1280}, height:{ideal:1920}}, audio:false};
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.camera.stream = stream;
    els.cameraVideo.srcObject = stream;
    await els.cameraVideo.play().catch(()=>{});
  }catch(e){
    toast('In-app camera unavailable. Opening Android picker…', 3200);
    closeCamera(false);
    fallbackFilePicker(state.camera.slot || 'front', 'camera');
  }
}
function stopCameraStream(clear=true){
  if(state.camera.stream){
    for(const track of state.camera.stream.getTracks()) track.stop();
    state.camera.stream = null;
  }
  if(clear && els.cameraVideo) els.cameraVideo.srcObject = null;
}
function closeCamera(clearSlot=true){
  stopCameraStream(true);
  els.cameraModal.classList.add('hidden');
  els.cameraModal.setAttribute('aria-hidden','true');
  if(clearSlot) state.camera.slot = null;
}
async function captureCameraFrame(){
  const slot = state.camera.slot || 'front';
  const video = els.cameraVideo;
  if(!video.videoWidth || !video.videoHeight) return toast('Camera is not ready yet.');
  const canvas = els.cameraCanvas;
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  const c = canvas.getContext('2d');
  c.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg', .92);
  const data = await imageFromDataUrl(dataUrl, `${slot}-camera.jpg`);
  setSlot(slot, data);
  closeCamera();
  switchScreen(slot === 'front' ? 'home' : 'capture');
  toast(`${slotLabel(slot)} captured.`);
}
function switchCamera(){
  state.camera.facingMode = state.camera.facingMode === 'user' ? 'environment' : 'user';
  startCameraStream();
}
function switchScreen(name){
  state.activeScreen = name;
  document.querySelectorAll('.screen').forEach(s=>s.classList.toggle('active', s.dataset.screen === name));
  document.querySelectorAll('.bottom-tab').forEach(b=>b.classList.toggle('active', b.dataset.go === name));
  window.scrollTo({top:0, behavior:'smooth'});
  requestAnimationFrame(()=>{ drawViewer(); renderModel(); renderCurve(); renderRadar(); });
}
function updateAnalyzeEnabled(){
  const disabled = !state.slots.front || state.busy;
  els.analyzeBtn.disabled = disabled;
  if(els.mobileAnalyzeBtn) els.mobileAnalyzeBtn.disabled = disabled;
  if(els.mobileAnalyzeBtn) els.mobileAnalyzeBtn.textContent = state.busy ? 'Analyzing…' : state.analysis ? 'Re-analyze' : 'Analyze';
  setStatus(state.busy ? 'Analyzing' : state.analysis ? 'Analyzed' : state.slots.front ? 'Ready' : 'Capture');
}
async function handleFile(slot, input){
  const file = input.files?.[0]; if (!file) return;
  try { setSlot(slot, await loadImageFile(file)); toast(`${slotLabel(slot)} loaded.`); }
  catch(e){ toast(e.message || "Could not load image"); }
}

els.frontInput.addEventListener("change", e=>handleFile("front", e.target));
els.profileInput.addEventListener("change", e=>handleFile("profile", e.target));
els.bodyFrontInput.addEventListener("change", e=>handleFile("bodyFront", e.target));
els.bodySideInput.addEventListener("change", e=>handleFile("bodySide", e.target));
for (const btn of document.querySelectorAll('.upload-trigger')) btn.addEventListener('click', ()=> triggerSlotInput(btn.dataset.slot, btn.dataset.source));
for (const btn of document.querySelectorAll('[data-go]')) btn.addEventListener('click', ()=> switchScreen(btn.dataset.go));
els.cameraCloseBtn?.addEventListener('click', ()=>closeCamera());
els.cameraCaptureBtn?.addEventListener('click', captureCameraFrame);
els.cameraSwitchBtn?.addEventListener('click', switchCamera);
els.cameraFallbackBtn?.addEventListener('click', ()=>{ const slot = state.camera.slot || 'front'; closeCamera(); fallbackFilePicker(slot, 'gallery'); });
for (const btn of document.querySelectorAll(".tab")) btn.addEventListener("click", () => { document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active")); btn.classList.add("active"); state.overlay = btn.dataset.overlay; drawViewer(); });
els.modeSelect.addEventListener("change", ()=> state.analysis && analyze());
els.strictnessSelect.addEventListener("change", ()=> state.analysis && analyze());
els.analyzeBtn.addEventListener("click", analyze);
els.mobileAnalyzeBtn?.addEventListener("click", analyze);
els.resetImagesBtn.addEventListener("click", resetImages);
els.mobileResetBtn?.addEventListener("click", resetImages);
els.fullResetBtn.addEventListener("click", fullReset);
els.metricSearch.addEventListener("input", renderFeatureGrid);
els.categoryFilter.addEventListener("change", renderFeatureGrid);
window.addEventListener("resize", () => { drawViewer(); renderModel(); renderCurve(); renderRadar(); });
els.wrap.addEventListener("dragover", e=>{e.preventDefault(); els.wrap.classList.add("drag");});
els.wrap.addEventListener("drop", async e=>{ e.preventDefault(); const f=e.dataTransfer.files?.[0]; if(f) setSlot("front", await loadImageFile(f)); });

function resetImages(){
  for (const k of Object.keys(state.slots)) { if (state.slots[k]?.url && state.slots[k].url.startsWith('blob:')) URL.revokeObjectURL(state.slots[k].url); state.slots[k]=null; updateSlotPreview(k, null); }
  for (const input of [els.frontInput, els.profileInput, els.bodyFrontInput, els.bodySideInput]) input.value = "";
  document.querySelectorAll(".upload-card").forEach(c=>c.classList.remove("loaded","error"));
  updateUploadSummary();
  clearAnalysis(true); updateAnalyzeEnabled(); drawViewer(); toast("Images cleared.");
}
function fullReset(){
  resetImages(); state.overlay="proportions";
  els.modeSelect.value="balanced"; els.strictnessSelect.value="normal";
  document.querySelectorAll(".tab").forEach(b=>b.classList.toggle("active", b.dataset.overlay==="proportions"));
  toast("Full reset complete.");
}
function clearAnalysis(clearCanvas){
  state.analysis = null;
  els.mainScore.textContent = "—"; els.scoreRange.textContent = state.slots.front ? "Ready to analyze" : "Add a front photo to start";
  els.qualityRing.style.setProperty("--q", 0); els.qualityRing.querySelector("span").textContent = "—";
  els.verdictText.textContent = "Aestra analyzes proportions, symmetry, pose quality, and visible feature signals without pretending bad photos are truth.";
  els.percentileStat.textContent = "—"; els.validityStat.textContent = "—"; els.poseStat.textContent = "—";
  els.coverageStat.textContent = coverageText();
  els.gapList.className = "chip-list empty"; els.gapList.textContent = "Waiting for analysis.";
  els.strengthList.className = "chip-list empty"; els.strengthList.textContent = "Waiting for analysis.";
  els.auditGrid.innerHTML = ""; els.featureGrid.className = "feature-grid empty-state"; els.featureGrid.textContent = "Run analysis to populate the breakdown.";
  els.percentileLabel.textContent = "—"; els.radarLabel.textContent = "—"; els.poseBadge.classList.add("hidden");
  clearSmallCanvas(els.curveCanvas, curveCtx); clearSmallCanvas(els.radarCanvas, radarCtx); clearSmallCanvas(els.modelCanvas, modelCtx);
  els.playbookList.className="playbook-list empty"; els.playbookList.textContent="Analysis required.";
  els.modelNotes.className="note-list empty"; els.modelNotes.textContent="Analysis required.";
  if (clearCanvas) drawViewer();
}
function coverageText(){ return `${Object.values(state.slots).filter(Boolean).length}/4`; }
function clearSmallCanvas(canvas, c){
  const r = canvas.getBoundingClientRect(); const dpr = devicePixelRatio || 1; canvas.width = Math.max(2, r.width*dpr); canvas.height=Math.max(2, r.height*dpr); c.clearRect(0,0,canvas.width,canvas.height);
}

async function initVision(){
  if (state.ready) return;
  toast("Loading vision models…", 4500);
  const resolver = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
  state.faceLandmarker = await FaceLandmarker.createFromOptions(resolver, {
    baseOptions: { modelAssetPath:"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", delegate:"GPU" },
    outputFaceBlendshapes:true, outputFacialTransformationMatrixes:true, numFaces:1, runningMode:"IMAGE"
  });
  state.poseLandmarker = await PoseLandmarker.createFromOptions(resolver, {
    baseOptions: { modelAssetPath:"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task", delegate:"GPU" },
    runningMode:"IMAGE", numPoses:1
  });
  state.ready = true;
}

async function analyze(){
  if (!state.slots.front || state.busy) return;
  state.busy = true; setStatus("Analyzing"); updateAnalyzeEnabled(); els.analyzeBtn.textContent = "Analyzing…";
  try {
    await initVision();
    const front = state.slots.front.img;
    const faceRes = state.faceLandmarker.detect(front);
    if (!faceRes.faceLandmarks?.length) throw new Error("No face landmarks found. Use a clearer front-facing image.");
    const lm = faceRes.faceLandmarks[0];
    const blend = blendshapeMap(faceRes.faceBlendshapes?.[0]);
    const frontAnalysis = buildFaceAnalysis(front, lm, blend, faceRes.facialTransformationMatrixes?.[0]);

    let profileAnalysis = null;
    if (state.slots.profile) {
      try {
        const pr = state.faceLandmarker.detect(state.slots.profile.img);
        if (pr.faceLandmarks?.length) profileAnalysis = buildProfileAnalysis(state.slots.profile.img, pr.faceLandmarks[0]);
      } catch (_) {}
    }
    let bodyAnalysis = null;
    if (state.slots.bodyFront || state.slots.bodySide) {
      try { bodyAnalysis = await buildBodyAnalysis(); } catch (_) { bodyAnalysis = null; }
    }
    state.analysis = assembleAnalysis(frontAnalysis, profileAnalysis, bodyAnalysis);
    renderAll();
    switchScreen("home");
    toast("Analysis complete.");
  } catch(e){ toast(e.message || "Analysis failed", 5200); console.error(e); }
  finally { state.busy = false; els.analyzeBtn.textContent = "Analyze"; updateAnalyzeEnabled(); }
}
function blendshapeMap(item){
  const map = {}; (item?.categories || []).forEach(c => map[c.categoryName] = c.score); return map;
}
function pointAt(lm, idx, img){ const p = lm[idx]; return {x:p.x*img.naturalWidth, y:p.y*img.naturalHeight, z:(p.z||0)*img.naturalWidth, idx}; }
function safePoint(lm, idx, img){ return lm[idx] ? pointAt(lm, idx, img) : null; }
function centerOf(lm, idxs, img){ return avgPoint(idxs.map(i=>safePoint(lm,i,img)).filter(Boolean)); }
function buildFrame(img, lm){
  const L = centerOf(lm, [IDX.leftEyeOuter, IDX.leftEyeInner, IDX.leftEyeTop, IDX.leftEyeBottom], img);
  const R = centerOf(lm, [IDX.rightEyeOuter, IDX.rightEyeInner, IDX.rightEyeTop, IDX.rightEyeBottom], img);
  const eyeMid = avgPoint([L,R]);
  const xAxis = normalize({x:R.x-L.x, y:R.y-L.y});
  const yAxis = {x:-xAxis.y, y:xAxis.x};
  const nose = pointAt(lm, IDX.noseBridge2, img);
  const center = {x:lerp(eyeMid.x, nose.x, .38), y:lerp(eyeMid.y, nose.y, .38)};
  const local = (p) => ({x:dot({x:p.x-center.x,y:p.y-center.y}, xAxis), y:dot({x:p.x-center.x,y:p.y-center.y}, yAxis), z:p.z ?? 0});
  const fromLocal = (x,y) => ({x:center.x+xAxis.x*x+yAxis.x*y, y:center.y+xAxis.y*x+yAxis.y*y});
  const ipd = dist(L,R);
  const faceLeft = pointAt(lm, IDX.leftCheek, img), faceRight = pointAt(lm, IDX.rightCheek, img);
  const chin = pointAt(lm, IDX.chin, img), top = pointAt(lm, IDX.faceTop, img);
  const width = Math.abs(local(faceRight).x - local(faceLeft).x);
  const height = Math.abs(local(chin).y - local(top).y);
  const roll = deg(Math.atan2(R.y-L.y, R.x-L.x));
  return {L,R,eyeMid,center,xAxis,yAxis,local,fromLocal,ipd,width,height,roll,imgW:img.naturalWidth,imgH:img.naturalHeight};
}
function qualityFromImage(img, frame, lm, blend){
  const box = faceBox(lm, img);
  const faceArea = (box.w*box.h)/(img.naturalWidth*img.naturalHeight);
  const margin = Math.min(box.x, box.y, img.naturalWidth-(box.x+box.w), img.naturalHeight-(box.y+box.h)) / Math.max(box.w, box.h);
  const resolution = Math.min(1, Math.sqrt((img.naturalWidth*img.naturalHeight)/(900*900)));
  const faceSize = scoreBand(faceArea, .16, .54, .20) / 10;
  const crop = clamp((margin + .08) / .24, 0, 1);
  const mouthOpen = (blend.jawOpen || 0) + (blend.mouthFunnel || 0)*.25;
  const smile = (blend.mouthSmileLeft || 0) + (blend.mouthSmileRight || 0);
  const eyeSquint = (blend.eyeSquintLeft || 0) + (blend.eyeSquintRight || 0);
  const expressionPenalty = clamp(mouthOpen*1.05 + smile*.45 + eyeSquint*.28, 0, 1);
  const rollPenalty = clamp((Math.abs(frame.roll)-18)/30, 0, 1) * .18;
  const yaw = yawProxy(frame, lm, img);
  const yawPenalty = clamp((yaw-.14)/.38,0,1);
  const quality = clamp((resolution*.30 + faceSize*.28 + crop*.22 + (1-expressionPenalty)*.20) * (1-rollPenalty), 0, 1);
  const validity = clamp(quality - expressionPenalty*.22 - yawPenalty*.18, 0, 1);
  return {box, faceArea, margin, resolution, faceSize, crop, expressionPenalty, yaw, yawPenalty, rollPenalty, quality, validity};
}
function faceBox(lm, img){
  const xs = lm.map(p=>p.x*img.naturalWidth), ys = lm.map(p=>p.y*img.naturalHeight);
  const x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
  return {x:x0,y:y0,w:x1-x0,h:y1-y0};
}
function yawProxy(frame, lm, img){
  const lc = frame.local(pointAt(lm, IDX.leftCheek, img)), rc = frame.local(pointAt(lm, IDX.rightCheek, img));
  const nose = frame.local(pointAt(lm, IDX.noseTip, img));
  const cheekImbalance = Math.abs(Math.abs(lc.x) - Math.abs(rc.x)) / (frame.width || 1);
  const noseShift = Math.abs(nose.x) / (frame.width || 1);
  return clamp(cheekImbalance*.85 + noseShift*.95, 0, 1);
}
function buildFaceAnalysis(img, lm, blend){
  const frame = buildFrame(img, lm);
  const q = qualityFromImage(img, frame, lm, blend);
  const skin = skinAnalysis(img, lm, frame);
  const features = [];
  const add = (feature) => features.push(feature);
  const f = (idx) => pointAt(lm, idx, img);
  const L = (idx) => frame.local(f(idx));
  const ratio = (a,b) => a/(b || 1);
  const mode = els.modeSelect.value;
  const eyeLW = Math.abs(L(IDX.leftEyeInner).x - L(IDX.leftEyeOuter).x);
  const eyeRW = Math.abs(L(IDX.rightEyeOuter).x - L(IDX.rightEyeInner).x);
  const eyeW = (eyeLW + eyeRW)/2;
  const eyeOpenL = Math.abs(L(IDX.leftEyeBottom).y - L(IDX.leftEyeTop).y) / eyeLW;
  const eyeOpenR = Math.abs(L(IDX.rightEyeBottom).y - L(IDX.rightEyeTop).y) / eyeRW;
  const eyeOpen = (eyeOpenL+eyeOpenR)/2;
  const innerGap = Math.abs(L(IDX.rightEyeInner).x - L(IDX.leftEyeInner).x);
  const faceRatio = frame.width / (frame.height || 1);
  const jawWidth = Math.abs(L(IDX.rightJaw).x - L(IDX.leftJaw).x);
  const cheekWidth = Math.abs(L(IDX.rightCheek).x - L(IDX.leftCheek).x);
  const templeWidth = Math.abs(L(IDX.rightTemple).x - L(IDX.leftTemple).x);
  const jawCheek = jawWidth/(cheekWidth||1);
  const topY = L(IDX.faceTop).y, browY = frame.local(centerOf(lm,[IDX.glabella,IDX.leftBrowInner,IDX.rightBrowInner], img)).y;
  const noseBaseY = L(IDX.noseBottom).y, chinY = L(IDX.chin).y;
  const thirds = [browY-topY, noseBaseY-browY, chinY-noseBaseY].map(Math.abs);
  const thirdVar = Math.max(...thirds)/(Math.min(...thirds)||1);
  const noseW = Math.abs(L(IDX.rightNostril).x - L(IDX.leftNostril).x);
  const noseLen = Math.abs(noseBaseY - L(IDX.noseBridge).y);
  const mouthW = Math.abs(L(IDX.mouthRight).x - L(IDX.mouthLeft).x);
  const lipFull = Math.abs(L(IDX.lowerLip).y - L(IDX.upperLip).y) / (mouthW || 1);
  const mouthOpenGeo = Math.abs(L(IDX.lowerLip).y - L(IDX.upperLip).y)/(mouthW||1);
  const chinH = Math.abs(chinY - L(IDX.lowerLip2).y);
  const midface = Math.abs(L(IDX.upperLip).y - browY)/(frame.height||1);
  const lowerThird = Math.abs(chinY - noseBaseY)/(frame.height||1);
  const browEyeL = Math.abs(frame.local(centerOf(lm,[IDX.leftBrowMid,IDX.leftBrowOuter,IDX.leftBrowInner], img)).y - frame.local(centerOf(lm,[IDX.leftEyeTop,IDX.leftEyeOuter,IDX.leftEyeInner], img)).y) / eyeLW;
  const browEyeR = Math.abs(frame.local(centerOf(lm,[IDX.rightBrowMid,IDX.rightBrowOuter,IDX.rightBrowInner], img)).y - frame.local(centerOf(lm,[IDX.rightEyeTop,IDX.rightEyeOuter,IDX.rightEyeInner], img)).y) / eyeRW;
  const browEye = (browEyeL+browEyeR)/2;
  const browSlope = (Math.atan2(L(IDX.leftBrowOuter).y-L(IDX.leftBrowInner).y, L(IDX.leftBrowOuter).x-L(IDX.leftBrowInner).x) + Math.atan2(L(IDX.rightBrowInner).y-L(IDX.rightBrowOuter).y, L(IDX.rightBrowInner).x-L(IDX.rightBrowOuter).x))/2;
  const leftTilt = deg(Math.atan2(L(IDX.leftEyeInner).y-L(IDX.leftEyeOuter).y, L(IDX.leftEyeInner).x-L(IDX.leftEyeOuter).x));
  const rightTilt = deg(Math.atan2(L(IDX.rightEyeOuter).y-L(IDX.rightEyeInner).y, L(IDX.rightEyeOuter).x-L(IDX.rightEyeInner).x));
  const canthal = (leftTilt+rightTilt)/2;
  const symmetry = symmetryScore(frame, lm, img);
  const noseAlign = 10 - clamp((Math.abs(L(IDX.noseBridge).x)+Math.abs(L(IDX.noseTip).x)+Math.abs(L(IDX.noseBottom).x))/(frame.width||1)*42, 0, 9.4);
  const cheekbone = clamp(scoreBand(jawCheek, mode==="masculine"?.74:.62, mode==="masculine"?.93:.84, .22) + scoreBand(cheekWidth/(templeWidth||1), .96, 1.12, .18)*.15, 0, 9.7);
  const jawScore = mode === "feminine" ? scoreBand(jawCheek, .54, .78, .20) : mode === "masculine" ? scoreBand(jawCheek, .74, .94, .18) : scoreBand(jawCheek, .62, .87, .20);
  const faceShapeScore = mode === "masculine" ? scoreBand(faceRatio, .69, .82, .13) : mode === "feminine" ? scoreBand(faceRatio, .61, .75, .13) : scoreBand(faceRatio, .64, .79, .13);
  const eyeSpacingScore = scoreBand(innerGap/eyeW, .92, 1.16, .32);
  const eyeOpenScore = mode === "feminine" ? scoreBand(eyeOpen, .255, .42, .18) : scoreBand(eyeOpen, .20, .35, .15);
  const canthalScore = scoreBand(canthal, -1.5, 7.2, 8.5);
  const browEyeScore = mode === "masculine" ? scoreBand(browEye, .34, .63, .28) : scoreBand(browEye, .40, .78, .32);
  const thirdScore = clamp(10 - (thirdVar-1)*8.0, 1.2, 9.4);
  const foreheadRatio = thirds[0]/(frame.height || 1);
  const foreheadScore = scoreBand(foreheadRatio, .23, .36, .17);
  const noseWidthScore = scoreBand(noseW/(cheekWidth||1), .195, .285, .12);
  const noseLenScore = scoreBand(noseLen/(frame.height||1), .245, .37, .14);
  const noseOverall = weightedMean([{value:noseWidthScore,weight:1},{value:noseLenScore,weight:1},{value:noseAlign,weight:.8}]);
  const mouthWidthScore = scoreBand(mouthW/(frame.width||1), .33, .48, .15);
  const lipScore = mode === "feminine" ? scoreBand(lipFull, .105, .195, .09) : scoreBand(lipFull, .075, .165, .09);
  const mouthScore = weightedMean([{value:mouthWidthScore,weight:.8},{value:lipScore,weight:1},{value:scoreBand(mouthOpenGeo,.045,.12,.10),weight:.35}]);
  const chinScore = mode === "masculine" ? scoreBand(chinH/(frame.height||1), .115, .19, .10) : scoreBand(chinH/(frame.height||1), .09, .165, .09);
  const midfaceScore = scoreBand(midface, .315, .43, .11);
  const lowerThirdScore = scoreBand(lowerThird, .255, .39, .13);
  const underEyeScore = skin.underEyeScore;
  const neckScore = 5.8 + (mode==="masculine" ? .2 : 0);
  const structuralCore = weightedMean([
    {value:symmetry.score,weight:1.3},{value:faceShapeScore,weight:1.0},{value:thirdScore,weight:.9},{value:eyeSpacingScore,weight:.7},
    {value:midfaceScore,weight:.8},{value:noseOverall,weight:.8},{value:jawScore,weight:1.0},{value:chinScore,weight:.7}
  ]);
  const harmonyScore = weightedMean([
    {value:structuralCore, weight:1.2},{value:Math.min(...[symmetry.score,thirdScore,noseOverall,jawScore].filter(Number.isFinite))+1.4, weight:.7},
    {value:weightedMean([{value:skin.evennessScore,weight:.4},{value:underEyeScore,weight:.25},{value:q.validity*10,weight:.2}]), weight:.35}
  ]);

  add(metric("Overall harmony", harmonyScore, .88*q.validity+.08, "structure", `coherence ${fmt(structuralCore)}/10`, "How well the main facial regions work together instead of only scoring isolated features.", "Highest ROI usually comes from improving the lowest structural anchor, not chasing tiny details."));
  add(metric("Facial symmetry", symmetry.score, symmetry.conf, "structure", `mirror error ${fmt(symmetry.error*100,1)}% IPD`, "Roll/tilt-corrected left-right comparison. Head turn lowers confidence rather than faking asymmetry.", "Use a centered lens and neutral head position for a cleaner symmetry read."));
  add(metric("Face shape", faceShapeScore, .82-q.yawPenalty*.25, "structure", `width/height ${fmt(faceRatio,2)}`, "General outline balance: vertical length, cheek width, jaw width, and compactness.", "Haircut, facial hair, leanness and camera distance change perceived outline strongly."));
  add(metric("Facial thirds", thirdScore, .62-q.yawPenalty*.14, "structure", `third spread ${fmt(thirdVar,2)}×`, "Upper, mid and lower third balance. Upper third is partially estimated because face meshes have sparse forehead coverage.", "A clean hairline-visible image improves this metric; don't overread it from covered foreheads."));
  add(metric("Forehead", foreheadScore, .55, "structure", `upper third ${fmt(foreheadRatio*100,1)}%`, "Estimated vertical forehead proportion from sparse upper-face landmarks.", "Fringe/hairline makes this uncertain; styling changes perception more than anatomy here."));
  add(metric("Brows", scoreBand(deg(browSlope), -5, 10, 12), .77, "eyes", `brow slope ${fmt(deg(browSlope),1)}°`, "Brow direction and placement relative to the eye area.", "Grooming and density can sharpen this area quickly."));
  add(metric("Brow-eye area", browEyeScore, .78-q.yawPenalty*.15, "eyes", `gap ratio ${fmt(browEye,2)}`, "How compact or open the brow-to-eye region is.", "Brows, eyelid exposure, lighting and lens height influence this a lot."));
  add(metric("Eyes", weightedMean([{value:eyeSpacingScore,weight:.6},{value:eyeOpenScore,weight:.7},{value:canthalScore,weight:.7},{value:browEyeScore,weight:.5}]), .86-q.yawPenalty*.18, "eyes", `open ${fmt(eyeOpen,2)} · tilt ${fmt(canthal,1)}°`, "Combined eye area score: spacing, openness, tilt and support from brows.", "For photos, camera height, relaxed eyelids and avoiding wide-angle distortion matter immediately."));
  add(metric("Eye spacing", eyeSpacingScore, .90-q.yawPenalty*.12, "eyes", `inner gap / eye ${fmt(innerGap/eyeW,2)}`, "Distance between eyes relative to eye width.", "Mostly structural; camera angle can still distort it."));
  add(metric("Eye openness", eyeOpenScore, .78, "eyes", `opening / width ${fmt(eyeOpen,2)}`, "Vertical eye aperture relative to eye width.", "Expression matters; forced wide eyes can hurt validity even if the number rises."));
  add(metric("Canthal tilt", canthalScore, .76, "eyes", `${fmt(canthal,1)}°`, "Tilt of the eye axis after roll correction.", "Do not judge this from a tilted photo unless the overlay axis lines up correctly."));
  add(metric("Under-eyes", underEyeScore, skin.conf, "skin-photo", `contrast ${fmt(skin.underEyeContrast,2)}`, "Brightness/texture proxy under the eyes, sampled locally from the image.", "Sleep, lighting angle, hydration, concealer, and softer overhead shadows help in photos."));
  add(metric("Cheekbones", cheekbone, .70-q.yawPenalty*.2, "structure", `jaw/cheek ${fmt(jawCheek,2)}`, "Cheek-to-jaw transition and perceived midface structure from front view.", "Leanness and side lighting create more visible cheek definition."));
  add(metric("Midface", midfaceScore, .82-q.yawPenalty*.12, "structure", `ratio ${fmt(midface,2)}`, "Vertical midface compactness relative to total facial height.", "Mostly structural; lens distance can fake midface length."));
  add(metric("Nose overall", noseOverall, .80-q.yawPenalty*.2, "nose-mouth", `W ${fmt(noseW/cheekWidth,2)} · L ${fmt(noseLen/frame.height,2)}`, "Combined width, length and midline alignment from front view.", "Side profile is needed for a serious projection read."));
  add(metric("Nose width", noseWidthScore, .84-q.yawPenalty*.15, "nose-mouth", `nostril/face ${fmt(noseW/cheekWidth,2)}`, "Nostril/base width relative to cheek width.", "Lighting and wide-angle selfies can exaggerate this."));
  add(metric("Nose length", noseLenScore, .78-q.yawPenalty*.12, "nose-mouth", `length/height ${fmt(noseLen/frame.height,2)}`, "Bridge-to-base length relative to face height.", "Camera height changes perceived length; side profile refines this."));
  add(metric("Nose alignment", noseAlign, .80-q.yawPenalty*.25, "nose-mouth", `midline drift ${fmt(Math.abs(L(IDX.noseTip).x)/frame.width*100,1)}%`, "Bridge, tip and base relation to the face midline.", "Head turn can look like deviation; pose usability matters here."));
  add(metric("Lips", lipScore, .78, "nose-mouth", `fullness ${fmt(lipFull,2)}`, "Lip vertical fullness relative to mouth width.", "Hydration, relaxed mouth posture and lighting improve the read."));
  add(metric("Mouth width", mouthWidthScore, .82-q.yawPenalty*.10, "nose-mouth", `mouth/face ${fmt(mouthW/frame.width,2)}`, "Mouth width relative to facial width.", "Expression stretches this; neutral mouth is needed for ranking."));
  add(metric("Jawline", jawScore, .80-q.yawPenalty*.18, "lower-face", `jaw/cheek ${fmt(jawCheek,2)}`, "Lower-face width and jaw support relative to cheek width.", "Body fat, beard styling and lighting are the practical levers."));
  add(metric("Mandibular angle", scoreBand(Math.abs(L(IDX.leftJaw).y - L(IDX.chin).y)/(frame.height||1), .18, .31, .14), .65, "lower-face", `jaw drop ${fmt(Math.abs(L(IDX.leftJaw).y - L(IDX.chin).y)/frame.height,2)}`, "Approximate jaw angle/vertical drop from frontal landmarks.", "Side profile improves this; front view alone is only medium confidence."));
  add(metric("Chin", chinScore, .76-q.yawPenalty*.12, "lower-face", `chin height ${fmt(chinH/frame.height,2)}`, "Chin height and lower-face support from frontal geometry.", "Side profile is required for projection; front view mainly sees height/width balance."));
  add(metric("Neck", neckScore, .34, "profile-body", "front estimate only", "Front face alone cannot measure neck well, so this stays low-confidence unless body photos are added.", "Use body front/side photos for actual neck/frame/posture analysis."));
  add(metric("Skin tone evenness", skin.evennessScore, skin.conf, "skin-photo", `patch Δ ${fmt(skin.patchDelta,1)}`, "Color consistency across cheeks, under-eyes, nose and lower face patches.", "Even lighting matters more than filters; avoid hard overhead light."));
  add(metric("Skin texture", skin.textureScore, skin.conf, "skin-photo", `local variance ${fmt(skin.texture,1)}`, "Local micro-contrast proxy from face patches, not a medical skin assessment.", "Clean lighting, sharpness control and basic skincare improve photo texture."));
  add(metric("Skin undertone", skin.undertoneScore, .44, "skin-photo", skin.undertone, "Very rough warm/cool/neutral color read from visible face patches.", "Use daylight photos for this; artificial light can fully change the result."));
  add(metric("Photo validity", q.validity*10, .96, "skin-photo", `quality ${fmt(q.quality*100,0)}%`, "Whether the image is suitable for attractiveness scoring: crop, resolution, expression, pose and face size.", "Neutral expression, non-wide lens, camera at eye height, visible hairline, centered head."));
  add(metric("Head pose", (1-q.yawPenalty*.8-q.rollPenalty*.5)*10, .95, "skin-photo", `roll ${fmt(frame.roll,1)}° · yaw proxy ${fmt(q.yaw,2)}`, "Pose usability. Roll is corrected visually; yaw/head turn reduces confidence.", "Take a centered front shot; tilted head is okay if landmarks are clean, but head turn is not."));

  return {img,lm,blend,frame,quality:q,features,skin,symmetry,raw:{structuralCore,harmonyScore,faceRatio,jawCheek,thirdVar,canthal,eyeOpen,innerGapEye:innerGap/eyeW}};
}
function metric(name, score, conf, category, measure, explanation, suggestion){
  return {name, score:clamp(score,0,10), conf:clamp(conf,0,1), category, measure, explanation, suggestion};
}
function symmetryScore(frame, lm, img){
  const errs = [];
  for (const [a,b] of PAIRS){
    if (!lm[a] || !lm[b]) continue;
    const la = frame.local(pointAt(lm,a,img));
    const lb = frame.local(pointAt(lm,b,img));
    const dx = Math.abs(Math.abs(la.x)-Math.abs(lb.x));
    const dy = Math.abs(la.y-lb.y)*.72;
    errs.push(Math.hypot(dx,dy)/(frame.ipd||1));
  }
  const error = median(errs);
  const yaw = yawProxy(frame,lm,img);
  const score = clamp(9.6 - error*34 - clamp(yaw-.18,0,1)*2.1, 1.0, 9.6);
  const conf = clamp(.94 - yaw*.85, .35, .95);
  return {score,error,conf,pairs:PAIRS};
}
function samplePixelStats(img, points, frame){
  const off = document.createElement("canvas");
  const maxW = 720; const scale = Math.min(1, maxW/img.naturalWidth);
  off.width = Math.round(img.naturalWidth*scale); off.height = Math.round(img.naturalHeight*scale);
  const c = off.getContext("2d", {willReadFrequently:true}); c.drawImage(img,0,0,off.width,off.height);
  const patches = [];
  for (const p of points){
    const x = clamp(Math.round(p.x*scale), 0, off.width-1), y = clamp(Math.round(p.y*scale),0,off.height-1);
    const r = Math.max(5, Math.round(frame.ipd*scale*.055));
    const data = c.getImageData(clamp(x-r,0,off.width-1), clamp(y-r,0,off.height-1), Math.min(r*2, off.width), Math.min(r*2, off.height)).data;
    const vals=[]; let R=0,G=0,B=0,n=0;
    for(let i=0;i<data.length;i+=4){
      const rr=data[i], gg=data[i+1], bb=data[i+2];
      const lum=.2126*rr+.7152*gg+.0722*bb;
      vals.push(lum); R+=rr; G+=gg; B+=bb; n++;
    }
    const m=mean(vals), v=mean(vals.map(v=>(v-m)*(v-m)));
    patches.push({lum:m,var:v,r:R/n,g:G/n,b:B/n});
  }
  return patches;
}
function skinAnalysis(img, lm, frame){
  const pts = [
    frame.fromLocal(-frame.ipd*.55, frame.ipd*.72), frame.fromLocal(frame.ipd*.55, frame.ipd*.72),
    frame.fromLocal(-frame.ipd*.38, frame.ipd*1.03), frame.fromLocal(frame.ipd*.38, frame.ipd*1.03),
    frame.fromLocal(0, frame.ipd*.95), frame.fromLocal(0, frame.ipd*1.45)
  ];
  let patches=[];
  try { patches = samplePixelStats(img, pts, frame); } catch(e) { patches=[]; }
  if (!patches.length) return {evennessScore:6,textureScore:6,underEyeScore:6,undertoneScore:6,underEyeContrast:0,patchDelta:0,texture:0,undertone:"unavailable",conf:.25};
  const lums = patches.map(p=>p.lum), vars=patches.map(p=>p.var);
  const patchDelta = Math.max(...lums)-Math.min(...lums);
  const texture = mean(vars);
  const evennessScore = scoreBand(patchDelta, 0, 34, 42);
  const textureScore = scoreBand(texture, 18, 360, 500);
  const underEyeContrast = Math.abs(mean([patches[0].lum,patches[1].lum])-mean([patches[2].lum,patches[3].lum]))/25;
  const underEyeScore = clamp(9.2 - underEyeContrast*2.4 - Math.sqrt(mean([patches[0].var,patches[1].var]))/22, 2.0, 9.4);
  const avg = avgPoint(patches.map(p=>({x:p.r,y:p.g,z:p.b})));
  const warmth = (avg.x - avg.z) / 255;
  const undertone = warmth > .055 ? "warm leaning" : warmth < .015 ? "cool/neutral leaning" : "neutral-warm";
  const undertoneScore = 7.0;
  return {evennessScore,textureScore,underEyeScore,undertoneScore,underEyeContrast,patchDelta,texture,undertone,conf:.68};
}
function buildProfileAnalysis(img, lm){
  const frame = buildFrame(img,lm);
  const nose = frame.local(pointAt(lm,IDX.noseTip,img));
  const chin = frame.local(pointAt(lm,IDX.chin,img));
  const upper = frame.local(pointAt(lm,IDX.upperLip,img));
  const lower = frame.local(pointAt(lm,IDX.lowerLip,img));
  const projectionBalance = scoreBand(Math.abs(nose.x-chin.x)/(frame.ipd||1), .18, .58, .52);
  const lipLine = Math.abs((upper.x+lower.x)/2 - lerp(nose.x,chin.x,.52))/(frame.ipd||1);
  const eLineScore = scoreBand(lipLine, .02, .22, .28);
  return {features:[
    metric("Side-profile projection", projectionBalance, .52, "profile-body", `nose/chin offset ${fmt(Math.abs(nose.x-chin.x)/frame.ipd,2)}`, "Profile projection estimate from side image landmarks.", "Use a true 90° side profile for this; partial turns are unreliable."),
    metric("E-line / lip projection", eLineScore, .50, "profile-body", `lip-line offset ${fmt(lipLine,2)}`, "Rough lip relation to nose-chin line.", "Only meaningful with a clean side profile.")
  ]};
}
async function buildBodyAnalysis(){
  const out=[];
  const img = state.slots.bodyFront?.img || state.slots.bodySide?.img;
  if (!img || !state.poseLandmarker) return null;
  const res = state.poseLandmarker.detect(img);
  if (!res.landmarks?.length) return null;
  const lm = res.landmarks[0], W=img.naturalWidth,H=img.naturalHeight;
  const P = i => ({x:lm[i].x*W, y:lm[i].y*H, z:(lm[i].z||0)*W, v:lm[i].visibility ?? .5});
  const ls=P(11), rs=P(12), lh=P(23), rh=P(24), la=P(27), ra=P(28), nk=P(0);
  const shW=dist(ls,rs), hipW=dist(lh,rh), torso=dist(avgPoint([ls,rs]),avgPoint([lh,rh])), leg=mean([dist(lh,la),dist(rh,ra)]);
  out.push(metric("Shoulder-to-hip ratio", scoreBand(shW/(hipW||1), 1.12, 1.65, .55), .66, "profile-body", `ratio ${fmt(shW/hipW,2)}`, "Frame width balance from pose landmarks.", "Posture, lens distance, clothing and stance influence this."));
  out.push(metric("Leg-to-torso ratio", scoreBand(leg/(torso||1), 1.25, 1.95, .75), .62, "profile-body", `ratio ${fmt(leg/torso,2)}`, "Approximate vertical proportion from full-body landmarks.", "Requires full body visible and camera not tilted."));
  out.push(metric("Posture", scoreBand(Math.abs(nk.x - avgPoint([lh,rh]).x)/(shW||1), 0, .20, .30), .58, "profile-body", "head/hip alignment", "Simple frontal posture/symmetry estimate.", "Stand straight, camera chest height, feet visible."));
  return {features:out};
}
function assembleAnalysis(front, profile, body){
  const features = front.features.slice();
  if (profile?.features) features.push(...profile.features); else features.push(metric("Side profile", NaN, 0, "profile-body", "unavailable", "No side-profile image was analyzed, so E-line/projection should not be scored.", "Add true side profile for projection metrics."));
  if (body?.features) features.push(...body.features); else features.push(metric("Full-body proportions", NaN, 0, "profile-body", "unavailable", "No usable body photo was analyzed, so posture/frame are excluded from final score.", "Add full-body front and side photos for frame analysis."));
  const final = scoreModel(features, front.quality, front.raw);
  return {front, profile, body, features, final};
}
function scoreModel(features, q, raw){
  const included = features.filter(m => Number.isFinite(m.score) && m.conf > .22 && m.name !== "Photo validity" && m.name !== "Head pose" && !["Skin undertone","Neck"].includes(m.name));
  const weightMap = {
    "Overall harmony":1.6,"Facial symmetry":1.25,"Face shape":1.0,"Facial thirds":.85,"Eyes":1.25,"Brow-eye area":.65,"Cheekbones":.95,"Midface":.80,"Nose overall":.90,"Lips":.62,"Mouth width":.45,"Jawline":1.0,"Chin":.72,"Skin tone evenness":.35,"Skin texture":.28,"Under-eyes":.32,
    "Side-profile projection":.65,"E-line / lip projection":.55,"Shoulder-to-hip ratio":.30,"Leg-to-torso ratio":.18,"Posture":.16
  };
  const values = included.map(m => ({value:m.score, weight:(weightMap[m.name] ?? .35) * (.55 + .45*m.conf)}));
  const robust = weightedMean(values);
  const sorted = included.map(m=>m.score).filter(Number.isFinite).sort((a,b)=>a-b);
  const lowAnchors = sorted.slice(0, Math.min(5, sorted.length));
  const lowMean = mean(lowAnchors);
  const highMean = mean(sorted.slice(-Math.min(7, sorted.length)));
  const coherence = clamp((highMean - Math.max(0, 7.3-lowMean)*.52 + raw.harmonyScore*.5)/1.5, 0, 10);
  const bottleneckDrag = clamp((6.2 - lowMean) * .18, 0, .65);
  const captureDrag = clamp(q.expressionPenalty*.78 + (1-q.crop)*.26 + Math.max(0,.32-q.faceArea)*.35, 0, .85);
  const poseDrag = clamp(q.yawPenalty*.58 + q.rollPenalty*.22, 0, .65);
  const strictness = els.strictnessSelect.value;
  const strictShift = strictness === "strict" ? -.18 : strictness === "generous" ? .16 : 0;
  const rawCore = robust*.68 + raw.structuralCore*.22 + coherence*.10;
  // Distribution model: transform core quality into latent attractiveness z.
  // 5.5 is distribution center. 7.0 is notably above average. 8.4+ with coherence can reach 9+.
  let z = (rawCore - 5.55) / 1.05;
  if (rawCore > 8.05 && lowMean > 6.55 && q.validity > .66) z += (rawCore-8.05)*.36 + clamp((coherence-8.0)*.08,0,.18);
  z += strictShift;
  z -= bottleneckDrag + captureDrag + poseDrag;
  const percentile = clamp(normalCdf(z), .001, .999);
  const score = clamp(1 + 9*percentile, 1, 10);
  const confidence = clamp(q.validity*.58 + mean(included.map(m=>m.conf))*.38 + (included.length>18?.04:0), .18, .96);
  const range = lerp(.95,.25,confidence) + captureDrag*.45 + poseDrag*.35;
  return {score, lo:clamp(score-range,1,10), hi:clamp(score+range,1,10), percentile, z, confidence, rawCore, robust, lowMean, highMean, coherence, bottleneckDrag, captureDrag, poseDrag};
}

function renderAll(){ drawViewer(); renderVerdict(); renderLists(); renderAudit(); renderFeatureGrid(); renderCurve(); renderRadar(); renderPlaybook(); renderModel(); }
function fitCanvas(canvas, img){
  const wrap = canvas.parentElement; const dpr = devicePixelRatio || 1;
  const rect = wrap.getBoundingClientRect(); canvas.width = Math.max(2, Math.floor(rect.width*dpr)); canvas.height = Math.max(2, Math.floor(rect.height*dpr));
  canvas.style.width = `${rect.width}px`; canvas.style.height = `${rect.height}px`;
  const scale = Math.min(canvas.width/img.naturalWidth, canvas.height/img.naturalHeight);
  const w = img.naturalWidth*scale, h = img.naturalHeight*scale;
  return {x:(canvas.width-w)/2,y:(canvas.height-h)/2,w,h,scale,dpr,cw:canvas.width,ch:canvas.height};
}
function drawViewer(){
  const slot = state.slots.front;
  if (!slot){
    const r = els.wrap.getBoundingClientRect(), dpr = devicePixelRatio || 1; els.canvas.width=r.width*dpr; els.canvas.height=r.height*dpr; ctx.clearRect(0,0,els.canvas.width,els.canvas.height); els.dropHint.classList.remove("hidden"); return;
  }
  const img = slot.img; const fit = fitCanvas(els.canvas,img); state.fit = fit;
  ctx.clearRect(0,0,fit.cw,fit.ch); ctx.fillStyle="#050609"; ctx.fillRect(0,0,fit.cw,fit.ch); ctx.drawImage(img, fit.x, fit.y, fit.w, fit.h);
  els.dropHint.classList.add("hidden");
  if (state.analysis) drawOverlay(ctx, fit, state.analysis.front, state.overlay);
}
function toC(p, fit){ return {x:fit.x + p.x*fit.scale, y:fit.y + p.y*fit.scale}; }
function fromLocalC(frame, fit, x, y){ return toC(frame.fromLocal(x,y), fit); }
function clampLabelPos(p, fit, pad=8){ return {x:clamp(p.x, fit.x+pad, fit.x+fit.w-pad), y:clamp(p.y, fit.y+pad, fit.y+fit.h-pad)}; }
function drawOverlay(c, fit, analysis, overlay){
  const {frame,lm,img,features,quality} = analysis;
  els.poseBadge.classList.remove("hidden");
  els.poseBadge.textContent = `roll ${fmt(frame.roll,1)}° corrected · yaw ${quality.yaw < .14 ? "low" : quality.yaw < .28 ? "medium" : "high"}`;
  c.save(); c.lineCap="round"; c.lineJoin="round"; c.font=`${12*(devicePixelRatio||1)}px Inter, system-ui`; c.textBaseline="middle";
  if (overlay === "clean") { c.restore(); return; }
  if (overlay === "mesh") { drawMesh(c, fit, analysis); c.restore(); return; }
  if (overlay === "heat") { drawGapHeat(c, fit, analysis); c.restore(); return; }
  if (overlay === "symmetry") { drawSymmetry(c, fit, analysis); c.restore(); return; }
  drawProportions(c, fit, analysis);
  c.restore();
}
function drawLine(c,a,b,color="#e7c572",w=2,alpha=1){ c.save(); c.globalAlpha=alpha; c.strokeStyle=color; c.lineWidth=w*(devicePixelRatio||1); c.beginPath(); c.moveTo(a.x,a.y); c.lineTo(b.x,b.y); c.stroke(); c.restore(); }
function drawDot(c,p,color="#e7c572",r=3){ c.save(); c.fillStyle=color; c.beginPath(); c.arc(p.x,p.y,r*(devicePixelRatio||1),0,Math.PI*2); c.fill(); c.restore(); }
function drawLabel(c, text, p, color="#dce6f6"){
  const dpr = devicePixelRatio || 1; c.save(); c.font=`${11*dpr}px Inter, system-ui`; const pad=5*dpr; const w=c.measureText(text).width+pad*2, h=22*dpr; p={x:clamp(p.x, 6*dpr, c.canvas.width-w-6*dpr), y:clamp(p.y, 13*dpr, c.canvas.height-13*dpr)}; c.fillStyle="rgba(5,7,11,.78)"; roundRect(c,p.x,p.y-h/2,w,h,9*dpr); c.fill(); c.strokeStyle="rgba(255,255,255,.18)"; c.stroke(); c.fillStyle=color; c.fillText(text,p.x+pad,p.y); c.restore();
}
function drawProportions(c, fit, a){
  const f=a.frame, ipd=f.ipd;
  const yTop=f.local(pointAt(a.lm,IDX.faceTop,a.img)).y, yBrow=f.local(centerOf(a.lm,[IDX.glabella,IDX.leftBrowInner,IDX.rightBrowInner],a.img)).y, yNose=f.local(pointAt(a.lm,IDX.noseBottom,a.img)).y, yChin=f.local(pointAt(a.lm,IDX.chin,a.img)).y;
  const xL=-f.width*.56, xR=f.width*.56;
  const rows = [[yTop,"hairline est"],[yBrow,"brow/glabella"],[yNose,"nose base"],[yChin,"chin"]];
  rows.forEach(([y,label],i)=>{
    const A=fromLocalC(f,fit,xL,y), B=fromLocalC(f,fit,xR,y); drawLine(c,A,B,i===0?"#7ed7ff":"#e7c572",2,.92); drawLabel(c,label,clampLabelPos(B,fit),"#edf4ff");
  });
  drawLine(c, fromLocalC(f,fit,0,yTop-ipd*.25), fromLocalC(f,fit,0,yChin+ipd*.18), "#ffffff", 1.3, .72);
  drawLabel(c,"face-local axis", clampLabelPos(fromLocalC(f,fit,ipd*.12,yTop-ipd*.18),fit), "#cfe8ff");
  const eyeL=toC(pointAt(a.lm,IDX.leftEyeOuter,a.img),fit), eyeLi=toC(pointAt(a.lm,IDX.leftEyeInner,a.img),fit), eyeRi=toC(pointAt(a.lm,IDX.rightEyeInner,a.img),fit), eyeR=toC(pointAt(a.lm,IDX.rightEyeOuter,a.img),fit);
  drawLine(c,eyeL,eyeLi,"#81d4fa",2,.9); drawLine(c,eyeRi,eyeR,"#81d4fa",2,.9);
  drawLine(c,toC(pointAt(a.lm,IDX.leftNostril,a.img),fit),toC(pointAt(a.lm,IDX.rightNostril,a.img),fit),"#ffcf5a",2,.9);
  drawLine(c,toC(pointAt(a.lm,IDX.mouthLeft,a.img),fit),toC(pointAt(a.lm,IDX.mouthRight,a.img),fit),"#c59cff",2,.9);
  drawLabel(c,`roll corrected ${fmt(f.roll,1)}°`,clampLabelPos(fromLocalC(f,fit,-ipd*.95,yTop-ipd*.28),fit),"#f6dd94");
}
function drawSymmetry(c, fit, a){
  const f=a.frame, ipd=f.ipd; const yTop=f.local(pointAt(a.lm,IDX.faceTop,a.img)).y-ipd*.35, yChin=f.local(pointAt(a.lm,IDX.chin,a.img)).y+ipd*.25;
  drawLine(c, fromLocalC(f,fit,0,yTop), fromLocalC(f,fit,0,yChin), "#ffffff", 2.2, .82);
  drawLabel(c,`symmetry ${fmt(a.symmetry.score)}/10`, clampLabelPos(fromLocalC(f,fit,ipd*.15,yTop),fit), "#e9f2ff");
  for (const [li,ri] of PAIRS){
    const lp = f.local(pointAt(a.lm,li,a.img)); const rp = f.local(pointAt(a.lm,ri,a.img));
    const lmC = fromLocalC(f,fit,lp.x,lp.y); const mirrorC = fromLocalC(f,fit,-rp.x,rp.y); const realRC = fromLocalC(f,fit,rp.x,rp.y);
    drawDot(c,lmC,"#81d4fa",2.2); drawDot(c,realRC,"#e7c572",2.2); drawLine(c,lmC,mirrorC,"#ff6d7a",1,.42);
  }
}
function drawGapHeat(c, fit, a){
  const high = a.features.filter(m=>Number.isFinite(m.score) && m.score<6.2).sort((x,y)=>x.score-y.score).slice(0,8);
  const map = {
    "Facial thirds":[0,-.35],"Forehead":[0,-.75],"Eyes":[0,-.05],"Brow-eye area":[0,-.17],"Under-eyes":[0,.16],"Cheekbones":[0,.52],"Nose overall":[0,.44],"Nose width":[0,.45],"Nose length":[0,.23],"Lips":[0,1.0],"Mouth width":[0,.95],"Jawline":[0,1.35],"Chin":[0,1.72],"Skin texture":[-.45,.65],"Skin tone evenness":[.45,.65]
  };
  const dpr=devicePixelRatio||1;
  for (const m of high){
    const loc = map[m.name] || [0,.5]; const p = fromLocalC(a.frame,fit,loc[0]*a.frame.ipd,loc[1]*a.frame.ipd);
    const r = lerp(46,76,(6.2-m.score)/6.2)*dpr;
    const g = c.createRadialGradient(p.x,p.y,0,p.x,p.y,r); g.addColorStop(0,`rgba(255,109,122,${.42 + (6-m.score)*.045})`); g.addColorStop(1,"rgba(255,109,122,0)");
    c.fillStyle=g; c.beginPath(); c.arc(p.x,p.y,r,0,Math.PI*2); c.fill(); drawLabel(c,`${m.name} ${fmt(m.score)}`, clampLabelPos({x:p.x+r*.28,y:p.y-r*.15},fit), "#ffd1d6");
  }
  if (!high.length) drawLabel(c,"no major visual bottleneck detected", {x:fit.x+18*dpr,y:fit.y+28*dpr}, "#b8ffd6");
}
function drawMesh(c, fit, a){
  c.save();
  for (const [i,j] of MESH_LINES){
    const pi=toC(pointAt(a.lm,i,a.img),fit), pj=toC(pointAt(a.lm,j,a.img),fit);
    drawLine(c,pi,pj,"#81d4fa",1.4,.65);
  }
  [10,9,168,1,2,152,33,133,362,263,234,454,172,397,61,291,13,14].forEach(i=>drawDot(c,toC(pointAt(a.lm,i,a.img),fit),"#e7c572",2.3));
  c.restore();
}
function roundRect(c,x,y,w,h,r){ c.beginPath(); c.moveTo(x+r,y); c.arcTo(x+w,y,x+w,y+h,r); c.arcTo(x+w,y+h,x,y+h,r); c.arcTo(x,y+h,x,y,r); c.arcTo(x,y,x+w,y,r); c.closePath(); }

function renderVerdict(){
  const A=state.analysis, f=A.final, q=A.front.quality;
  els.mainScore.textContent = fmt(f.score,1);
  els.scoreRange.textContent = `likely ${fmt(f.lo,1)}–${fmt(f.hi,1)} · ${els.strictnessSelect.value} · ${els.modeSelect.value}`;
  els.qualityRing.style.setProperty("--q", Math.round(f.confidence*100)); els.qualityRing.querySelector("span").textContent = `${Math.round(f.confidence*100)}%`;
  els.percentileStat.textContent = `${Math.round(f.percentile*1000)/10}%`;
  els.validityStat.textContent = q.validity > .78 ? "clean" : q.validity > .56 ? "usable" : "distorted";
  els.poseStat.textContent = q.yaw < .14 ? "front/clean" : q.yaw < .28 ? "some turn" : "head turn";
  els.coverageStat.textContent = coverageText();
  const caveat = q.validity < .6 ? " This photo is not a clean rating reference, so the range is wider." : "";
  const high = f.score >= 8.8 ? "High-end coherent result: strong features line up without obvious structural anchors collapsing." : f.score >= 7.4 ? "Clearly above-average read, but at least one bottleneck is still capping the upper range." : f.score >= 5.2 ? "Middle-distribution read: some attractive anchors, but the big gaps matter more than small positives." : "Below-middle read for this input; capture quality or structural anchors are dragging the estimate.";
  els.verdictText.textContent = `${high}${caveat}`;
}
function renderLists(){
  const m=state.analysis.features.filter(x=>Number.isFinite(x.score));
  const gaps=m.filter(x=>x.name!=="Photo validity"&&x.score<7.0).sort((a,b)=>a.score-b.score).slice(0,5);
  const strengths=m.filter(x=>x.conf>.45).sort((a,b)=>b.score-a.score).slice(0,5);
  renderChipList(els.gapList,gaps,"gap"); renderChipList(els.strengthList,strengths,"good");
}
function renderChipList(el, items, type){
  el.className="chip-list"; el.innerHTML="";
  if(!items.length){el.className="chip-list empty"; el.textContent=type==="gap"?"No major bottleneck detected.":"No stable strengths yet."; return;}
  for(const x of items){
    const div=document.createElement("div"); div.className=`chip ${type==="good"?"good":x.score<5.8?"gap-high":"gap-med"}`;
    div.innerHTML=`<b>${x.name}<span class="chip-score">${fmt(x.score)}</span></b><span>${x.measure} · conf ${Math.round(x.conf*100)}%</span>`;
    el.appendChild(div);
  }
}
function renderAudit(){
  const f=state.analysis.final;
  const rows=[['latent z',fmt(f.z,2)],['percentile',`${fmt(f.percentile*100,1)}%`],['raw core',fmt(f.rawCore)],['robust avg',fmt(f.robust)],['low anchors',fmt(f.lowMean)],['coherence',fmt(f.coherence)],['bottleneck drag',fmt(f.bottleneckDrag,2)],['capture drag',fmt(f.captureDrag,2)],['pose drag',fmt(f.poseDrag,2)],['confidence',`${Math.round(f.confidence*100)}%`]];
  els.auditGrid.innerHTML = rows.map(([k,v])=>`<div><span>${k}</span><b>${v}</b></div>`).join("");
}
function renderFeatureGrid(){
  if(!state.analysis) return;
  const q=els.metricSearch.value.trim().toLowerCase(), cat=els.categoryFilter.value;
  const data=state.analysis.features.filter(m => (cat==="all"||m.category===cat) && (!q || `${m.name} ${m.measure} ${m.explanation}`.toLowerCase().includes(q)));
  els.featureGrid.className="feature-grid"; els.featureGrid.innerHTML="";
  for(const m of data){
    const card=document.createElement("div");
    const missing=!Number.isFinite(m.score); const score=missing?0:m.score;
    card.className=`feature-card ${!missing && score<5.8?'high-gap':!missing&&score<7?'med-gap':''}`;
    const tag = missing ? `<span class="tag warn">missing</span>` : score>=8 ? `<span class="tag good">strong</span>` : score<5.8 ? `<span class="tag bad">gap</span>` : score<7 ? `<span class="tag warn">watch</span>` : `<span class="tag">solid</span>`;
    card.innerHTML=`<div class="feature-top"><div><div class="feature-title">${m.name}</div><div class="feature-meta"><span>${m.measure}</span><span>conf ${Math.round(m.conf*100)}%</span></div></div><div class="feature-score">${missing?'—':fmt(score)}</div></div><div class="score-bar"><i style="width:${missing?0:score*10}%"></i></div><p>${m.explanation}</p><small>${m.suggestion}</small><div>${tag}</div>`;
    els.featureGrid.appendChild(card);
  }
  if(!data.length){ els.featureGrid.className="feature-grid empty-state"; els.featureGrid.textContent="No matching metrics."; }
}
function renderCurve(){
  if(!state.analysis) return;
  const canvas=els.curveCanvas, c=curveCtx, r=canvas.getBoundingClientRect(), dpr=devicePixelRatio||1; canvas.width=r.width*dpr; canvas.height=r.height*dpr;
  c.clearRect(0,0,canvas.width,canvas.height); const W=canvas.width,H=canvas.height,pad=22*dpr;
  c.strokeStyle="rgba(255,255,255,.12)"; c.lineWidth=1*dpr; c.beginPath(); c.moveTo(pad,H-pad); c.lineTo(W-pad,H-pad); c.stroke();
  c.beginPath();
  for(let i=0;i<260;i++){
    const x=lerp(-3,3,i/259); const pdf=Math.exp(-.5*x*x)/Math.sqrt(2*Math.PI); const px=lerp(pad,W-pad,(x+3)/6); const py=H-pad-pdf*(H-pad*2)*2.15;
    if(i===0)c.moveTo(px,py); else c.lineTo(px,py);
  }
  c.strokeStyle="#e7c572"; c.lineWidth=2*dpr; c.stroke();
  const z=state.analysis.final.z; const px=lerp(pad,W-pad,clamp((z+3)/6,0,1));
  c.strokeStyle="#81d4fa"; c.lineWidth=2*dpr; c.beginPath(); c.moveTo(px,pad*.7); c.lineTo(px,H-pad); c.stroke();
  c.fillStyle="#dfe9f6"; c.font=`${11*dpr}px Inter, system-ui`; c.fillText(`score ${fmt(state.analysis.final.score,1)}`, px+6*dpr, pad*1.2);
  els.percentileLabel.textContent = `${fmt(state.analysis.final.percentile*100,1)} percentile`;
}
function renderRadar(){
  if(!state.analysis) return;
  const canvas=els.radarCanvas,c=radarCtx,r=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1; canvas.width=r.width*dpr; canvas.height=r.height*dpr;
  c.clearRect(0,0,canvas.width,canvas.height);
  const groups=["Structure","Eyes","Nose/mouth","Lower face","Skin/photo","Profile/body"];
  const catMap={"Structure":"structure","Eyes":"eyes","Nose/mouth":"nose-mouth","Lower face":"lower-face","Skin/photo":"skin-photo","Profile/body":"profile-body"};
  const vals=groups.map(g=>{
    const arr=state.analysis.features.filter(m=>m.category===catMap[g]&&Number.isFinite(m.score)&&m.conf>.2).map(m=>m.score);
    return arr.length?mean(arr):NaN;
  });
  const W=canvas.width,H=canvas.height,cx=W/2,cy=H/2+4*dpr,R=Math.min(W,H)*.34;
  c.strokeStyle="rgba(255,255,255,.10)"; c.fillStyle="rgba(129,212,250,.16)";
  for(let ring=2; ring<=10; ring+=2){ c.beginPath(); for(let i=0;i<groups.length;i++){ const a=-Math.PI/2+i/groups.length*Math.PI*2; const rr=R*ring/10; const x=cx+Math.cos(a)*rr,y=cy+Math.sin(a)*rr; i?c.lineTo(x,y):c.moveTo(x,y);} c.closePath(); c.stroke(); }
  c.beginPath(); vals.forEach((v,i)=>{ const a=-Math.PI/2+i/groups.length*Math.PI*2; const rr=R*(Number.isFinite(v)?v:0)/10; const x=cx+Math.cos(a)*rr,y=cy+Math.sin(a)*rr; i?c.lineTo(x,y):c.moveTo(x,y); }); c.closePath(); c.fill(); c.strokeStyle="#81d4fa"; c.lineWidth=2*dpr; c.stroke();
  c.fillStyle="#dce6f6"; c.font=`${10*dpr}px Inter, system-ui`; groups.forEach((g,i)=>{ const a=-Math.PI/2+i/groups.length*Math.PI*2; const x=cx+Math.cos(a)*(R+22*dpr),y=cy+Math.sin(a)*(R+15*dpr); c.fillText(g,x-28*dpr,y); });
  els.radarLabel.textContent = `core ${fmt(state.analysis.final.rawCore)}`;
}

function renderPlaybook(){
  if(!state.analysis){ els.playbookList.className="playbook-list empty"; els.playbookList.textContent="Analysis required."; return; }
  const feats = state.analysis.features.filter(m=>Number.isFinite(m.score) && m.name!=="Photo validity" && m.name!=="Head pose");
  const gaps = feats.filter(m=>m.conf>.28 && m.score<7.0).sort((a,b)=>a.score-b.score).slice(0,4);
  const strengths = feats.filter(m=>m.conf>.4).sort((a,b)=>b.score-a.score).slice(0,3);
  const q = state.analysis.front.quality;
  const capture = [];
  if(q.validity < .82) capture.push({name:"Use a cleaner rating photo", detail:"Neutral expression, less crop distortion, and even lighting will tighten the score range and make symmetry/texture reads cleaner."});
  if(q.yaw > .14) capture.push({name:"Reduce head turn", detail:"A straighter front photo improves symmetry, thirds, and eye spacing interpretation."});
  if(!state.slots.profile) capture.push({name:"Add a side profile", detail:"That unlocks stronger reads for convexity, E-line, and projection instead of leaving profile quality partly unresolved."});
  if(!capture.length) capture.push({name:"Capture quality is already solid", detail:"You are getting a reasonably clean read from the current upload set."});

  const groups = [
    {
      title:"Highest-ROI fixes",
      blurb:"The biggest visible bottlenecks currently capping the estimate.",
      items: gaps.length ? gaps.map(m=>({title:m.name, meta:`${fmt(m.score)}/10 · conf ${Math.round(m.conf*100)}%`, body:m.suggestion || m.explanation})) : [{title:"No major structural bottleneck", meta:"stable", body:"There is no single obvious weak point dominating the read right now."}]
    },
    {
      title:"Keep / amplify",
      blurb:"Strong areas worth preserving in styling, grooming, and photo choices.",
      items: strengths.length ? strengths.map(m=>({title:m.name, meta:`${fmt(m.score)}/10`, body:m.explanation})) : [{title:"Need cleaner input", meta:"—", body:"Upload a clearer front image to surface stable strengths."}]
    },
    {
      title:"Better input & next steps",
      blurb:"Practical ways to make the analysis more complete and more trustworthy.",
      items: capture.map(m=>({title:m.name, meta:"input", body:m.detail}))
    }
  ];

  els.playbookList.className="playbook-list";
  els.playbookList.innerHTML = groups.map(g=>`<div class="playbook-group"><h4>${g.title}</h4><p>${g.blurb}</p><div class="playbook-items">${g.items.map(i=>`<div class="playbook-item"><b>${i.title}<span>${i.meta}</span></b><span>${i.body}</span></div>`).join("")}</div></div>`).join("");
}

function renderModel(){
  const canvas=els.modelCanvas, c=modelCtx, r=canvas.getBoundingClientRect(), dpr=devicePixelRatio||1; canvas.width=r.width*dpr; canvas.height=r.height*dpr; c.clearRect(0,0,canvas.width,canvas.height); c.fillStyle="#050609"; c.fillRect(0,0,canvas.width,canvas.height);
  if(!state.analysis){ els.modelNotes.className="note-list empty"; els.modelNotes.textContent="Analysis required."; return; }
  const A=state.analysis.front, f=A.frame, W=canvas.width,H=canvas.height,cx=W/2,cy=H*.52,scale=Math.min(W,H)/(f.ipd*3.35);
  const localCanvas = (idx) => { const p=f.local(pointAt(A.lm,idx,A.img)); return {x:cx+p.x*scale, y:cy+(p.y-f.ipd*.55)*scale, z:p.z}; };
  c.strokeStyle="rgba(129,212,250,.32)"; c.lineWidth=1.2*dpr;
  for(const [i,j] of MESH_LINES){ const a=localCanvas(i),b=localCanvas(j); c.beginPath(); c.moveTo(a.x,a.y); c.lineTo(b.x,b.y); c.stroke(); }
  const pts=[10,9,168,6,1,2,152,33,133,362,263,234,454,127,356,172,397,61,291,13,14,98,327,70,105,107,336,334,300];
  for(const i of pts){ const p=localCanvas(i); const depth=clamp(1.0-(p.z/(f.ipd||1))*1.8,.3,1.0); c.fillStyle=`rgba(${Math.round(231*depth)},${Math.round(197*depth+50)},${Math.round(114*depth+80)},.92)`; c.beginPath(); c.arc(p.x,p.y,3.0*dpr,0,Math.PI*2); c.fill(); }
  c.fillStyle="#aebbd0"; c.font=`${12*dpr}px Inter, system-ui`; c.fillText("2.5D landmark relief (front image)",16*dpr,24*dpr);
  renderModelNotes();
}
function renderModelNotes(){
  if(!state.analysis){ els.modelNotes.className="note-list empty"; els.modelNotes.textContent="Analysis required."; return; }
  const A=state.analysis, q=A.front.quality, f=A.front.frame;
  const mainGap = A.features.filter(m=>Number.isFinite(m.score) && m.name!=="Photo validity" && m.name!=="Head pose").sort((a,b)=>a.score-b.score)[0];
  const strongest = A.features.filter(m=>Number.isFinite(m.score)).sort((a,b)=>b.score-a.score)[0];
  const notes = [
    {title:"Measurement reliability", value:`${Math.round(A.final.confidence*100)}%`, body:`Photo validity is ${q.validity>.78?"clean":q.validity>.58?"usable":"limited"}; pose usability is ${q.yaw<.14?"good":q.yaw<.28?"moderate":"reduced"}.`, foot:`Roll correction ${fmt(f.roll,1)}° · percentile ${fmt(A.final.percentile*100,1)}%`},
    {title:"Primary score cap", value: mainGap ? `${mainGap.name} ${fmt(mainGap.score)}` : "—", body: mainGap ? mainGap.explanation : "No single metric is clearly capping the score.", foot: mainGap ? mainGap.suggestion : ""},
    {title:"Strongest anchor", value: strongest ? `${strongest.name} ${fmt(strongest.score)}` : "—", body: strongest ? strongest.explanation : "No stable strength surfaced yet.", foot: strongest ? `Confidence ${Math.round((strongest.conf||0)*100)}%` : ""},
    {title:"Coverage", value: coverageText(), body:`Front photo is required; profile/body images expand the analysis instead of being treated as negative scores.`, foot: `${state.slots.profile?"Profile added":"No profile"} · ${state.slots.bodyFront||state.slots.bodySide?"Body image added":"No body image"}`}
  ];
  els.modelNotes.className="note-list";
  els.modelNotes.innerHTML = notes.map(n=>`<div class="note-card"><b>${n.title}<span>${n.value}</span></b><p>${n.body}</p>${n.foot?`<small>${n.foot}</small>`:""}</div>`).join("");
}

// Initial state
registerPwaSupport();
clearAnalysis(false); drawViewer(); updateUploadSummary(); updateAnalyzeEnabled(); switchScreen('home');
