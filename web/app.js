"use strict";
const RC = window.ReconCore;
const TILE = 256;
const R_EYE=[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,469,470,471,472,468];
const L_EYE=[362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,474,475,476,477,473];

const $=id=>document.getElementById(id);
const S={
  mode:'recon', region:'rEye', reg:'translation',
  stack:false, auto:true, contrast:1, cmp:1,
  pad:0.40, rlIter:15, psfMode:'catchlight', psfSize:15,
  running:false, srcKind:null, landmarks:null, roiRect:null, dragging:false, dragStart:null,
  accum:null, accN:0, refTile:null, fpsT:performance.now(), fpsC:0,
  lastProcessed:null, lastRaw:null,
  // sensor
  sview:'residual', sigma:3.0, levels:4, prState:null, K:null, ncc:0, pce:0, lastResidual:null
};

const srcCanvas=$('srcCanvas'),srcCtx=srcCanvas.getContext('2d',{willReadFrequently:true});
const outCanvas=$('outCanvas'),outCtx=outCanvas.getContext('2d',{willReadFrequently:true});
const tile=document.createElement('canvas');tile.width=tile.height=TILE;
const tileCtx=tile.getContext('2d',{willReadFrequently:true});
let video=null,faceMesh=null,camera=null,rafId=null,imageEl=null;

/* ---- tabs ---- */
$('tabRecon').onclick=()=>setMode('recon');
$('tabSensor').onclick=()=>setMode('sensor');
function setMode(m){S.mode=m;
  $('tabRecon').classList.toggle('active',m==='recon');$('tabSensor').classList.toggle('active',m==='sensor');
  $('reconCtl').classList.toggle('hide',m!=='recon');$('sensorCtl').classList.toggle('hide',m==='recon');
  $('outTitle').textContent=m==='recon'?'02 · reconstruction':'02 · sensor view';}

function seg(sel,attr,cb){document.querySelectorAll(sel).forEach(b=>b.onclick=()=>{
  document.querySelectorAll(sel).forEach(x=>x.classList.remove('active'));b.classList.add('active');cb(b.dataset[attr]);});}
seg('[data-region]','region',v=>{S.region=v;resetStack();$('srcHint').textContent=v==='roi'?'drag to set ROI':'region: '+v;});
seg('[data-reg]','reg',v=>{S.reg=v;resetStack();});
seg('[data-psf]','psf',v=>{S.psfMode=v;});
seg('[data-sview]','sview',v=>{S.sview=v;});
function toggle(id,key,after){const el=$(id);el.classList.toggle('on',S[key]);el.onclick=()=>{S[key]=!S[key];el.classList.toggle('on',S[key]);if(after)after();};}
toggle('stackTog','stack',()=>{$('stackState').textContent=S.stack?'on':'off';resetStack();});
toggle('autoTog','auto');
function slider(id,out,key,fmt,map){const el=$(id),o=$(out);const u=()=>{const r=+el.value;S[key]=map?map(r):r;o.textContent=fmt(r);};el.oninput=u;u();}
slider('pad','padOut','pad',v=>v+'%',v=>v/100);
slider('contrast','coOut','contrast',v=>(v/100).toFixed(2)+'×',v=>v/100);
slider('cmp','cmpOut','cmp',v=>v+'%',v=>v/100);
slider('rlIter','rlIterOut','rlIter',v=>''+v);
slider('psfSize','psfOut','psfSize',v=>''+v);
slider('sigma','sigOut','sigma',v=>(v/10).toFixed(1),v=>v/10);
slider('levels','lvlOut','levels',v=>''+v);
$('resetStack').onclick=resetStack;
function resetStack(){S.accum=null;S.accN=0;S.refTile=null;S.lastProcessed=null;$('framesOut').textContent='0';$('alignOut').textContent='—';}

/* ---- sensor buttons ---- */
$('enrollBtn').onclick=()=>{S.enrolling=!S.enrolling;$('enrollBtn').textContent=S.enrolling?'● enrolling…':'● enroll fingerprint';$('enrollBtn').classList.toggle('stop',S.enrolling);};
$('verifyBtn').onclick=()=>{if(!S.K){$('verdict').textContent='enroll first';return;}runVerify();};
$('resetPrnu').onclick=()=>{S.prState=null;S.K=null;S.enrolling=false;$('enrollN').textContent='0';
  $('enrollBtn').textContent='● enroll fingerprint';$('enrollBtn').classList.remove('stop');
  gauge(0,0);$('verdict').className='verdict';$('verdict').textContent='awaiting verify';};

/* ---- deconvolve ---- */
$('deconvBtn').onclick=()=>{if(!S.accum){return;}deconvolveStack();};

/* ---- source ---- */
$('startBtn').onclick=async()=>{if(S.running&&S.srcKind==='cam'){stopAll();return;}stopAll();
  try{video=document.createElement('video');video.muted=true;video.playsInline=true;
    const st=await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720},audio:false});
    video.srcObject=st;await video.play();S.srcKind='cam';setLive(true,'▶ stop webcam','cam');
    await ensureFaceMesh();startPump();loop();
  }catch(e){$('srcHint').textContent='camera unavailable — '+e.message;}};
$('loadVideo').onclick=()=>$('videoFile').click();$('loadImage').onclick=()=>$('imageFile').click();
$('videoFile').onchange=e=>{const f=e.target.files[0];if(!f)return;stopAll();
  video=document.createElement('video');video.src=URL.createObjectURL(f);video.loop=true;video.muted=true;video.playsInline=true;
  video.onloadeddata=async()=>{await video.play();S.srcKind='video';setLive(true,'▶ start webcam','video');await ensureFaceMesh();startPump();loop();};};
$('imageFile').onchange=e=>{const f=e.target.files[0];if(!f)return;stopAll();
  imageEl=new Image();imageEl.onload=async()=>{S.srcKind='image';setLive(true,'▶ start webcam','image');await ensureFaceMesh();loop();};imageEl.src=URL.createObjectURL(f);};
function setLive(on,t,k){S.running=on;const s=$('liveStamp'),b=$('startBtn');s.classList.toggle('on',on);
  s.innerHTML='<span class="dot"></span>'+(on?(k==='cam'?'live capture':'playback'):'standby');
  $('srcState').textContent=on?k:'off';b.textContent=t;b.classList.toggle('stop',on&&k==='cam');}
function stopAll(){if(rafId)cancelAnimationFrame(rafId);rafId=null;if(camera){try{camera.stop()}catch(e){}camera=null;}
  if(video&&video.srcObject)video.srcObject.getTracks().forEach(t=>t.stop());
  video=null;imageEl=null;S.landmarks=null;resetStack();setLive(false,'▶ start webcam',null);
  $('srcMeta').textContent='no signal';$('outMeta').textContent='idle';}
async function ensureFaceMesh(){if(faceMesh)return;
  faceMesh=new FaceMesh({locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`});
  faceMesh.setOptions({maxNumFaces:1,refineLandmarks:true,minDetectionConfidence:.5,minTrackingConfidence:.5});
  faceMesh.onResults(r=>{S.landmarks=(r.multiFaceLandmarks&&r.multiFaceLandmarks[0])||null;});}
function startPump(){camera=new Camera(video,{onFrame:async()=>{
  if(S.mode==='recon'&&S.region!=='roi'&&S.region!=='full')await faceMesh.send({image:video});},width:1280,height:720});camera.start();}

/* ---- ROI ---- */
function frameSize(){if(S.srcKind==='image')return[imageEl.naturalWidth,imageEl.naturalHeight];return[video.videoWidth||640,video.videoHeight||480];}
function lmBox(idx,W,H){let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;for(const i of idx){const p=S.landmarks[i];if(!p)continue;
  x0=Math.min(x0,p.x);y0=Math.min(y0,p.y);x1=Math.max(x1,p.x);y1=Math.max(y1,p.y);}return{x0:x0*W,y0:y0*H,x1:x1*W,y1:y1*H};}
function padBox(b,W,H){let w=b.x1-b.x0,h=b.y1-b.y0,px=w*S.pad,py=h*S.pad;let x=b.x0-px,y=b.y0-py;w+=2*px;h+=2*py;
  x=Math.max(0,x);y=Math.max(0,y);w=Math.min(W-x,w);h=Math.min(H-y,h);return{x,y,w,h};}
function computeROI(W,H){if(S.region==='full')return{x:0,y:0,w:W,h:H};
  if(S.region==='roi')return S.roiRect||{x:W*.35,y:H*.35,w:W*.3,h:H*.3};
  if(!S.landmarks)return null;
  if(S.region==='rEye')return padBox(lmBox(R_EYE,W,H),W,H);
  if(S.region==='lEye')return padBox(lmBox(L_EYE,W,H),W,H);
  const b=lmBox([...R_EYE,...L_EYE],W,H);
  if(S.region==='glasses'){const cx=(b.x0+b.x1)/2,cy=(b.y0+b.y1)/2,w=(b.x1-b.x0)*1.35,h=(b.y1-b.y0)*2.2;return padBox({x0:cx-w/2,y0:cy-h/2,x1:cx+w/2,y1:cy+h/2},W,H);}
  return padBox(b,W,H);}

/* ---- helpers: canvas <-> float luma tile ---- */
function roiToLuma(srcImg,roi){tileCtx.imageSmoothingEnabled=true;tileCtx.imageSmoothingQuality='high';
  tileCtx.drawImage(srcImg,roi.x,roi.y,roi.w,roi.h,0,0,TILE,TILE);
  const d=tileCtx.getImageData(0,0,TILE,TILE).data,L=new Float64Array(TILE*TILE);
  for(let i=0,j=0;i<d.length;i+=4,j++)L[j]=0.2126*d[i]+0.7152*d[i+1]+0.0722*d[i+2];return L;}
function frameToLuma(srcImg){tileCtx.imageSmoothingEnabled=true;tileCtx.drawImage(srcImg,0,0,TILE,TILE);
  const d=tileCtx.getImageData(0,0,TILE,TILE).data,L=new Float64Array(TILE*TILE);
  for(let i=0,j=0;i<d.length;i+=4,j++)L[j]=0.2126*d[i]+0.7152*d[i+1]+0.0722*d[i+2];return L;}
function stddev(a){let m=0;for(let i=0;i<a.length;i++)m+=a[i];m/=a.length;let v=0;for(let i=0;i<a.length;i++){const d=a[i]-m;v+=d*d;}return Math.sqrt(v/a.length);}

/* ---- main loop ---- */
function loop(){const[W,H]=frameSize();if(!W){rafId=requestAnimationFrame(loop);return;}
  srcCanvas.width=W;srcCanvas.height=H;const srcImg=S.srcKind==='image'?imageEl:video;
  srcCtx.drawImage(srcImg,0,0,W,H);$('srcMeta').textContent=W+'×'+H;
  if(S.mode==='recon'){
    if(S.srcKind==='image'&&S.region!=='roi'&&S.region!=='full'&&!S.landmarks)faceMesh.send({image:imageEl});
    const roi=computeROI(W,H);drawReticle(W,H,roi);
    $('tFace').textContent=S.landmarks?'locked':(S.region==='roi'||S.region==='full'?'n/a':'searching');
    if(roi&&roi.w>2&&roi.h>2)processRecon(roi,srcImg);else{blank();$('outMeta').textContent='no roi';}
  }else{processSensorLive(srcImg);}
  S.fpsC++;const now=performance.now();if(now-S.fpsT>500){$('fpsStamp').textContent=Math.round(S.fpsC*1000/(now-S.fpsT))+' fps';S.fpsT=now;S.fpsC=0;}
  rafId=requestAnimationFrame(loop);}
function blank(){outCtx.fillStyle='#0a0f16';outCtx.fillRect(0,0,outCanvas.width,outCanvas.height);}

/* ---- RECON pipeline (uses ReconCore) ---- */
function processRecon(roi,srcImg){
  const L=roiToLuma(srcImg,roi);              // 0..255 float tile
  S.lastRaw=L;
  $('tRoi').textContent=Math.round(roi.w)+'×'+Math.round(roi.h)+' px';
  $('tNoise').textContent=stddev(L).toFixed(1);
  if(S.stack){
    let cur=L;
    if(!S.accum){S.accum=Float64Array.from(L);S.accN=1;S.refTile=Float64Array.from(L);$('alignOut').textContent='ref';}
    else{
      let aligned=cur,al='—';
      if(S.reg==='translation'){const t=RC.phaseCorr(S.refTile,cur,TILE,TILE);aligned=RC.shiftImg(cur,TILE,TILE,t.dy,t.dx);al=t.dy.toFixed(1)+','+t.dx.toFixed(1)+'px';}
      else if(S.reg==='rst'){const r=RC.registerRST(S.refTile,cur,TILE,TILE);const rot=RC.rotateScale(cur,TILE,TILE,r.angle,r.scale);aligned=RC.shiftImg(rot,TILE,TILE,r.dy,r.dx);al=r.angle.toFixed(1)+'° ×'+r.scale.toFixed(3);}
      for(let i=0;i<TILE*TILE;i++)S.accum[i]+=aligned[i];S.accN++;$('alignOut').textContent=al;
    }
    $('framesOut').textContent=S.accN;
    const avg=new Float64Array(TILE*TILE);for(let i=0;i<TILE*TILE;i++)avg[i]=S.accum[i]/S.accN;
    S.lastProcessed=avg;presentRecon(avg);
  }else{S.lastProcessed=L;presentRecon(L);}
}
function deconvolveStack(){
  if(!S.lastProcessed)return;$('busy').classList.add('on');
  setTimeout(()=>{
    const base=S.lastProcessed;const norm=new Float64Array(TILE*TILE);
    let mx=1;for(let i=0;i<base.length;i++)mx=Math.max(mx,base[i]);
    for(let i=0;i<base.length;i++)norm[i]=Math.min(1,Math.max(1e-6,base[i]/mx));
    const psf=S.psfMode==='catchlight'?RC.estimatePSF(norm,TILE,TILE,S.psfSize):RC.gaussPSF(S.psfSize/6);
    const u=RC.richardsonLucy(norm,TILE,TILE,psf,S.rlIter);
    const out=new Float64Array(TILE*TILE);for(let i=0;i<u.length;i++)out[i]=u[i]*mx;
    S.lastProcessed=out;presentRecon(out);$('busy').classList.remove('on');
    $('outMeta').textContent='deconvolved · '+S.rlIter+'it';
  },30);
}
function presentRecon(arr){
  let a=Float64Array.from(arr);
  if(S.auto){const s=arr.slice().sort();const lo=s[(s.length*0.01)|0],hi=s[(s.length*0.99)|0];const sc=hi>lo?255/(hi-lo):1;
    for(let i=0;i<a.length;i++)a[i]=Math.min(255,Math.max(0,(a[i]-lo)*sc));}
  const con=S.contrast;for(let i=0;i<a.length;i++)a[i]=Math.min(255,Math.max(0,(a[i]-128)*con+128));
  if(S.cmp<1&&S.lastRaw){for(let i=0;i<a.length;i++)a[i]=S.lastRaw[i]*(1-S.cmp)+a[i]*S.cmp;}
  drawTile(a,S.stack?'int·'+S.accN:'single');
}
function drawTile(lumaArr,tag){
  const img=tileCtx.createImageData(TILE,TILE);
  for(let i=0,j=0;i<lumaArr.length;i++,j+=4){const v=Math.min(255,Math.max(0,lumaArr[i]));img.data[j]=img.data[j+1]=img.data[j+2]=v;img.data[j+3]=255;}
  tileCtx.putImageData(img,0,0);
  outCtx.imageSmoothingEnabled=false;outCtx.fillStyle='#0a0f16';outCtx.fillRect(0,0,outCanvas.width,outCanvas.height);
  const sc=Math.min(outCanvas.width/TILE,outCanvas.height/TILE),d=TILE*sc,dx=(outCanvas.width-d)/2,dy=(outCanvas.height-d)/2;
  outCtx.drawImage(tile,0,0,TILE,TILE,dx,dy,d,d);outCtx.strokeStyle='#1e2836';outCtx.strokeRect(dx+.5,dy+.5,d-1,d-1);
  if(tag)$('outMeta').textContent=tag;
}

/* ---- SENSOR (uses ReconCore) ---- */
function processSensorLive(srcImg){
  const L=frameToLuma(srcImg);
  if(S.enrolling){S.prState=RC.fingerprintAccumulate(S.prState,L,TILE,TILE,S.sigma,S.levels);
    S.K=RC.fingerprintFinalize(S.prState);$('enrollN').textContent=S.prState.n;}
  // live residual view
  S.lastResidual=RC.noiseResidual(L,TILE,TILE,S.sigma,S.levels);
  const arr=(S.sview==='print'&&S.K)?S.K:S.lastResidual;
  drawSigned(arr,(S.sview==='print'?'fingerprint K̂':'residual W'));
}
function runVerify(){const srcImg=S.srcKind==='image'?imageEl:video;const L=frameToLuma(srcImg);
  const r=RC.verify(L,S.K,TILE,TILE,S.sigma,S.levels);S.ncc=r.ncc;S.pce=r.pce;gauge(r.ncc,r.pce);verdict(r.pce);}
function drawSigned(arr,tag){let mn=1e9,mx=-1e9;for(let i=0;i<arr.length;i++){if(arr[i]<mn)mn=arr[i];if(arr[i]>mx)mx=arr[i];}
  const sc=mx>mn?255/(mx-mn):1;const img=tileCtx.createImageData(TILE,TILE);
  for(let i=0,j=0;i<arr.length;i++,j+=4){const v=(arr[i]-mn)*sc;img.data[j]=img.data[j+1]=img.data[j+2]=v;img.data[j+3]=255;}
  tileCtx.putImageData(img,0,0);
  outCtx.imageSmoothingEnabled=false;outCtx.fillStyle='#0a0f16';outCtx.fillRect(0,0,outCanvas.width,outCanvas.height);
  const s=Math.min(outCanvas.width/TILE,outCanvas.height/TILE),d=TILE*s,dx=(outCanvas.width-d)/2,dy=(outCanvas.height-d)/2;
  outCtx.drawImage(tile,0,0,TILE,TILE,dx,dy,d,d);outCtx.strokeStyle='#1e2836';outCtx.strokeRect(dx+.5,dy+.5,d-1,d-1);
  $('outMeta').textContent=tag+' 256×256';}
function gauge(c0,pce){$('nccVal').textContent=c0.toFixed(4);$('pceVal').textContent=pce.toFixed(1);
  $('nccBar').style.width=Math.min(100,Math.abs(c0)/0.06*100)+'%';
  $('pceBar').style.width=Math.min(100,Math.abs(pce)/80*100)+'%';
  $('pceBar').style.background=pce>50?'var(--signal)':pce>15?'var(--signal-dim)':'var(--line-2)';}
function verdict(pce){const v=$('verdict');if(pce>50){v.className='verdict match';v.textContent='same sensor · likely';}
  else if(pce>15){v.className='verdict';v.textContent='weak / inconclusive';}else{v.className='verdict no';v.textContent='no match';}}

/* ---- reticle + manual ROI ---- */
function drawReticle(W,H,roi){if(!roi)return;srcCtx.save();srcCtx.fillStyle='rgba(8,11,17,0.55)';
  srcCtx.beginPath();srcCtx.rect(0,0,W,H);srcCtx.rect(roi.x,roi.y,roi.w,roi.h);srcCtx.fill('evenodd');
  srcCtx.strokeStyle=S.region==='roi'?'#ffb454':'#54e6c1';srcCtx.lineWidth=Math.max(1.5,W/500);srcCtx.strokeRect(roi.x,roi.y,roi.w,roi.h);
  const t=Math.min(roi.w,roi.h)*.18;srcCtx.beginPath();
  for(const[cx,cy,sx,sy]of[[roi.x,roi.y,1,1],[roi.x+roi.w,roi.y,-1,1],[roi.x,roi.y+roi.h,1,-1],[roi.x+roi.w,roi.y+roi.h,-1,-1]]){srcCtx.moveTo(cx,cy);srcCtx.lineTo(cx+sx*t,cy);srcCtx.moveTo(cx,cy);srcCtx.lineTo(cx,cy+sy*t);}srcCtx.stroke();srcCtx.restore();}
function cxy(e){const r=srcCanvas.getBoundingClientRect(),sx=srcCanvas.width/r.width,sy=srcCanvas.height/r.height,p=e.touches?e.touches[0]:e;return[(p.clientX-r.left)*sx,(p.clientY-r.top)*sy];}
srcCanvas.addEventListener('mousedown',e=>{if(S.mode!=='recon'||S.region!=='roi')return;S.dragging=true;S.dragStart=cxy(e);e.preventDefault();});
srcCanvas.addEventListener('mousemove',e=>{if(!S.dragging)return;const[x,y]=cxy(e),[x0,y0]=S.dragStart;S.roiRect={x:Math.min(x,x0),y:Math.min(y,y0),w:Math.abs(x-x0),h:Math.abs(y-y0)};e.preventDefault();});
window.addEventListener('mouseup',()=>{if(S.dragging){S.dragging=false;resetStack();}});

/* ---- snapshots ---- */
$('snap').onclick=dl;$('snap2').onclick=dl;
function dl(){const a=document.createElement('a');a.download='recon_'+Date.now()+'.png';a.href=outCanvas.toDataURL('image/png');a.click();}

blank();srcCtx.fillStyle='#0a0f16';srcCtx.fillRect(0,0,srcCanvas.width,srcCanvas.height);
