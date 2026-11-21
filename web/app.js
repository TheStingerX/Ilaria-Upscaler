'use strict';
const $ = id => document.getElementById(id);
const state = {mode:'fast', catalog:null, file:null, previewUrl:null, resultUrl:null, running:false};
const modeButtons = [...document.querySelectorAll('[data-mode]')];
let pollTimer = null;
function labelStatus(mode, enabled){
  $(mode+'Status').textContent=enabled?'Ready':'Setup';
  $(mode+'Dot').className='status-dot '+(enabled?'ready':'warn');
}
function modeInfo(){ return state.catalog?.modes?.[state.mode]; }
function updateControls(){
  const info=modeInfo(), select=$('modelSelect'); select.replaceChildren();
  (info?.models||[]).forEach(model=>{
    const option=document.createElement('option');option.value=model.id;option.textContent=model.label;
    select.append(option);
  });
  const ready=Boolean(state.file && info?.available && !state.running);
  $('upscaleBtn').disabled=!ready;
  $('readyText').textContent=state.running?'Enhancing your image…':!state.file?'Upload an image to begin':!info?.available?'This engine needs configuration':'Ready to process with '+state.mode.toUpperCase();
  $('modelHint').textContent=info?.reason||'Unable to load engine information';
  const notice=$('modeNotice');notice.hidden=Boolean(info?.available);notice.textContent=info?.reason||'Engine unavailable.';
  $('tileSelect').disabled=state.mode==='ultra';
  $('tileHint').textContent=state.mode==='ultra'?'SeedVR2 uses its own VAE tiling':'Adjust memory vs. processing speed';
  modeButtons.forEach(btn=>{const active=btn.dataset.mode===state.mode;btn.classList.toggle('active',active);btn.setAttribute('aria-pressed',String(active));});
}
async function init(){
  try{
    const response=await fetch('/api/status');if(!response.ok)throw new Error('Server response '+response.status);
    state.catalog=await response.json();
    const hw=state.catalog.hardware||{};
    $('hardwareBadge').textContent='● '+(hw.cuda?`${hw.device} · ${hw.vram_gb} GB VRAM`:hw.nvidia_detected?'NVIDIA GPU · CUDA OFF':'CPU MODE');
    for(const name of ['fast','quality','ultra'])labelStatus(name,Boolean(state.catalog.modes[name]?.available));
    $('ultraFoot').textContent=state.catalog.modes.ultra.available?'✦ SeedVR2 installed':'✦ Setup required';
  }catch(error){$('hardwareBadge').textContent='● Server unavailable';showError('Cannot reach processing server: '+error.message);}
  updateControls();
}
function showError(message){const box=$('modeNotice');box.hidden=false;box.textContent=message;}
function resetResult(){
  if(state.resultUrl)URL.revokeObjectURL(state.resultUrl);
  state.resultUrl=null;$('resultBanner').hidden=true;$('compareDivider').style.display='none';
  $('afterLayer').style.display='none';$('compareRange').disabled=true;$('comparePlaceholder').hidden=false;
}
function setFile(file){
  if(!file)return;
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)){showError('Please use PNG, JPEG or WebP.');return;}
  if(file.size>18*1024*1024){showError('The image must be 18 MB or smaller.');return;}
  const url=URL.createObjectURL(file), probe=new Image();
  probe.onload=()=>{
    if(probe.naturalWidth*probe.naturalHeight>12_000_000){URL.revokeObjectURL(url);showError('The image must be no larger than 12 megapixels.');return;}
    if(state.previewUrl)URL.revokeObjectURL(state.previewUrl);
    state.previewUrl=url;state.file=file;resetResult();
    $('originalImg').src=url;$('dropzone').hidden=true;$('previewArea').hidden=false;
    $('fileName').textContent=file.name;$('fileDimensions').textContent=`${probe.naturalWidth} × ${probe.naturalHeight} · ${(file.size/1024/1024).toFixed(2)} MB`;
    $('modeNotice').hidden=true;updateControls();
  };
  probe.onerror=()=>{URL.revokeObjectURL(url);showError('The selected file is not a valid image.');};probe.src=url;
}
$('dropzone').addEventListener('click',()=>$('imageInput').click());
$('dropzone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('imageInput').click();}});
$('replaceImage').addEventListener('click',()=>$('imageInput').click());
$('imageInput').addEventListener('change',e=>setFile(e.target.files?.[0]));
for(const type of ['dragenter','dragover'])$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.add('dragging');});
for(const type of ['dragleave','drop'])$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.remove('dragging');});
$('dropzone').addEventListener('drop',e=>setFile(e.dataTransfer.files?.[0]));
modeButtons.forEach(btn=>btn.addEventListener('click',()=>{if(state.running)return;state.mode=btn.dataset.mode;updateControls();}));
$('scaleRange').addEventListener('input',e=>$('scaleValue').textContent=e.target.value+'×');
$('compareRange').addEventListener('input',e=>setCompare(e.target.value));
function setCompare(percent){$('afterLayer').style.clipPath=`inset(0 0 0 ${percent}%)`;$('compareDivider').style.left=percent+'%';}
async function processImage(){
  if(state.running||!state.file||!modeInfo()?.available)return;
  state.running=true;resetResult();updateControls();$('progressBox').hidden=false;progress(0,'Uploading your image…');
  const fd=new FormData();fd.append('image',state.file);fd.append('mode',state.mode);fd.append('model',$('modelSelect').value);
  fd.append('scale',$('scaleRange').value);fd.append('tile',$('tileSelect').value);
  try{
    const response=await fetch('/api/jobs',{method:'POST',body:fd});
    if(!response.ok){const content=await response.json().catch(()=>({}));throw new Error(content.detail||`Upload failed (${response.status})`);}
    const {id}=await response.json();await pollJob(id);
  }catch(e){progress(0,'Error: '+e.message);$('progressTitle').textContent='Processing failed';$('progressPercent').textContent='!';}
  finally{state.running=false;updateControls();}
}
async function pollJob(id){
  for(;;){
    await new Promise(resolve=>{pollTimer=setTimeout(resolve,950);});
    const response=await fetch('/api/jobs/'+encodeURIComponent(id));
    if(!response.ok)throw new Error('Unable to get processing status');
    const job=await response.json();progress(job.progress||0,job.message||'Processing…');
    if(job.state==='error')throw new Error(job.message||'Upscaling failed');
    if(job.state==='done'){
      const result=await fetch('/api/jobs/'+encodeURIComponent(id)+'/result');if(!result.ok)throw new Error('Could not download the processed image');
      state.resultUrl=URL.createObjectURL(await result.blob());
      $('resultImg').src=state.resultUrl;$('afterLayer').style.display='block';$('comparePlaceholder').hidden=true;
      $('compareDivider').style.display='block';$('compareRange').disabled=false;setCompare(50);
      $('downloadBtn').href=state.resultUrl;$('resultInfo').textContent=`${job.metadata?.width||'?'} × ${job.metadata?.height||'?'} · PNG`;
      $('resultBanner').hidden=false;$('progressBox').hidden=true;
      $('previewArea').scrollIntoView({block:'nearest',behavior:'smooth'});return;
    }
  }
}
function progress(value,message){$('progressBar').style.width=value+'%';$('progressPercent').textContent=value+'%';$('progressTitle').textContent='Processing your image';$('progressMessage').textContent=message;}
$('upscaleBtn').addEventListener('click',processImage);
window.addEventListener('pagehide',()=>{if(pollTimer)clearTimeout(pollTimer);if(state.previewUrl)URL.revokeObjectURL(state.previewUrl);if(state.resultUrl)URL.revokeObjectURL(state.resultUrl);});
init();
