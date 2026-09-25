const fourier = { file:null, data:null, sum:null, wave:null, count:0, timer:null, request:null, loading:false, error:null };

function pauseFourier() {
    clearTimeout(fourier.timer); fourier.timer=null;
    $("fourierPlay").textContent=fourier.data && fourier.count===fourier.data.groups.length ? "Replay" : "Play";
}

function clearFourierData() {
    pauseFourier();
    fourier.data=fourier.sum=fourier.wave=null; fourier.count=0;
    for (const id of ["fourierOriginal","fourierResult","fourierWave","fourierSpectrum"]) {
        const canvas=$(id);canvas.getContext("2d").clearRect(0,0,canvas.width,canvas.height);
    }
    $("fourierSampleLabel").textContent="";
    $("fourierMetrics").textContent="MSE and PSNR compare the displayed reconstruction with the sampled reference.";
    $("fourierWaveLabel").textContent="Neutral gray means zero contribution. Wave contrast is scaled for visibility.";
    $("fourierDetail").textContent="Upload an image to begin Fourier synthesis.";
}

function resetFourier() {
    if(fourier.request)fourier.request.abort();fourier.request=null;
    fourier.file=null;fourier.loading=false;fourier.error=null;
    $("fourierInput").value="";$("fourierFileName").textContent="";
    clearFourierData();renderFourier();
}

function selectFourierFile(file) {
    if(!file)return;
    if(!file.type.startsWith("image/")){showToast("Please select an image.",true);return;}
    fourier.file=file;$("fourierFileName").textContent=file.name;prepareFourier();
}

async function prepareFourier() {
    if(!fourier.file)return;
    if(fourier.request)fourier.request.abort();
    const controller=new AbortController();fourier.request=controller;
    clearFourierData();fourier.loading=true;fourier.error=null;renderFourier();
    const form=new FormData();form.append("image",fourier.file);
    form.append("mode",$("fourierMode").value);form.append("order",$("fourierOrder").value);form.append("sample_size",$("fourierSize").value);
    try {
        const response=await fetch("/api/fourier-canvas",{method:"POST",body:form,signal:controller.signal});
        const data=await response.json();if(!response.ok)throw Error(data.error||"Unable to prepare Fourier components");
        if(fourier.request!==controller)return;
        fourier.data=data;fourier.loading=false;
        initializeFourierSum();renderFourier();
    } catch(error) {
        if(controller!==fourier.request || error.name==="AbortError")return;
        fourier.loading=false;fourier.error=error.message;renderFourier();
    } finally {if(fourier.request===controller)fourier.request=null;}
}

function initializeFourierSum() {
    const d=fourier.data;fourier.count=0;
    fourier.sum=new Float64Array(d.width*d.height*d.channels);
    for(let i=0;i<fourier.sum.length;i++)fourier.sum[i]=d.dc[i%d.channels];
    fourier.wave=null;
}

// Real inverse-DFT synthesis. Coefficients are already divided by W*H.
// Paired bins contribute twice the real part; DC/Nyquist bins only once.
function addFourierGroup(group) {
    const d=fourier.data,w=d.width,h=d.height,c=d.channels;
    const contribution=new Float64Array(fourier.sum.length);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++) {
        const phase=2*Math.PI*(group.fx*x/w+group.fy*y/h);
        const cos=Math.cos(phase),sin=Math.sin(phase);
        for(let channel=0;channel<c;channel++) {
            const i=(y*w+x)*c+channel;
            contribution[i]=group.multiplier*(group.re[channel]*cos-group.im[channel]*sin);
            fourier.sum[i]+=contribution[i];
        }
    }
    fourier.wave=contribution;
}

function seekFourier(target) {
    if(!fourier.data)return;
    const d=fourier.data;target=Math.max(0,Math.min(d.groups.length,Math.trunc(target)));
    if(target<fourier.count)initializeFourierSum();
    while(fourier.count<target) {addFourierGroup(d.groups[fourier.count]);fourier.count++;}
    renderFourier();
}

function drawFourierImage(id, values, wave=false) {
    const d=fourier.data,canvas=$(id);canvas.width=d.width;canvas.height=d.height;
    const ctx=canvas.getContext("2d"),pixels=ctx.createImageData(d.width,d.height);
    let peak=0;
    if(wave && values)for(const v of values)peak=Math.max(peak,Math.abs(v));
    for(let i=0;i<d.width*d.height;i++) {
        for(let channel=0;channel<3;channel++) {
            const v=values ? values[i*d.channels+(d.channels===1?0:channel)] : 0;
            pixels.data[i*4+channel]=wave ? (peak>1e-10 ? Math.round(128+127*v/peak) : 128) : Math.round(Math.max(0,Math.min(255,v)));
        }
        pixels.data[i*4+3]=255;
    }
    ctx.putImageData(pixels,0,0);return pixels.data;
}

function spectrumColor(value) {
    const stops=[[8,7,21],[66,18,102],[158,47,127],[240,112,78],[252,248,184]];
    const position=Math.max(0,Math.min(1,value))*4,index=Math.min(3,Math.floor(position)),f=position-index;
    return `rgb(${stops[index].map((v,i)=>Math.round(v+(stops[index+1][i]-v)*f)).join(",")})`;
}

function drawFourierSpectrum() {
    const d=fourier.data,canvas=$("fourierSpectrum"),ctx=canvas.getContext("2d");
    canvas.width=Math.max(1,Math.round(384*d.width/Math.max(d.width,d.height)));
    canvas.height=Math.max(1,Math.round(384*d.height/Math.max(d.width,d.height)));
    const cw=canvas.width/d.width,ch=canvas.height/d.height;
    function coordinates(y,x){return [((x+Math.floor(d.width/2))%d.width)*cw,((y+Math.floor(d.height/2))%d.height)*ch];}
    for(let y=0;y<d.height;y++)for(let x=0;x<d.width;x++) {
        const [px,py]=coordinates(y,x);ctx.fillStyle=spectrumColor(d.spectrum[y*d.width+x]);ctx.fillRect(px,py,cw+0.2,ch+0.2);
    }
    function mark(y,x,latest=false){const [px,py]=coordinates(y,x);ctx.fillStyle="#aaf0d0";ctx.fillRect(px+cw*.35,py+ch*.35,Math.max(1,cw*.3),Math.max(1,ch*.3));if(latest){ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.strokeRect(px,py,cw,ch);}}
    mark(0,0);
    for(let i=0;i<fourier.count;i++){const g=d.groups[i];mark(g.y,g.x,i===fourier.count-1);mark(...g.partner,i===fourier.count-1);}
}

function renderFourier() {
    // Other tools own the shared toolbar while Fourier Canvas is hidden.
    if(state.activeTool!=="fourier")return;
    const d=fourier.data;
    $("resetBtn").disabled=!fourier.file;
    $("previewStatus").textContent=fourier.loading ? "Preparing Fourier components…" : fourier.error || (d ? `${d.width} × ${d.height} ${d.mode} sample · ${fourier.count===d.groups.length ? "reconstruction complete" : "ready to build"}` : "Upload any image to begin");
    for(const id of ["fourierPlay","fourierStep","fourierRestart","fourierComplete","fourierCount"])$(id).disabled=!d;
    $("fourierCount").max=d?d.groups.length:0;$("fourierCount").value=fourier.count;
    $("fourierCountLabel").textContent=d?`${fourier.count+1} / ${d.groups.length+1} (including DC)` : "Upload an image";
    state.outputUrl=null;$("downloadBtn").disabled=!d;
    if(!d)return;
    $("fourierStep").disabled=fourier.count===d.groups.length;
    if(fourier.timer===null)$("fourierPlay").textContent=fourier.count===d.groups.length?"Replay":"Play";
    $("fourierSampleLabel").textContent=`${d.width} × ${d.height} · ${d.channels===1?"grayscale":"RGB"}`;
    drawFourierImage("fourierOriginal",d.samples);
    const pixels=drawFourierImage("fourierResult",fourier.sum);
    drawFourierImage("fourierWave",fourier.wave,true);drawFourierSpectrum();
    let squared=0;
    for(let i=0;i<d.width*d.height;i++)for(let c=0;c<d.channels;c++)squared+=(pixels[i*4+c]-d.samples[i*d.channels+c])**2;
    const mse=squared/d.samples.length,psnr=mse===0?"∞":(10*Math.log10(255*255/mse)).toFixed(2);
    $("fourierMetrics").textContent=`MSE ${mse.toFixed(3)} · PSNR ${psnr} dB · compared with sampled original`;
    if(fourier.count) {
        const g=d.groups[fourier.count-1];
        const peak=fourier.wave.reduce((max,v)=>Math.max(max,Math.abs(v)),0);
        $("fourierWaveLabel").textContent=`Latest wave: fx=${g.fx}, fy=${g.fy} cycles/sample image · peak contribution ±${peak.toFixed(3)}. Contrast scaled for visibility.`;
        $("fourierDetail").textContent=`${fourier.count} wave groups added to average brightness. ${g.multiplier===2?"The latest conjugate pair contributes a real sinusoidal wave.":"The latest self-conjugate frequency is added once."}${d.channels===3?" RGB contributions are synthesized independently.":""}`;
    } else {
        $("fourierWaveLabel").textContent="No oscillating wave yet. Neutral gray means zero contribution.";
        $("fourierDetail").textContent=`Starting with average brightness (DC): ${d.dc.map(v=>v.toFixed(2)).join(", ")}${d.channels===3?" (R, G, B)":""}. Press Play or Step wave.`;
    }
    state.outputUrl=$("fourierResult").toDataURL("image/png");
}

$("fourierInput").addEventListener("change",event=>selectFourierFile(event.target.files[0]));
for(const id of ["fourierMode","fourierOrder","fourierSize"])$(id).addEventListener("change",prepareFourier);
$("fourierStep").addEventListener("click",()=>{pauseFourier();seekFourier(fourier.count+1);});
$("fourierRestart").addEventListener("click",()=>{pauseFourier();initializeFourierSum();renderFourier();});
$("fourierComplete").addEventListener("click",()=>{pauseFourier();seekFourier(fourier.data.groups.length);});
$("fourierCount").addEventListener("input",event=>{pauseFourier();seekFourier(Number(event.target.value));});
$("fourierPlay").addEventListener("click",()=>{
    if(fourier.timer!==null){pauseFourier();return;}
    if(fourier.count===fourier.data.groups.length)initializeFourierSum();
    function tick(){
        const speed=Number($("fourierSpeed").value),batch=Math.max(1,Math.ceil(speed/30));
        seekFourier(fourier.count+batch);
        if(fourier.count<fourier.data.groups.length){fourier.timer=setTimeout(tick,1000*batch/speed);$("fourierPlay").textContent="Pause";}
        else{pauseFourier();renderFourier();}
    }
    tick();
});
document.addEventListener("visibilitychange",()=>{if(document.hidden)pauseFourier();});
