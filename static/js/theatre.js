const theatre = { size: 12, signal: null, kernel: null, output: [], index: 0, timer: null, revision: 0, key: null, file: null };

function theatrePause() {
    clearTimeout(theatre.timer);
    theatre.timer = null;
    $("theatrePlay").textContent = "Play";
}

function theatreReflect(index, length) {
    while (index < 0 || index >= length) index = index < 0 ? -index : 2 * length - 2 - index;
    return index;
}

// Signed floating-point sum before clipping. No image-filter library is used.
function theatreCalculate(signal, kernel, row, column) {
    const n = kernel.length, radius = Math.floor(n / 2);
    let sum = 0;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const input = signal[theatreReflect(row + y - radius, signal.length)][theatreReflect(column + x - radius, signal[0].length)];
        const weight = kernel[n - 1 - y][n - 1 - x];
        const product = input * weight;
        sum += product;
    }
    return {sum, value: Math.trunc(Math.max(0, Math.min(255, sum)))};
}

function theatreDrawGrid(id, values, current = -1) {
    const canvas = $(id), ctx = canvas.getContext("2d"), cell = canvas.width / theatre.size;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < 144; i++) {
        const v = values[i], x = (i % 12) * cell, y = Math.floor(i / 12) * cell;
        ctx.fillStyle = v === undefined ? "#172235" : `rgb(${v},${v},${v})`;
        ctx.fillRect(x, y, cell, cell);
        ctx.strokeStyle = "rgba(110,145,155,.3)";
        ctx.lineWidth = 1; ctx.strokeRect(x, y, cell, cell);
    }
    if (current >= 0) {
        const x = current % 12, y = Math.floor(current / 12);
        if (id === "theatreInput") {
            const radius = Math.floor(theatre.kernel.length / 2);
            ctx.fillStyle = "rgba(170,240,208,.18)";
            ctx.fillRect((x-radius)*cell, (y-radius)*cell, theatre.kernel.length*cell, theatre.kernel.length*cell);
            ctx.strokeStyle = "#d69a59"; ctx.lineWidth = 3;
            ctx.strokeRect((x-radius)*cell, (y-radius)*cell, theatre.kernel.length*cell, theatre.kernel.length*cell);
        }
        ctx.strokeStyle = "#aaf0d0"; ctx.lineWidth = 3;
        ctx.strokeRect(x*cell+2, y*cell+2, cell-4, cell-4);
    }
}

function theatreStep() {
    if (!theatre.signal || !theatre.kernel || theatre.index >= 144) return;
    const index = theatre.index, row = Math.floor(index/12), column = index%12;
    const result = theatreCalculate(theatre.signal, theatre.kernel, row, column);
    theatre.output[index] = result.value;
    theatreDrawGrid("theatreInput", theatre.signal.flat(), index);
    theatreDrawGrid("theatreOutput", theatre.output, index);
    theatre.index++;
    $("theatreProgress").value = theatre.index;
    $("theatrePosition").textContent = `${theatre.index} / 144 pixels${theatre.index === 144 ? " · complete" : ""}`;
    if (theatre.index === 144) { theatrePause(); $("theatrePlay").textContent="Replay"; $("theatreStep").disabled=true; }
}

function theatreRestart() {
    theatrePause(); theatre.index=0; theatre.output=[];
    $("theatreProgress").value=0; $("theatrePosition").textContent="0 / 144 pixels";
    theatreDrawGrid("theatreInput", theatre.signal ? theatre.signal.flat() : []);
    theatreDrawGrid("theatreOutput", []);
    $("theatreStep").disabled=!theatre.signal;
}

async function syncTheatre() {
    const active = state.activeTool === "convolution";
    $("convolutionTheatre").classList.toggle("hidden", !active);
    theatrePause();
    if (!active) { theatre.revision++; return; }
    const preset = $("presetSelect").value;
    const allowed = preset !== "custom";
    for (const id of ["theatrePlay","theatreStep","theatreRestart"]) $(id).disabled=true;
    const key = preset + JSON.stringify(state.kernelMatrix);
    const revision = ++theatre.revision;
    if (!allowed) {
        theatre.signal=null; theatre.kernel=null; theatre.key=null;
        theatreRestart();
        $("theatreStatus").textContent="Choose Identity, Gaussian, Average or Laplacian sharpen. Custom painted kernels are not animated.";
        return;
    }
    if (key === theatre.key && state.convFile === theatre.file && theatre.signal) {
        for (const id of ["theatrePlay","theatreStep","theatreRestart"]) $(id).disabled=false;
        $("theatreStep").disabled=theatre.index>=144;
        return;
    }
    theatre.signal=null; theatreRestart();
    $("theatreStatus").textContent="Preparing teaching sample…";
    try {
        let signal;
        if (state.convFile) {
            const bitmap = await createImageBitmap(state.convFile);
            const canvas=document.createElement("canvas"); canvas.width=canvas.height=12;
            const ctx=canvas.getContext("2d"); ctx.drawImage(bitmap,0,0,12,12); bitmap.close();
            const pixels=ctx.getImageData(0,0,12,12).data;
            signal=Array.from({length:12},(_,y)=>Array.from({length:12},(_,x)=>{
                const i=(y*12+x)*4; return Math.trunc(.299*pixels[i]+.587*pixels[i+1]+.114*pixels[i+2]);
            }));
        } else {
            signal=Array.from({length:12},(_,y)=>Array.from({length:12},(_,x)=>
                x>=3&&x<=8&&y>=3&&y<=8 ? 210 : 25+x*5));
        }
        if (revision !== theatre.revision) return;
        theatre.signal=signal; theatre.kernel=state.kernelMatrix.map(r=>[...r]);
        theatre.key=key; theatre.file=state.convFile;
        theatreRestart();
        $("theatreStatus").textContent=`${$("presetSelect").selectedOptions[0].textContent} · ${theatre.kernel.length} × ${theatre.kernel.length} · ${state.convFile ? "resized grayscale upload" : "built-in test signal"}`;
        for (const id of ["theatrePlay","theatreStep","theatreRestart"]) $(id).disabled=false;
    } catch (error) {
        if (revision===theatre.revision) $("theatreStatus").textContent="Unable to prepare the uploaded image. Try another image or reset to use the test signal.";
    }
}

$("theatrePlay").addEventListener("click",()=>{
    if (theatre.timer !== null) { theatrePause(); return; }
    if (theatre.index>=144) theatreRestart();
    $("theatrePlay").textContent="Pause";
    function tick() {
        theatreStep();
        if (theatre.index<144) theatre.timer=setTimeout(tick,1000/Number($("theatreSpeed").value));
    }
    tick();
});
$("theatreStep").addEventListener("click",()=>{theatrePause();theatreStep();});
$("theatreRestart").addEventListener("click",theatreRestart);
document.addEventListener("visibilitychange",()=>{if(document.hidden)theatrePause();});
syncTheatre();
