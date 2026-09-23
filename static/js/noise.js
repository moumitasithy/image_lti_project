// Keep this experiment separate from the existing Wiener deblurring workspace.
let noiseFile = null;
let noiseSource = null;
let noiseResult = null;
let noiseRequest = null;

function cancelNoiseRequest() {
    if (noiseRequest) noiseRequest.abort();
    noiseRequest = null;
    $("noiseRun").disabled = false;
    $("noiseRun").textContent = $("noiseOperation").value === "add" ? "Add Noise" : "Clean Noise";
}

function setNoiseImage(name, source) {
    const img = $("noise" + name);
    img.hidden = !source;
    if (source) img.src = source;
    else img.removeAttribute("src");
    $("noise" + name + "Empty").hidden = Boolean(source);
}

function refreshNoisePreview() {
    setNoiseImage("Original", noiseResult?.original_image || noiseSource);
    const adding = $("noiseOperation").value === "add";
    setNoiseImage("Output", noiseResult?.output_image);
    $("noiseOutputLabel").textContent = adding ? "Noise added" : "Cleaned image";
    $("noiseMetricsTable").classList.toggle("hidden", !adding);
    const metrics = noiseResult?.metrics;
    $("noiseMse").textContent = metrics ? metrics.mse.toFixed(2) : "—";
    $("noisePsnr").textContent = metrics ? (metrics.psnr === null ? "∞ dB (identical)" : metrics.psnr.toFixed(2) + " dB") : "—";
    $("noiseSummary").textContent = adding
        ? "MSE and PSNR show how much the added noise changed your uploaded image."
        : "Cleaning quality cannot be measured with MSE/PSNR without a clean reference image. Compare the uploaded and cleaned images visually.";
    if (state.activeTool !== "noise") return;
    state.outputUrl = noiseResult?.output_image || null;
    $("downloadBtn").disabled = !state.outputUrl;
    $("resetBtn").disabled = !noiseFile;
    $("previewStatus").textContent = noiseResult ? (adding ? "Noise added" : "Noise cleaning complete")
        : noiseFile ? "Choose settings, then press " + (adding ? "Add Noise" : "Clean Noise")
        : adding ? "Upload an image to add noise" : "Upload a noisy image to clean";
}

function selectNoiseFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) { showToast("Please select an image file.", true); return; }
    cancelNoiseRequest();
    if (noiseSource) URL.revokeObjectURL(noiseSource);
    noiseFile = file;
    noiseSource = URL.createObjectURL(file);
    noiseResult = null;
    $("noiseFileName").textContent = file.name;
    refreshNoisePreview();
}

function resetNoise() {
    cancelNoiseRequest();
    if (noiseSource) URL.revokeObjectURL(noiseSource);
    noiseFile = noiseSource = noiseResult = null;
    $("noiseInput").value = "";
    $("noiseFileName").textContent = "";
    refreshNoisePreview();
}

function updateNoiseControls() {
    const adding = $("noiseOperation").value === "add";
    document.querySelectorAll("[data-noise-mode]").forEach(element => {
        element.classList.toggle("hidden", element.dataset.noiseMode !== $("noiseOperation").value);
    });
    $("noiseUploadLabel").textContent = adding ? "Upload an image" : "Upload a noisy image";
    const gaussian = $("noiseType").value === "gaussian";
    $("noiseAmountControl").classList.toggle("hidden", !adding || gaussian);
    $("noiseSigmaControl").classList.toggle("hidden", !adding || !gaussian);
    $("noiseAmountValue").textContent = $("noiseAmount").value + "%";
    $("noiseSigmaValue").textContent = $("noiseSigma").value;
    $("noiseWindowValue").textContent = `${$("noiseWindow").value} × ${$("noiseWindow").value}`;
    cancelNoiseRequest();
    noiseResult = null;
    refreshNoisePreview();
}

async function runNoiseCleaner() {
    if (!noiseFile) { showToast("Upload an image first.", true); return; }
    const adding = $("noiseOperation").value === "add";
    if (adding && (!$("noiseSeed").value || !$("noiseSeed").reportValidity())) return;
    cancelNoiseRequest();
    const controller = new AbortController();
    noiseRequest = controller;
    $("noiseRun").disabled = true;
    $("noiseRun").textContent = adding ? "Adding noise…" : "Cleaning…";
    $("previewStatus").textContent = adding ? "Adding noise…" : "Cleaning the uploaded image…";
    $("downloadBtn").disabled = true;
    const form = new FormData();
    form.append("image", noiseFile);
    for (const [key, value] of Object.entries({operation: $("noiseOperation").value, noise_type: $("noiseType").value,
        amount: Number($("noiseAmount").value) / 100, sigma: $("noiseSigma").value,
        filter_type: $("noiseFilter").value, window_size: $("noiseWindow").value,
        seed: $("noiseSeed").value, border: $("noiseBorder").value})) form.append(key, value);
    try {
        const response = await fetch("/api/noise-cleaner", {method: "POST", body: form, signal: controller.signal});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Noise cleaning failed");
        if (noiseRequest !== controller || state.activeTool !== "noise") return;
        noiseResult = data;
        refreshNoisePreview();
    } catch (error) {
        if (error.name !== "AbortError" && noiseRequest === controller) {
            noiseResult = null;
            refreshNoisePreview();
            $("previewStatus").textContent = "Noise cleaning failed";
            showToast(error.message, true);
        }
    } finally {
        if (noiseRequest === controller) cancelNoiseRequest();
    }
}

$("noiseInput").addEventListener("change", event => selectNoiseFile(event.target.files[0]));
$("noiseRun").addEventListener("click", runNoiseCleaner);
for (const id of ["noiseOperation", "noiseType", "noiseAmount", "noiseSigma", "noiseFilter", "noiseWindow", "noiseSeed", "noiseBorder"]) {
    $(id).addEventListener("input", updateNoiseControls);
}

updateNoiseControls();
