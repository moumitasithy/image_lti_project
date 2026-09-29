# EYE_Maze 👁️

**An interactive web studio for 2D signal processing and LTI spatial filtering.**

EYE_Maze treats an image as what it really is — a two-dimensional discrete signal — and lets you experiment with it. Upload an image, paint your own convolution kernel and watch the output change, inspect directional edges, study a kernel's frequency response, reverse a blur, clean noisy images, and rebuild a picture wave by wave from its Fourier components.

The app is built with Python and Flask. All of the core filtering (2D convolution, edge detection, median/mean filtering) is implemented by hand with NumPy array operations rather than library filter calls such as `cv2.filter2D` or `scipy.signal.convolve2d`, so the code itself shows how each operation works.

---

## 📚 Course Information

| | |
|---|---|
| **Course** | CSE 220 — Signals (Level-2, Term-2) |
| **Project type** | Course final project |
| **Group** | Capybara |
| **Advisor** | Ashrafur Rahman |

### Team Members

| Name | Student ID |
|---|---|
| Moumita Hassan Sithy | 2305127 |
| Surya Ahmed Lara | 2305128 |

---

## ✨ Features

### 1. Instant Grayscale Converter
One-click conversion from RGB/RGBA to a single-channel grayscale signal using the weighted luminance formula

```
Gray = 0.299·R + 0.587·G + 0.114·B
```

This reduces a color image to one 2D signal, which simplifies analysis and makes edge maps easier to read.

### 2. Interactive Blur & Sharpening Suite (Visual Kernel Painter)
- Edit a convolution kernel in real time (odd square sizes from 3×3 up to 31×31) and instantly see the filtered output.
- Build Gaussian/average blur, Laplacian sharpening, or any custom kernel.
- Convolution is implemented from scratch: the kernel is flipped 180° (true convolution), the image is reflect-padded, and the output is accumulated as a sum of shifted, weighted copies of the input:

```
y[m, n] = Σ Σ h[i, j] · x[m − i, n − j]
```

### 3. Multi-directional Edge Detector
- Detect **horizontal**, **vertical** and **combined** edges using **Sobel**, **Prewitt** or **Laplacian** operators.
- Kernel size is adjustable (3×3 to 31×31). Larger kernels are built by combining the base derivative stencil with binomial (Sobel/Laplacian) or uniform (Prewitt) smoothing.
- Combined gradient magnitude for Sobel/Prewitt: `|G| = √(Gx² + Gy²)`; for the Laplacian the two second-derivative responses are summed.
- Side-by-side edge maps with strength metrics: mean strength, maximum strength and the percentage of edge pixels.
- The exact kernel coefficients being used can be displayed alongside the result.

### 4. Visual Kernel Laboratory
Analyze any kernel as an LTI system:
- **Automatic classification** — identity, blur/low-pass, edge/high-pass, sharpening or custom, based on the coefficient pattern and DC gain.
- **DC gain** (sum of coefficients), which tells you whether the filter preserves overall brightness.
- **180° symmetry** check.
- **Separability** test using SVD — if the kernel has rank one, it is split into a vertical and a horizontal 1D vector.
- **Frequency response** — a centered, log-scaled magnitude spectrum of the kernel (via zero-padded 2D FFT), showing which spatial frequencies it passes or suppresses.

### 5. Reverse Blur / Image Restoration
Undo a known blur with **Wiener deconvolution** in the frequency domain:

```
W(u, v) = H*(u, v) / ( |H(u, v)|² + K )
```

- Blur models: Gaussian (size + σ), linear motion (size + angle) or a custom non-negative kernel.
- The balance parameter **K** controls the trade-off between sharp restoration and noise amplification.

### 6. Noise Cleaner & Restorer
- **Add noise:** salt-and-pepper (impulse) noise or additive Gaussian noise (generated with the Box–Muller transform). A seed makes results repeatable.
- **Clean noise:** median filter (non-linear, rank-based) or mean filter (a normalized box kernel, which is a linear low-pass LTI filter), with odd window sizes and reflect/replicate border handling.
- **Quality metrics:** MSE and PSNR are reported when noise is added, comparing the noisy image with the clean original.

```
MSE  = (1/MN) Σ (f − g)²
PSNR = 10 · log₁₀(255² / MSE)  dB
```

### 7. Fourier Canvas
- Decomposes a downsampled image (32×32 or 64×64, grayscale or color) into its 2D Fourier components.
- Conjugate-symmetric frequency pairs are grouped into real-valued 2D waves, so the image can be rebuilt progressively — either from the **lowest frequencies first** or from the **strongest components first**.
- Shows the log-magnitude spectrum, making it clear how much of an image lives in its low frequencies.

---

## 🧠 Signal Processing Concepts Demonstrated

- Images as 2D discrete signals
- Linear Time-Invariant (shift-invariant) systems and impulse response
- 2D spatial convolution and kernel flipping
- Border handling (padding) in finite-length signals
- Low-pass vs. high-pass filtering
- Gradient (first-derivative) and Laplacian (second-derivative) edge detection
- Separable filters and rank-one decomposition
- 2D Discrete Fourier Transform and frequency response
- Deconvolution and the Wiener filter
- Linear vs. non-linear filtering (mean vs. median)
- Noise models and objective quality metrics (MSE, PSNR)

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Signal processing:** NumPy (convolution, FFT, SVD, filtering)
- **Image I/O:** OpenCV (decoding, encoding, color-order conversion and resizing only)
- **Frontend:** HTML, CSS, JavaScript (`templates/index.html`)

---

## 📁 Project Structure

```
EYE_Maze/
├── app.py              # Flask server, API routes and input validation
├── dsp_engine.py       # All signal-processing logic
├── templates/
│   └── index.html      # Web interface
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or newer
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/EYE_Maze.git
cd EYE_Maze

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main web interface |
| `/api/grayscale` | POST | Convert an uploaded image to grayscale |
| `/api/convolve` | POST | Apply a custom kernel to an image |
| `/api/edge-kernels` | POST | Return the kernel coefficients for an edge operator and size |
| `/api/edges` | POST | Horizontal, vertical and combined edge maps with metrics |
| `/api/kernel-analysis` | POST | Kernel classification, DC gain, symmetry, separability and frequency response |
| `/api/deconvolve` | POST | Wiener deconvolution with a Gaussian, motion or custom blur model |
| `/api/noise-cleaner` | POST | Add noise or clean an image with median/mean filtering |
| `/api/fourier-canvas` | POST | Fourier components for progressive image reconstruction |

All image endpoints accept `multipart/form-data` with an `image` field. Kernels are sent as a JSON-encoded 2D array.

---

## 🧪 Usage Tips

- Start with **Grayscale** to see an image as a single 2D signal.
- In the **Kernel Painter**, try a 3×3 averaging kernel (all values `1/9`), then a sharpening kernel such as `[[0,-1,0],[-1,5,-1],[0,-1,0]]`, and compare them in the **Kernel Laboratory** — one has a low-pass spectrum, the other boosts high frequencies.
- In **Edge Detection**, compare Sobel and Laplacian on the same image to see first- vs. second-derivative behavior.
- In **Noise Cleaner**, add salt-and-pepper noise and compare a median filter with a mean filter — the median removes impulses while keeping edges much sharper.
- In **Restoration**, a very small balance value sharpens aggressively but amplifies noise; increase it gradually to find a stable result.

---

## 🙏 Acknowledgements

We thank our advisor, **Ashrafur Rahman**, for his guidance throughout this project.

---

*Developed as the final project for CSE 220 (Signals), Level-2 Term-2.*
