import base64

import cv2
import numpy as np


def convert_to_grayscale(img_array):
    """
    Convert an RGB or RGBA image into a 2D grayscale image.

    app.py converts OpenCV's BGR/BGRA ordering into RGB/RGBA
    before calling this function.
    """
    img_array = np.asarray(img_array)

    # Image is already grayscale.
    if img_array.ndim == 2:
        return np.clip(img_array, 0, 255).astype(np.uint8)

    if img_array.ndim != 3:
        raise ValueError(
            'Image must be a 2D grayscale or 3D color array'
        )

    channels = img_array.shape[2]

    # Remove the alpha channel from RGBA.
    if channels == 4:
        img_array = img_array[..., :3]
        channels = 3

    # Handle an image stored as H x W x 1.
    if channels == 1:
        return np.clip(
            img_array[..., 0],
            0,
            255
        ).astype(np.uint8)

    if channels != 3:
        raise ValueError(
            'Only grayscale, RGB and RGBA images are supported'
        )

    rgb_float = img_array.astype(np.float32)

    # Weighted luminance formula:
    # Gray = 0.299R + 0.587G + 0.114B
    grayscale = np.dot(
        rgb_float[..., :3],
        np.array(
            [0.299, 0.587, 0.114],
            dtype=np.float32
        )
    )

    return np.clip(
        grayscale,
        0,
        255
    ).astype(np.uint8)


def array_to_base64(img_array):
    """
    Convert a NumPy image array into a PNG Base64 data URL.

    The processing functions use RGB/RGBA ordering, while
    OpenCV's PNG encoder expects BGR/BGRA ordering.
    """
    img_array = np.asarray(img_array)

    if img_array.ndim == 2:
        image_for_encoding = np.clip(
            img_array,
            0,
            255
        ).astype(np.uint8)

    elif img_array.ndim == 3:
        channels = img_array.shape[2]

        if channels == 1:
            image_for_encoding = np.clip(
                img_array[..., 0],
                0,
                255
            ).astype(np.uint8)

        elif channels == 3:
            image_for_encoding = cv2.cvtColor(
                np.clip(
                    img_array,
                    0,
                    255
                ).astype(np.uint8),
                cv2.COLOR_RGB2BGR
            )

        elif channels == 4:
            image_for_encoding = cv2.cvtColor(
                np.clip(
                    img_array,
                    0,
                    255
                ).astype(np.uint8),
                cv2.COLOR_RGBA2BGRA
            )

        else:
            raise ValueError(
                'Unsupported number of image channels'
            )

    else:
        raise ValueError(
            'Image must be a 2D or 3D NumPy array'
        )

    success, buffer = cv2.imencode(
        '.png',
        image_for_encoding
    )

    if not success:
        raise ValueError(
            'Unable to encode the processed image'
        )

    encoded_string = base64.b64encode(
        buffer
    ).decode('utf-8')

    return f'data:image/png;base64,{encoded_string}'


def validate_convolution_kernel(kernel):
    """
    Validate a kernel before performing spatial convolution.

    Kernel Painter uses odd square kernels from 3x3 to 31x31.
    """
    try:
        kernel_array = np.asarray(
            kernel,
            dtype=np.float32
        )
    except (TypeError, ValueError):
        raise ValueError(
            'Kernel must contain numeric values'
        )

    if kernel_array.ndim != 2:
        raise ValueError(
            'Kernel must be a two-dimensional matrix'
        )

    kernel_height, kernel_width = kernel_array.shape

    if kernel_height != kernel_width:
        raise ValueError(
            'Kernel must be square'
        )

    if kernel_height < 3 or kernel_height > 31:
        raise ValueError(
            'Kernel size must be between 3x3 and 31x31'
        )

    if kernel_height % 2 == 0:
        raise ValueError(
            'Kernel dimensions must be odd'
        )

    if not np.isfinite(kernel_array).all():
        raise ValueError(
            'Kernel contains a non-finite value'
        )

    if np.max(np.abs(kernel_array)) > 100:
        raise ValueError(
            'Kernel coefficients must remain between -100 and 100'
        )

    return kernel_array


def custom_2d_convolution(img_array, kernel):
    """
    Apply direct 2D spatial convolution without using
    cv2.filter2D or scipy.signal.convolve2d.

    The operation is vectorized across image pixels while
    looping over the kernel positions.
    """
    img_array = np.asarray(img_array)
    kernel = validate_convolution_kernel(kernel)

    kernel_height, kernel_width = kernel.shape

    # True convolution rotates the kernel by 180 degrees.
    flipped_kernel = np.flip(
        kernel,
        axis=(0, 1)
    )

    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    def process_channel(channel):
        """
        Apply the same 2D kernel to one image channel.
        """
        channel = channel.astype(np.float32)
        image_height, image_width = channel.shape

        padded_channel = np.pad(
            channel,
            (
                (padding_height, padding_height),
                (padding_width, padding_width)
            ),
            mode='reflect'
        )

        output = np.zeros(
            (image_height, image_width),
            dtype=np.float32
        )

        # Each loop selects a shifted image region,
        # multiplies it by one kernel coefficient,
        # and accumulates the result.
        for kernel_row in range(kernel_height):
            for kernel_column in range(kernel_width):
                image_region = padded_channel[
                    kernel_row:
                    kernel_row + image_height,
                    kernel_column:
                    kernel_column + image_width
                ]

                output += (
                    image_region *
                    flipped_kernel[
                        kernel_row,
                        kernel_column
                    ]
                )

        return output

    # Grayscale image
    if img_array.ndim == 2:
        output_array = process_channel(img_array)

    # Color image
    elif img_array.ndim == 3:
        channels = img_array.shape[2]

        if channels not in (1, 3, 4):
            raise ValueError(
                'Unsupported number of image channels'
            )

        # Remove alpha before convolution.
        if channels == 4:
            img_array = img_array[..., :3]
            channels = 3

        if channels == 1:
            output_array = process_channel(
                img_array[..., 0]
            )
        else:
            output_array = np.zeros(
                img_array.shape,
                dtype=np.float32
            )

            # Apply the same kernel independently
            # to red, green and blue channels.
            for channel_index in range(3):
                output_array[..., channel_index] = (
                    process_channel(
                        img_array[..., channel_index]
                    )
                )

    else:
        raise ValueError(
            'Image must be a 2D or 3D array'
        )

    # Convolution may produce negative values or values
    # greater than 255. Convert them to valid uint8 pixels.
    return np.clip(
        output_array,
        0,
        255
    ).astype(np.uint8)


def apply_custom_kernel(img_array, kernel):
    """
    Public function used by the /api/convolve route.

    The Visual Kernel Painter sends its painted matrix here.
    """
    return custom_2d_convolution(
        img_array,
        kernel
    )


EDGE_KERNELS = {
    'sobel': {
        'horizontal': np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ],
            dtype=np.float32
        ),
        'vertical': np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            dtype=np.float32
        )
    },

    'prewitt': {
        'horizontal': np.array(
            [
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]
            ],
            dtype=np.float32
        ),
        'vertical': np.array(
            [
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]
            ],
            dtype=np.float32
        )
    },

    'laplacian': {
        'horizontal': np.array(
            [
                [0, 0, 0],
                [1, -2, 1],
                [0, 0, 0]
            ],
            dtype=np.float32
        ),
        'vertical': np.array(
            [
                [0, 1, 0],
                [0, -2, 0],
                [0, 1, 0]
            ],
            dtype=np.float32
        )
    }
}


def _convolve_float(img_array, kernel):
    """
    Return the signed convolution response before
    absolute value and display normalization.
    """
    image = np.asarray(
        img_array,
        dtype=np.float32
    )

    kernel = np.asarray(
        kernel,
        dtype=np.float32
    )

    if image.ndim != 2:
        raise ValueError(
            'Edge convolution requires a grayscale image'
        )

    if kernel.ndim != 2:
        raise ValueError(
            'Edge kernel must be two-dimensional'
        )

    kernel_height, kernel_width = kernel.shape

    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    # Rotate kernel for true convolution.
    flipped_kernel = np.flip(
        kernel,
        axis=(0, 1)
    )

    padded_image = np.pad(
        image,
        (
            (padding_height, padding_height),
            (padding_width, padding_width)
        ),
        mode='reflect'
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    image_height, image_width = image.shape

    for kernel_row in range(kernel_height):
        for kernel_column in range(kernel_width):
            image_region = padded_image[
                kernel_row:
                kernel_row + image_height,
                kernel_column:
                kernel_column + image_width
            ]

            output += (
                image_region *
                flipped_kernel[
                    kernel_row,
                    kernel_column
                ]
            )

    return output


def _normalize_edge(response):
    """
    Convert a signed edge response into a visible uint8 image.
    """
    response = np.asarray(
        response,
        dtype=np.float32
    )

    strength = np.abs(response)

    maximum = float(
        strength.max()
    )

    if maximum <= 0:
        empty_image = np.zeros_like(
            strength,
            dtype=np.uint8
        )

        return empty_image, strength

    normalized = (
        strength *
        (255.0 / maximum)
    )

    normalized_image = np.clip(
        normalized,
        0,
        255
    ).astype(np.uint8)

    return normalized_image, strength


def detect_edges(img_array, operator='sobel'):
    """
    Generate horizontal, vertical and combined edge maps.
    """
    operator = str(operator).lower()

    if operator not in EDGE_KERNELS:
        raise ValueError(
            f'Unsupported edge operator: {operator}'
        )

    grayscale_image = convert_to_grayscale(
        img_array
    )

    kernels = EDGE_KERNELS[operator]

    horizontal_response = _convolve_float(
        grayscale_image,
        kernels['horizontal']
    )

    vertical_response = _convolve_float(
        grayscale_image,
        kernels['vertical']
    )

    # Overall gradient magnitude:
    # sqrt(G_horizontal^2 + G_vertical^2)
    combined_response = np.hypot(
        horizontal_response,
        vertical_response
    )

    maps = {}

    responses = (
        ('horizontal', horizontal_response),
        ('vertical', vertical_response),
        ('combined', combined_response)
    )

    for name, response in responses:
        image, strength = _normalize_edge(
            response
        )

        maps[name] = {
            'image': image,

            'mean_strength': round(
                float(strength.mean()),
                2
            ),

            'max_strength': round(
                float(strength.max()),
                2
            ),

            # Pixels with normalized intensity above 32
            # are counted as edge pixels.
            'edge_pixels': round(
                float(
                    (image > 32).mean() * 100
                ),
                2
            )
        }

    return maps


# ============================================================
# Feature 4: Visual Kernel Laboratory
# ============================================================


def _classify_kernel(kernel, tolerance=0.02):
    """Classify a kernel from its coefficient pattern and DC gain."""
    values = kernel.ravel()
    kernel_sum = float(values.sum())
    center = kernel.shape[0] // 2

    nonzero_count = int(
        np.count_nonzero(np.abs(values) > 1e-8)
    )

    has_positive = bool(np.any(values > 1e-8))
    has_negative = bool(np.any(values < -1e-8))

    if nonzero_count == 0:
        return 'Zero filter'

    if (
        nonzero_count == 1 and
        abs(float(kernel[center, center]) - 1.0) < 1e-8
    ):
        return 'Identity'

    if (
        has_positive and
        not has_negative and
        abs(kernel_sum - 1.0) < tolerance
    ):
        return 'Blur / low-pass'

    if (
        has_positive and
        has_negative and
        abs(kernel_sum) < tolerance
    ):
        return 'Edge / high-pass'

    if (
        has_positive and
        has_negative and
        abs(kernel_sum - 1.0) < tolerance
    ):
        return 'Sharpening'

    return 'Custom spatial filter'


def _kernel_frequency_response(kernel, spectrum_size=256):
    """Create a centered logarithmic magnitude response for display."""
    spectrum_size = int(spectrum_size)

    if spectrum_size < kernel.shape[0]:
        spectrum_size = kernel.shape[0]

    # Put the kernel center at spatial coordinate (0, 0) before FFT.
    padded_kernel = np.zeros(
        (spectrum_size, spectrum_size),
        dtype=np.float32
    )

    kernel_height, kernel_width = kernel.shape
    padded_kernel[:kernel_height, :kernel_width] = kernel

    padded_kernel = np.roll(
        padded_kernel,
        shift=(
            -(kernel_height // 2),
            -(kernel_width // 2)
        ),
        axis=(0, 1)
    )

    frequency_response = np.fft.fft2(
        padded_kernel
    )

    frequency_response = np.fft.fftshift(
        frequency_response
    )

    magnitude = np.log1p(
        np.abs(frequency_response)
    )

    maximum = float(magnitude.max())

    if maximum <= 1e-12:
        return np.zeros_like(
            magnitude,
            dtype=np.uint8
        )

    return np.clip(
        magnitude * (255.0 / maximum),
        0,
        255
    ).astype(np.uint8)


def _separable_components(kernel, tolerance=1e-5):
    """Use SVD to determine whether a 2D kernel has rank one."""
    left, singular_values, right = np.linalg.svd(
        kernel.astype(np.float64),
        full_matrices=False
    )

    if singular_values[0] <= 1e-12:
        zero_vector = [0.0] * kernel.shape[0]
        return True, zero_vector, zero_vector.copy()

    residual = float(
        np.linalg.norm(singular_values[1:]) /
        singular_values[0]
    )

    if residual > tolerance:
        return False, None, None

    scale = np.sqrt(singular_values[0])

    vertical = (
        left[:, 0] * scale
    ).astype(np.float64)

    horizontal = (
        right[0, :] * scale
    ).astype(np.float64)

    return (
        True,
        np.round(vertical, 6).tolist(),
        np.round(horizontal, 6).tolist()
    )


def analyze_kernel(kernel, spectrum_size=256):
    """
    Analyze a painted kernel for the Visual Kernel Laboratory.

    Returns its spatial classification, DC gain, 180-degree
    symmetry, separability and display-ready frequency response.
    """
    kernel = validate_convolution_kernel(kernel)

    kernel_sum = float(kernel.sum())

    symmetric = bool(
        np.allclose(
            kernel,
            np.flip(kernel, axis=(0, 1)),
            atol=1e-6,
            rtol=0.0
        )
    )

    (
        separable,
        vertical_vector,
        horizontal_vector
    ) = _separable_components(kernel)

    return {
        'kernel_sum': round(kernel_sum, 6),
        'dc_gain': round(kernel_sum, 6),
        'filter_type': _classify_kernel(kernel),
        'symmetric': symmetric,
        'separable': separable,
        'vertical_vector': vertical_vector,
        'horizontal_vector': horizontal_vector,
        'frequency_response': _kernel_frequency_response(
            kernel,
            spectrum_size
        )
    }


# ============================================================
# Feature 5: Reverse Blur / Image Restoration
# ============================================================


def _psf_to_otf(kernel, output_shape):
    """Convert a centered spatial blur kernel (PSF) into an OTF."""
    kernel_height, kernel_width = kernel.shape
    output_height, output_width = output_shape

    if (
        kernel_height > output_height or
        kernel_width > output_width
    ):
        raise ValueError(
            'The blur kernel cannot be larger than the image'
        )

    padded_kernel = np.zeros(
        output_shape,
        dtype=np.float32
    )

    padded_kernel[:kernel_height, :kernel_width] = kernel

    padded_kernel = np.roll(
        padded_kernel,
        shift=(
            -(kernel_height // 2),
            -(kernel_width // 2)
        ),
        axis=(0, 1)
    )

    return np.fft.fft2(padded_kernel)


def wiener_deconvolution(img_array, blur_kernel, balance=0.01):
    """
    Approximately reverse a known blur using Wiener deconvolution.

    balance controls the trade-off between inverse restoration and
    noise amplification. A larger value produces a more stable but
    less aggressive restoration.
    """
    image = np.asarray(img_array)
    kernel = validate_convolution_kernel(blur_kernel)

    try:
        balance = float(balance)
    except (TypeError, ValueError):
        raise ValueError('Wiener balance must be numeric')

    if not np.isfinite(balance) or balance <= 0:
        raise ValueError(
            'Wiener balance must be a positive finite number'
        )

    if np.any(kernel < -1e-8):
        raise ValueError(
            'A restoration blur kernel cannot contain negative values'
        )

    kernel_sum = float(kernel.sum())

    if kernel_sum <= 1e-8:
        raise ValueError(
            'A restoration blur kernel must have a positive sum'
        )

    kernel = kernel / kernel_sum

    if image.ndim == 2:
        working_image = image[..., np.newaxis]
        return_grayscale = True

    elif image.ndim == 3:
        if image.shape[2] == 4:
            image = image[..., :3]

        if image.shape[2] not in (1, 3):
            raise ValueError(
                'Restoration supports grayscale, RGB or RGBA images'
            )

        working_image = image
        return_grayscale = image.shape[2] == 1

    else:
        raise ValueError(
            'Image must be a 2D or 3D array'
        )

    working_image = (
        working_image.astype(np.float32) / 255.0
    )

    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2

    padded_image = np.pad(
        working_image,
        (
            (pad_height, pad_height),
            (pad_width, pad_width),
            (0, 0)
        ),
        mode='reflect'
    )

    image_shape = padded_image.shape[:2]
    optical_transfer = _psf_to_otf(
        kernel,
        image_shape
    )

    wiener_filter = (
        np.conj(optical_transfer) /
        (np.abs(optical_transfer) ** 2 + balance)
    )

    restored = np.empty_like(
        padded_image,
        dtype=np.float32
    )

    for channel_index in range(padded_image.shape[2]):
        blurred_spectrum = np.fft.fft2(
            padded_image[..., channel_index]
        )

        restored[..., channel_index] = np.real(
            np.fft.ifft2(
                blurred_spectrum * wiener_filter
            )
        )

    restored = restored[
        pad_height:
        pad_height + working_image.shape[0],
        pad_width:
        pad_width + working_image.shape[1]
    ]

    restored = np.clip(
        restored * 255.0,
        0,
        255
    ).astype(np.uint8)

    if return_grayscale:
        return restored[..., 0]

    return restored
