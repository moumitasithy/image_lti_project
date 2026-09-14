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