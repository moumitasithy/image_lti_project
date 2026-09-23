from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import json
import dsp_engine


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def decode_uploaded_image(file, mode=cv2.IMREAD_UNCHANGED):
    """
    Convert an uploaded image file into an OpenCV NumPy array.
    Raises ValueError if the uploaded data is not a valid image.
    """
    file_bytes = file.read()

    if not file_bytes:
        raise ValueError('The uploaded image is empty')

    np_image = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_image, mode)

    if image is None:
        raise ValueError('Unable to decode the uploaded image')

    return image


def convert_opencv_to_rgb(image):
    """
    OpenCV loads images in BGR/BGRA order.
    Convert them to RGB/RGBA before sending them to dsp_engine.
    """
    if len(image.shape) != 3:
        return image

    channels = image.shape[2]

    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    return image


def validate_kernel(kernel_string):
    """
    Parse and validate a kernel received from the Visual Kernel Painter.
    Only odd square kernels from 3x3 to 31x31 are accepted.
    """
    if not kernel_string:
        raise ValueError('Kernel matrix is required')

    try:
        kernel_data = json.loads(kernel_string)
        kernel = np.asarray(kernel_data, dtype=np.float32)
    except (json.JSONDecodeError, TypeError, ValueError):
        raise ValueError('Kernel must be a valid numeric matrix')

    if kernel.ndim != 2:
        raise ValueError('Kernel must be a two-dimensional matrix')

    rows, columns = kernel.shape

    if rows != columns:
        raise ValueError('Kernel must be square')

    if rows < 3 or rows > 31:
        raise ValueError(
            'Kernel size must be between 3x3 and 31x31'
        )

    if rows % 2 == 0:
        raise ValueError(
            'Kernel size must be odd, such as 3x3, 5x5 or 7x7'
        )

    if not np.isfinite(kernel).all():
        raise ValueError(
            'Kernel contains an invalid or non-finite value'
        )

    if np.max(np.abs(kernel)) > 100:
        raise ValueError(
            'Kernel coefficients must remain between -100 and 100'
        )

    return kernel


# ---------------------------------------------------------------------------
# Helpers used only by Feature 4 and Feature 5
# ---------------------------------------------------------------------------


def parse_float_field(name, default, minimum, maximum):
    """Read and validate one finite floating-point form value."""
    raw_value = request.form.get(name, str(default))

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a number')

    if not np.isfinite(value):
        raise ValueError(f'{name} must be finite')

    if value < minimum or value > maximum:
        raise ValueError(
            f'{name} must be between {minimum} and {maximum}'
        )

    return value


def parse_odd_size(name='kernel_size', default=9):
    """Read an odd kernel size between 3 and 31."""
    raw_value = request.form.get(name, str(default))

    try:
        size = int(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be an integer')

    if size < 3 or size > 31 or size % 2 == 0:
        raise ValueError(
            f'{name} must be an odd integer between 3 and 31'
        )

    return size


def create_gaussian_blur_kernel(size, sigma):
    """Create a normalized two-dimensional Gaussian point-spread function."""
    coordinates = np.arange(size, dtype=np.float32) - size // 2
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)

    kernel = np.exp(
        -(x_grid ** 2 + y_grid ** 2) /
        (2.0 * sigma ** 2)
    )

    kernel_sum = float(kernel.sum())

    if kernel_sum <= 0:
        raise ValueError('Unable to create the Gaussian blur kernel')

    return (kernel / kernel_sum).astype(np.float32)


def create_motion_blur_kernel(size, angle):
    """Create a normalized linear-motion point-spread function."""
    mask = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radius = center
    radians = np.deg2rad(angle)

    delta_x = int(round(np.cos(radians) * radius))
    delta_y = int(round(np.sin(radians) * radius))

    start = (center - delta_x, center - delta_y)
    end = (center + delta_x, center + delta_y)

    cv2.line(mask, start, end, color=255, thickness=1)

    kernel = mask.astype(np.float32) / 255.0
    kernel_sum = float(kernel.sum())

    if kernel_sum <= 0:
        raise ValueError('Unable to create the motion blur kernel')

    return kernel / kernel_sum


def get_restoration_kernel():
    """Build or validate the blur model selected in Feature 5."""
    kernel_type = request.form.get(
        'kernel_type',
        'gaussian'
    ).lower()

    if kernel_type == 'custom':
        kernel = validate_kernel(
            request.form.get('kernel')
        )

        # A blur point-spread function should not contain negative weights.
        if np.any(kernel < -1e-8):
            raise ValueError(
                'A custom restoration kernel cannot contain negative weights'
            )

        kernel_sum = float(kernel.sum())

        if abs(kernel_sum) < 1e-8:
            raise ValueError(
                'A custom restoration kernel must have a non-zero sum'
            )

        # Normalize defensively so small rounding errors do not change brightness.
        return kernel_type, (kernel / kernel_sum).astype(np.float32)

    size = parse_odd_size()

    if kernel_type == 'gaussian':
        sigma = parse_float_field(
            'sigma',
            default=2.0,
            minimum=0.1,
            maximum=20.0
        )

        return kernel_type, create_gaussian_blur_kernel(
            size,
            sigma
        )

    if kernel_type == 'motion':
        angle = parse_float_field(
            'angle',
            default=0.0,
            minimum=0.0,
            maximum=180.0
        )

        return kernel_type, create_motion_blur_kernel(
            size,
            angle
        )

    raise ValueError(
        'kernel_type must be gaussian, motion or custom'
    )


@app.route('/api/grayscale', methods=['POST'])
def handle_grayscale():
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image uploaded'
            }), 400

        image = decode_uploaded_image(
            request.files['image'],
            cv2.IMREAD_UNCHANGED
        )

        # convert_to_grayscale() expects RGB/RGBA ordering.
        image = convert_opencv_to_rgb(image)

        grayscale_image = dsp_engine.convert_to_grayscale(image)

        base64_string = dsp_engine.array_to_base64(
            grayscale_image
        )

        return jsonify({
            'grayscale_image': base64_string
        })

    except ValueError as error:
        return jsonify({
            'error': str(error)
        }), 400

    except Exception as error:
        app.logger.exception('Grayscale processing failed')

        return jsonify({
            'error': str(error)
        }), 500


@app.route('/api/convolve', methods=['POST'])
def handle_convolve():
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image uploaded'
            }), 400

        # Receive and validate the matrix generated by Kernel Painter.
        kernel = validate_kernel(
            request.form.get('kernel')
        )

        image = decode_uploaded_image(
            request.files['image'],
            cv2.IMREAD_COLOR
        )

        # IMREAD_COLOR produces a three-channel BGR image.
        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        output_image = dsp_engine.apply_custom_kernel(
            image_rgb,
            kernel
        )

        base64_string = dsp_engine.array_to_base64(
            output_image
        )

        return jsonify({
            'output_image': base64_string
        })

    except ValueError as error:
        return jsonify({
            'error': str(error)
        }), 400

    except Exception as error:
        app.logger.exception('Convolution processing failed')

        return jsonify({
            'error': str(error)
        }), 500


@app.route('/api/edges', methods=['POST'])
def handle_edges():
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image uploaded'
            }), 400

        operator = request.form.get(
            'operator',
            'sobel'
        ).lower()

        allowed_operators = {
            'sobel',
            'prewitt',
            'laplacian'
        }

        if operator not in allowed_operators:
            return jsonify({
                'error': 'Unsupported edge operator'
            }), 400

        image = decode_uploaded_image(
            request.files['image'],
            cv2.IMREAD_UNCHANGED
        )

        image = convert_opencv_to_rgb(image)

        edge_maps = dsp_engine.detect_edges(
            image,
            operator
        )

        results = {}

        for name, result in edge_maps.items():
            results[name] = {
                'image': dsp_engine.array_to_base64(
                    result['image']
                ),
                'mean_strength': result['mean_strength'],
                'max_strength': result['max_strength'],
                'edge_pixels': result['edge_pixels']
            }

        return jsonify({
            'operator': operator,
            'results': results
        })

    except ValueError as error:
        return jsonify({
            'error': str(error)
        }), 400

    except Exception as error:
        app.logger.exception('Edge detection failed')

        return jsonify({
            'error': str(error)
        }), 500


# ---------------------------------------------------------------------------
# Feature 4: Visual Kernel Laboratory
# ---------------------------------------------------------------------------


@app.route('/api/kernel-analysis', methods=['POST'])
def handle_kernel_analysis():
    try:
        # Accept normal form data from index.html and JSON for easy API testing.
        if request.is_json:
            body = request.get_json(silent=True) or {}
            kernel_value = body.get('kernel')
            kernel_string = json.dumps(kernel_value)
        else:
            kernel_string = request.form.get('kernel')

        kernel = validate_kernel(kernel_string)
        analysis = dsp_engine.analyze_kernel(kernel)

        response = {
            'size': int(kernel.shape[0]),
            'kernel_sum': analysis['kernel_sum'],
            'dc_gain': analysis['dc_gain'],
            'filter_type': analysis['filter_type'],
            'symmetric': analysis['symmetric'],
            'separable': analysis['separable'],
            'frequency_response': dsp_engine.array_to_base64(
                analysis['frequency_response']
            )
        }

        if analysis.get('separable'):
            response['vertical_vector'] = analysis[
                'vertical_vector'
            ]
            response['horizontal_vector'] = analysis[
                'horizontal_vector'
            ]

        return jsonify(response)

    except ValueError as error:
        return jsonify({
            'error': str(error)
        }), 400

    except Exception as error:
        app.logger.exception('Kernel analysis failed')

        return jsonify({
            'error': str(error)
        }), 500


# ---------------------------------------------------------------------------
# Feature 5: Reverse Blur / Image Restoration
# ---------------------------------------------------------------------------


@app.route('/api/deconvolve', methods=['POST'])
def handle_deconvolution():
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No blurred image uploaded'
            }), 400

        image = decode_uploaded_image(
            request.files['image'],
            cv2.IMREAD_COLOR
        )

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        kernel_type, blur_kernel = get_restoration_kernel()

        balance = parse_float_field(
            'balance',
            default=0.01,
            minimum=0.000001,
            maximum=1.0
        )

        restored_image = dsp_engine.wiener_deconvolution(
            image_rgb,
            blur_kernel,
            balance
        )

        return jsonify({
            'restored_image': dsp_engine.array_to_base64(
                restored_image
            ),
            'kernel_type': kernel_type,
            'kernel_size': int(blur_kernel.shape[0]),
            'balance': balance
        })

    except ValueError as error:
        return jsonify({
            'error': str(error)
        }), 400

    except Exception as error:
        app.logger.exception('Image restoration failed')

        return jsonify({
            'error': str(error)
        }), 500


@app.route('/api/noise-cleaner', methods=['POST'])
def handle_noise_cleaner():
    try:
        if 'image' not in request.files:
            raise ValueError('Upload an image first')
        image = convert_opencv_to_rgb(decode_uploaded_image(request.files['image']))
        if image.dtype != np.uint8:
            raise ValueError('Please upload an 8-bit image')
        operation = request.form.get('operation', 'clean')
        if operation not in ('add', 'clean'):
            raise ValueError('Operation must be add or clean')
        try:
            seed = int(request.form.get('seed', '0')) if operation == 'add' else 0
        except ValueError:
            raise ValueError('Seed must be an integer')
        output, metrics = dsp_engine.clean_noise(
            image,
            operation=operation,
            noise_type=request.form.get('noise_type', 'salt_pepper'),
            amount=parse_float_field('amount', 0.1, 0, 1) if operation == 'add' else 0.1,
            sigma=parse_float_field('sigma', 20, 0, 100) if operation == 'add' else 20,
            filter_type=request.form.get('filter_type', 'median'),
            window_size=parse_odd_size('window_size', 3) if operation == 'clean' else 3,
            seed=seed,
            border=request.form.get('border', 'reflect')
        )
        return jsonify(original_image=dsp_engine.array_to_base64(image),
                       output_image=dsp_engine.array_to_base64(output),
                       operation=operation, metrics=metrics)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except Exception:
        app.logger.exception('Noise cleaning failed')
        return jsonify(error='Unable to clean this image'), 500


if __name__ == '__main__':
    app.run(
        debug=True,
        port=5000
    )
