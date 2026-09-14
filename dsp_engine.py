import numpy as np
import cv2
import base64

def convert_to_grayscale(img_array):
    #rgb/rgba ke 2d grayscale covert function
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[..., :3]#this line convert (R,G,B,A) to (R,G,B)

    if len(img_array.shape) == 3 and img_array.shape[2] == 3:#color image er (heigh,width,3) 3 indicate rgb
        # Weighted Luminance Formula
        gray_img = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
        return gray_img.astype(np.uint8)#why unit8  like abr value ashlo 140.75.image pixel man unit 8 format
     #format e thake 0<=pixel<+255
        #each pixel jonno ekta pixel thake
    return img_array

def array_to_base64(img_array):
    #convert graysacle numpy array to ong image
   #image 2d hole
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
   #cv2 normalyb imgae ke bgr order load kore tai amader  ektake rgb te convert korte hbe
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    _, buffer = cv2.imencode('.png', img_array)
    encoded_string = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"
def custom_2d_convolution(img_array, kernel):
   
    kernel = np.array(kernel, dtype=np.float32)
    k_h, k_w = kernel.shape
    
    # Convolution rule: 180 degree kernel flip
    kernel_flipped = np.flipud(np.fliplr(kernel))
    
    pad_h = k_h // 2
    pad_w = k_w // 2

    def process_channel(channel):
        h, w = channel.shape
        padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect').astype(np.float32)
        output = np.zeros((h, w), dtype=np.float32)

        # Vectorized Sliding Window Operations (No Pixel Nested Loop)
        for i in range(k_h):
            for j in range(k_w):
                output += padded[i:i+h, j:j+w] * kernel_flipped[i, j]
                
        return output

    # Grayscale image (2D)
    if len(img_array.shape) == 2:
        output_array = process_channel(img_array)
    # RGB image (3D)
    else:
        output_array = np.zeros_like(img_array, dtype=np.float32)
        for c in range(3):
            output_array[..., c] = process_channel(img_array[..., c])

    return np.clip(output_array, 0, 255).astype(np.uint8)


def apply_custom_kernel(img_array, kernel):
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[..., :3]
    return custom_2d_convolution(img_array, kernel)


EDGE_KERNELS = {
    'sobel': {
        'horizontal': [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        'vertical': [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    },
    'prewitt': {
        'horizontal': [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        'vertical': [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
    },
    'laplacian': {
        'horizontal': [[0, 0, 0], [1, -2, 1], [0, 0, 0]],
        'vertical': [[0, 1, 0], [0, -2, 0], [0, 1, 0]],
    },
}


def _convolve_float(img_array, kernel):
    """Return the signed convolution response before display normalization."""
    kernel = np.asarray(kernel, dtype=np.float32)
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(img_array.astype(np.float32),
                    ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros(img_array.shape, dtype=np.float32)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            output += padded[i:i + img_array.shape[0], j:j + img_array.shape[1]] * kernel[i, j]
    return output


def _normalize_edge(response):
    strength = np.abs(response)
    maximum = float(strength.max())
    if maximum == 0:
        return np.zeros_like(strength, dtype=np.uint8), strength
    return np.clip(strength * 255.0 / maximum, 0, 255).astype(np.uint8), strength


def detect_edges(img_array, operator='sobel'):
    """Create horizontal, vertical, and combined normalized edge maps."""
    if operator not in EDGE_KERNELS:
        raise ValueError(f'Unsupported edge operator: {operator}')

    gray_img = convert_to_grayscale(img_array)
    kernels = EDGE_KERNELS[operator]
    horizontal_response = _convolve_float(gray_img, kernels['horizontal'])
    vertical_response = _convolve_float(gray_img, kernels['vertical'])
    combined_response = np.sqrt(horizontal_response ** 2 + vertical_response ** 2)

    maps = {}
    for name, response in (
        ('horizontal', horizontal_response),
        ('vertical', vertical_response),
        ('combined', combined_response),
    ):
        image, strength = _normalize_edge(response)
        maps[name] = {
            'image': image,
            'mean_strength': round(float(strength.mean()), 2),
            'max_strength': round(float(strength.max()), 2),
            'edge_pixels': round(float((image > 32).mean() * 100), 2),
        }
    return maps