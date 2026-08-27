import numpy as np
import cv2
import base64

def convert_to_grayscale(img_array):
    """RGB/RGBA ইমেজকে ২ডি Grayscale এ কনভার্ট করে"""
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[..., :3]

    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Weighted Luminance Formula
        gray_img = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
        return gray_img.astype(np.uint8)
    
    return img_array

def array_to_base64(img_array):
    """
    Numpy Array কে সঠিকভাবে PNG Base64 এ কনভার্ট করে।
    Grayscale ইমেজকে 3-channel (RGB) এ রূপান্তর করে পাঠানো হয় যেন রঙ না বদলায়।
    """
    # যদি ইমেজটি ১-চ্যানেল Grayscale (2D Array) হয়
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    # যদি ৩-চ্যানেল RGB হয়, OpenCV এর জন্য RGB থেকে BGR করতে হবে
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    _, buffer = cv2.imencode('.png', img_array)
    encoded_string = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"
def custom_2d_convolution(img_array, kernel):
    """
    Built-in convolve ছাড়া Optimized Vectorized 2D Spatial Convolution
    """
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