from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import json
import dsp_engine

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/grayscale', methods=['POST'])
def handle_grayscale():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
        
        gray_img = dsp_engine.convert_to_grayscale(img)
        base64_str = dsp_engine.array_to_base64(gray_img)
        
        return jsonify({'grayscale_image': base64_str})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/convolve', methods=['POST'])
def handle_convolve():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        kernel_str = request.form.get('kernel', '[[0,0,0],[0,1,0],[0,0,0]]')
        kernel = json.loads(kernel_str)

        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR) # Ensure BGR 3-channel
        
        # Convert BGR to RGB for consistent processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        output_img = dsp_engine.apply_custom_kernel(img_rgb, kernel)
        base64_str = dsp_engine.array_to_base64(output_img)
        
        return jsonify({'output_image': base64_str})
    except Exception as e:
        print(f"Convolution Error: {e}") # Terminal print for debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)