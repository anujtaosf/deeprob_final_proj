from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename

from classify import warmup_resource_constrained, run_pipeline_resource_constrained

app = Flask(__name__)
os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

warmup_resource_constrained()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, GIF, or WEBP image.'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        results = run_pipeline_resource_constrained(save_path)
    finally:
        os.remove(save_path)

    if not results:
        return jsonify({'error': 'No animals detected in the image.'}), 200

    # If multiple animals detected, surface the most threatening one.
    best = max(results, key=lambda r: r['threat'])
    return jsonify({
        'species':    best['species'],
        'emotion':    best['emotion'],
        'confidence': best['confidence'],
        'threat':     best['threat'],
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
