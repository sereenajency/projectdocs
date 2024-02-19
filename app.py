from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64

app = Flask(__name__)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Get the uploaded image, category, and style prompt from the request
        image_file = request.files['image']
        image = Image.open(image_file)
        category = request.form['category']
        style_prompt = request.form['stylePrompt']

        # Convert the image to a suitable format for the model
        image = image.convert("RGB").resize((512, 512))
        input_image = torch.tensor(np.array(image)).unsqueeze(0).to(pipe.device)

        # Generate the image using the provided style prompt
        generated_image = pipe(prompt=style_prompt, image=input_image, strength=0.5, num_inference_steps=50)

        # Save the generated image
        output_buffer = BytesIO()
        generated_image.images[0].save(output_buffer, format="PNG")
        output_buffer.seek(0)

        return jsonify({"imageUrl": "data:image/png;base64," + base64.b64encode(output_buffer.read()).decode()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
