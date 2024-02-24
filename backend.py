# Import necessary libraries
from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64
from serpapi import GoogleSearch
import pandas as pd
import requests
import os

# Initialize Flask app
app = Flask(__name__)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Define upload folder
image_folder = "uploads"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)


def upload_image_to_imgbb(image_path, api_key):
    # Function to upload image to imgbb
    url = "https://api.imgbb.com/1/upload"
    files = {"image": (image_path, open(image_path, "rb"))}
    params = {"key": api_key}

    response = requests.post(url, files=files, params=params)
    data = response.json()

    return data.get("data", {}).get("url")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file:
                image_filename = file.filename
                image_path = os.path.join(image_folder, image_filename)
                file.save(image_path)

                # Image search and generation process
                api_key = "fe4095abd2b2d313d039c4d7e28fb628"
                url = upload_image_to_imgbb(image_path, api_key)

                if url:
                    params = {
                        "engine": "google_lens",
                        "url": url,
                        "no_cache": "true",
                        "api_key": "YOUR_SERPAPI_KEY",  # Replace with your SerpApi key
                    }

                    search = GoogleSearch(params)
                    results = search.get_dict()

                    name_price_url = []

                    important_keys = ["source", "title", "price", "thumbnail"]

                    if results['search_metadata']['status'] == "Success":
                        for item in results['visual_matches']:
                            name_price_url.append({
                                "source": item.get('source'),
                                "title": item.get('title'),
                                "link": item.get('link'),
                                "price": item['price'].get('extracted_value') if 'price' in item else None,
                                "currency": item['price'].get('currency') if 'price' in item else None,
                                "thumbnail": item.get('thumbnail')
                            })

                        df = pd.DataFrame(name_price_url)
                        filtered_df = df[df['link'].notnull()]
                        filtered_df = df[df['currency'].notnull()]
                        filtered_df = filtered_df.sort_values(by='price')
                        filtered_df = filtered_df.reset_index()
                        filtered_df = filtered_df[["source", "link", "price"]]

                        # Save the search results to CSV
                        filtered_df.to_csv("results.csv", index=False)

                        # Generate image using Stable Diffusion
                        style_prompt = getCategoryPrompt(request.form['category'])
                        input_image = torch.tensor(np.array(Image.open(image_path)).convert("RGB").resize((512, 512))).unsqueeze(0).to(pipe.device)
                        generated_image = pipe(prompt=style_prompt, image=input_image, strength=0.8, num_inference_steps=50)

                        # Save the generated image
                        output_buffer = BytesIO()
                        generated_image.images[0].save(output_buffer, format="PNG")
                        output_buffer.seek(0)

                        # Return generated image URL
                        return jsonify({"imageUrl": "data:image/png;base64," + base64.b64encode(output_buffer.read()).decode()})

                    else:
                        return jsonify({"error": "No item found in the search."})

                else:
                    return jsonify({"error": "Error uploading image... retry again :)"})

        except Exception as e:
            return jsonify({"error": f"Error: {e}"})

    return render_template('index.html', data=[])


def getCategoryPrompt(category):
    # Function to get category-specific prompts
    switcher = {
        'kitchen': 'Design a stylish kitchen interior.',
        'living-room': 'Create a cozy living room setting.',
        'bedroom': 'Design a comfortable bedroom space.',
        'study-room': 'Generate a modern study room environment.'
        # Add more categories and prompts as needed
    }
    return switcher.get(category, '')


if __name__ == '__main__':
    app.run(debug=True)
