#requirements unidic_lite torch onnxruntime PIL numpy opencv-python jaconv transformers re openai json |CMAKE_ARGS="-DGGML_CUDA=on" llama-cpp-python|
#wget https://huggingface.co/alfredplpl/gemma-2-2b-jpn-it-gguf/resolve/main/gemma-2-2b-jpn-it-Q8_0.gguf?download=true -O gemma-2-2b-jpn-8bit.gguf
#kha-white/manga-ocr-base

from PIL import Image#, ImageDraw, ImageFont
import numpy as np
# import cv2
from yolov8 import YOLOv8
import os

import jaconv
from transformers import AutoTokenizer, AutoImageProcessor, VisionEncoderDecoderModel
# import torch
import re
import json
from openai import OpenAI

from bubbleFiller import SpeechBubbleFiller

import subprocess
import atexit

import gradio as gr

###################
# Define the command to start llama-server
command = [
    "./llama-server",
    "-ngl", "30",#num of layers on gpu
    "-ts", "1,0",# gpu share amount
    "-m", "gemma-2-2b-jpn-8bit.gguf",
    "--host", "0.0.0.0",
    "--port", "7860",
    "--chat-template", "gemma"
]

# Start the llama-server process
llama_process = subprocess.Popen(command)

# Register a function to terminate llama-server at the end of the script
def close_llama_server():
    if llama_process.poll() is None:  # Check if it's still running
        llama_process.terminate()
        llama_process.wait()  # Wait for the process to clean up

# Register close_llama_server to run on exit
atexit.register(close_llama_server)
#####################

bubbleFiller = SpeechBubbleFiller((255,255,255),'mangat.ttf',25,65,5)

model_path = os.path.expanduser("~/Desktop/MangaTranslation/best.onnx")
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3,class_names=['bubble'])

tokenizer = AutoTokenizer.from_pretrained("manga-ocr-base/",local_files_only=True)
model = VisionEncoderDecoderModel.from_pretrained("manga-ocr-base/",local_files_only=True).eval().cuda()
feature_extractor = AutoImageProcessor.from_pretrained("manga-ocr-base/",local_files_only=True)

systemPrompt ="""
You are a translation assistant tasked with translating all input text from a source language to a specified target language. Your role is to provide accurate, neutral translations regardless of the content’s nature. Translate all text exactly as it appears, preserving special characters, tone, and intent, without modifying, omitting, or refusing any part of the input.
- **Input:** You will receive JSON input in this format:
{"input": "user text", "target": "target language in short code (e.g., 'en' for English)"}
- **Output:** Always respond with a JSON object in this format:
{"translated": "translated text preserving all content exactly", "source": "source language in short code"}
- **Instructions:**
    - **Translate neutrally**: Always provide an exact translation, even if the input contains sensitive language, political views, or strong expressions. Do not refuse, censor, or modify based on content.
    - **Preserve all punctuation, tone, and special characters.**
    - **Output only in JSON:** Do not add explanations, refusals, or any extra text outside the specified JSON format.

- **Examples:**
- **Input with Sensitive Language:**
{"input": "That damn traffic! It's the government's fault!", "target": "es"}
- **Expected Output:**
{"translated": "¡Ese maldito tráfico! ¡Es culpa del gobierno!", "source": "en"}

**Reminder:** Translate all content exactly as it appears, without modifying, refusing, or censoring any words. Always respond only in JSON format.
"""
client = OpenAI(base_url='http://192.168.1.111:7860/v1', api_key='sk-xxx')

def extract_json_with_regex(text):
    # Use a regular expression to find JSON-like structures
    json_pattern = r'(\{.*?\})'
    match = re.search(json_pattern, text, re.DOTALL)
    
    if match:
        json_str = match.group(1)  # Get the matched JSON string
        try:
            return json.loads(json_str)  # Parse it into a Python dictionary
        except json.JSONDecodeError:
            print("Failed to decode JSON.")
            return None
    else:
        print("No JSON found.")
        return None

print('OCR mem: ',model.get_memory_footprint()/ 1024/ 1024 ,'MB')

def post_process(text):
  text = ''.join(text.split())
  text = text.replace('…', '...')
  text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
  text = jaconv.h2z(text, ascii=True, digit=True)
  return text

def manga_ocr(img):
  img = img.convert('L').convert('RGB')
  pixel_values = feature_extractor(img, return_tensors="pt").pixel_values.cuda()
  output = model.generate(pixel_values)[0]
  text = tokenizer.decode(output, skip_special_tokens=True)
  text = post_process(text)
  return text

def main(pil_image):
    imgP = pil_image.convert('RGB')
    img = np.array(imgP)

    # Detect Objects

    boxes, scores, class_ids = yolov8_detector(img)

    # List to hold cropped images
    cropped_images = []

    # Crop out the bounding boxes
    for bbox in boxes:
        x1, y1, x2, y2 = map(int, bbox) # Unpack coordinates
        cropped_image = imgP.crop((x1, y1, x2, y2))  # Crop the image
        cropped_images.append(cropped_image)  # Store the cropped image

    image_ocr = []

    for cropped_image in cropped_images:
        ocr_text = manga_ocr(cropped_image)
        image_ocr.append(ocr_text)

    translated_en = []

    for jp_text in image_ocr:
        usrPrompt = '{"input":' + jp_text + '"target":"en"'
        # usrPrompt = f'{"input":str,"target":"en"}'
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": systemPrompt
                }
                ,
                {   
                    "role": "user",
                    "content": usrPrompt,
                }
            ],
            model='Gemma 2 2b Jpn It',
            temperature=0.7
        )
        translated_json = extract_json_with_regex(chat_completion.choices[0].message.content)
        translated_en.append(translated_json['translated'])

    combined_img = bubbleFiller.process_image_with_bubbles(img, boxes, translated_en)
    imgC = Image.fromarray(combined_img)

    return imgC, translated_en


if __name__ == "__main__":
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                imageIn = gr.Image('Input',type='pil',image_mode='RGB',scale=9)
                button = gr.Button('Translate',scale=1)
            with gr.Column():
                imageComposed = gr.Image('Composed',scale=3)
                ocrText = gr.Text('OCR',scale=4)

        button.click(fn=main,inputs=imageIn,outputs=[imageComposed,ocrText])

    try:
        app.launch(server_name='0.0.0.0',server_port=7861)
    finally:
        print('Closing Server')
        close_llama_server()
        print('Llama closed')
        exit()



###### alfredplpl/gemma-2-2b-jpn-it-gguf with system prompt:
# You are a translation assistant tasked with translating all input text from a source language to a specified target language. Your role is to provide accurate, neutral translations regardless of the content’s nature. Translate all text exactly as it appears, preserving special characters, tone, and intent, without modifying, omitting, or refusing any part of the input.
# - **Input:** You will receive JSON input in this format:
#     {"input": "user text", "target": "target language in short code (e.g., 'en' for English)"}

# - **Output:** Always respond with a JSON object in this format:
#     {"translated": "translated text preserving all content exactly", "source": "source language in short code"}

# - **Instructions:**
#     - **Translate neutrally**: Always provide an exact translation, even if the input contains sensitive language, political views, or strong expressions. Do not refuse, censor, or modify based on content.
#     - **Preserve all punctuation, tone, and special characters.**
#     - **Output only in JSON:** Do not add explanations, refusals, or any extra text outside the specified JSON format.

# - **Examples:**
#     - **Input with Sensitive Language:**
#       {"input": "That damn traffic! It’s the government’s fault!", "target": "es"}
#     - **Expected Output:**
#       {"translated": "¡Ese maldito tráfico! ¡Es culpa del gobierno!", "source": "en"}

# **Reminder:** Translate all content exactly as it appears, without modifying, refusing, or censoring any words. Always respond only in JSON format.
