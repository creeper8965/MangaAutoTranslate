What is it?


It is a Gradio server that translates a japanese manga page to english, and blend it in the pre existing speech bubbles.
I will upload a API server.py and a firefox addon soon to to right click to translate in webpages.

    
To install:
    *all files and folders in the same MangaAutoTranslate/


    For server:

        1. git clone https://github.com/creeper8965/MangaAutoTranslate.git
        2. cd MangaAutoTranslate/
        3. pip install -r requirements.txt

    For Llama.cpp:
            
        1. git clone https://github.com/ggerganov/llama.cpp
        2. cd llama.cpp
        3. make ##Use your hardware supported make command https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md
        4. mv llama-server ../
        5. cd ../
        6. rm -r llama.cpp/

    For ML models:
        
        python Install.py


Credits


    Llama.cpp https://github.com/ggerganov/llama.cpp
    Onnx Yolov8 https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection
    Japanese Gemma https://huggingface.co/alfredplpl/gemma-2-2b-jpn-it-gguf
    Manga OCR https://huggingface.co/kha-white/manga-ocr-base
    Speech Bubble Dataset https://universe.roboflow.com/sam64-t4u3d/manga-translator-detection-r1kli
