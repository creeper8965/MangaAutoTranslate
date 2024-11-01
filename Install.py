import transformers
from transformers import AutoTokenizer, AutoImageProcessor, VisionEncoderDecoderModel
import subprocess

print('Installing OCR')

tokenizer = AutoTokenizer.from_pretrained("kha-white/manga-ocr-base")
model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base").eval()
feature_extractor = AutoImageProcessor.from_pretrained("kha-white/manga-ocr-base")

tokenizer.save_pretrained('manga-ocr-base/')
model.save_pretrained('manga-ocr-base/')
feature_extractor.save_pretrained('manga-ocr-base/')

print('OCR Installed!')

print('Installing LLM')
cmd = ['wget','https://huggingface.co/alfredplpl/gemma-2-2b-jpn-it-gguf/resolve/main/gemma-2-2b-jpn-it-Q8_0.gguf?download=true','-O','gemma-2-2b-jpn-8bit.gguf']
# Start the llama-server process
gemma_download = subprocess.Popen(cmd)
print('LLM Installed')
