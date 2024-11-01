To install:


For server:
git clone https://github.com/creeper8965/MangaAutoTranslate.git
cd MangaAutoTranslate/
pip install -r requirements.txt

For Llama.cpp:
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make ##Use your hardware supported make command https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md
mv llama-server ../
rm -r llama.cpp

For ML models:
python Install.py

