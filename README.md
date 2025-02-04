# YachayMed: Transcripción automatizada de la conversación entre paciente y doctor

Hay muchas obligaciones tediosas en el ámbito médico lo cual dificulta una interacción de alta calidad entre doctor y paciente. En este proyecto, desarrollamos un sistema de inteligencia artificial que automatiza la transcripción de una conversación entre doctor y paciente con el fin de apoyar la generación de un reporte médico estructurado. 

## Demo

<video src="demo/demo.mp4" controls></video>

## Explicación del sistema

sudo apt update && sudo apt install ffmpeg 
brew install ffmpeg
pip install Flask
pip install openai-whisper
pip install llama-cpp-python
Use quantized Apollo https://huggingface.co/itlwas/Apollo-7B-Q4_K_M-GGUF
The model is probably https://huggingface.co/itlwas/Apollo-7B-Q4_K_M-GGUF/resolve/main/apollo-7b-q4_k_m.gguf