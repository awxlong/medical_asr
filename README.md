# YachayMed: Transcripción automatizada de la conversación entre paciente y doctor

Hay muchas obligaciones tediosas en el ámbito médico lo cual dificulta una interacción de alta calidad entre doctor y paciente. En este proyecto, desarrollamos un sistema de inteligencia artificial que automatiza la transcripción de una conversación entre doctor y paciente con el fin de apoyar la generación de un reporte médico estructurado. 

## Demo

<img src="demo/demo.gif" alt="Demo of YachayMed" width="50%" />

## Sistema
### Preinstalaciones
Dependiendo de tu sistema operativo, hay algunas instalaciones que hay que hacer para ejecutar el sistema: 
1. Linux: `sudo apt update && sudo apt install ffmpeg`. Mac: `brew install ffmpeg` y `brew install llama.cpp`
2. Python: pip install Flask openai-whisper llama-cpp-python
3. `llama-cli --hf-repo itlwas/Apollo-7B-Q4_K_M-GGUF --hf-file apollo-7b-q4_k_m.gguf -p "Medicina e inteligencia artificial son "` según las instrucciones indicadas en https://huggingface.co/itlwas/Apollo-7B-Q4_K_M-GGUF

Con estas librerías, debería ser suficiente convocar el sistema a través de `python app.py`.

### Explicación del mecanismo
Seguimos las recomendaciones publicadas en este estudio https://arxiv.org/abs/2402.07658. Este equipo de Google argumentan que la precisión de la transcripción de una conversación entre doctor y paciente puede ser mejorada a través del uso sinergético entre un transcriptor de audio y texto cuya salida es mejorada por un modelo de lenguaje gigante (LLM).

Inspirado por este estudio, usamos openai-whisper para la primera fase de la transcripción de audio a texto, lo cual observamos varios errores en términos médicos, así como la falta de la identifficaión del interlocutor. Usar openai-whisper es debido a que es plurilingüe, así como la librería más accesible y versátil de integrar en nuestro sistema. 

Después utilizamos un LLM llamado Apollo (https://arxiv.org/abs/2403.03640) para poder corregir estos errores e identificar el interlocutor. Elegir Apollo es justificado por varias razones:
1. Es también plurilingüe
2. Es un LLM relativamente ligero que puede ser ejecutado localmente en una laptop. Sus pesas cuestan alrededor de 7GB. Además, ofrece una versión cuantizada lo cual reduce la precisión de las pesas sin compromenter mucho la calidad del texto que puede producir. Cuantización garantiza asequibilidad en recursos computacionales restringidos. 

Como caso ejemplar, exploramos transcribir la conversación en este video https://www.youtube.com/watch?v=gpWmbBHetzg 

### Limitaciones
Actualmente podemos ver que tanto Whisper y Apollo cometen errores de mencionar palabras nunca dichas, aunque a veces Apollo es capaz de inferir lo que la paciente se refería pero no dijo explícitamente. Tal vez usar un Apollo relativamente menos cuantizado (pero más caro de ejecutar) podría resolver algunas discrepancias. Apollo viene en varias versiones cuantizadas según https://huggingface.co/itlwas/Apollo-7B-Q4_K_M-GGUF/resolve/main/apollo-7b-q4_k_m.gguf

Analizar el audio en segmentos en vez de toda la transcripción al mismo tiempo con una mejor ingenería de prompt como lo indica el estudio de Google también podría solucionar los problemas de precisión.

### Extensiones futuras
Planeamos integrar la herramienta MedCAT del paquete CogStack https://github.com/CogStack/MedCAT para poder anotar automáticamente palabras claves de la transcripción y relacionarlos con alguna base de datos médicos. La visión es construir un sistema comprensivo similar a Sanivert de España https://ceur-ws.org/Vol-3729/p12_rev.pdf 