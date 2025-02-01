import os
import whisper
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from llama_cpp import Llama

# Cargar el modelo Whisper (base)
whisper_model = whisper.load_model("base")

# Configurar el modelo Apollo (modelo cuantizado)
# Asegúrate de que 'model_path' apunte al archivo descargado, por ejemplo:
apollo_model_path = "/Users/awxlong/Library/Caches/llama.cpp/itlwas_Apollo-7B-Q4_K_M-GGUF_apollo-7b-q4_k_m.gguf" # medium size, balanced quality according to https://huggingface.co/tensorblock/Apollo-7B-GGUF. Next time should try Apollo-7B-Q5_K_M.gguf
apollo_model = Llama(
    model_path=apollo_model_path,
    n_ctx=2048,          # Ajusta según sea necesario
    n_threads=4,         # Número de hilos (ajusta según tu CPU)
    seed=42
)

def improve_transcription(transcript: str) -> str:
    """
    Usa el modelo Apollo para mejorar la transcripción.
    Se envía un prompt en español que le indica al modelo:
      - Identificar cambios de interlocutor.
      - Corregir errores médicos.
    """
    prompt = (
        "Mejora la siguiente transcripción de una conversación médico-paciente. "
        "Identifica los cambios de interlocutor y corrige errores, especialmente en "
        "terminología médica. Devuelve el resultado en un formato estructurado con "
        "etiquetas claras (por ejemplo, Doctor:, Paciente:). Mantén el contenido original de la transcripción. \n\n"
        f"Transcripción original: {transcript}\n\nRespuesta:"
    )
    
    # Generar respuesta usando el modelo Apollo
    output = apollo_model(
        prompt,
        max_tokens=2048,      # Ajusta según la longitud necesaria
        temperature=0.42,      # Menor temperatura para respuestas más deterministas
        top_p=0.95,
    )
    # El output es un diccionario con la clave "choices"
    improved = output["choices"][0]["text"].strip()
    return improved

# Configurar la aplicación Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a"}

# Asegúrate de que la carpeta de uploads existe
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_and_transcribe():
    if request.method == "POST":
        if "file" not in request.files:
            return "No se encontró el archivo", 400
        file = request.files["file"]
        if file.filename == "":
            return "No se ha seleccionado ningún archivo", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            # Paso 1: Transcribir el audio usando Whisper
            result = whisper_model.transcribe(filepath)
            transcript_whisper = result["text"]
            
            # Paso 2: Mejorar la transcripción usando Apollo-7B
            transcript_apollo = improve_transcription(transcript_whisper)
            
            # Usar url_for para que el archivo de audio esté disponible en el navegador.
            audio_url = url_for('static', filename=f'uploads/{filename}')
            # Mueve el archivo subido a la carpeta static/uploads para servirlo
            static_uploads = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(static_uploads, exist_ok=True)
            os.rename(filepath, os.path.join(static_uploads, filename))
            
            return render_template(
                "result.html",
                audio_url=audio_url,
                transcript_whisper=transcript_whisper,
                transcript_apollo=transcript_apollo
            )
    return render_template("upload.html")

if __name__ == "__main__":
    # Asegúrate de usar un puerto que no genere conflictos
    app.run(port=5001, debug=True)
