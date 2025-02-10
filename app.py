import os
import pdb
import whisper
from flask import Flask, request, render_template, redirect, url_for
from markupsafe import Markup
from werkzeug.utils import secure_filename
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForTokenClassification

from medspanner_utils import *

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

# modelos de MedSpaNer
EXCEPTIONS_LIST = "medspanner/list_except.txt"
ExceptionsDict = read_exceptions_list(EXCEPTIONS_LIST)

umls_tokenizer = AutoTokenizer.from_pretrained("medspaner/roberta-es-clinical-trials-umls-7sgs-ner")
umls_token_classifier = AutoModelForTokenClassification.from_pretrained("medspaner/roberta-es-clinical-trials-umls-7sgs-ner")

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
        temperature=0.7,      # Menor temperatura para respuestas más deterministas
        top_p=0.95,
    )
    # El output es un diccionario con la clave "choices"
    improved = output["choices"][0]["text"].strip()
    return improved

def annotate_transcription(transcript: str, use_umls=True, use_nested=False, use_lexicon=False, normalize=False):
    """
    Modified version of your annotation function with explicit parameters
    """
    # Initialize entities storage
    AllFlatEnts = {}
    AllNestedEnts = {}
    Entities = {}

    # Clean and prepare text
    text = transcript.lstrip()
    text = re.sub("\r", "", text)

    # Load normalization data if needed
    UMLSData = {}
    if normalize:
        with open("medspanner/umls_data.pickle", "rb") as DataFile:
            UMLSData = pickle.load(DataFile)

    # Process text
    Sentences = sentences_spacy(text)
    LexiconData, POSData = read_lexicon("medspanner/MedLexSp.pickle")
    Tokens = tokenize_spacy_text(text, POSData)

    # Apply lexicon if selected
    if use_lexicon:
        Entities, NestedEnts = apply_lexicon(text, LexiconData, use_nested)
        AllFlatEnts = merge_dicts(AllFlatEnts, Entities)
        AllNestedEnts = merge_dicts(AllNestedEnts, NestedEnts)

    # Apply neural model if selected
    if use_umls:
        Output = annotate_sentences_with_model(Sentences, text, umls_token_classifier, umls_tokenizer, device)
        for i, Ent in enumerate(Output):
            Entities[i] = {
                'start': Ent['start'],
                'end': Ent['end'],
                'ent': Ent['word'],
                'label': Ent['entity_group']
            }
        AllFlatEnts = merge_dicts(AllFlatEnts, Entities)
        AllFlatEnts, NestedEntities = remove_overlap(AllFlatEnts)
        AllNestedEnts = merge_dicts(AllNestedEnts, NestedEntities)

    # Clean entities
    AllFlatEnts = remove_entities(AllFlatEnts, Tokens, ExceptionsDict, 'label', AllNestedEnts)
    NestedEntsCleaned = {}
    if use_nested:
        NestedEntsCleaned = remove_entities(AllNestedEnts, Tokens, ExceptionsDict, 'label', AllNestedEnts)

    # Prepare final entities
    EntitiesNoOverlap, NestedEntities = remove_overlap_gui(AllFlatEnts)
    NestedEntsCleaned = merge_dicts(NestedEntities, NestedEntsCleaned)
    NestedEntsCleaned, NestedEntities = remove_overlap_gui(NestedEntsCleaned)
    NestedEntsCleaned = remove_nested_entity(EntitiesNoOverlap, NestedEntsCleaned)

    # Generate HTML and BRAT data
    annotated_html = Markup(EntsDict2html(text, EntitiesNoOverlap, LexiconData, NestedEntsCleaned, UMLSData))
    
    if use_nested:
        # Merge both dictionaries of nesting (outer) and nested entities to output in BRAT format
        AllFinalEntities = merge_dicts(AllFlatEnts,AllNestedEnts)
    else:
        AllFinalEntities = AllFlatEnts

    FinalHash = codeAttribute(AllFinalEntities)

    FinalHash = codeAttribute(AllFinalEntities)
    brat_data = convert2brat_gui(FinalHash, LexiconData if normalize else None, UMLSData if normalize else None)

    return annotated_html, brat_data


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
            
            try:
               
                # pdb.set_trace()
                # Perform annotation
                transcript_apollo_annotated, annotation_data = annotate_transcription(
                    transcript=transcript_apollo,
                    use_nested=True,
                    use_lexicon=True,
                    normalize=True
                )

                return render_template(
                    "result.html",
                    audio_url=audio_url,
                    transcript_whisper=transcript_whisper,
                    transcript_apollo=transcript_apollo,
                    transcript_apollo_annotated=transcript_apollo_annotated,
                    ann_data=annotation_data,
                )
            except Exception as e:
                app.logger.error(f"Annotation error: {str(e)}")
                return render_template("error.html", error_message="Error processing annotation")

    return render_template("upload.html")

# @app.route("/annotate_text", methods=["POST"])
# def annotate_text():
#     """Handle text annotation from the integrated interface"""
#     try:
#         # Get all form data
#         text = request.form.get("text", "")
        
#         # Get annotation parameters
#         # use_umls = "neu" in request.form
#         use_nested = "nest" in request.form
#         use_lexicon = "lex" in request.form
#         normalize = "norm" in request.form
#         # pdb.set_trace()
#         # Perform annotation
#         annotated_text, annotation_data = annotate_transcription(
#             transcript=text,
#             use_nested=use_nested,
#             use_lexicon=use_lexicon,
#             normalize=normalize
#         )
#         # pdb.set_trace()
#         return render_template(
#             "result.html",
#             annotated_text=annotated_text,
#             ann_data=annotation_data,
#             selected_checkboxes={
#                 "nest": use_nested,
#                 "lex": use_lexicon,
#                 "norm": normalize
#             }
#         )
#     except Exception as e:
#         app.logger.error(f"Annotation error: {str(e)}")
#         return render_template("error.html", error_message="Error processing annotation")

@app.route('/ayuda')
def ayuda():
    return render_template('ayuda.html')

@app.route('/acerca_de')
def acerca_de():
    return render_template('acerca_de.html')

if __name__ == "__main__":
    # Asegúrate de usar un puerto que no genere conflictos
    app.run(port=5001, debug=True)
