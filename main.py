from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import speech_recognition as sr


# load model and processor
def load_model(tamaño="tiny"):
    global processor
    global model 
    processor = WhisperProcessor.from_pretrained("openai/whisper-%s" % tamaño)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-%s" % tamaño)
    model.config.forced_decoder_ids = None
    return None

# record audio
def save_audio(path):
    if path is None:
        path = "audio.wav"
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("Di algo:")
        audio = r.listen(source)
    with open(path, "wb") as f:
        f.write(audio.get_wav_data())
    return None

def transcribe_audio(audio_path):
    # load audio file
    audio = torchaudio.load(audio_path)
    # Prepara el audio para el modelo
    input_features = processor(audio, return_tensors="pt")

    # Realiza la predicción
    outputs = model(**input_features)

    # Obtiene la transcripción
    predicted_tokens = outputs.predicted_tokens
    transcript = processor.decode(predicted_tokens)

    # Imprime la transcripción
    print(transcript)
    return None

def main():
    load_model()
    save_audio()
    transcribe_audio("audio.wav")
    return None

if __name__ == "__main__":
    main()
