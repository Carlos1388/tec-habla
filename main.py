import speech_recognition as sr
from gtts import gTTS
import pyphen


# Crear un objeto de reconocimiento de voz
r = sr.Recognizer()

# Configurar el micrófono
mic = sr.Microphone()

# Iniciar la grabación
with mic as source:
    print("Di algo:")
    audio = r.listen(source)

# Reconocer el texto del audio
try:
    text = r.recognize_google(audio)
except sr.RequestError:
    print("No se pudo conectar al servicio de Google Speech Recognition")
except sr.UnknownValueError:
    print("No se pudo entender el audio")

# Mostrar el texto reconocido
print(text)

# Seleccionar el idioma
language = 'es'

# Convertir el texto a fonemas
tts = gTTS(text, lang=language)

# Guardar el archivo de audio con los fonemas
tts.save("fonemas.wav")


def fonemas_a_afi(fonemas):
    # Create a Pyphen object with the desired language
    dic = pyphen.Pyphen(lang='es_ES')

    # Convert phonemes to AFI symbols
    simbolos_afi = []
    for fonema in fonemas:
        simbolos_afi.append(dic.inserted(fonema))

    # Save AFI symbols to a text file
    with open("fonemas_afi.txt", "w") as f:
        f.write(" ".join(simbolos_afi))

    # Return list of IPA symbols
    print(simbolos_afi)

fonemas_a_afi(tts.text.split())
