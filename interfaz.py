from transformers import WhisperProcessor, WhisperForConditionalGeneration
import speech_recognition as sr
import soundfile
import torch
from transformers import pipeline
import numpy as np
import librosa

from phonemizer.backend.espeak.wrapper import EspeakWrapper

import phonemizer 
import Levenshtein

EspeakWrapper.set_library("C:\Program Files\eSpeak NG\libespeak-ng.dll") # path to the espeak-ng library

MODEL_SIZE = "tiny"
CONVERSATION = "conversation.txt"


# load model and processor
def load_model(tamaño=MODEL_SIZE):
    global processor
    global model 
    global forced_decoder_ids
    processor = WhisperProcessor.from_pretrained("openai/whisper-%s" % tamaño)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-%s" % tamaño)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    return None

# record audio
def save_audio(path):
    if path is None:
        path = "audio.wav"
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        audio = r.listen(source)
    with open(path, "wb") as f:
        f.write(audio.get_wav_data())
    return None


def transcribe_audio(audio_path):
    # Load the audio file
    audio, sample_rate = soundfile.read(audio_path)

    # Resample the audio to 16000Hz if necessary
    if sample_rate != 16000:
        audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Ensure the audio data is 2D
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=0)

    # Prepare the audio for the model
    input_features = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features

    # Generate token ids
    predicted_ids = model.generate(input_features,forced_decoder_ids=forced_decoder_ids)

    # Decode token ids to text
    #transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

    # Print the transcription
    #print(transcription)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)

def transcribe_ipa(text):
    phonemes = phonemizer.phonemize(text,language="en-us",backend="espeak")
    return phonemes


def process_conversation(conversation):
    # read conversation.txt
    # split conversation into messages
    g = open("conversation_clean.txt", "w")
    with open(conversation, "r") as f:
        messages = f.readlines()
        from_A = []
        from_B = []
        for i in range(len(messages)):
            message = messages[i]
            list_message = message.split("**")
            message = list_message[-1]
            if i % 2 == 0:
                from_A.append(message)
            else:
                from_B.append(message)
            # write messages in conversation_clean.txt
            g.write(message)
    g.close()
    return from_A, from_B


from_A, from_B = process_conversation(CONVERSATION)
# take out (and delete after) first message from A and print it
class Demo:
    def __init__(self):
        self.message_num = 0
        self.data = {
            'speaker1_pre': ' ',
            'speaker2_pre': ' ',
            'speaker1': ' ',
            'speaker2': ' ',
            'speaker1_post': ' ',
            'speaker2_post': ' ',
            'phonemes_theo': ' ',
            'phonemes': ' '
        }
    
import Levenshtein

def word_distance(word1, word2):
    return Levenshtein.distance(word1, word2)

def calculate_path(seq1_words, seq2_words):
    matrix = [[0] * (len(seq2_words) + 1) for _ in range(len(seq1_words) + 1)]

    for i in range(1, len(seq1_words) + 1):
        matrix[i][0] = i
    for j in range(1, len(seq2_words) + 1):
        matrix[0][j] = j

    for i in range(1, len(seq1_words) + 1):
        for j in range(1, len(seq2_words) + 1):
            if seq1_words[i - 1] == seq2_words[j - 1]:
                cost = 0
            else:
                cost = word_distance(seq1_words[i - 1], seq2_words[j - 1])

            matrix[i][j] = min(matrix[i - 1][j] + 1,        # Deletion
                               matrix[i][j - 1] + 1,        # Insertion
                               matrix[i - 1][j - 1] + cost)  # Substitution/Match

    return matrix

def backtrack_path(matrix, seq1_words, seq2_words):
    i = len(seq1_words)
    j = len(seq2_words)
    operations = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and matrix[i][j] == matrix[i - 1][j - 1] + word_distance(seq1_words[i - 1], seq2_words[j - 1]):
            if seq1_words[i - 1] != seq2_words[j - 1]:
                operations.append(("Substitution", seq1_words[i - 1], seq2_words[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and matrix[i][j] == matrix[i - 1][j] + 1:
            operations.append(("Deletion", seq1_words[i - 1], ""))
            i -= 1
        else:
            operations.append(("Insertion", "", seq2_words[j - 1]))
            j -= 1

    return operations[::-1]

def print_matrix(matrix, seq1_words, seq2_words):
    print("      ", end="")
    for word in seq2_words:
        print(f"  {word:5}", end="")  # Column headers (seq2)
    print()

    for i, word in enumerate(seq1_words):
        print(f"{word:5} ", end="")  # Row headers (seq1)
        for j in range(len(seq2_words) + 1):
            print(f"{matrix[i][j]:3} ", end="") 
        print()

demo = Demo()
load_model()
import time

def update_demo_data(demo,message_num):
    if demo.message_num < 1:
        demo.data['speaker1_pre'] = ' '
        demo.data['speaker2_pre'] = ' '
        demo.message_num = 0
    else:
        demo.data['speaker1_pre'] = from_A[demo.message_num-1]
        demo.data['speaker2_pre'] = from_B[demo.message_num-1]
    demo.data['speaker1'] = from_A[demo.message_num]
    demo.data['speaker2'] = from_B[demo.message_num]
    demo.data['speaker1_post'] = from_A[demo.message_num+1]
    demo.data['speaker2_post'] = from_B[demo.message_num+1] 
    return None

import PySimpleGUI as sg
# import PySimpleGUIQt as sg


MLINE_KEY = '-MLINE-'

layout = [  [sg.Text('Try to pronounce the highlighted sentence')],
            [sg.Multiline(size=(170,40), key=MLINE_KEY, reroute_cprint=True, write_only=False)],
            [sg.Button('Start'),sg.Button('Exit'),sg.Button('Back'),sg.Button('Talk'), sg.B('Next'),sg.Button('Show Errors')]  ]

window = sg.Window('TRPHonemenes', layout)

# print = lambda *args, **kwargs: window[MLINE_KEY].print(*args, **kwargs, text_color='red')
mline:sg.Multiline = window[MLINE_KEY]
pass_reading = False

while True:
    while demo.message_num < len(from_B):
        if pass_reading:
            pass_reading = False
        else:
            event, values = window.read() 

        if event in ('WIN_CLOSED', 'Exit'):
            window.close()
            break
        if 'Start' in event:
            demo.message_num = 0
            update_demo_data(demo,demo.message_num)
            mline.update(demo.data['speaker1_pre'], text_color_for_value='grey', append=False)   
            mline.update(demo.data['speaker2_pre'], text_color_for_value='grey', append=True)   
            mline.update(demo.data['speaker1'], text_color_for_value='black', append=True)   
            mline.update(demo.data['speaker2'], text_color_for_value='blue', background_color_for_value='yellow', append=True)   
            mline.update(demo.data['speaker1_post'], text_color_for_value='grey', append=True)
            mline.update(demo.data['speaker2_post'], text_color_for_value='grey', append=True)

        if 'Back' in event:
            demo.message_num -= 1
            update_demo_data(demo,demo.message_num)
            mline.update(demo.data['speaker1_pre'], text_color_for_value='grey', append=False)   
            mline.update(demo.data['speaker2_pre'], text_color_for_value='grey', append=True)   
            mline.update(demo.data['speaker1'], text_color_for_value='black', append=True)   
            mline.update(demo.data['speaker2'], text_color_for_value='blue',  background_color_for_value='yellow', append=True)   
            mline.update(demo.data['speaker1_post'], text_color_for_value='grey', append=True)
            mline.update(demo.data['speaker2_post'], text_color_for_value='grey', append=True)


        if 'Next' in event:
            demo.message_num += 1    
            update_demo_data(demo,demo.message_num)

            mline.update(demo.data['speaker1_pre'], text_color_for_value='grey', append=False)   
            mline.update(demo.data['speaker2_pre'], text_color_for_value='green', append=True)   
            mline.update(demo.data['speaker1'], text_color_for_value='black', append=True)   
            mline.update(demo.data['speaker2'], text_color_for_value='blue', background_color_for_value='yellow', append=True)   
            mline.update(demo.data['speaker1_post'], text_color_for_value='grey', append=True)
            mline.update(demo.data['speaker2_post'], text_color_for_value='grey', append=True)

        if 'Talk' in event:

            filename = "audio" + str(demo.message_num) + ".wav"

            save_audio(filename)

            mline.update(demo.data['speaker1_pre'], text_color_for_value='grey', append=False)   
            mline.update(demo.data['speaker2_pre'], text_color_for_value='grey', append=True)   
            mline.update(demo.data['speaker1'], text_color_for_value='black', append=True)   
            mline.update(demo.data['speaker2'], text_color_for_value='black', background_color_for_value='yellow', append=True)   
            mline.update(demo.data['speaker1_post'], text_color_for_value='grey', append=True)
            mline.update(demo.data['speaker2_post'], text_color_for_value='grey', append=True)

            texto = transcribe_audio(filename)
            texto_ipa = transcribe_ipa(texto)
            demo.data['phonemes'] = texto_ipa[0]
            theoretical_ipa = transcribe_ipa(demo.data['speaker2'])
            #split the phonemes into words
            ipa_list = texto_ipa[0].split()
            theoretical_ipa_list = theoretical_ipa.split()
            #split the phonemes into characters
            ipa_lista = list(texto_ipa[0])
            theoretical_ipa_lista = list(theoretical_ipa)

            if ipa_lista == theoretical_ipa_lista:
                mline.update(demo.data['speaker1_pre'], text_color_for_value='grey', append=False)   
                mline.update(demo.data['speaker2_pre'], text_color_for_value='grey', append=True)   
                mline.update(demo.data['speaker1'], text_color_for_value='black', append=True)   
                mline.update(demo.data['speaker2'], text_color_for_value='black', background_color_for_value='green', append=True)   
                mline.update(demo.data['speaker1_post'], text_color_for_value='grey', append=True)
                mline.update(demo.data['speaker2_post'], text_color_for_value='grey', append=True)
                time.sleep(0.5)
                event = 'Next'
                pass_reading = True
            else:
            # compare the transcribed text with the message from B
                #print("Incorrect! --------- You said -----------------")
                # text[0] to list of words  
                mline.update("Incorrect! --------- You said --------------------\n", text_color_for_value='black', append=False) 
                list_text = texto[0].split()
                output = ""
                output_ipa = ""
                for item in list_text:
                    output += item + '\t'
                output += '\n'
                mline.update(output, text_color_for_value='black', append=True) 
                for item in ipa_list:
                    output_ipa += item + '\t'
                output_ipa += '\n'
                mline.update(output_ipa, text_color_for_value='black', append=True) 
                list_theoretical = demo.data['speaker2'].split()
                #print("--------- You have to say -------------------------")
                mline.update("--------- You have to say -------------------------\n", text_color_for_value='blue', append=True) 

                output = ""
                output_ipa = ""
                for item in list_theoretical:
                    output += item + '\t'
                output += '\n'
                mline.update(output, text_color_for_value='blue', append=True) 
                for item in theoretical_ipa_list:
                    output_ipa += item + '\t'
                output_ipa += '\n'
                mline.update(output_ipa, text_color_for_value='black', append=True) 
                compara_ipa = ""
                event, values = window.read() 
                if 'Show Errors' in event:
                    matrix = calculate_path(ipa_list, theoretical_ipa_list)
                    edit_operations = backtrack_path(matrix, ipa_list, theoretical_ipa_list)
                    #print("--------- Edit operations: -------------------------")
                    mline.update("--------- Edit operations: -------------------------\n", text_color_for_value='black', append=True) 
                    for operation in edit_operations:
                        show_op = f" * {operation[0]}: {operation[1]} with {operation[2]}" if operation[0] != "Deletion" else f" * {operation[0]}: {operation[1]}"
                        show_op += '\n'
                        mline.update(show_op, text_color_for_value='black', append=True) 

                    #mline.update("--------- Edit operations (matrix): -----------------\n", text_color_for_value='black', append=True) 
                    print_matrix(matrix, ipa_list, theoretical_ipa_list)
                    # print = "      " + ""
                    # for word in theoretical_ipa_list:
                    #     print += f"  {word:5}" + ""  # Column headers (seq2)
                    # print += '\n'
                    # mline.update(print, text_color_for_value='black', append=True) 

                    # for i, word in enumerate(ipa_list):
                    #     print = f"{word:5} " + ""  # Row headers (seq1)
                    #     for j in range(len(theoretical_ipa_list) + 1):
                    #         print += f"{matrix[i][j]:3} " + "" 
                    #     print += '\n'   
                    #     mline.update(print, text_color_for_value='black', append=True) 


