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
EspeakWrapper.set_library("C:\Program Files\eSpeak NG\libespeak-ng.dll")

# load model and processor
def load_model(tamaño="tiny"):
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

def wavelet_inspired_compare(seq1, seq2, window_size=3):
    score = 0
    for i in range(len(seq1) - window_size + 1):
        subseq1 = seq1[i:i + window_size]
        subseq2 = seq2[i:i + window_size]

        if subseq1 == subseq2:
            score += 5  # High bonus for matching subsequence
        else:
            for p1, p2 in zip(subseq1, subseq2):
                score += 1 if p1 == p2 else -1  # Simple mismatch penalty 

    return score

path = "conversation.txt"
from_A, from_B = process_conversation(path)
# take out (and delete after) first message from A and print it
class Demo:
    def __init__(self):
        self.message_num = 0
        self.data = {
            'speaker1_pre': ' ',
            'speaker2_pre': ' ',
            'speaker1': 'I am a robot',
            'speaker2': 'I am a human',
            'speaker1_post': 'I am a robot',
            'speaker2_post': 'I am a human',
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

import pandas as pd

def create_matrix_dataframe(matrix, seq1_words, seq2_words):
    # Add empty column for row headers
    seq2_words.insert(0, "") 

    # Create DataFrame 
    df = pd.DataFrame(matrix, columns=seq2_words, index=seq1_words)

    # Remove the extra row header (empty string)
    df.index.name = None

    return df

demo = Demo()
load_model()
import time
# loop until both lists are empty
while demo.message_num < len(from_B):
    demo.data['speaker1'] = from_A[demo.message_num]
    demo.data['speaker2'] = from_B[demo.message_num]
    demo.data['speaker1_post'] = from_A[demo.message_num+1]
    demo.data['speaker2_post'] = from_B[demo.message_num+1] 
    
    filename = "audio" + str(demo.message_num) + ".wav"
    print("-------------------- Message number: " + str(demo.message_num) + " --------------------")
    print("Other one:" + demo.data['speaker1'])
    print("Your turn:" + demo.data['speaker2'])
    save_audio(filename)
    print("Transcribing audio...")
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
        print("Correct! ")
        demo.data['speaker1_pre'] = demo.data['speaker1']
        demo.data['speaker2_pre'] = demo.data['speaker2']
        demo.message_num += 1    
    else:
    # compare the transcribed text with the message from B
        print("Incorrect! --------- You said -----------------")
        # text[0] to list of words  
    
        list_text = texto[0].split()
        output = ""
        output_ipa = ""
        for item in list_text:
            output += item + '\t'
        print(output)
        for item in ipa_list:
            output_ipa += item + '\t'
        print(output_ipa)
        list_theoretical = demo.data['speaker2'].split()
        print("--------- Theoretical -----------------")
        output = ""
        output_ipa = ""
        for item in list_theoretical:
            output += item + '\t'
        print(output)
        for item in theoretical_ipa_list:
            output_ipa += item + '\t'
        print(output_ipa)
        compara_ipa = ""
        # for i in range(len(theoretical_ipa_list)):
        #     if ipa_list[i] == theoretical_ipa_list[i]:
        #         compara_ipa += ipa_list[i] + '\t'
        #     else:
        #         position = i
        #         compara_ipa += "X" + '\t'
        # print(compara_ipa)
        matrix = calculate_path(ipa_list, theoretical_ipa_list)
        edit_operations = backtrack_path(matrix, ipa_list, theoretical_ipa_list)

        for operation in edit_operations:
            print(f" * {operation[0]}: {operation[1]} with {operation[2]}") 
        # print_matrix(matrix, ipa_list, theoretical_ipa_list)
        df = create_matrix_dataframe(matrix, ipa_list, theoretical_ipa_list)
        print(df)
        try:
            input("Press Enter to continue...")
        except KeyboardInterrupt:
            pass


