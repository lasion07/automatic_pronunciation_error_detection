import io
import time

import torch
import librosa

from itertools import groupby
from datasets import load_dataset
from pydub import AudioSegment, silence
from pydub.playback import play
from pynput import keyboard
from speech_recognition import Recognizer, Microphone
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2Processor

def decode_phonemes(
    ids: torch.Tensor, processor: Wav2Vec2Processor, ignore_stress: bool = False
) -> str:
    """CTC-like decoding. First removes consecutive duplicates, then removes special tokens."""
    # removes consecutive duplicates
    ids = [id_ for id_, _ in groupby(ids)]

    special_token_ids = processor.tokenizer.all_special_ids + [
        processor.tokenizer.word_delimiter_token_id
    ]
    # converts id to token, skipping special tokens
    phonemes = [processor.decode(id_) for id_ in ids if id_ not in special_token_ids]

    # joins phonemes
    prediction = " ".join(phonemes)

    # whether to ignore IPA stress marks
    if ignore_stress == True:
        prediction = prediction.replace("ˈ", "").replace("ˌ", "")

    return prediction


def infer(waveform, feature_extractor, model, tokenizer) -> list:
  # forward sample through model to get greedily predicted transcription ids
    input_values = feature_extractor(waveform, sampling_rate=16_000, return_tensors="pt", padding="longest").input_values
    logits = model(input_values).logits[0]
    pred_ids = torch.argmax(logits, axis=-1)

    # retrieve word stamps (analogous commands for `output_char_offsets`)
    outputs = tokenizer.decode(pred_ids, output_char_offsets=True)
    # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
    time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

    special_token_ids = tokenizer.all_special_ids + [tokenizer.word_delimiter_token_id]

    ipa_char_offsets = [
        {
            "ipa_char": d["char"],
            "start_time": round(d["start_offset"] * time_offset, 2),
            "end_time": round(d["end_offset"] * time_offset, 2),
        }
        for d in outputs.char_offsets if d["char"] != ' '
    ]

    ipa_recording_transcript = outputs.text.lower()

    return ipa_recording_transcript, ipa_char_offsets

def on_press(key):
    global recording
    if recording == False and key == keyboard.Key.shift:
        recording = True

# Input
# reference_audio = ''

# import model, feature extractor, tokenizer
checkpoint = "bookbot/wav2vec2-ljspeech-gruut"
model = AutoModelForCTC.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)
sr = processor.feature_extractor.sampling_rate

recognizer = Recognizer()
recognizer.energy_threshold = 400

recording = False

listener = keyboard.Listener(on_press=on_press)
listener.start()

start_time = time.time()
with Microphone(sample_rate=sr) as source:
  while True:
    if not recording:
      print('Press Shift to start record', end='\r')
      time.sleep(1)
      continue

    recording = False
    print("\nYou can start speaking now...")
    audio = recognizer.listen(source, phrase_time_limit=3) # Bytes
    
    data = io.BytesIO(audio.get_wav_data()) # Object(Bytes) Ex: 96300
    audio_segment = AudioSegment.from_file(data) # Object(Object) Ex: 96300
    waveform = torch.FloatTensor(audio_segment.get_array_of_samples()) # Tensor(Array(Int)) Ex: 48128 =>  1 Int = 2 Bytes

    start_time = time.time()
    # result = infer(waveform, phoneme_feature_extractor, phoneme_model, phoneme_tokenizer)
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs["input_values"]).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    prediction = decode_phonemes(predicted_ids[0], processor, ignore_stress=True)
    word_time = time.time() - start_time

    # text, char_offsets = result
    play(audio_segment) 
    print(prediction)
    # print(char_offsets)
    