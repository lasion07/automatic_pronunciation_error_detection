import io
import time
import torch
from pydub import AudioSegment
from speech_recognition import Recognizer, Microphone
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

torch.random.manual_seed(0)

# import model, feature extractor, tokenizer
model_name = 'facebook/wav2vec2-lv-60-espeak-cv-ft'
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

recognizer = Recognizer()

start_time = time.time()

with Microphone(sample_rate=16000) as source:
  print("You can start speaking now...")
  while True:
    audio = recognizer.listen(source, phrase_time_limit=3) # Bytes
    print(time.time() - start_time)

    start_time = time.time()
    data = io.BytesIO(audio.get_wav_data()) # Object(Bytes) Ex: 96300
    clip = AudioSegment.from_file(data) # Object(Object) Ex: 96300
    x = torch.FloatTensor(clip.get_array_of_samples()) # Tensor(Array(Int)) Ex: 48128 =>  1 Int = 2 Bytes

    # tokenize
    input_values = processor(x, return_tensors="pt").input_values
    
    # retrieve logits
    with torch.no_grad():
      logits = model(input_values).logits
    
    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    wav2vec2_time = time.time() - start_time

    print('You said:', str(transcription), wav2vec2_time)
    