import io
import time

import torch
import torchaudio

from pydub import AudioSegment
from speech_recognition import Recognizer, Microphone


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])
    

print(torch.__version__)
print(torchaudio.__version__)

torch.random.manual_seed(0)
device = torch.device("mps" if torch.has_mps else "cpu")

print(device)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print("Sample Rate:", bundle.sample_rate)

print("Labels:", bundle.get_labels())

model = bundle.get_model().to(device)

print(model.__class__)

recognizer = Recognizer()

with Microphone(sample_rate=16000) as source:
  print("You can start speaking now...")
  while True:
    audio = recognizer.listen(source, phrase_time_limit=3) # Bytes

    start_time = time.time()
    data = io.BytesIO(audio.get_wav_data()) # Object(Bytes) Ex: 96300
    clip = AudioSegment.from_file(data) # Object(Object) Ex: 96300
    waveform = torch.FloatTensor([clip.get_array_of_samples()]).to(device) # Tensor(Array(Int)) Ex: 48128 =>  1 Int = 2 Bytes

    with torch.inference_mode():
        # features, _ = model.extract_features(waveform)
        emission, _ = model(waveform)
        decoder = GreedyCTCDecoder(labels=bundle.get_labels())
        transcript = decoder(emission[0])
    end_time = time.time()

    print('You said:', transcript)
    print(end_time - start_time)