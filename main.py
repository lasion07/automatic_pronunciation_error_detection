import os

import numpy as np
import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from src.utils import LogMelSpectrogram
from src.text_process import TextProcess


def inference_pipeline(audio_input):
    """
        Input: audio_input
        Output: label sequence, predicted sequence, and error
    """
    n_fft = 1024
    n_mels = 128
    win_length = 400  # 40ms
    hop_length = 200  # 20ms

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    text_process = TextProcess()
    vocab_size = text_process.n_class

    teacher_checkpoint = torch.load(
        f"models/teacher_2_hidden_libri_subsampling_swish_final.pt"
    )
    teacher_ckpt = dict()
    for key, val in teacher_checkpoint["conformer_state_dict"].items():
        teacher_ckpt[key.split(".", 1)[-1]] = val  # remove the [module.]...

    wav2vec2_conformer_teacher = load_conformer_pretrained(teacher_hiddens)
    teacher_model = ConformerModel(
        wav2vec2_conformer_teacher, input_dim=n_mels, vocab_size=vocab_size
    )
    teacher_model = teacher_model.to(device)
    teacher_model.load_state_dict(teacher_ckpt)

    feature_transform = LogMelSpectrogram(
            n_fft=n_fft, n_mels=n_mels, win_length=win_length, hop_length=hop_length
        )

    waveform, sample_rate = audio_input

    spectrogram = feature_transform(waveform).permute(0, 2, 1).squeeze()

    pred_sequence = recognize(spectrogram.unsqueeze(0).to(device), torch.tensor([spectrogram.shape[0]]), teacher_model)

    return pred_sequence


if __name__ == '__main__':
    """
        Pronunciation Assessment
    """
    
    # Read audio_input
    file_audio_path = ''
    audio_input = torchaudio.load(file_audio_path)

    # Infer
    inference_pipeline(audio_input)