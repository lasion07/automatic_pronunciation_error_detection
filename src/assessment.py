import os
import time
import torch
import json
import string
import base64
import random
import audioread
import numpy as np
import src.utils.WordMatching as wm

from torchaudio.transforms import Resample
from src.model import EnglishModel


model = EnglishModel()

transform = Resample(orig_freq=48000, new_freq=16000)


def lambda_handler(event, context):
    data = json.loads(event['body'])

    # Get values
    real_text = data['title']
    audio_bytes = base64.b64decode(
        data['base64Audio'][22:].encode('utf-8')) # 22 is default

    # Check real_text exists
    if len(real_text) == 0:
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Credentials': "true",
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': ''
        }

    # Write audio to a temp file
    start = time.time()
    letters = string.ascii_lowercase
    random_string = ''.join(random.choice(letters) for i in range(20))
    audio_chunk_path = 'audio_chunks/' + random_string + '.ogg'
    f = open(audio_chunk_path, 'wb')
    f.write(audio_bytes)
    f.close()
    print('Time for saving binary in file: ', str(time.time()-start))

    # Read the audio file and transform it
    start = time.time()
    signal, fs = audioread_load(audio_chunk_path)

    signal = transform(torch.Tensor(signal)).unsqueeze(0)

    print('Time for loading .ogg file file: ', str(time.time()-start))

    # Run model
    result = model.processAudioForGivenText(
        signal, real_text)

    # Remove the audio file
    start = time.time()
    os.remove(audio_chunk_path)
    print('Time for deleting file: ', str(time.time()-start))

    start = time.time()
    real_transcripts_ipa = ' '.join(
        [word[0] for word in result['real_and_transcribed_words_ipa']])
    matched_transcripts_ipa = ' '.join(
        [word[1] for word in result['real_and_transcribed_words_ipa']])

    real_transcripts = ' '.join(
        [word[0] for word in result['real_and_transcribed_words']])
    matched_transcripts = ' '.join(
        [word[1] for word in result['real_and_transcribed_words']])

    words_real = real_transcripts.lower().split()
    mapped_words = matched_transcripts.split()

    is_letter_correct_all_words = ''
    for idx, word_real in enumerate(words_real):

        mapped_letters, mapped_letters_indices = wm.get_best_mapped_words(
            mapped_words[idx], word_real)

        is_letter_correct = wm.getWhichLettersWereTranscribedCorrectly(
            word_real, mapped_letters)  # , mapped_letters_indices)

        is_letter_correct_all_words += ''.join([str(is_correct)
                                                for is_correct in is_letter_correct]) + ' '

    pair_accuracy_category = ' '.join(
        [str(category) for category in result['pronunciation_categories']])
    print('Time to post-process results: ', str(time.time()-start))

    res = {'real_transcript': result['recording_transcript'],
           'ipa_transcript': result['recording_ipa'],
           'pronunciation_accuracy': str(int(result['pronunciation_accuracy'])),
           'real_transcripts': real_transcripts, 'matched_transcripts': matched_transcripts,
           'real_transcripts_ipa': real_transcripts_ipa,
           'matched_transcripts_ipa': matched_transcripts_ipa,
           'pair_accuracy_category': pair_accuracy_category,
           'start_time': result['start_time'],
           'end_time': result['end_time'],
           'is_letter_correct_all_words': is_letter_correct_all_words}

    # {'real_transcript': 'what regisable do you useer grove', 
    #  'ipa_transcript': 'wət regisable du ju useer groʊv', 
    #  'pronunciation_accuracy': '63', 
    #  'real_transcripts': 'What vegetables do you usually grow?', 
    #  'matched_transcripts': 'what regisable do you useer grove', 
    #  'real_transcripts_ipa': 'wət ˈvɛʤtəbəlz du ju ˈjuʒəwəli groʊ?', 
    #  'matched_transcripts_ipa': 'wət regisable du ju useer groʊv', 
    #  'pair_accuracy_category': '0 1 0 0 2 2', 
    #  'start_time': '0.6198125 1.25025 2.1959375 2.432375 2.66875 3.2204375', 
    #  'end_time': '0.95625 2.217125 2.532375 2.76875 3.241625 3.8720625', 
    #  'is_letter_correct_all_words': '1111 0110011110 11 111 1100000 11101 '}

    return json.dumps(res)

# From Librosa


def audioread_load(path, offset=0.0, duration=None, dtype=np.float32):
    """Load an audio buffer using audioread.

    This loads one block at a time, and then concatenates the results.
    """

    y = []
    with audioread.audio_open(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + \
                (int(np.round(sr_native * duration)) * n_channels)

        n = 0

        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[: s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, sr_native

# From Librosa


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)
