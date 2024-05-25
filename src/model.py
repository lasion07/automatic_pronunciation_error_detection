import torch
import numpy as np
import src.utils.WordMetrics as WordMetrics
import src.utils.WordMatching as wm

from itertools import groupby
from string import punctuation
from src.utils.eng2ipa import eng2ipa
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2Processor
from phonemizer.backend.espeak.wrapper import EspeakWrapper

_ESPEAK_LIBRARY = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib'  #use the Path to the library.
EspeakWrapper.set_library(_ESPEAK_LIBRARY)


class EnglishModel:
    current_transcript: str
    current_ipa: str

    current_recorded_audio: torch.Tensor
    current_recorded_transcript: str
    current_recorded_word_locations: list
    current_recorded_intonations: torch.tensor
    current_words_pronunciation_accuracy = []
    categories_thresholds = np.array([80, 60, 59])

    sampling_rate = 16000

    def __init__(self) -> None:
        checkpoint = "bookbot/wav2vec2-ljspeech-gruut"
        self.model = AutoModelForCTC.from_pretrained(checkpoint)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.sr = self.processor.feature_extractor.sampling_rate

    ##################### ASR Functions ###########################
    
    def recognize(self, recordedAudio: torch.Tensor = None):
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
        
        inputs = self.processor(recordedAudio, sampling_rate=self.sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs["input_values"]).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        pred_string = decode_phonemes(predicted_ids[0], self.processor, ignore_stress=True)

        return pred_string

    ##################### END ASR Functions ###########################

    ##################### Evaluation Functions ###########################
    def matchSampleAndRecordedWords(self, real_text, recorded_transcript):
        words_estimated = recorded_transcript.split()

        if real_text is None:
            words_real = self.current_transcript[0].split()
        else:
            words_real = real_text.split()

        mapped_words, mapped_words_indices = wm.get_best_mapped_words(
            words_estimated, words_real)

        real_and_transcribed_words = []
        real_and_transcribed_words_ipa = []
        for word_idx in range(len(words_real)):
            if word_idx >= len(mapped_words)-1:
                mapped_words.append('-')
            real_and_transcribed_words.append(
                (words_real[word_idx],    mapped_words[word_idx]))
            real_and_transcribed_words_ipa.append((eng2ipa(words_real[word_idx]),
                                                   eng2ipa(mapped_words[word_idx])))
        return real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices

    def getPronunciationAccuracy(self, real_and_transcribed_words_ipa) -> float:
        total_mismatches = 0.
        number_of_phonemes = 0.
        current_words_pronunciation_accuracy = []
        for pair in real_and_transcribed_words_ipa:

            real_without_punctuation = self.removePunctuation(pair[0]).lower()
            number_of_word_mismatches = WordMetrics.edit_distance_python(
                real_without_punctuation, self.removePunctuation(pair[1]).lower())
            total_mismatches += number_of_word_mismatches
            number_of_phonemes_in_word = len(real_without_punctuation)
            number_of_phonemes += number_of_phonemes_in_word

            current_words_pronunciation_accuracy.append(float(
                number_of_phonemes_in_word-number_of_word_mismatches)/number_of_phonemes_in_word*100)

        percentage_of_correct_pronunciations = (
            number_of_phonemes-total_mismatches)/number_of_phonemes*100

        return np.round(percentage_of_correct_pronunciations), current_words_pronunciation_accuracy

    def removePunctuation(self, word: str) -> str:
        return ''.join([char for char in word if char not in punctuation])

    def getWordsPronunciationCategory(self, accuracies) -> list:
        categories = []

        for accuracy in accuracies:
            categories.append(
                self.getPronunciationCategoryFromAccuracy(accuracy))

        return categories

    def getPronunciationCategoryFromAccuracy(self, accuracy) -> int:
        return np.argmin(abs(self.categories_thresholds-accuracy))
