import torch
import numpy as np
import WordMetrics
import WordMatching as wm
from string import punctuation

import eng_to_ipa as ei
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC


def eng2ipa(text):
    ipa_text = ei.convert(text).replace('*','')
    return ipa_text

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
        model_name = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    ##################### ASR Functions ###########################
    
    def processAudioForGivenText(self, recordedAudio: torch.Tensor = None, real_text=None):
        def speech_recognize(recordedAudio: torch.Tensor, return_offsets=True):
            # data = io.BytesIO(audio)
            # clip = AudioSegment.from_file(data)
            waveform = recordedAudio[0] # torch.FloatTensor(clip.get_array_of_samples())

            # forward sample through model to get greedily predicted transcription ids
            input_values = self.feature_extractor(waveform, sampling_rate=16_000, return_tensors="pt", padding="longest").input_values
            logits = self.model(input_values).logits[0]
            pred_ids = torch.argmax(logits, axis=-1)

            # retrieve word stamps (analogous commands for `output_char_offsets`)
            outputs = self.tokenizer.decode(pred_ids, output_word_offsets=True)
            # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
            time_offset = self.model.config.inputs_to_logits_ratio / self.feature_extractor.sampling_rate

            start_time, end_time = "", ""
            for d in outputs.word_offsets:
                start_time += str(round(d["start_offset"] * time_offset, 2)) + " "
                end_time += str(round(d["end_offset"] * time_offset, 2)) + " "

            recording_transcript = outputs.text.lower()
            recording_ipa = eng2ipa(recording_transcript)

            return recording_transcript, recording_ipa, start_time, end_time
    
        recording_transcript, recording_ipa, start_time, end_time = speech_recognize(recordedAudio)

        print(recording_transcript, recording_ipa, start_time, end_time, sep='\n')

        real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices = self.matchSampleAndRecordedWords(
            real_text, recording_transcript)

        pronunciation_accuracy, current_words_pronunciation_accuracy = self.getPronunciationAccuracy(
            real_and_transcribed_words)  # _ipa

        pronunciation_categories = self.getWordsPronunciationCategory(
            current_words_pronunciation_accuracy)
        
        # print({'recording_transcript': recording_transcript,
        #         'real_and_transcribed_words': real_and_transcribed_words,
        #         'recording_ipa': recording_ipa,
        #         'start_time': start_time,
        #         'end_time': end_time,
        #         'real_and_transcribed_words_ipa': real_and_transcribed_words_ipa, 'pronunciation_accuracy': pronunciation_accuracy,
        #         'pronunciation_categories': pronunciation_categories})

        return {'recording_transcript': recording_transcript,
                'real_and_transcribed_words': real_and_transcribed_words,
                'recording_ipa': recording_ipa,
                'start_time': start_time,
                'end_time': end_time,
                'real_and_transcribed_words_ipa': real_and_transcribed_words_ipa, 'pronunciation_accuracy': pronunciation_accuracy,
                'pronunciation_categories': pronunciation_categories}

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
