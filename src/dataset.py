import json
import random

import pandas as pd

from src.utils.eng2ipa import eng2ipa

dataset_name = 'cv-valid-test'
dataset_information = pd.read_csv(f'data/{dataset_name}.csv')

def lambda_handler(event, context):
    body = json.loads(event['body'])
    
    sample_index = random.randint(0, len(dataset_information))
    sample_audio_path, sample_transcription, *_ = dataset_information.iloc[sample_index]
    translated_trascript = ""

    sample_ipa = eng2ipa(sample_transcription)

    result = {'real_transcript': sample_transcription,
              'ipa_transcript': sample_ipa,
              'transcript_translation': translated_trascript}

    return json.dumps(result)
