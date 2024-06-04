from flask import Flask, render_template, request
import webbrowser
import os
from flask_cors import CORS
import json
import pandas as pd 

import src.assessment as assessment
import src.dataset as dataset
import base64

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = '*'

rootPath = ''
#=================================
@app.route(rootPath+'/getall')
def getall():
    df = pd.read_csv('./data/cv-valid-test.csv')
    file_audio = df['filename'][:6]
    text = df['text'][:6]
    audio_files_base64 = []
    for filename in file_audio:
        filename = filename.split('/')[1]
        with open(f'./data/{filename}', 'rb') as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
            audio_files_base64.append(audio_base64)
    data = {
        'status': 200,
        'text': list(text),
        'audio': audio_files_base64,
    }
    return data

#===============================================================


@app.route(rootPath+'/')
def main():
    return render_template('main.html')


@app.route(rootPath+'/getSample', methods=['POST'])
def getNext():
    event = {'body':  json.dumps(request.get_json(force=True))}
    return dataset.lambda_handler(event, [])


@app.route(rootPath+'/GetAccuracyFromRecordedAudio', methods=['POST'])
def GetAccuracyFromRecordedAudio():
    event = {'body': json.dumps(request.get_json(force=True))}
    print(assessment.lambda_handler(event, []))
    return assessment.lambda_handler(event, [])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
