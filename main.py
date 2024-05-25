from flask import Flask, render_template, request
import webbrowser
import os
from flask_cors import CORS
import json

import src.assessment as assessment
import src.dataset as dataset

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = '*'

rootPath = ''


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
    return assessment.lambda_handler(event, [])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
