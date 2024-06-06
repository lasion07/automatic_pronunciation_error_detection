import os
import json

from flask_cors import CORS
from flask import Flask, render_template, request

import src.assessment as assessment
import src.dataset as dataset

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = '*'

rootPath = ''

data = {
    'status': 200, 
    'text_list': ['without the dataset the article is useless', "i've got to go to him", 'and you know it', 'strange images passed through my mind', 'the sheep had taught him that', "the shower's in there", 'follow the instructions here', 'the shop is closed on mondays', 'even coming down on the train together she wrote me', "i'm going away he said"], 
    'ipa_text_list': ['wɪˈθaʊt ðə ˈdeɪtəˌsɛt ðə ˈɑrtɪkəl ɪz ˈjusləs', 'aɪv gɑt tɪ goʊ tɪ ɪm', 'ənd ju noʊ ɪt', 'streɪnʤ ˈɪmɪʤɪz pæst θru maɪ maɪnd', 'ðə ʃip hæd tɔt ɪm ðət', "ðə shower's ɪn ðɛr", 'ˈfɑloʊ ðə ˌɪnˈstrəkʃənz hir', 'ðə ʃɑp ɪz kloʊzd ɔn ˈmənˌdeɪz', 'ˈivɪn ˈkəmɪŋ daʊn ɔn ðə treɪn təˈgɛðər ʃi roʊt mi', 'əm goʊɪŋ əˈweɪ hi sɛd']}

data_acc = {
  "score": 0.48,
  "error_char_indexes": ["false", "false", "false", "false", "true", "false", "false", "false", "true", "false", "false", "true", "true", "true", "true", "false", "true", "true", "true", "false", "true", "true", "true", "true", "true", "false", "false", "true", "true", "false", "false", "false", "false"],
  "word_scores": [0.17, 0.0, 0.62, 1.0, 0.86, 0.0, 0.33]
}


@app.route(rootPath+'/')
def main():
    return render_template('main.html')

@app.route(rootPath+'/getData')
def GetData():
    # return data
    return dataset.get_data()

@app.route(rootPath+'/GetAccuracyFromRecordedAudio', methods=['POST'])
def GetAccuracyFromRecordedAudio():
    event = {'body': json.dumps(request.get_json(force=True))}
    # return assessment.lambda_handler(event, [])
    return data_acc

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
