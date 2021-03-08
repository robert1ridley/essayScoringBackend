import json
from flask import Flask, request, Response
from models.scoring_model import ScoringModel

app = Flask(__name__)

model = ScoringModel(
        'resources/vocab.json',
        'resources/org_word_hi_att_text_only_model',
        [
            'resources/HSK_Level_1.xls',
            'resources/HSK_Level_2.xls',
            'resources/HSK_Level_3.xls',
            'resources/HSK_Level_4.xls',
            'resources/HSK_Level_5.xls',
            'resources/HSK_Level_6.xls',
        ],
        'resources/idiomlist.txt'
    )


def validate_request(req_json):
    try:
        original_text = req_json['text']
    except (TypeError, KeyError):
        return json.dumps({"error": "Please send the parameter 'text' with your request."}), None
    if len(original_text) == 0:
        return json.dumps({"error": "Input text too short."}), None
    return None, original_text


@app.route("/api/v1/score", methods=['POST'])
def get_score():
    req_json = request.get_json()

    validation_result, original_text = validate_request(req_json)
    if validation_result:
        return Response(validation_result,
                        status=200, mimetype='application/json')
    model.set_essay_text(original_text)
    model.process_essay_text()
    score = model.predict_score()
    model.get_idiom_prop()
    model.get_high_vocab_prop()
    vocab_score = model.get_vocab_score()
    readability_score = model.get_readability_score()
    res = json.dumps({"score": score, 'vocabulary': vocab_score, 'readability_score': readability_score, "error": None})
    res = Response(res, status=200, mimetype='application/json')
    return res


@app.errorhandler(404)
def not_found(error):
    return Response(json.dumps({'error': 'Endpoint not found'}), status=404, mimetype='application/json')


if __name__ == '__main__':
    app.run(
        host='0.0.0.0'
    )