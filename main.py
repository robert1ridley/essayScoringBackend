import json
from flask import Flask, request, Response
from models.scoring_model import ScoringModel

application = app = Flask(__name__)


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
    model = ScoringModel(original_text, 'resources/vocab.json', 'resources/org_word_hi_att_text_only_model')
    model.load_vocab()
    model.load_model()
    model.process_essay_text()
    score = model.predict_score()
    res = json.dumps({"score": score, "error": None})
    res = Response(res, status=200, mimetype='application/json')
    return res


@app.errorhandler(404)
def not_found(error):
    return Response(json.dumps({'error': 'Endpoint not found'}), status=404, mimetype='application/json')


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=3000,
    )