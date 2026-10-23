import os
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "fake_news_model.joblib"
)

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        news_text = str(data.get("news", "")).strip()

        if not news_text:
            return jsonify({"error": "Please enter news text."}), 400

        if len(news_text) < 30:
            return jsonify({
                "error": "Please enter at least 30 characters."
            }), 400

        if len(news_text) > 20000:
            return jsonify({
                "error": "News text must be under 20,000 characters."
            }), 400

        prediction = int(model.predict([news_text])[0])
        probabilities = model.predict_proba([news_text])[0]

        class_index = list(model.classes_).index(prediction)
        score = float(probabilities[class_index]) * 100

        return jsonify({
            "prediction": "REAL NEWS" if prediction == 1 else "FAKE NEWS",
            "score": round(score, 2),
            "disclaimer": (
                "This is an ML prediction, not a fact-check. "
                "Verify important claims using reliable sources."
            )
        })

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )