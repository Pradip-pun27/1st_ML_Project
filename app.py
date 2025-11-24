import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Try to load model and pipeline; if missing, we'll show a friendly message in the UI
MODEL_PATH = 'model.pkl'
PIPELINE_PATH = 'pipeline.pkl'

model = None
pipeline = None
_load_error = None
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(PIPELINE_PATH):
        model = joblib.load(MODEL_PATH)
        pipeline = joblib.load(PIPELINE_PATH)
    else:
        _load_error = 'Saved model or pipeline not found. Run the training script to create them.'
except Exception as e:
    _load_error = f'Error loading model/pipeline: {e}'


NUMERIC_FIELDS = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income'
]
CAT_FIELD = 'ocean_proximity'
ALL_FIELDS = NUMERIC_FIELDS + [CAT_FIELD]


def parse_and_validate_form(form):
    """Parse request.form and return (df, None) on success or (None, error_message)."""
    values = {}
    # numeric
    for name in NUMERIC_FIELDS:
        raw = form.get(name, '')
        if raw is None or str(raw).strip() == '':
            return None, f'Missing value for {name}.'
        try:
            values[name] = float(str(raw).strip())
        except ValueError:
            return None, f'Invalid numeric value for {name}: "{raw}"'

    # categorical
    cat = form.get(CAT_FIELD, '')
    if cat is None or str(cat).strip() == '':
        return None, f'Missing value for {CAT_FIELD}.'
    values[CAT_FIELD] = str(cat).strip()

    # Build one-row dataframe preserving column order
    df = pd.DataFrame([values], columns=ALL_FIELDS)
    return df, None


@app.route('/')
def home():
    note = None
    if _load_error:
        note = _load_error
    return render_template('index.html', prediction_text=None, load_note=note)


@app.route('/predict', methods=['POST'])
def predict():
    # Validate inputs
    df, err = parse_and_validate_form(request.form)
    if err:
        return render_template('index.html', prediction_text=f'Input error: {err}')

    if model is None or pipeline is None:
        return render_template('index.html', prediction_text=(
            'Model is not available. Please train the model first by running the training script.'
        ))

    try:
        X_trans = pipeline.transform(df)
        pred = model.predict(X_trans)[0]
    except Exception as e:
        return render_template('index.html', prediction_text=f'Prediction error: {e}')

    return render_template('index.html', prediction_text=f'Predicted median house value: ${pred:,.2f}')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint. Accepts JSON body with the same field names and returns JSON."""
    data = None
    # Accept JSON or form-encoded
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    df, err = parse_and_validate_form(data)
    if err:
        return jsonify({'error': err}), 400

    if model is None or pipeline is None:
        return jsonify({'error': 'Model not available. Train the model first.'}), 503

    try:
        X_trans = pipeline.transform(df)
        pred = float(model.predict(X_trans)[0])
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    return jsonify({'median_house_value': pred})


if __name__ == '__main__':
    # When running locally this prints helpful info
    if _load_error:
        print('WARNING:', _load_error)
    app.run(debug=True)
