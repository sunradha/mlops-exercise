import os
import pytest
import app
import json


@pytest.fixture(scope="module")
def ensure_data():
    filepath = 'data/credit_card_records.csv'
    if not os.path.exists(filepath):
        pytest.skip(f"Test skipped: Required data file not found at {filepath}")
    return filepath


def test_model_file_created(ensure_data):
    if not os.path.exists('models'):
        os.makedirs('models')

    app.main()  # Trains and saves the model
    assert os.path.exists('models/model.pkl')


def test_model_score():
    score = app.main()  # Assuming the main function returns the score
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    # Load the model scores
    with open('model_scores.json', 'r') as f:
        model_scores = json.load(f)

    # Get the latest model score
    latest_score = model_scores[-1]['score']

    # Compare the latest score with the current score
    assert score >= latest_score
