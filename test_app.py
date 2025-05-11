import os
import pytest
import app


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


def test_model_score(ensure_data):
    score = app.main()  # Returns the accuracy
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
