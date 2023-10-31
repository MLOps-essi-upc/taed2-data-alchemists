from http import HTTPStatus
import os
import sys
from io import BytesIO
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import pytest
from fastapi.testclient import TestClient
import io
from api import app


@pytest.fixture(scope="module", autouse=True)
def client():
    # Use the TestClient with a `with` statement to trigger the startup and shutdown events.
    with TestClient(app) as client:
        return client

"""
@pytest.fixture
def payload():
    return '../../data/test/image_1.png'
"""

def test_root(client):
    response = client.get("/")
    json = response.json()
    assert response.status_code == 200
    assert (
        json["data"]["message"]
        == "This is a digit recognizer model. Please update an image of a digit and our model will identify it!"
    )
    assert json["message"] == "OK"
    assert json["status-code"] == 200
    assert json["method"] == "GET"
    assert json["timestamp"] is not None

def test_models(client):
    response = client.get("/models")
    json = response.json()
    assert response.status_code == 200
    assert (
        json["data"]["models"]
        == [
              {
                "name": "cnn_digit_recognizer",
                "metrics": {
                  "batch_size": 100,
                  "learning_rate": 0.001,
                  "num_classes": 10,
                  "dataset_size": 5250
                }
              }
            ]

    )
    assert json["message"] == "OK"
    assert json["status-code"] == 200
    assert json["method"] == "GET"
    assert json["timestamp"] is not None


def test_model_prediction(client):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the image file name
    image_filename = "image_prova.png"
    # Construct the full path to the image file
    image_path = os.path.join(script_dir, image_filename)
    if os.path.isfile(image_path):
        #_files = {'uploadFile': open(image, 'rb')}
        # Define the URL and additional headers
        url = "http://127.0.0.1:8000/models"
        #headers = {"accept": "application/json"}

        # Send the POST request
        response = client.post("http://127.0.0.1:8000/models/main", files={"file": ("filename", open(image_filename, "rb"), "image/jpeg")})

        assert response.status_code == 200
    else:
        pytest.fail("File does not exist.")
