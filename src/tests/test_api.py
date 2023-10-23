from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient

from app.api import app

from PIL import Image

import json
import numpy as np


@pytest.fixture(scope="module", autouse=True)
def client():
    # Use the TestClient with a `with` statement to trigger the startup and shutdown events.
    with TestClient(app) as client:
        return client


@pytest.fixture
def payload():
    return {'../data/test/image_1.png'}


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


def test_model_prediction(client, payload):
    # Use constants if fixture created
    if os.path.isfile(payload):
        _files = {'uploadFile': open(pyload, 'rb')}
        response = client.post('/models}}',
                           params={
                               "accept": {{application/json}}
                           },
                           files=_files
                           )
        assert response.status_code == 200
    else:
        pytest.fail("Scratch file does not exists.")
