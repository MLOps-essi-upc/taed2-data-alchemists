import pickle
import numpy as np
import pandas as pd
import os
from io import BytesIO

from PIL import Image
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from torchvision import datasets
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from schemas import UploadImage, DigitClass


app = FastAPI(title="Digit Classifier API",
        description="This API identifies which digit is hand-written in an image.",
        version="1.0",)

def construct_response(f):
    @wraps(f)
    async def wrap(request: Request, *args, **kwargs):
        results = await f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now(),
            "url": request.url._url,
        }
        # Add data
        if "data" in results:
            response["data"] = results["data"]


        return response

    return wrap



@app.on_event("startup")
def _load_models():
    # Define the CNN architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            ## Define layers of a CNN
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=1)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 11 * 11, 2048)
            self.fc2 = nn.Linear(2048, 10)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            ## Define forward behavior
            x = self.conv1(x)
            x = self.pool1(F.relu(self.conv2(x)))
            x = self.dropout(x)
            x = self.conv3(x)
            x = self.pool2(F.relu(self.conv4(x)))
            x = self.dropout(x)
            #print(x.shape)
            x = x.view(-1, 64 * 11 * 11)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # instantiate the CNN
    model = Net()

    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    file_path = os.path.join(parent_directory, 'models/cnn_digit_recognizer.pt')

    model.load_state_dict(torch.load(file_path))

    return model


@app.get("/")
@construct_response
async def _root(request: Request):

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "This is a digit recognizer model. Please update an image of a digit and our model will identify it!"},
    }

    return response

"""
def preprocessat(payload: UploadFile):
    # Access the uploaded image using payload.file
    img = payload.file

    # Perform any necessary preprocessing on the uploaded image
    img = Image.open(img)

    img = img.convert('L')

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Normalize pixel values to [0, 255]
    img_normalized = (img_array / 255.0 * 255).astype(int)

    # Flatten the 2D array into a 1D array
    img_flat = img_normalized.flatten()

    # Get the 'id' (last character of the image name) and convert it to an integer
    image_id = [int(image_name[-5])]


    image_id.extend(img_flat)
    # Append the 'id' and the flattened, normalized pixel values to the list
    normalized_pixel_data.append(image_id)

    # Create a DataFrame from the list of normalized pixel values
    pixel_df = pd.DataFrame(normalized_pixel_data, columns=['id'] + ['pixel{}'.format(i) for i in range(len(normalized_pixel_data[0])-1)])

    return pixel_df
"""
@app.post("/models", tags=["Predict"])
@construct_response
async def _predict(request: Request, file: UploadFile):  # Change payload to accept image file
     # Load the image and perform any necessary preprocessing
    normalized_pixel_data = []

     # Read the content of the uploaded file
    file_contents = await file.read()

    # Create a BytesIO object to simulate a file-like object
    img_data = BytesIO(file_contents)

    # Use BytesIO object as input to Image.open()
    img = Image.open(img_data)

    img = img.convert('L')

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Normalize pixel values to [0, 255]
    img_normalized = (img_array / 255.0 * 255).astype(int)

    # Flatten the 2D array into a 1D array
    img_flat = img_normalized.flatten()

    # Get the 'id' (last character of the image name) and convert it to an integer
    #image_id = [int(image_name[-5])]
    image_id = [int(0)]

    image_id.extend(img_flat)
    # Append the 'id' and the flattened, normalized pixel values to the list
    normalized_pixel_data.append(image_id)

    # Create a DataFrame from the list of normalized pixel values
    prova = pd.DataFrame(normalized_pixel_data, columns=['id'] + ['pixel{}'.format(i) for i in range(len(normalized_pixel_data[0])-1)])


    labels_prova = prova['id'].values
    img_prova = prova.drop(labels='id', axis=1).values / 255 # Normalization

    img_prova = img_prova.reshape(-1, 1, 28, 28)

    # COnvert prova set to tensors
    img_prova = torch.from_numpy(img_prova).float()
    labels_prova = torch.from_numpy(labels_prova).type(torch.LongTensor)

    prova = data_utils.TensorDataset(img_prova, labels_prova)

    # Define batch_size, epoch and iteration
    batch_size = 100
    n_iters = 2000
    num_epochs = 30
    num_classes = 10

    prova_loader = data_utils.DataLoader(prova,
                                        batch_size=batch_size,
                                        shuffle=False, num_workers=16)


    # check if CUDA is available
    use_cuda = torch.cuda.is_available()


    model = _load_models()

    # move tensors to GPU if CUDA is available
    if use_cuda:
        model.cuda()

    classes = ['zero', 'one', 'two', 'three', 'four',
               'five', 'six', 'seven', 'eight', 'nine']

    predictions_output = []
    labels_output = []
    # Obtain one batch of test images
    for images, labels in prova_loader:

        # Move model inputs to CUDA, if GPU available
        if use_cuda:
            images = images.cuda()

        # Get sample outputs
        output = model(images)
        # Convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy()) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "Class": preds.tolist(),
        },
    }

    return response
"""
@app.get("/")
async def root():
    return {"message": "This is a digit recognizer model. Please update a folder with image of digits and our model will return the prediction of each"}
"""
