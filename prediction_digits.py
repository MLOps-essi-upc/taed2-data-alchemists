# Data analysis packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Data visualizaiton packages
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

# Deep learning packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from torchvision import datasets

def predict():
    prova = pd.read_csv('data/test.csv', dtype=np.float32)
    labels_prova = prova['id'].values
    img_prova = prova.drop(labels='id', axis=1).values / 255 # Normalization

    img_prova = img_prova.reshape(-1, 1, 28, 28)

    # COnvert prova set to tensors
    img_prova = torch.from_numpy(img_prova)
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
            self.fc2 = nn.Linear(2048, num_classes)
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

    # move tensors to GPU if CUDA is available
    if use_cuda:
        model.cuda()

    model.load_state_dict(torch.load('digit_recognizer.pth'))

    # helper function to un-normalize and display an image
    def imshow(img):
        img = img.numpy() * 255  # unnormalize and convert from Tensor image
        plt.imshow(img[0])  # show image


    classes = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']

    predictions_output = []
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

        # Plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(20, 5))
        for idx in range(10):
            ax = fig.add_subplot(2, 10 // 2, idx + 1, xticks=[], yticks=[])
            # You can use images[idx].cpu() instead of images.cpu()[idx]
            imshow(images[idx].cpu())
            ax.set_title("Predicted: {}".format(classes[preds[idx]]))
            #print("Imatge:{} és {}".format(classes[labels[idx]], classes[preds[idx]]))
            predictions_output.append(classes[preds[idx]])
    return predictions_output

def main():
    pred = predict()
    print(pred)

main()
