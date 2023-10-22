

# Model Card for DIGIT RECOGNIZER🔢🔍

This model is used to identify digits from a dataset of tens of thousands of handwritten images, by utilizing convolutional neural networks (CNN). MNIST handwritten digit dataset is used to train and test the CNN model.

## Model Details

### Model Description

The rationale behind this is to recognize, as accurately as possible, the numbers from 1 to 9 by analizing images of handwritten digits.

- **Developed by:*andikarachman* 

- **Shared by:** andikarachman
- **Model type:** NLP model
- **Language(s):** English
- **License:** CC BY-SA 3.0
- **Finetuned from model:** 'CNN-Digit-Recognizer'

### Model Sources

The model can be found on the followig link. Inside it can also be found the original model the fine tuned version is based on.

- **Repository:** https://github.com/MLOps-essi-upc/taed2-data-alchemists/edit/main/modelcard.md

## Uses

Handwritten Digit Recognition model, has a wide range of practical applications.
Some of the most common uses are the following:

- **Digit Recognition in Postal Services:** Handwritten digit recognition is often used by postal services to automate the sorting of mail. It can read the handwritten postal codes, making the process more efficient.
- **Bank Check Processing:** Banks use digit recognition models to automatically read the handwritten amounts on checks, reducing errors and processing time.
- **Digital Document Scanning:** Scanning handwritten documents or forms and converting them into machine-readable text is made easier with digit recognition. This can be useful in digitizing historical documents and records.
- **Handwriting Recognition in Tablets and Smartphones:** Digit recognition models are often incorporated into tablets and smartphones for applications like note-taking and converting handwritten notes to text.
- **Accessibility Features:** Digit recognition can be used in assistive technologies to help individuals with disabilities interact with digital devices, such as converting handwritten notes to text or reading handwritten numbers aloud.

### Out-of-Scope Use

If not used ethically, this model can be used to commit forgery and fraud, for example by using the model to create counterfeit documents or to manipulate handwritten numbers for fraudulent purposes, such as altering checks or identification documents. Users are asked to use the model in an ethical and correct way. 

## Bias, Risks, and Limitations

The model can be biased if the training data used to build it is not representative of the population it will be applied to. For example, if the training data contains predominantly one type of handwriting, the model may perform poorly on other handwriting styles. Moreover, since a big majority of the population is righ-handed, if the dataset doesn't have enough left-handed representation, it could lead to difficulties identifying the digits written by the second group.
This model also presents some legal risks. Using digit recognition technology for unlawful purposes, like forging documents, can lead to legal consequences.
As for limitations, the model may struggle to generalize to handwriting styles not present in the training data, leading to errors in recognition. Noisy or smudged handwritten characters may lead to incorrect recognition, affecting the model's robustness.

### Recommendations

Adhering to ethical guidelines, data privacy regulations, and security best practices can help ensure responsible and safe use of digit recognition models. Regular audits and updates of the model's performance can also help maintain its effectiveness and fairness.

## Training Details

### Training Data

This model was trained on a dataset with 42000 images that contain a handwritten digit. The file has an extension .csv.
The dataset can be found on:

- **Dataset:** (https://www.kaggle.com/competitions/digit-recognizer/data)

### Training Procedure 
This model was trained fine-tuning a convolutional neural network. 
The dataset is slipt in three parts:
- 60% of the dataset become the train set.
- 20% of the dataset become the validation set.
- 20% of the dataset become the test set.

#### Training Hyperparameters
The following hyperparameters were used during training:

- num_classes: 10
- lr: 0.001
- train_batch_size: 100
- eval_batch_size: 100
- optimizer: alpha=0.9 and eps=1e-08 
n_iters = 2000


## Model Card Authors
The authors of this Model Card are Roger Bel Clapés, Queralt Benito Martín and Martí Farré Farrús.
