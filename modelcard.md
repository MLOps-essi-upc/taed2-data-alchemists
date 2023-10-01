

# Model Card for MENTAL HEALTH CHAT BOT🤖🏥

This model is a chat bot trained to answer health related questions. It is the fine tuned version of the following model: 'ybelkada/falcon-7b-sharded-bf16', on HuggingFace, on a data set with pairs of questions and answers of the subject, also from HuggingFace.

## Model Details

### Model Description

The rationale behind this is to answer mental health related queries that can be factually verified without responding gibberish words.

- **Developed by:** ybelkada

- **Shared by:** heliosbrahma
- **Model type:** NLP model
- **Language(s):** English
- **License:** mit
- **Finetuned from model:** 'ybelkada/falcon-7b-sharded-bf16'

### Model Sources [optional]

The model can be found on the followig link. Inside it can also be found the original model the fine tuned version is based on.

- **Repository:** https://github.com/MLOps-essi-upc/taed2-data-alchemists/edit/main/modelcard.md

## Uses

A chatbot focused on mental health can be a valuable tool to provide support, information, and resources to individuals dealing with mental health issues. Here are some possible uses of a mental health chatbot:

- **Information and Education:** The chatbot can offer information about various mental health conditions, their symptoms, causes, and treatment options. It can help raise awareness and reduce stigma surrounding mental health.
- **Medication and Treatment Information:** For users who are prescribed medication or undergoing therapy, the chatbot can provide information about medications, potential side effects, and the importance of adhering to treatment plans.
- **24/7 Availability:** Unlike human support, a chatbot can be available 24/7, making it a valuable resource for users who need assistance during non-business hours or in different time zones.
- **Anonymous and Non-Judgmental:** Users may feel more comfortable discussing their mental health concerns with a chatbot, as it offers anonymity and does not pass judgment.
- **Preventative Education:** The chatbot can offer proactive tips for maintaining good mental health and preventing mental health issues, such as stress management techniques and healthy lifestyle advice.

### Out-of-Scope Use

While mental health chatbots can be a valuable resource, they are not a replacement for professional mental health care. Users with severe or persistent mental health issues should be encouraged to seek help from qualified mental health professionals. Chatbots can complement and enhance mental health support but should not be relied upon as the sole source of care.

## Bias, Risks, and Limitations

The model underwent training using a dataset that could potentially contain sensitive mental health-related information. It is crucial to emphasize that while mental health chatbots developed with this model can provide assistance, they should not be considered a substitute for professional mental health care. Chatbots have inherent limitations as they lack subjectivity and the ability to interpret non-verbal cues or reactions. Furthermore, there could be a bias depending on the dataset used for training and the individuals involved in creating it.

### Recommendations

This model should be utilized as a supplementary tool in medical procedures. It is not capable of making diagnoses or prescribing treatments

## Training Details

### Training Data

This model was trained on a dataset with 172 rows of conversational pair of questions and answers stored as strings. The file has an extension .parquet.
The dataset can be found on:

- **Dataset:** heliosbrahma/mental_health_chatbot_dataset

### Training Procedure 
This model was trained using QLoRA technique to fine-tune on a custom dataset on free-tier GPU available in Google Colab.


#### Training Hyperparameters
The following hyperparameters were used during training:

- learning_rate: 0.0002
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.03
- training_steps: 320


## Model Card Authors [optional]
The authors of this Model Card are Roger Bel Clapés, Queralt Benito Martín and Martí Farré Farrús.
