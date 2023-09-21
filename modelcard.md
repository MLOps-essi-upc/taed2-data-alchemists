---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for HEALTH CHAT BOT🤖🏥

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

- **Repository:** {{ [repo](https://huggingface.co/heliosbrahma/falcon-7b-sharded-bf16-finetuned-mental-health-conversational) | default("[More Information Needed]", true)}}

## Uses

A chatbot focused on mental health can be a valuable tool to provide support, information, and resources to individuals dealing with mental health issues. Here are some possible uses of a mental health chatbot:

- **Information and Education:** The chatbot can offer information about various mental health conditions, their symptoms, causes, and treatment options. It can help raise awareness and reduce stigma surrounding mental health.
- **Medication and Treatment Information:** For users who are prescribed medication or undergoing therapy, the chatbot can provide information about medications, potential side effects, and the importance of adhering to treatment plans.
- **24/7 Availability:** Unlike human support, a chatbot can be available 24/7, making it a valuable resource for users who need assistance during non-business hours or in different time zones.
- **Anonymous and Non-Judgmental:** Users may feel more comfortable discussing their mental health concerns with a chatbot, as it offers anonymity and does not pass judgment.
- **Preventative Education:** The chatbot can offer proactive tips for maintaining good mental health and preventing mental health issues, such as stress management techniques and healthy lifestyle advice.


### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

{{ downstream_use | default("[More Information Needed]", true)}}

### Out-of-Scope Use

While mental health chatbots can be a valuable resource, they are not a replacement for professional mental health care. Users with severe or persistent mental health issues should be encouraged to seek help from qualified mental health professionals. Chatbots can complement and enhance mental health support but should not be relied upon as the sole source of care.

## Bias, Risks, and Limitations

The model underwent training using a dataset that could potentially contain sensitive mental health-related information. It is crucial to emphasize that while mental health chatbots developed with this model can provide assistance, they should not be considered a substitute for professional mental health care. Chatbots have inherent limitations as they lack subjectivity and the ability to interpret non-verbal cues or reactions. Furthermore, there could be a bias depending on the dataset used for training and the individuals involved in creating it.

### Recommendations

This model should be utilized as a supplementary tool in medical procedures. It is not capable of making diagnoses or prescribing treatments

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

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

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Data Card if possible. -->

{{ testing_data | default("[More Information Needed]", true)}}

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

{{ model_examination | default("[More Information Needed]", true)}}

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** {{ hardware | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}

## Technical Specifications [optional]

### Model Architecture and Objective

{{ model_specs | default("[More Information Needed]", true)}}

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

{{ hardware | default("[More Information Needed]", true)}}

#### Software

{{ software | default("[More Information Needed]", true)}}

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}

**APA:**

{{ citation_apa | default("[More Information Needed]", true)}}

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

{{ glossary | default("[More Information Needed]", true)}}

## More Information [optional]

{{ more_information | default("[More Information Needed]", true)}}

## Model Card Authors [optional]

{{ model_card_authors | default("[More Information Needed]", true)}}

## Model Card Contact

{{ model_card_contact | default("[More Information Needed]", true)}}
