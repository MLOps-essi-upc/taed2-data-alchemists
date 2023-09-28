---
base_model: ybelkada/falcon-7b-sharded-bf16
tags:
- generated_from_trainer
model-index:
- name: falcon-7b-sharded-bf16-finetuned-mental-health-conversational
  results: []
license: mit
datasets:
- heliosbrahma/mental_health_chatbot_dataset
language:
- en
metrics:
- rouge
pipeline_tag: conversational
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# falcon-7b-sharded-bf16-finetuned-mental-health-conversational

This model is a fine-tuned version of [ybelkada/falcon-7b-sharded-bf16](https://huggingface.co/ybelkada/falcon-7b-sharded-bf16) on a custom [heliosbrahma/mental_health_chatbot_dataset](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset) dataset.

## Model description

This model is fine-tuned on custom mental health conversational dataset. The rationale behind this is to answer mental health related queries that can be factually verified without responding gibberish words.

## Intended uses & limitations

The model was trained on the dataset which may contain sensitive information related to mental health. It is important to note that while mental health chatbots built using this model can be helpful, they are not a replacement for professional mental health care.

## Training and evaluation data

This model was trained on custom [heliosbrahma/mental_health_chatbot_dataset](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset) dataset which 172 rows of conversational pair of questions and answers.

## Training procedure

This model was trained using QLoRA technique to fine-tune on a custom dataset on free-tier GPU available in Google Colab.

### Training hyperparameters

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

### Training results



### Framework versions

- Transformers 4.31.0
- Pytorch 2.0.1+cu118
- Datasets 2.14.2
- Tokenizers 0.13.3