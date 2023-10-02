

# Dataset Card for mental_health_chatbot_dataset

## Dataset Description

- **Repository** [repo_url](https://github.com/MLOps-essi-upc/taed2-data-alchemists/tree/main/heliosbrahmamental_health_chatbot_dataset)
- **Homepage:** [homepage_url](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset)
- **Original Repository:** [og_repo_url](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset/tree/main)

### Dataset Summary

This dataset comprises conversational pairs of questions and corresponding answers, all centered around the topic of Mental Health. The data curation process involved sourcing content from reputable healthcare sources such as WebMD, Mayo Clinic, HealthLine, and online FAQs. To ensure privacy and data security, all questions and answers have undergone anonymization, eliminating any personally identifiable information (PII). Furthermore, thorough pre-processing has been applied to eliminate extraneous characters, resulting in a clean and structured dataset for analysis and model development.

### Languages

The text in the dataset is in English.

## Dataset Structure

### Data Instances

A data instance include a text columns which is a conversational pair of questions and answers. Questions were asked by the patients and answers were given by healthcare providers.

### Data Fields

- 'text': conversational pair of questions and answers between patient and healthcare provider.

## Dataset Creation

### Curation Rationale

Chatbots offer a readily available and accessible platform for individuals seeking support. They can be accessed anytime and anywhere, providing immediate assistance to those in need. Chatbots can offer empathetic and non-judgmental responses, providing emotional support to users. While they cannot replace human interaction entirely, they can be a helpful supplement, especially in moments of distress. Hence, this dataset was curated to help finetune a conversational AI bot using this custom dataset which can then be deployed and be provided to the end patient as a chatbot.

### Source Data

This dataset was curated from popular healthcare blogs like WebMD, Mayo Clinic and HeatlhLine, online FAQs etc.

## Considerations for Using the Data

### Personal and Sensitive Information

The dataset may contain sensitive information related to mental health. All questions and answers have been anonymized to remove any PII data.
