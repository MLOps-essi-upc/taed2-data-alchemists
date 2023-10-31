TAED2
==============================

Project Organization
------------

    ├── .dvc
    │   ├── .gitignore       
    │   └── config  
    │
    ├── __pycache__      
    │   └── test_api.cpython-39-pytest-7.4.3.pyc    
    │
    ├── data
    │   ├── test
    │       ├── image_0.png
    │       ├── image_1.png
    │       ├── image_2.png
    │       ├── image_3.png
    │       ├── image_4.png
    │       ├── image_5.png
    │       ├── image_6.png
    │       ├── image_7.png
    │       ├── image_8.png
    │       └── image_9.png
    │   ├── sample_submission.csv
    │   ├── test.csv
    │   ├── train.csv
    │   └── train_aug.csv
    │
    ├── great expectations        <- Great Expectations files
    │   ├── checkpoints
    │       └── my_checkpoint.yml
    │   ├── expectations
    │       ├── .ge_store_backend_id
    │       └── my_suite.json
    │   ├── notebooks
    │       ├── pandas
    │           └── validation_playground.ipynb
    │       ├── spark
    │           └── validation_playground.ipynb
    │       ├── sql
    │           └── validation_playground.ipynb
    │       └── .DS_Store
    │   ├── plugins
    │       ├── custom_data_docs/styles
    │           └── data_docs_custom_styles.css
    │       └── .DS_Store   
    │   ├── .DS_Store
    │   ├── .gitignore
    │   └── great_expectations.yml
    │
    ├── models            
    │   └── cnn_digit_recognizer.pt
    │
    ├── src                  <- Source code
    │   ├── __pycache__
    │       └── __init__.cpython-39.pyc
    │   ├── app              <- FastAPI
    │       ├── api.py
    │       ├── image_prova.png
    │       ├── schemas.py
    │       └── test_api.py
    │
    ├── tests/__pycache__            
    │   └── test_api.cpython-39-pytest-7.4.3.pyc
    │
    ├── .dvcignore               
    │
    ├── .gitignore               
    │
    ├── README.md           <- The top-level README for developers using this project.
    │
    ├── LICENSE
    │
    ├── creacio_img.ipynb
    │
    ├── data.dvc
    │
    ├── datasetcard.md      <- Dataset card with the dataset information.
    │
    ├── digit_recognizer.py <- Trained model
    │
    ├── emissions.csv
    │
    ├── modelcard.md        <- Model card with the model information.
    │
    ├── params.yaml
    │
    ├── prediction_digits.py
    │
    ├── preprocessat.py
    │
    ├── requirements.txt
    │
    ├── setup.py
    │
    ├── test_environment.py
    │
    └── tox.ini             <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
