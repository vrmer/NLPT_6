# Joint Processing of Attribution Components with Contextual Embeddings
The github for the Attribute Relation Detection project for the course Natural Language Processing Technology at the Vrije Universiteit Amsterdam (2021).
Students: Myrthe Buckens, Bas Diender, Marcell Fekete and Adrielli Lopes Rego
### Requirements 
The needed requirements can be found in `requirements` and installed by running
```pip install requirements``` from your terminal.

### Scripts 
This folder contains the following subfolders with the code files to run the project code: 

#### data exploration
This subfolder contains code for data exploration on file and annotation level.
* `annotation_level_data_collection.py`: code to explore data on annotation level.
* `file_level_data_collection.py`: code to explore data on file level. 
* `specific_data_exploration.py`: code to generate statistics and provide insights in the data for this project for annotation and file level insights combined.

#### encoding
This subfolder contains the code to generate the DistilBERT encodings for the tokens in the articles.
* `bert.py`: code to generate DistilBERT representations for the sentences and tokens in the file given as input.
* `extract_encodings.py`: code that runs the bert.py scripts and saves the output to `data/encodings`. 

#### models
This subfolder contains the code for running and training the models used in this project.
* `baseline.py`: code to run a baseline model based on syntactic features. 
* `lstm.py`: code to train the lstm network, after training, the model is saved. 
* `final_model_predictions.py`: code that loads the trained and saved model and makes predictions on the test data.

#### processing
This subfolder contains code for processing the data in the right format for the models.
* `preprocessing.py`: code to clean and prepare the data set for the model input.
* `training_instances.py`: code to generate files per sentence as training instances for the lstm model and saves the output to `data/instances`.
* `postprocessing.py`: code to add the BIO-scheme after model predictions.

#### evaluation
This subfolder contains code for evaluating the final output of the models.
* `evaluation.attribution.v5.pl`: Evaluation script in pearl provided by Roser Morante Vallejo to compare the output of the models on partial and exact match. 
* `results_to_conll.py`: code to convert the output files with results to conll files.

### Data 
This folder contains data and empty folders for output for data.

* `cue_list.csv`: csv file containing a list with cues for the baseline model.

#### models
This subfolder holds the saved and trained models, output from the `lstm.py` script.
* `lstm_classifier_one_sentence_instances.json`: json with the saved model configurations for the trained lstm network.

#### encodings
The following subfolders need to be created on your local device for saving the encodings output from BERT; these are too big to upload to github. 
    
* dev-conll-foreval
* test-conll-foreval
* train-conll-foreval

#### instances
The following subfolders need to be created on your local device for saving the single sentence instances; these are too big to upload to github. 
* dev-conll-foreval
* test-conll-foreval
* train-conll-foreval