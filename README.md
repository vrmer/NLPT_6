# Joint Processing of Attribution Components with Contextual Embeddings
The github for the Attribute Relation Detection project for the course Natural Language Processing Technology at the Vrije Universiteit Amsterdam (2021).
Students: Myrthe Buckens, Bas Diender, Marcell Fekete and Adrielli Lopes Rego
### requirements 
The needed requirements can be found in `requirements` and installed by running
```pip install requirements``` from your terminal.

### scripts 
This folder contains the following subfolders with the code files to run the project code: 

#### data exploration
* `annotation_level_data_collection.py`: code to explore data on annotation level.
* `file_level_data_collection.py`: code to explore data on file level. 
* `specific_data_exploration.py`: code to generate statistics and provide insights in the data for this project for annotation and file level insights combined.

#### encoding
* `bert.py`: code to generate DistilBERT representations for the sentences and tokens in the file given as input.
* `extract_encodings.py`: code that runs the bert.py scripts and saves the output to `data/encodings`. 

#### models
* `baseline.py`: code to run a baseline model based on syntactic features. 
* `lstm.py`: code to train the lstm network, after training, the model is saved. 
* `final_model_predictions.py`: code that loads the trained and saved model and makes predictions on the test data.

#### processing
* `preprocessing.py`: code to clean and prepare the data set for the model input.
* `postprocessing.py`: code to add the BIO-scheme after model predictions.
* `results_to_conll.py`: code to convert the output files with results to conll files.


