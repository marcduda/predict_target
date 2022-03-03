# predict_target
install the mandatory libraries in `requirements.txt`\

to launch the package and the training :
```
python3  __main__.py       
```
with the following options :\
`--input_file pathToTrainData`: path to the data on which we want to train the model (default `train_technical_test.csv`)\
`--train_full_model`: option to add if we want to train a model\
`--out_directory pathToOutputDir`: where to save the model (default `.`)\
`--predict`: option to add if we want to make a prediction\
`--test_file pathToTestData`: path to the data on which we want to do a prediction (default `test_technical_test.csv`)\
`--previous_accuracy value`: option to add if we want to compare the new trained model to a previous accuracy value and only save the new model if this accuracy is better.\