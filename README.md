# Transfer learning as a feature engineering mean
Transfer learning by using Alexnet convolutional layer as feature engineering. An intermediate csv is used to save convolutional layer pre-processing. The NN layers can then be replaced by another classifier. The training time is tremendously reduced for a very accurate result.

There need to be a data/dataset folder containing n folder (containing the pictures) named after the n classes. The notebook takes care of the rest

The app.py script provides a "paint-like" interface that the classifier will use to predict user drawing 

The classifier is trained on a very reduced subset of the original quick draw database. Therefore, the performances are limited.
