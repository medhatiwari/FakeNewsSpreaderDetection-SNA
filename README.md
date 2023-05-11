# FakeNewsSpreaderDetection-SNA

## Group-9
Anupam Raj (IIT2020034)   
Arjun Yadav (IIT2020120)  
Kaustubh Kale (IIT2020129)  
Satyam Gupta (IIT2020143)  
Medha Tiwari (IEC2020063)  


** The task of detecting fake news spreaders is very crucial in today’s world. With the advancement
of AI it has become even more crucial to differentiate between what is fake and what is real. It
would be a really interesting task to identify whether an AI is impersonating a human and
spreading fake news. Here we have only applied classical machine learning methods for detection
of fake news spreaders on textual data using n-grams as features extracted from the data.
Here we are trying to determine if the author of a particular twitter feed can be a fake news
spreader or not. 
**
***
## Reading Data

The script reads in some data from files , processes the data, and merges the resulting dataframes. Specifically, it reads in the contents of the file **truth.txt** and creates two lists, one containing the IDs of each tweet and the other containing whether or not the tweet was classified as a "spreader." It then creates a pandas dataframe called meta_data from these lists.

Next, the script uses the Path object to find all .xml files in the data/en directory and reads in the tweets contained in those files. It extracts the author ID from the filename and creates a list of these IDs along with a list of the tweets themselves. It then creates another pandas dataframe called **text_data** from these lists.

Finally, the script merges the meta_data and text_data dataframes on the ID column and creates a new dataframe called en_data. It then saves this dataframe to a tab-separated file called en_data.tsv.

Overall, this script is processing and merging English language tweet data for analysis or further processing.

***

## Cleaning Data

The script first imports the re, pandas, and emoji packages. It then reads in the en_data.tsv file as a pandas dataframe and extracts the tweet text into a list called feed_list.

The script defines two functions for cleaning the tweet text: cleaning_v1 and cleaning_v2. The first function removes any characters that are not alphanumeric or spaces and replaces newline characters with spaces. The second function also converts all text to lowercase, removes any commas or quotation marks, adds spaces between emojis and other characters, splits words and punctuation into separate tokens, and removes any trailing spaces.

The script applies cleaning_v1 and cleaning_v2 to the feed_list and saves the resulting cleaned tweet text back into the en_data dataframe. It then saves the cleaned data to two separate tab-separated value files: clean_en_data_v1.tsv and clean_en_data_v2.tsv.

Overall, this script appears to be cleaning and processing English language tweets for analysis or further processing.

***

## Modelling
Testing four different learning methods with 5-fold cross-validation and grid-search for hyper-parameters

### Logistic Regression
### Random forest
### SVM
### XGBoost

### Logistic Regression
Python code that trains and tests logistic regression models using two different text vectorization methods (TfidfVectorizer) on cleaned tweet data. The code first imports the necessary packages, including pandas, sklearn, pickle, and time. It then reads in the cleaned tweet data from two separate files (clean_en_data_v1.tsv and clean_en_data_v2.tsv) as pandas dataframes.

The code defines two separate pipelines, one for each vectorization method, each consisting of a TfidfVectorizer and a logistic regression model. The pipeline is then run with a GridSearchCV object, which searches for the best hyperparameters for the model using a cross-validation approach.

The resulting models are saved in two separate pickle files (normalized_v1_gs.pickle and normalized_v2_gs.pickle) for future use. Finally, the results of the cross-validation are saved in two separate tab-separated value files (lr_tfidf_results_v1.tsv and lr_tfidf_results_v2.tsv), each containing the hyperparameters and mean test scores for each combination of hyperparameters.

___

### Random Forest

The code imports necessary libraries and modules such as pandas, sklearn's RandomForestClassifier, ParameterGrid, TfidfVectorizer, CountVectorizer, and time. It then reads in two datasets from two different files, data_v1.tsv and data_v2.tsv.

The code then creates a ParameterGrid with various parameters for the TfidfVectorizer and RandomForestClassifier models. It then fits a model with each combination of parameters in the grid for both datasets, storing the resulting out-of-bag (OOB) score for each model run.

The OOB scores and corresponding parameters are then saved in two separate files, rf_tfidf_results_v1.tsv and rf_tfidf_results_v2.tsv. The code then reads in each of these files, sorts the resulting OOB scores in descending order, and prints out the parameters with the highest OOB score for each dataset, along with the highest OOB score.

___

### SVM
The code is implementing cross-validation for a linear support vector machine (SVM) classifier on two datasets (data_v1 and data_v2). The goal is to find the optimal hyperparameters for the classifier using a grid search.

The code uses the GridSearchCV function from the scikit-learn library to perform a 5-fold cross-validation on the SVM classifier with different hyperparameters. The hyperparameters being optimized include the n-gram range, the minimum document frequency, and the regularization parameter C for the SVM. The cross-validation is stratified to ensure that each fold contains roughly the same proportions of the different classes.

The code uses the TfidfVectorizer function from scikit-learn to transform the text data into a numerical representation that can be used by the SVM classifier. The Pipeline function is used to define a pipeline that combines the vectorizer and the SVM classifier.

After running the cross-validation, the code saves the resulting trained SVM classifiers in pickle files (svm_tfidf_v1 and svm_tfidf_v2). It also saves the results of the grid search (best hyperparameters and corresponding accuracy scores) in two separate TSV files (svm_tfidf_results_v1.tsv and svm_tfidf_results_v2.tsv). Finally, the code prints the best hyperparameters and the corresponding accuracy score for each dataset.

***
### XGBoost
The code is for training and evaluating an XGBoost model for tweet classification using a pipeline that includes a TfidfVectorizer and GridSearchCV to find the best hyperparameters for the model. The data is read from two TSV files, preprocessed and used for training and testing the model. The best hyperparameters for the model are determined using a 5-fold StratifiedKFold cross-validation with the parameter grid specified in the parameters dictionary. The resulting grid_search object is saved as a pickle file. Additionally, the results of the cross-validation for both datasets are saved as TSV files. Finally, the best model is trained on the entire dataset and saved as pickle files.

***
### Finally training the best models on the complete set
The code imports several Python libraries including "pickle", "pandas", "svm", "LogisticRegression", "RandomForestClassifier", and "TfidfVectorizer" from scikit-learn.

It then reads two TSV files into pandas dataframes, data_v1 and data_v2, which contain pre-processed tweets.

Next, the code applies three different models to the data: logistic regression (LR), random forest (RF), and support vector machines (SVM). For each model, the code uses the TfidfVectorizer to convert the text data into a numerical format, and then trains the model on the numerical data. The best hyperparameters for each model are also specified.

Finally, the trained models are saved using the "pickle" module, which allows them to be loaded and used later without retraining. The resulting saved models are lr_v1.sav, rf_v2.sav, and svm_v1.sav. The trained vectorizers are also saved, and they are used later for transforming new data before making predictions.

***



![img](https://github.com/medhatiwari/FakeNewsSpreaderDetection-SNA/assets/75640645/51c08b75-d717-4e9e-b62f-14e3c9fae657)

### Predicting Result

The code is a Python script that loads pre-trained models and vectorizers for four different machine learning algorithms (Random Forest, SVM, Logistic Regression, and XGBoost) for the English language. It then applies these models to predict whether a user is a "spreader" or "non-spreader" of information based on their tweets.

The script begins by importing necessary packages such as numpy, pandas, and sklearn. It then defines several helper functions for cleaning tweets and handling emojis.

Next, the script reads in XML files of Twitter data for the English language and extracts the tweet text. It then creates two data frames, en_data_v1 and en_data_v2, which contain the tweet text cleaned using two different methods. The script then loads four vectorizers and uses them to transform the tweet data for each model.

Finally, the script loads the four pre-trained models and applies them to the transformed tweet data. The predicted spreader/non-spreader labels for each user are stored in four separate arrays (en_preds_RF, en_preds_SVM, en_preds_LR, and en_preds_XGB).

***

## Creating a stacking ensemble for the above learning methods

### Creating training and dev set for the ensemble model

The code starts by importing necessary libraries: pandas for data manipulation, pickle for serializing and deserializing Python objects, and various machine learning algorithms and tools from the scikit-learn library.

The code then reads in two separate TSV files containing cleaned tweet data into pandas DataFrames, data_en_v1 and data_en_v2. The best hyperparameters for each model are then set using the Pipeline function from scikit-learn.

Next, a train and dev set are constructed for an ensemble model. A stratified k-fold cross-validation is performed with 5 folds to split the data and the best-performing models - logistic regression (lr_pl), support vector machine (svm_pl), and random forest (rf_pl) - are trained on the training set and used to make predictions on the test set. The predicted probabilities of each model are then concatenated into a single DataFrame, result_en and result_en_v2, respectively, and saved as TSV files.

Overall, the code reads in tweet data, sets hyperparameters for three different machine learning models, performs cross-validation, trains the models on the training set, and saves the predicted probabilities for use in an ensemble model.

***
### Searching best stacking model
The code starts by importing necessary libraries such as joblib, pandas, and scikit-learn. It then loads two datasets, result_en and result_en_v2, and computes the mean and majority predictions of the models on these datasets.

Next, the code performs a hyperparameter search for two linear models: Logistic Regression and Ridge Classifier. GridSearchCV is used to perform the hyperparameter search for both models, and the best parameters are printed along with the performance metrics (classification report and confusion matrix) on both the training and dev datasets. Finally, the best performing Logistic Regression model is saved using joblib.

**Results**
precision    recall  f1-score   support

           0       0.65      0.67      0.66       150
           1       0.66      0.63      0.65       150

    accuracy                           0.65       300
   macro avg       0.65      0.65      0.65       300
weighted avg       0.65      0.65      0.65       300

[[101  49]
 [ 55  95]]
              precision    recall  f1-score   support

           0       0.63      0.68      0.65       150
           1       0.65      0.60      0.63       150

    accuracy                           0.64       300
   macro avg       0.64      0.64      0.64       300
weighted avg       0.64      0.64      0.64       300

[[102  48]
 [ 60  90]]
              precision    recall  f1-score   support

           0       0.65      0.68      0.66       150
           1       0.66      0.63      0.65       150

    accuracy                           0.66       300
   macro avg       0.66      0.66      0.66       300
weighted avg       0.66      0.66      0.66       300

[[102  48]
 [ 55  95]]
              precision    recall  f1-score   support

           0       0.63      0.67      0.65       150
           1       0.65      0.61      0.63       150

    accuracy                           0.64       300
   macro avg       0.64      0.64      0.64       300
weighted avg       0.64      0.64      0.64       300

[[100  50]
 [ 59  91]]
Fitting 5 folds for each of 110 candidates, totalling 550 fits

precision    recall  f1-score   support

           0       0.71      0.77      0.74       150
           1       0.75      0.69      0.72       150

    accuracy                           0.73       300
   macro avg       0.73      0.73      0.73       300
weighted avg       0.73      0.73      0.73       300

[[115  35]
 [ 47 103]]
              precision    recall  f1-score   support

           0       0.70      0.75      0.72       150
           1       0.73      0.67      0.70       150
           
 ***
We can see that **Logistic Regression is the best stacking model
**
## Getting the final predictions from the ensemble model

The is a Python script that performs sentiment analysis on a set of tweets written in English. It imports several libraries including joblib, numpy, pandas, pickle, re, and sklearn. The script defines some auxiliary functions including cleaning_v1, cleaning_v2, is_emoji, emoji_space, and a dictionary for prediction. It then reads in data from an XML file containing tweets, and stores the tweets in two pandas data frames, one for each text cleaning function. The script then loads three pre-trained models, a random forest classifier, a support vector machine, and a logistic regression model, each trained on a different text vectorizer. The models are used to predict whether each tweet is written by a "spreader" or "non-spreader". Finally, the script combines the output from the three models using an ensemble method and saves the final predictions in a CSV file.

***
## Final Accuracy Measurement

The code imports the Pandas library as 'pd' and the accuracy, precision, recall, and f1-score metrics from the Scikit-learn library. It then reads in a text file of English metadata ('test.txt') and separates the id and the label (0 or 1) of each data instance. It then reads in a final predictions CSV file ('final_predictions.csv') and extracts the 'pred' column.

Next, it creates an empty Pandas DataFrame and adds the actual labels and predicted labels as columns to the DataFrame. It then calculates the accuracy score using the actual and predicted labels in the DataFrame and stores it in the variable 'accuracy_ensem'. Finally, it prints the accuracy score for the ensemble.

**Accuracy for ensemble: 0.715**
         





