<h2>Sentiment_analysis_of_tweets<h2></h2>
Project to implement a Twitter sentiment analysis model for overcoming the challenges to identify the Twitter tweets text sentiments (positive, negative)

<h3>Introduction</h3>
Sentiment analysis refers to identifying as well as classifying the sentiments that are expressed in the text source. Tweets are often useful in generating a vast amount of sentiment data upon analysis. These data are useful in understanding the opinion of the people about a variety of topics. Therefore, we need to develop an Automated Machine Learning Sentiment Analysis Model in order to compute the customer perception using NER. Due to the presence of non-useful characters (collectively termed as the noise) along with useful data, it becomes difficult to implement models on them.

<h3>Dataset</h3>
Twitter Sentiment Analysis Detecting hatred tweets, provided by Analytics Vidhya Link to the datasets :
• https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-atredspeech?select=train.csv

<h3>Project Pipeline</h3>
• Understand the problem statement
• Import Necessary Dependencies
• Read and Load the Dataset
• Exploratory Data Analysis
• Data Visualization of Target Variables
• Data Pre-processing
• Feature selection
• Splitting our data into Train and Test Subset
• Transforming Dataset using TF-IDF Vectorizer
• Model Building
• Determining which model is best (Hypothesis testing)

<h3>Pre-processing</h3>
The pre-processing of the text data is an essential step as it makes the raw text ready for mining, i.e., it becomes easier to extract information from the text and apply machine learning algorithms to it.

<h3>Cleaning data</h3>
The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text.
Further, we will be extracting numeric features from our data. This feature space is created using all the unique words present in the entire data. So, if we pre-process our data well, then we would be able to get a better-quality feature space.
A) Removing Twitter Handles (@user)
B) Removing Punctuations, Numbers, and Special Characters
C) Removing Short Words
D) Tokenization
E) Stemming

<h3>Exploratory Data Analysis & visualization</h3>
Will explore the cleaned tweets text. Exploring and visualizing data, no matter whether its text or any other data, is an essential step in gaining insights.
Did the following:
• Plot various graphs for positive and negative tweets on the basis of Label
• Visualizing the words in the tweets before and after cleaning through word clouds
• Plot ngrams for most occurring 1,2,3 words in our tweets
• Find the difference between the word frequency in our data through histograms
• Named entity recognition of tweets of different categories & Displaying sample observations
• Plotting named entities mentioned most times in Non-Offensive & Offensive tweets
• Finding count of text by each named entity of Non offensive & Offensive tweets
• Visualize most repetitive Non offensive & Offensive text phrases from each named entity
• Understanding the impact of Hashtags on tweets sentiment
• Displaying count of hashtags in each entity and plotting word counts of Non offensive & Offensive hashtags
• Plotting Bar plots of top hashtag counts in Non-offensive & Offensive tweets
• Plotting a pie chart counting number of rows containing a hashtag

<h3>Feature selection</h3>
TF-IDF Features
TF-IDF works by penalizing the common words by assigning them lower weights while giving importance to words which are rare in the entire corpus but appear in good numbers in few documents.
Important terms related to TF-IDF:
• TF = (Number of times term t appears in a document)/(Number of terms in the document)
• IDF = log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.
• TF-IDF = TF*IDF

<h3>Model building:</h3>
In the problem statement we have used three different models respectively
• Logistic Regression
• Decision Tree Classifier
• Random Forest Classifier
Want to try all the classifiers on the dataset and then try to find out the one which gives the best performance among them.

Determining which model is best (Hypothesis testing)
After training the model we then apply the evaluation measures to check how the model is performing. Accordingly, we use the following evaluation parameters to check the performance of the models respectively :
• Accuracy Score
• F1 Scores
• Confusion Matrix

<h3>Hypothesis testing:</h3>
Examining machine learning models via statistical significance tests requires some expectations that will influence the statistical tests used.
An approach to evaluate each model on the same k-fold cross-validation split of the data and calculates each split score. That would give a sample of ten scores for ten-fold cross-validation. Then, we can compare those scores using the paired statistical test.
Due to using the same data rows to train the model more than once, the assumption of independence is violated; hence, the test would be biased.
This statistical test could be adjusted to overcome the lack of independence. Also, the number of folds and repeats of the method can be configured to achieve a better sampling of model performance.
Steps to hypothesis testing:
The first step would be to to state the null hypothesis statement.
H0: Both models have the same performance on the dataset.
H1: Both models doesn’t have the same performance on the dataset.
Significance level is 0.05
let’s assume a significance threshold of α=0.05 for rejecting the null hypothesis that both algorithms perform equally well on the dataset and conduct the 5x2_cv _t_test.
used the paired_ttest_5x2cv function from the evaluation module to calculate the t and p value for both models.

<h3>Conclusion</h3>
Upon evaluating all the models we can conclude the following details i.e.

<h4>Accuracy:</h4>
As far as the accuracy of the model is concerned Random Forest Classifier performs better than Decision Tree Classifier which in turn performs better than Logistic Regression.

F1-score:
The F1 Scores for class 0 and class 1 are :
(a) For class 0: Logistic Regression (accuracy = 0.97) < Decision Tree Classifier (accuracy =0.97) < Random Forest Classifier (accuracy = 0.98)
(b) For class 1: Logistic Regression (accuracy = 0.94) < Decision Tree Classifier (accuracy =0.95) < Random Forest Classifier (accuracy = 0.96)
We, therefore, conclude that the Random Forest Classifier is the best model for the above-given dataset.
