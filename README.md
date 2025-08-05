# Sentiment-tagging with Naive-Bayes for CSE150A-Group-Project

Note: To run this agent on your device, you must first download the dataset in csv format from kaggle (https://www.kaggle.com/datasets/kazanova/sentiment140/data), then add it to the project folder so that the notebook and the csv file are in the same directory.

## PEAS/Agent Analysis:
* Describe your agent in terms of PEAS and give a background of your task at hand. 

| Component in PEAS | Specification for component |
|-------------------|-----------------------------|
| **Performance**   | Maximize tagging accuracy, optimize run-time speed for real-time use, handling of newly encountered words that havent been encountered previously in the corpus|
| **Environment**   | A list of 16 million user tweets from the kaggle sentiment140 dataset, tagged with sentiment for positive or negative sentiments.|
| **Actuators**     | Output sentiment tag|
| **Sensors**       | Reads in incoming word sequences as tweets, or raw text.|


* What problem are you solving? Why does probabilistic modeling make sense to tackle this problem?

Sentiment tagging is used in a variety of different fields, including customer analysis, opinion mining, political stance evaluation, and more. We aim to automatically classify the sentiment of incoming tweets using uncertainty modeling. Uncertainty modeling is important for this problem because it helps us determine the sentiment of an unknown string of text that has potentially never been encountered before. In non-probabilistic approaches, various issues can occur, such as unique words that haven’t been seen before, difficulty discerning noisy text with lots of ambiguous language. These kinds of approaches can include keyword matching, or threshold-based heuristics. 

## Agent Setup, Data Preprocessing, Training setup:
Probabilistic modeling, specifically utilizing Naive Bayes is perfect in tackling this problem.
Intuitively, when we look at a sentence such as "Billy loves his dog", we **observe** the words "Billy", "loves", 
"his", "dog", from which we are trying to infer the **unobserved** sentiment or meaning of the sentence. 
Naive Bayes is best for this kind of approach because it works quickly in single word processing, and each word is dependent on the sentiment but independent of the other words in the sentence given the sentiment, making CPT value calculations easier.

We will be using the sentiment140 dataset from Kaggle, which compiles 16 million tweets, half of which have positive sentiment and half of which have negative sentiment.

Each tweet is assigned a value of either 0 or 4, with 0 being a negative sentiment, and 4 being positive sentiment. Since we are only analyzing the sentiment of newly written tweets, the tweet and the sentiment are the only variables we really care about.



Below is an image of the dataset as a table. The second column determines the sentiment value, and the last column is the tweet text.
<img width="1011" height="357" alt="image" src="https://github.com/user-attachments/assets/d72e078c-3dfe-4fd2-88de-a4a452d6693e" />

Our Naive Bayes model calculates the probability of a given word by the below formula (gotten from ChatGPT): 
<img width="810" height="426" alt="image" src="https://github.com/user-attachments/assets/260ebbd1-e0be-4713-8d78-fee8b9158f5d" />

We count the frequency of the words appearances in tweets of either negative or positive sentiment, depending on what the calculation is. We also apply a smoothing constant in the numerator and denominator to prevent the appearance of zero values. 

<img width="441" height="353" alt="image" src="https://github.com/user-attachments/assets/35803a5e-8331-47e5-8c67-82c41aba04e9" />

<img width="687" height="588" alt="image" src="https://github.com/user-attachments/assets/5dc03e2a-1bd6-4f01-a95a-dccdf2abcee1" />


We get a general formula for calculating the sentiment of a sentence given the probability by the formula below:
<img width="843" height="698" alt="image" src="https://github.com/user-attachments/assets/d71f153a-d8a4-41c0-84f5-e8e3b52e022b" />

This is gotten through Bayes Theorem, and the joint probability function for a sentence being a combination of the individual word probabilities.

To test our model, we randomly selected 100 tweets, 50 of which were positive, and 50 of which were negative. We then applied the tweets through our algorithm, and analyzed the results, comparing the original tweet sentiment with what our model predicted it to be. We reached an accuracy ranging between 75 to 78% depending on the sample, and by contrast, the randomized number picker got an accuracy rating of around 50% each time, just guessing the sentiment with no analysis.

<img width="1776" height="501" alt="image" src="https://github.com/user-attachments/assets/a6b1184f-d12d-4e76-b29d-9fe02a3c1a8b" />
<img width="1777" height="515" alt="image" src="https://github.com/user-attachments/assets/3af0934d-489c-4477-af68-0f5d6a2df373" />


We implemented improvements to our model by adjusting filler words and negative sentiment expressions, such as "not good" or"not awesome", so that they are recognized as single negative sentiment unigrams, rather than the words "not", "good", and "awesome" to be recognized separately/. This change allows for us to accurately capture sentiment nuances in statements that use words that are traditionally used together, as compared to separate. We also tried to exclude certain "stopwords" like "the" "and" "than" and other words that seemingly do nothing to alter the sentiment of a sentence. This alteration however, had minimal effect on the accuracy of our predictor and even somewhat decreased the accuracy of predictions for this dataset, but if applied to a different dataset that the model isn't trained on, the results may differ. We found that adding in Negative Weighting and removing least frequently used words led to our model having the same accuracy for both. This is most likely due to the least frequently used words not having much effect on the sentiment of a tweet regardless. we saw a 0.63% increase in accuracy this specific sample of the data, and tested various other sampling schemes to find an increase from around 0.3 to 0.95%. While this may not seem like much, a Naive Bayes Classifier without use of transformers or more complex structures usually reaches a peak accuracy of around 80%, and incremental increases like this stack up when used with other modifications as well.

Some improvements that could be made to the model to improve the accuracy at the cost of performance, is to add a bigram filter, like in hw2, or a trigram even, and calculate the LLRs of the tweets depending on these instead, to see if correlation between word neighbors would increase the accuracy. We considered doing this, however it would add a lot of latency to our preprocessing step, which was already taking 20 seconds for a unigram, and would grow exponentially in time with a bigram filter. We implemented bigram filtering for negative keywords instead, treating each negative bigram as a unigram, which increased our accuracy. Another thing that we considered was to test various different thresholding values for our LLR, to see if there is a certain threshold value that maximizes accuracy and use that instead of just using 0 as the divider between positive and negative sentiment, however this approach led to more inaccuracy in the sample space, due to some less frequent negative words being treated as positive ones. One factor that could play into our innacuracy is the skewing of keyword data. There are about 200,000 more negative words in the corpus than positive ones. We could also test the data by evaluating each unique word to see whether it has a positive or negative skew, and if there are significantly more unique words that have a negative skew than a positive one, then we can incorporate some kind of corrective algorithm to the calculation to normalize and equalize the number of unique words that are positive or negative.
