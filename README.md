# Senior Project
### Noah Segal-Gould's Senior Project for fulfillment of a degree in Computer Science and Experimental Humanities from Bard College in May 2018

#### Goal:
* Acquire and identify "subtweets" on Twitter.

#### Progress:
* The scripts can...
  * Download subtweets and non-subtweets based on the absence or presence of 
 "subtweet" in the replies to tweets
  * Combine the non-subtweets and subtweets into one dataset
  * Split the dataset into training data and test data
  * Train a Naive Bayes classifier using the training data
  * Calculate F1, precision, and recall
  * Create and visualize a confusion matrix of the results
  * Test and visualize the results of the classifier on other data
  
#### To-Do:
* Train a Naive Bayes Classifier...
  * Using Scikit-Learn MultinomialNB
  * With a Pipeline to add features for words which identify others, oneself, URLs, etc.
  * Training data statistics
    * Tweet lengths
    * Punctuation in tweets
    * Total number of stop words in tweets
    * Total number of unique words in each tweet (and in all tweets)
  * Apply K-Folds to the training data
  * Rename the classes from "positive" and "negative" to "subtweet" and "non-subtweet"
  
This project has a license.
