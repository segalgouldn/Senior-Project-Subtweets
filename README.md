# Senior Project
### Noah Segal-Gould's Senior Project for fulfillment of a degree in Computer Science and Experimental Humanities from Bard College in May 2018

#### Goal:
* Acquire and identify "subtweets" on Twitter.

#### Progress (Done):
* Download subtweets and non-subtweets based on the absence or presence of "subtweet" in the replies to tweets
* Combine the non-subtweets and subtweets into one dataset
* Split the dataset into training data and test data using K-Folds
* Train a Naive Bayes classifier using the training data
* Calculate F1, precision, and recall
* Create and visualize a confusion matrix of the results
* Test and visualize the results of the classifier on other data
* Provide statistics on tweet length, punctuation, stop words, and unique words
  
#### To-Do:
* Make the mentions of usernames and appearances of URLs generic (i.e. "[@username] and [url]")
* Fix the pipeline so the most informative features of the classifier can be shown
  
This project has a license.
