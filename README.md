# Senior Project
### Noah Segal-Gould's Senior Project for fulfillment of a degree in Computer Science and Experimental Humanities from Bard College in May 2018

#### Goal:
* Acquire and identify "subtweets" on Twitter.

#### Progress:
* The scripts can...
  * Download user-tagged subtweets live, by streaming the Twitter API
  * Download user-tagged subtweets for the previous day, every day at midnight
  * Download user-tagged subtweets for an entire prior month-long period
  * Create training data for a Naive Bayes classifier using a combination of the downloaded subtweets and the positive sentiment classified normal tweets from [Alec Go's dataset](http://help.sentiment140.com/for-students) which do not contain username mentions
  * Consolidate all the previously downloaded subtweets into larger corpora
  
#### To-Do:
* Train a Naive Bayes Classifier...
  * Using Scikit-Learn
  * With a Pipeline to add features for pronouns, mentions of names, non-mentions of usernames, etc.
  * Training data statistics
    * Lengths
    * Punctuation
    * Stop words
    * unique words
  * K-Folds
  
This project has a license.
