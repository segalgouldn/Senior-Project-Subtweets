
# coding: utf-8

# ## Using Scikit-Learn and NLTK to build a Naive Bayes Classifier that identifies subtweets

# #### In all tables, assume:
# * "➊" represents a single hashtag
# * "➋" represents a single URL
# * "➌" represents a single mention of username (e.g. "@noah")

# #### Import libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.externals import joblib
from nltk.corpus import stopwords
from random import choice
from string import punctuation

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import scipy.stats
import itertools
import enchant
import nltk
import json
import re


# #### Set up some regex patterns

# In[3]:


hashtags_pattern = r'(\#[a-zA-Z0-9]+)'


# In[4]:


urls_pattern = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))'


# In[5]:


at_mentions_pattern = r'(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z0-9_]+)'


# #### Prepare English dictionary for language detection

# In[6]:


english_dict = enchant.Dict("en_US")


# #### Use NLTK's tokenizer instead of Scikit's

# In[7]:


tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)


# #### Prepare for viewing long text in CSVs and ones with really big and small numbers

# In[8]:


pd.set_option("max_colwidth", 1000)


# In[9]:


pd.options.display.float_format = "{:.4f}".format


# #### Load the two data files

# In[10]:


subtweets_data = [t for t in json.load(open("../data/other_data/subtweets.json")) 
                  if t["tweet_data"]["user"]["lang"] == "en" 
                  and t["reply"]["user"]["lang"] == "en"]


# In[11]:


non_subtweets_data = [t for t in json.load(open("../data/other_data/non_subtweets.json")) 
                      if t["tweet_data"]["user"]["lang"] == "en" 
                      and t["reply"]["user"]["lang"] == "en"]


# #### Only use tweets with at least 50% English words
# #### Also, make the mentions of usernames, URLs, and hashtags generic

# In[12]:


get_ipython().run_cell_magic('time', '', 'subtweets_data = [(re.sub(hashtags_pattern, \n                          "➊", \n                          re.sub(urls_pattern, \n                                 "➋", \n                                 re.sub(at_mentions_pattern, \n                                        "➌", \n                                        t["tweet_data"]["full_text"])))\n                   .replace("\\u2018", "\'")\n                   .replace("\\u2019", "\'")\n                   .replace("&quot;", "\\"")\n                   .replace("&amp;", "&")\n                   .replace("&gt;", ">")\n                   .replace("&lt;", "<"))\n                  for t in subtweets_data]')


# In[13]:


new_subtweets_data = []
for tweet in subtweets_data:
    tokens = tokenizer.tokenize(tweet)
    english_tokens = [english_dict.check(token) for token in tokens]
    percent_english_words = sum(english_tokens)/len(english_tokens)
    if percent_english_words >= 0.5:
        new_subtweets_data.append(tweet)


# In[14]:


get_ipython().run_cell_magic('time', '', 'non_subtweets_data = [(re.sub(hashtags_pattern, \n                              "➊", \n                              re.sub(urls_pattern, \n                                     "➋", \n                                     re.sub(at_mentions_pattern, \n                                            "➌", \n                                            t["tweet_data"]["full_text"])))\n                       .replace("\\u2018", "\'")\n                       .replace("\\u2019", "\'")\n                       .replace("&quot;", "\\"")\n                       .replace("&amp;", "&")\n                       .replace("&gt;", ">")\n                       .replace("&lt;", "<"))\n                      for t in non_subtweets_data]')


# In[15]:


new_non_subtweets_data = []
for tweet in non_subtweets_data:
    tokens = tokenizer.tokenize(tweet)
    english_tokens = [english_dict.check(token) for token in tokens]
    percent_english_words = sum(english_tokens)/len(english_tokens)
    if percent_english_words >= 0.5:
        new_non_subtweets_data.append(tweet)


# #### Show examples

# In[16]:


print("Subtweets dataset example:")
print(choice(new_subtweets_data))


# In[17]:


print("Non-subtweets dataset example:")
print(choice(new_non_subtweets_data))


# #### Find the length of the smaller dataset

# In[18]:


smallest_length = len(min([new_subtweets_data, new_non_subtweets_data], key=len))


# #### Cut both down to be the same length

# In[19]:


subtweets_data = new_subtweets_data[:smallest_length]


# In[20]:


non_subtweets_data = new_non_subtweets_data[:smallest_length]


# In[21]:


print("Smallest dataset length: {}".format(len(non_subtweets_data)))


# #### Prepare data for training

# In[22]:


subtweets_data = [(tweet, "subtweet") for tweet in subtweets_data]


# In[23]:


non_subtweets_data = [(tweet, "non-subtweet") for tweet in non_subtweets_data]


# #### Combine them

# In[24]:


training_data = subtweets_data + non_subtweets_data


# #### Create custom stop words to include generic usernames, URLs, and hashtags, as well as common English first names

# In[25]:


names_lower = set([name.lower() for name in open("../data/other_data/first_names.txt").read().split("\n")])


# In[26]:


generic_tokens = {"➊", "➋", "➌"}


# In[27]:


stop_words = text.ENGLISH_STOP_WORDS | names_lower | generic_tokens


# #### Build the pipeline

# In[28]:


sentiment_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(tokenizer=tokenizer.tokenize, 
                                   ngram_range=(1, 3), 
                                   stop_words=stop_words)),
    ("classifier", MultinomialNB())
])


# #### K-Folds splits up and separates out 10 training and test sets from the data, from which the classifier is trained and the confusion matrix and classification reports are updated

# In[29]:


text_training_data = np.array([row[0] for row in training_data])


# In[30]:


class_training_data = np.array([row[1] for row in training_data])


# In[31]:


num_folds=10


# In[32]:


kf = KFold(n_splits=num_folds, random_state=42, shuffle=True)


# In[33]:


get_ipython().run_cell_magic('time', '', 'cnf_matrix = np.zeros((2, 2), dtype=int)\nfor i, (train_index, test_index) in enumerate(kf.split(text_training_data)):\n    \n    text_train, text_test = text_training_data[train_index], text_training_data[test_index]\n    class_train, class_test = class_training_data[train_index], class_training_data[test_index]\n    \n    sentiment_pipeline.fit(text_train, class_train)\n    predictions = sentiment_pipeline.predict(text_test)\n        \n    cnf_matrix += confusion_matrix(class_test, predictions)\n    \n    print("Iteration {}".format(i+1))\n    print(classification_report(class_test, predictions, digits=3))\n    print("null accuracy: {:.3f}\\n".format(max(pd.value_counts(pd.Series(class_test)))/float(len(class_test))))\n    print("="*53)')


# #### See the most informative features

# In[34]:


def most_informative_features(pipeline, n=50):
    vectorizer = pipeline.named_steps["vectorizer"]
    classifier = pipeline.named_steps["classifier"]
    
    class_labels = classifier.classes_
    
    feature_names = vectorizer.get_feature_names()
    
    top_n_class_1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    top_n_class_2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]
    
    return {class_labels[0]: pd.DataFrame({"Weight": [tup[0] for tup in top_n_class_1], 
                                           "Feature": [tup[1] for tup in top_n_class_1]}), 
            class_labels[1]: pd.DataFrame({"Weight": [tup[0] for tup in reversed(top_n_class_2)],
                                           "Feature": [tup[1] for tup in reversed(top_n_class_2)]})}


# In[35]:


most_informative_features_all = most_informative_features(sentiment_pipeline)


# In[36]:


most_informative_features_non_subtweet = most_informative_features_all["non-subtweet"]


# In[37]:


most_informative_features_subtweet = most_informative_features_all["subtweet"]


# In[38]:


most_informative_features_non_subtweet.join(most_informative_features_subtweet, 
                                            lsuffix=" (Non-subtweet)", 
                                            rsuffix=" (Subtweet)")


# #### Define function for visualizing confusion matrices

# In[39]:


def plot_confusion_matrix(cm, classes, normalize=False,
                          title="Confusion Matrix", cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted Label")


# #### Show the matrices

# In[40]:


class_names = ["non-subtweet", "subtweet"]

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# #### Update matplotlib style

# In[41]:


plt.style.use("fivethirtyeight")


# #### Save the classifier for another time

# In[42]:


joblib.dump(sentiment_pipeline, "../data/other_data/subtweets_classifier.pkl");


# #### Print tests for the classifier

# In[43]:


def tests_dataframe(tweets_dataframe, text_column="SentimentText", sentiment_column="Sentiment"):
    predictions = sentiment_pipeline.predict_proba(tweets_dataframe[text_column])
    negative_probability = predictions[:, 0].tolist()
    positive_probability = predictions[:, 1].tolist()
    return pd.DataFrame({"tweet": tweets_dataframe[text_column], 
                         "sentiment_score": tweets_dataframe[sentiment_column], 
                         "subtweet_negative_probability": negative_probability, 
                         "subtweet_positive_probability": positive_probability}).sort_values(by="subtweet_positive_probability", 
                                                                                             ascending=False)


# #### Make up some tweets

# In[44]:


test_tweets = ["Some people don't know their place.", 
               "Isn't it funny how some people don't know their place?", 
               "How come you people act like this?", 
               "You're such a nerd.",
               "I love Noah, he's so cool.",
               "Who the heck is Noah?",
               "This is a @NoahSegalGould subtweet. Go check out https://segal-gould.com.", 
               "This is a subtweet.", 
               "Hey @jack!", 
               "Hey Jack!",
               "http://www.google.com"]


# #### Make a dataframe from the list

# In[45]:


test_tweets_df = pd.DataFrame({"Tweet": test_tweets, "Sentiment": [None]*len(test_tweets)})


# #### Remove usernames, URLs, and hashtags

# In[46]:


test_tweets_df["Tweet"] = test_tweets_df["Tweet"].str.replace(hashtags_pattern, "➊")


# In[47]:


test_tweets_df["Tweet"] = test_tweets_df["Tweet"].str.replace(urls_pattern, "➋")


# In[48]:


test_tweets_df["Tweet"] = test_tweets_df["Tweet"].str.replace(at_mentions_pattern, "➌")


# #### Print the tests

# In[49]:


tests_dataframe(test_tweets_df, text_column="Tweet", 
                sentiment_column="Sentiment").drop(["sentiment_score", 
                                                    "subtweet_negative_probability"], axis=1)


# #### Tests on friends' tweets

# #### Aaron

# In[50]:


aaron_df = pd.read_csv("../data/data_for_testing/friends_data/akrapf96_tweets.csv").dropna()
aaron_df["Sentiment"] = None


# #### Remove usernames, URLs, and hashtags

# In[51]:


aaron_df["Text"] = aaron_df["Text"].str.replace(hashtags_pattern, "➊")


# In[52]:


aaron_df["Text"] = aaron_df["Text"].str.replace(urls_pattern, "➋")


# In[53]:


aaron_df["Text"] = aaron_df["Text"].str.replace(at_mentions_pattern, "➌")


# In[54]:


aaron_df = tests_dataframe(aaron_df, text_column="Text", 
                           sentiment_column="Sentiment").drop(["sentiment_score", 
                                                               "subtweet_negative_probability"], axis=1)


# In[55]:


aaron_df.to_csv("../data/data_from_testing/friends_data/akrapf96_tests.csv")


# In[56]:


aaron_df.head(10)


# In[57]:


aaron_df_for_plotting = aaron_df.drop(["tweet"], axis=1)


# #### Julia

# In[58]:


julia_df = pd.read_csv("../data/data_for_testing/friends_data/juliaeberry_tweets.csv").dropna()
julia_df["Sentiment"] = None


# #### Remove usernames, URLs, and hashtags

# In[59]:


julia_df["Text"] = julia_df["Text"].str.replace(hashtags_pattern, "➊")


# In[60]:


julia_df["Text"] = julia_df["Text"].str.replace(urls_pattern, "➋")


# In[61]:


julia_df["Text"] = julia_df["Text"].str.replace(at_mentions_pattern, "➌")


# In[62]:


julia_df = tests_dataframe(julia_df, text_column="Text", 
                           sentiment_column="Sentiment").drop(["sentiment_score", 
                                                               "subtweet_negative_probability"], axis=1)


# In[63]:


julia_df.to_csv("../data/data_from_testing/friends_data/juliaeberry_tests.csv")


# In[64]:


julia_df.head(10)


# In[65]:


julia_df_for_plotting = julia_df.drop(["tweet"], axis=1)


# #### Noah

# In[66]:


noah_df = pd.read_csv("../data/data_for_testing/friends_data/noahsegalgould_tweets.csv").dropna()
noah_df["Sentiment"] = None


# #### Remove usernames, URLs, and hashtags

# In[67]:


noah_df["Text"] = noah_df["Text"].str.replace(hashtags_pattern, "➊")


# In[68]:


noah_df["Text"] = noah_df["Text"].str.replace(urls_pattern, "➋")


# In[69]:


noah_df["Text"] = noah_df["Text"].str.replace(at_mentions_pattern, "➌")


# In[70]:


noah_df = tests_dataframe(noah_df, text_column="Text", 
                          sentiment_column="Sentiment").drop(["sentiment_score", 
                                                              "subtweet_negative_probability"], axis=1)


# In[71]:


noah_df.to_csv("../data/data_from_testing/friends_data/noahsegalgould_tests.csv")


# In[72]:


noah_df.head(10)


# In[73]:


noah_df_for_plotting = noah_df.drop(["tweet"], axis=1)


# #### Rename the columns for later

# In[74]:


aaron_df_for_plotting_together = aaron_df_for_plotting.rename(columns={"subtweet_positive_probability": "Aaron"})


# In[75]:


julia_df_for_plotting_together = julia_df_for_plotting.rename(columns={"subtweet_positive_probability": "Julia"})


# In[76]:


noah_df_for_plotting_together = noah_df_for_plotting.rename(columns={"subtweet_positive_probability": "Noah"})


# #### Prepare statistics on friends' tweets

# In[77]:


friends_df = pd.concat([aaron_df_for_plotting_together, 
                        julia_df_for_plotting_together, 
                        noah_df_for_plotting_together], ignore_index=True)


# In[78]:


friends_df.describe()


# In[79]:


aaron_mean = friends_df.describe().Aaron[1]
aaron_std = friends_df.describe().Aaron[2]

julia_mean = friends_df.describe().Julia[1]
julia_std = friends_df.describe().Julia[2]

noah_mean = friends_df.describe().Noah[1]
noah_std = friends_df.describe().Noah[2]


# #### Plot all the histograms

# In[80]:


get_ipython().run_cell_magic('time', '', 'fig = plt.figure(figsize=(16, 9))\nax = fig.add_subplot(111)\n\nn, bins, patches = ax.hist([aaron_df_for_plotting.subtweet_positive_probability, \n                            julia_df_for_plotting.subtweet_positive_probability, \n                            noah_df_for_plotting.subtweet_positive_probability], \n                           bins="scott",\n                           color=["#256EFF", "#46237A", "#3DDC97"],\n                           density=True, \n                           label=["Aaron", "Julia", "Noah"],\n                           alpha=0.75)\n\naaron_line = scipy.stats.norm.pdf(bins, aaron_mean, aaron_std)\nax.plot(bins, aaron_line, "--", color="#256EFF", linewidth=2)\n\njulia_line = scipy.stats.norm.pdf(bins, julia_mean, julia_std)\nax.plot(bins, julia_line, "--", color="#46237A", linewidth=2)\n\nnoah_line = scipy.stats.norm.pdf(bins, noah_mean, noah_std)\nax.plot(bins, noah_line, "--", color="#3DDC97", linewidth=2)\n\nax.set_xticks([float(x/10) for x in range(11)], minor=False)\nax.set_title("Friends\' Dataset Distribution of Subtweet Probabilities", fontsize=18)\nax.set_xlabel("Probability That Tweet is a Subtweet", fontsize=18)\nax.set_ylabel("Portion of Tweets with That Probability", fontsize=18)\n\nax.legend()\n\nplt.show()')


# #### Statisitics on training data

# #### Remove mentions of usernames for these statistics

# In[81]:


training_data = [" ".join([token for token in tokenizer.tokenize(pair[0]) if "@" not in token]) 
                 for pair in training_data]


# #### Lengths (Less than or equal to 280 characters and greater than or equal to 5 characters)

# In[82]:


length_data = [len(tweet) for tweet in training_data]


# In[83]:


length_data_for_stats = pd.DataFrame({"Length": length_data, "Tweet": training_data})


# In[84]:


# length_data_for_stats = length_data_for_stats[length_data_for_stats["Length"] <= 280]  


# In[85]:


# length_data_for_stats = length_data_for_stats[length_data_for_stats["Length"] >= 5]


# In[86]:


length_data = length_data_for_stats.Length.tolist()


# #### Top 10 longest tweets

# In[87]:


length_data_for_stats.sort_values(by="Length", ascending=False).head(10)


# #### Top 10 shortest tweets

# In[88]:


length_data_for_stats.sort_values(by="Length", ascending=True).head(10)


# #### Tweet length statistics

# In[89]:


length_data_for_stats.describe()


# #### Punctuation

# In[90]:


punctuation_data = [len(set(punctuation).intersection(set(tweet))) for tweet in training_data]


# In[91]:


punctuation_data_for_stats = pd.DataFrame({"Punctuation": punctuation_data, "Tweet": training_data})


# #### Top 10 most punctuated tweets

# In[92]:


punctuation_data_for_stats.sort_values(by="Punctuation", ascending=False).head(10)


# #### Tweets punctuation statistics

# In[93]:


punctuation_data_for_stats.describe()


# #### Stop words

# In[94]:


stop_words_data = [len(set(stopwords.words("english")).intersection(set(tweet.lower()))) 
                   for tweet in training_data]


# In[95]:


stop_words_data_for_stats = pd.DataFrame({"Stop words": stop_words_data, "Tweet": training_data})


# #### Top 10 tweets with most stop words

# In[96]:


stop_words_data_for_stats.sort_values(by="Stop words", ascending=False).head(10)


# #### Top 10 tweets with fewest stop words

# In[97]:


stop_words_data_for_stats.sort_values(by="Stop words", ascending=True).head(10)


# #### Tweets stop words statistics

# In[98]:


stop_words_data_for_stats.describe()


# #### Unique words (at least 2)

# In[99]:


unique_words_data = [len(set(tokenizer.tokenize(tweet))) for tweet in training_data]


# In[100]:


unique_words_data_for_stats = pd.DataFrame({"Unique words": unique_words_data, "Tweet": training_data})


# In[101]:


# unique_words_data_for_stats = unique_words_data_for_stats[unique_words_data_for_stats["Unique words"] >= 2]


# In[102]:


unique_words_data = unique_words_data_for_stats["Unique words"].tolist()


# #### Top 10 tweets with most unique words

# In[103]:


unique_words_data_for_stats.sort_values(by="Unique words", ascending=False).head(10)


# #### Top 10 tweets with fewest unique words

# In[104]:


unique_words_data_for_stats.sort_values(by="Unique words", ascending=True).head(10)


# #### Tweets unique words statistics

# In[105]:


unique_words_data_for_stats.describe()


# #### Plot them

# In[106]:


length_mean = length_data_for_stats.describe().Length[1]
length_std = length_data_for_stats.describe().Length[2]

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)

n, bins, patches = ax.hist(length_data, 
                           bins="scott", 
                           edgecolor="black", 
                           density=True, 
                           color="#12355b", 
                           alpha=0.5)

length_line = scipy.stats.norm.pdf(bins, length_mean, length_std)
ax.plot(bins, length_line, "--", linewidth=3, color="#415d7b")

ax.set_title("Training Dataset Distribution of Tweet Lengths", fontsize=18)
ax.set_xlabel("Tweet Length", fontsize=18);
ax.set_ylabel("Porton of Tweets with That Length", fontsize=18);

plt.show()


# In[107]:


punctuation_mean = punctuation_data_for_stats.describe().Punctuation[1]
punctuation_std = punctuation_data_for_stats.describe().Punctuation[2]

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)

n, bins, patches = ax.hist(punctuation_data, 
                           bins="scott", 
                           edgecolor="black", 
                           density=True, 
                           color="#420039",
                           alpha=0.5)

punctution_line = scipy.stats.norm.pdf(bins, punctuation_mean, punctuation_std)
ax.plot(bins, punctution_line, "--", linewidth=3, color="#673260")

ax.set_title("Training Dataset Distribution of Punctuation", fontsize=18)
ax.set_xlabel("Punctuating Characters", fontsize=18)
ax.set_ylabel("Porton of Punctuating Characters", fontsize=18)

plt.show()


# In[108]:


stop_words_mean = stop_words_data_for_stats.describe()["Stop words"][1]
stop_words_std = stop_words_data_for_stats.describe()["Stop words"][2]

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)

n, bins, patches = ax.hist(stop_words_data, 
                           bins="scott", 
                           edgecolor="black", 
                           density=True, 
                           color="#698f3f",
                           alpha=0.5)

stop_words_line = scipy.stats.norm.pdf(bins, stop_words_mean, stop_words_std)
ax.plot(bins, stop_words_line, "--", linewidth=3, color="#87a565")

ax.set_title("Training Dataset Distribution of Stop Words", fontsize=18)
ax.set_xlabel("Stop Words in Tweet", fontsize=18)
ax.set_ylabel("Porton of Tweets with That Number of Stop Words", fontsize=18)

plt.show()


# In[109]:


unique_words_mean = unique_words_data_for_stats.describe()["Unique words"][1]
unique_words_std = unique_words_data_for_stats.describe()["Unique words"][2]

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)

n, bins, patches = ax.hist(unique_words_data, 
                           bins="scott", 
                           edgecolor="black", 
                           density=True, 
                           color="#Ca2e55",
                           alpha=0.5)

unique_words_line = scipy.stats.norm.pdf(bins, unique_words_mean, unique_words_std)
ax.plot(bins, unique_words_line, "--", linewidth=3, color="#d45776")

ax.set_title("Training Dataset Distribution of Unique Words", fontsize=18)
ax.set_xlabel("Unique Words in Tweet", fontsize=18)
ax.set_ylabel("Porton of Tweets with That Number of Unique Words", fontsize=18)

plt.show()

