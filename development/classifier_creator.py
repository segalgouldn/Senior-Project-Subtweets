
# coding: utf-8

# ## Using Scikit-Learn and NLTK to build a Naive Bayes Classifier that identifies subtweets

# ### Goals:
# #### Use Scikit-Learn pipelines to define special features to add to a Naive Bayes Classifier
# #### Evaluate the accuracy of the classifier
# #### Maybe do it live, on a Twitter API stream

# ### Methods:
# #### Use the training set I made before

# #### Import libraries

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
from textblob import TextBlob
from time import time, sleep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import datetime
import tweepy
import nltk
import json
import re


# #### Set max column width for dataframes

# In[3]:


# pd.set_option("max_colwidth", 280)


# #### Load the CSV

# In[4]:


df = pd.read_csv("../data/data_for_training/final_training_data/Subtweets_Classifier_Training_Data.csv")


# #### Create training and test sets from the single training set I made before

# In[5]:


text_train, text_test, class_train, class_test = train_test_split(df.alleged_subtweet.tolist(), 
                                                                  df.is_subtweet.tolist())


# #### Use NLTK's tokenizer instead of Scikit's

# In[6]:


tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)


# #### Function for managing TextBlob polarities

# In[7]:


def simplify_polarity(polarity):
    if polarity >= 0:
        return 1
    return 0


# #### Class for distinguishing polarizing parts of speech as features

# In[8]:


class TweetStats(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        first_names = ["Aaliyah", "Aaron", "Abby", "Abigail", "Abraham", "Adam",
                       "Addison", "Adrian", "Adriana", "Adrianna", "Aidan", "Aiden",
                       "Alan", "Alana", "Alejandro", "Alex", "Alexa", "Alexander",
                       "Alexandra", "Alexandria", "Alexia", "Alexis", "Alicia", "Allison",
                       "Alondra", "Alyssa", "Amanda", "Amber", "Amelia", "Amy",
                       "Ana", "Andrea", "Andres", "Andrew", "Angel", "Angela",
                       "Angelica", "Angelina", "Anna", "Anthony", "Antonio", "Ariana",
                       "Arianna", "Ashley", "Ashlyn", "Ashton", "Aubrey", "Audrey",
                       "Austin", "Autumn", "Ava", "Avery", "Ayden", "Bailey",
                       "Benjamin", "Bianca", "Blake", "Braden", "Bradley", "Brady",
                       "Brandon", "Brayden", "Breanna", "Brendan", "Brian", "Briana",
                       "Brianna", "Brittany", "Brody", "Brooke", "Brooklyn", "Bryan",
                       "Bryce", "Bryson", "Caden", "Caitlin", "Caitlyn", "Caleb",
                       "Cameron", "Camila", "Carlos", "Caroline", "Carson", "Carter",
                       "Cassandra", "Cassidy", "Catherine", "Cesar", "Charles", "Charlotte",
                       "Chase", "Chelsea", "Cheyenne", "Chloe", "Christian", "Christina",
                       "Christopher", "Claire", "Cody", "Colby", "Cole", "Colin",
                       "Collin", "Colton", "Conner", "Connor", "Cooper", "Courtney",
                       "Cristian", "Crystal", "Daisy", "Dakota", "Dalton", "Damian",
                       "Daniel", "Daniela", "Danielle", "David", "Delaney", "Derek",
                       "Destiny", "Devin", "Devon", "Diana", "Diego", "Dominic",
                       "Donovan", "Dylan", "Edgar", "Eduardo", "Edward", "Edwin",
                       "Eli", "Elias", "Elijah", "Elizabeth", "Ella", "Ellie", 
                       "Emily", "Emma", "Emmanuel", "Eric", "Erica", "Erick",
                       "Erik", "Erin", "Ethan", "Eva", "Evan", "Evelyn",
                       "Faith", "Fernando", "Francisco", "Gabriel", "Gabriela", "Gabriella",
                       "Gabrielle", "Gage", "Garrett", "Gavin", "Genesis", "George",
                       "Gianna", "Giovanni", "Giselle", "Grace", "Gracie", "Grant",
                       "Gregory", "Hailey", "Haley", "Hannah", "Hayden", "Hector",
                       "Henry", "Hope", "Hunter", "Ian", "Isaac", "Isabel",
                       "Isabella", "Isabelle", "Isaiah", "Ivan", "Jack", "Jackson",
                       "Jacob", "Jacqueline", "Jada", "Jade", "Jaden", "Jake",
                       "Jalen", "James", "Jared", "Jasmin", "Jasmine", "Jason", 
                       "Javier", "Jayden", "Jayla", "Jazmin", "Jeffrey", "Jenna",
                       "Jennifer", "Jeremiah", "Jeremy", "Jesse", "Jessica", "Jesus",
                       "Jillian", "Jocelyn", "Joel", "John", "Johnathan", "Jonah",
                       "Jonathan", "Jordan", "Jordyn", "Jorge", "Jose", "Joseph",
                       "Joshua", "Josiah", "Juan", "Julia", "Julian", "Juliana",
                       "Justin", "Kaden", "Kaitlyn", "Kaleb", "Karen", "Karina",
                       "Kate", "Katelyn", "Katherine", "Kathryn", "Katie", "Kayla",
                       "Kaylee", "Kelly", "Kelsey", "Kendall", "Kennedy", "Kenneth",
                       "Kevin", "Kiara", "Kimberly", "Kyle", "Kylee", "Kylie",
                       "Landon", "Laura", "Lauren", "Layla", "Leah", "Leonardo",
                       "Leslie", "Levi", "Liam", "Liliana", "Lillian", "Lilly",
                       "Lily", "Lindsey", "Logan", "Lucas", "Lucy", "Luis",
                       "Luke", "Lydia", "Mackenzie", "Madeline", "Madelyn", "Madison",
                       "Makayla", "Makenzie", "Malachi", "Manuel", "Marco", "Marcus",
                       "Margaret", "Maria", "Mariah", "Mario", "Marissa", "Mark",
                       "Martin", "Mary", "Mason", "Matthew", "Max", "Maxwell",
                       "Maya", "Mckenzie", "Megan", "Melanie", "Melissa", "Mia",
                       "Micah", "Michael", "Michelle", "Miguel", "Mikayla", "Miranda",
                       "Molly", "Morgan", "Mya", "Naomi", "Natalia", "Natalie",
                       "Nathan", "Nathaniel", "Nevaeh", "Nicholas", "Nicolas", "Nicole",
                       "Noah", "Nolan", "Oliver", "Olivia", "Omar", "Oscar",
                       "Owen", "Paige", "Parker", "Patrick", "Paul", "Payton",
                       "Peter", "Peyton", "Preston", "Rachel", "Raymond", "Reagan",
                       "Rebecca", "Ricardo", "Richard", "Riley", "Robert", "Ruby",
                       "Ryan", "Rylee", "Sabrina", "Sadie", "Samantha", "Samuel",
                       "Sara", "Sarah", "Savannah", "Sean", "Sebastian", "Serenity",
                       "Sergio", "Seth", "Shane", "Shawn", "Shelby", "Sierra",
                       "Skylar", "Sofia", "Sophia", "Sophie", "Spencer", "Stephanie",
                       "Stephen", "Steven", "Summer", "Sydney", "Tanner", "Taylor", 
                       "Thomas", "Tiffany", "Timothy", "Travis", "Trenton", "Trevor",
                       "Trinity", "Tristan", "Tyler", "Valeria", "Valerie", "Vanessa",
                       "Veronica", "Victor", "Victoria", "Vincent", "Wesley", "William",
                       "Wyatt", "Xavier", "Zachary", "Zoe", "Zoey"]
        first_names_lower = set([name.lower() for name in first_names])

        pronouns = ["You", "You're", "Your", 
                    "She", "She's", "Her", "Hers", 
                    "He", "He's", "Him", "His", 
                    "They", "They're", "Them", "Their", "Theirs"]
        prounouns_lower = set([pronoun.lower() for pronoun in pronouns])
        
        first_person_pronouns = ["I", "I'm", "We", "We're", "Our", "My", "Us"]
        first_person_pronouns_lower = set([pronoun.lower() for pronoun in first_person_pronouns])
        
        pattern = "(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
        
        final_output = []
        for text in posts:
            tokenized_text = tokenizer.tokenize(text)
            
            num_pronouns = len(prounouns_lower.intersection(tokenized_text))
            num_names = len(first_names_lower.intersection(tokenized_text))
            num_first_person = len(first_person_pronouns_lower.intersection(tokenized_text))
            num_at_symbols = text.count("@")
            num_subtweet = text.count("subtweet") + text.count("Subtweet")
            num_urls = len(re.findall(pattern, text))
            
            weighted_dict = {"sentiment": simplify_polarity(TextBlob(text).sentiment.polarity), 
                             "num_subtweet": num_subtweet,
                             "num_at_symbols": num_at_symbols, 
                             "num_urls": num_urls,
                             "num_pronouns": num_pronouns,
                             "num_names": num_names, 
                             "num_first_person": num_first_person, 
                             "num_at_symbols": num_at_symbols,
                             "num_subtweet": num_subtweet,
                             "num_urls": num_urls}
            final_output.append(weighted_dict)
        return final_output


# #### Build the pipeline

# In[9]:


sentiment_pipeline = Pipeline([
    ("features", FeatureUnion([
        ("ngram_tf_idf", Pipeline([
            ("counts", CountVectorizer(tokenizer=tokenizer.tokenize)),
            ("tf_idf", TfidfTransformer())
        ])),
        ("stats_vect", Pipeline([
            ("tweet_stats", TweetStats()),
            ("vect", DictVectorizer())
        ]))
    ])),
    ("classifier", MultinomialNB())
])


# #### Show the results

# In[10]:


sentiment_pipeline.fit(text_train, class_train)
predictions = sentiment_pipeline.predict(text_test)


# In[11]:


print(classification_report(class_test, predictions))


# #### Define function for visualizing confusion matrices

# In[12]:


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# #### Show the matrices

# In[13]:


# class_names = ["negative", "positive"]

# cnf_matrix = confusion_matrix(class_test, predictions)
# np.set_printoptions(precision=2)

# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

# plt.show()


# #### Save the classifier for another time

# In[14]:


joblib.dump(sentiment_pipeline, "../data/other_data/subtweets_classifier.pkl") 


# #### Print tests for the classifier

# In[15]:


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

# In[16]:


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

# In[17]:


test_tweets_df = pd.DataFrame({"Tweet": test_tweets, "Sentiment": [None]*len(test_tweets)})


# #### Print the tests

# In[18]:


# tests_dataframe(test_tweets_df, text_column="Tweet", sentiment_column="Sentiment").head()


# #### Test on actual tweets

# In[19]:


# naji_df = pd.read_csv("../data/data_for_testing/other_data/naji_data.csv", error_bad_lines=False)


# #### Repair some leftover HTML

# In[20]:


# naji_df["SentimentText"] = naji_df["SentimentText"].str.replace("&quot;", "\"")
# naji_df["SentimentText"] = naji_df["SentimentText"].str.replace("&amp;", "&")
# naji_df["SentimentText"] = naji_df["SentimentText"].str.replace("&gt;", ">")
# naji_df["SentimentText"] = naji_df["SentimentText"].str.replace("&lt;", "<")


# #### Remove rows with non-English

# In[21]:


def is_english(s):
    return all(ord(char) < 128 for char in s)


# In[22]:


# naji_df = naji_df[naji_df["SentimentText"].map(is_english)]


# #### Show the length of the dataset

# In[23]:


# print("Length of dataset: {}".format(len(naji_df)))


# #### Use randomly selected 50K rows from dataset

# In[24]:


# naji_df = naji_df.sample(n=50000).reset_index(drop=True)


# #### Print and time the tests

# In[25]:


# get_ipython().run_cell_magic('time', '', 'naji_df = tests_dataframe(naji_df)')


# In[26]:


# naji_df.to_csv("../data/data_from_testing/other_data/naji_tests.csv")


# In[27]:


# naji_df.head()


# #### Plot the results

# In[28]:


# naji_df_columns = ["sentiment_score", "subtweet_negative_probability"]


# In[29]:


# naji_df = naji_df.set_index("tweet").drop(naji_df_columns, axis=1).head(10)


# In[30]:


# naji_df.plot.barh(logx=True);


# #### Tests on friends' tweets

# In[31]:


# aaron_df = pd.read_csv("../data/data_for_testing/friends_data/akrapf96_tweets.csv").dropna()
# aaron_df["Sentiment"] = None


# In[32]:


# get_ipython().run_cell_magic('time', '', 'aaron_df = tests_dataframe(aaron_df, text_column="Text", sentiment_column="Sentiment")')


# In[33]:


# aaron_df.to_csv("../data/data_from_testing/friends_data/akrapf96_tests.csv")


# In[34]:


# aaron_df["tweet"] = aaron_df["tweet"].str[:140]


# In[35]:


# aaron_df.head()


# #### Plot the results

# In[36]:


# aaron_df_columns = ["sentiment_score", "subtweet_negative_probability"]


# In[37]:


# aaron_df = aaron_df.set_index("tweet").drop(aaron_df_columns, axis=1).head(10)


# In[38]:


# aaron_df.plot.barh(logx=True);


# In[39]:


# julia_df = pd.read_csv("../data/data_for_testing/friends_data/juliaeberry_tweets.csv").dropna()
# julia_df["Sentiment"] = None


# In[40]:


# get_ipython().run_cell_magic('time', '', 'julia_df = tests_dataframe(julia_df, text_column="Text", sentiment_column="Sentiment")')


# In[41]:


# julia_df.to_csv("../data/data_from_testing/friends_data/juliaeberry_tests.csv")


# In[42]:


# julia_df["tweet"] = julia_df["tweet"].str[:140]


# In[43]:


# julia_df.head()


# #### Plot the results

# In[44]:


# julia_df_columns = ["sentiment_score", "subtweet_negative_probability"]


# In[45]:


# julia_df = julia_df.set_index("tweet").drop(julia_df_columns, axis=1).head(10)


# In[46]:


# julia_df.plot.barh(logx=True);


# In[47]:


# zoe_df = pd.read_csv("../data/data_for_testing/friends_data/zoeterhune_tweets.csv").dropna()
# zoe_df["Sentiment"] = None


# In[48]:


# get_ipython().run_cell_magic('time', '', 'zoe_df = tests_dataframe(zoe_df, text_column="Text", sentiment_column="Sentiment")')


# In[49]:


# zoe_df.to_csv("../data/data_from_testing/friends_data/zoeterhune_tests.csv")


# In[50]:


# zoe_df["tweet"] = zoe_df["tweet"].str[:140]


# In[51]:


# zoe_df.head()


# #### Plot the results

# In[52]:


# zoe_df_columns = ["sentiment_score", "subtweet_negative_probability"]


# In[53]:


# zoe_df = zoe_df.set_index("tweet").drop(zoe_df_columns, axis=1).head(10)


# In[54]:


# zoe_df.plot.barh(logx=True);


# In[55]:


# noah_df = pd.read_csv("../data/data_for_testing/friends_data/noahsegalgould_tweets.csv").dropna()
# noah_df["Sentiment"] = None


# In[56]:


# get_ipython().run_cell_magic('time', '', 'noah_df = tests_dataframe(noah_df, text_column="Text", sentiment_column="Sentiment")')


# In[57]:


# noah_df.to_csv("../data/data_from_testing/friends_data/noahsegalgould_tests.csv")


# In[58]:


# noah_df["tweet"] = noah_df["tweet"].str[:140]


# In[59]:


# noah_df.head()


# #### Plot the results

# In[60]:


# noah_df_columns = ["sentiment_score", "subtweet_negative_probability"]


# In[61]:


# noah_df = noah_df.set_index("tweet").drop(noah_df_columns, axis=1).head(10)


# In[62]:


# noah_df.plot.barh(logx=True);


# #### Test it in realtime
# #### Define some useful variables for later

# In[63]:


THRESHOLD = 0.925 # 92.5% positives and higher, only
DURATION = 60*60*24*7 # 1 week


# #### Load Twitter API credentials

# In[64]:


consumer_key, consumer_secret, access_token, access_token_secret = open("../../credentials.txt").read().split("\n")


# #### Use the API credentials to connect to the API

# In[65]:


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, retry_delay=1, timeout=120, # 2 minutes
                 compression=True, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# #### Prepare the final dataframe

# In[66]:


subtweets_live_list = []
non_subtweets_live_list = []


# #### Create a custom class for streaming subtweets

# In[67]:


class StreamListener(tweepy.StreamListener):
    def on_status(self, status):
        text = status.text
        text = text.replace("&quot;", "\"").replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
        
        # negative_probability = sentiment_pipeline.predict_proba([text]).tolist()[0][0]
        positive_probability = sentiment_pipeline.predict_proba([text]).tolist()[0][1]
        
        screen_name = status.user.screen_name
        created_at = status.created_at
        
        sentiment = TextBlob(text).sentiment
        
        sentiment_polarity = sentiment.polarity
        sentiment_subjectivity = sentiment.subjectivity
        
        row = {"tweet": text, 
               "screen_name": screen_name, 
               "time": created_at, 
               "subtweet_probability": positive_probability, 
               "sentiment_polarity": sentiment_polarity, 
               "sentiment_subjectivity": sentiment_subjectivity}
        print_list = pd.DataFrame([row]).values.tolist()[0]
        
        if all([positive_probability >= THRESHOLD,
                not status.retweeted,
                "RT @" not in text, 
                not status.in_reply_to_status_id]):
            
            api.update_status("{:.1%} \nhttps://twitter.com/{}/status/{}".format(positive_probability, 
                                                                                 screen_name, 
                                                                                 status.id))
            
            subtweets_live_list.append(row)
            subtweets_df = pd.DataFrame(subtweets_live_list).sort_values(by="subtweet_probability", 
                                                                         ascending=False)
            subtweets_df.to_csv("../data/data_from_testing/live_downloaded_data/subtweets_live_data.csv")
            
            print("Subtweet:\n{}\nGeographical Bounding Box: {}\nTotal tweets acquired: {}\n".format(str(print_list)[1:-1],
                                                                                                     None,
                                                                                                     # status.place.bounding_box.coordinates, 
                                                                                                     (len(subtweets_live_list) + len(non_subtweets_live_list))))
            
            return row
        else:
            non_subtweets_live_list.append(row)
            non_subtweets_df = pd.DataFrame(non_subtweets_live_list).sort_values(by="subtweet_probability", 
                                                                                 ascending=False)
            non_subtweets_df.to_csv("../data/data_from_testing/live_downloaded_data/non_subtweets_live_data.csv")
            
            # print("Not a Subtweet:\n{}\nTotal tweets acquired: {}\n".format(print_list, len(subtweets_live_list) + len(non_subtweets_live_list)))
            return row


# #### Get a list of the IDs of all my mutuals and my mutuals' followers

# In[68]:


# get_ipython().run_cell_magic('time', '', 'my_followers = [str(user_id) for ids_list in \n                tweepy.Cursor(api.followers_ids, \n                              screen_name="NoahSegalGould").pages() \n                for user_id in ids_list]\nusers_i_follow = [str(user_id) for ids_list in \n                  tweepy.Cursor(api.friends_ids, \n                                screen_name="NoahSegalGould").pages() \n                  for user_id in ids_list]\n\nmutuals = list(set(my_followers) & set(users_i_follow))\n\nmy_mutuals = mutuals[:]\nfor i, mutual in enumerate(mutuals):\n    start_time = time()\n    user = api.get_user(user_id=mutual)\n    if not user.protected:\n        individual_mutuals_followers = []\n        c = tweepy.Cursor(api.followers_ids, user_id=mutual).items()\n        while True:\n            try:\n                individual_mutuals_follower = c.next()\n                individual_mutuals_followers.append(str(individual_mutuals_follower))\n            except tweepy.TweepError:\n                sleep(600) # 10 minutes\n                continue\n            except StopIteration:\n                break\n        total = len(individual_mutuals_followers)\n        name = user.screen_name\n        print("{} followers for mutual {}: {}".format(total, i+1, name))\n        if total <= 2500:\n            my_mutuals.extend(individual_mutuals_followers)\n        else:\n            print("\\tMutual {0}: {1} has too many followers: {2}".format(i+1, name, total))\n    else:\n        continue\n    end_time = time()\n    with open("../data/other_data/NoahSegalGould_Mutuals_and_Mutuals_Followers_ids.json", "w") as outfile:\n        json.dump(my_mutuals, outfile)\n    print("{0:.2f} seconds for getting the followers\' IDs of mutual {1}: {2}\\n".format((end_time - start_time), \n                                                                                       i+1, user.screen_name))\n    sleep(5)\nmy_mutuals = list(set(my_mutuals))')


# In[69]:


# print("Total number of my mutuals: {}".format(len(mutuals)))


# In[70]:


# print("Total number of my mutuals' followers: {}".format(len(my_mutuals) - len(mutuals)))


# #### Instantiate the listener

# In[ ]:


stream_listener = StreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)


# #### Start the stream asynchronously, and stop it after some duration of seconds

# In[ ]:

my_mutuals = json.load(open("../data/other_data/NoahSegalGould_Mutuals_and_Mutuals_Followers_ids.json"))

stream.filter(follow=my_mutuals, async=True)
print("Columns:")
print("screen_name, sentiment_polarity, sentiment_subjectivity, subtweet_probability, time, text")
sleep(DURATION)
stream.disconnect()


# #### Plot the results

# In[ ]:


# subtweets_df = pd.read_csv("../data/data_from_testing/live_downloaded_data/subtweets_live_data.csv", index_col=0)


# In[ ]:


# subtweets_df["tweet"] = subtweets_df["tweet"].str[:140]


# In[ ]:


# subtweets_df_columns = ["screen_name", "time"]


# In[ ]:


# subtweets_df = subtweets_df.set_index("tweet").drop(subtweets_df_columns, axis=1).head(10)


# In[ ]:


# subtweets_df.plot.barh(logx=True);

