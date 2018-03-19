
# coding: utf-8

# ## Using Scikit-Learn and NLTK to build a Naive Bayes Classifier that identifies subtweets

# ### Goals:
# #### Use Scikit-Learn pipelines to define special features to add to a Naive Bayes Classifier
# #### Evaluate the accuracy of the classifier
# #### Maybe do it live, on a Twitter API stream

# ### Methods:
# #### Use the training set I made before

# #### Import libraries

# In[ ]:


# %matplotlib inline


# In[ ]:


from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
from textblob import TextBlob
from time import sleep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import datetime
import tweepy
import nltk
import re


# #### Set max column width for dataframes

# In[ ]:


# pd.set_option("max_colwidth", 280)


# #### Load the CSV

# In[ ]:


df = pd.read_csv("../data/data_for_training/final_training_data/Subtweets_Classifier_Training_Data.csv")


# #### Create training and test sets from the single training set I made before

# In[ ]:


text_train, text_test, class_train, class_test = train_test_split(df.alleged_subtweet.tolist(), 
                                                                  df.is_subtweet.tolist())


# #### Use NLTK's tokenizer instead of Scikit's

# In[ ]:


tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)


# #### Function for managing TextBlob polarities

# In[ ]:


def simplify_polarity(polarity):
    if polarity >= 0:
        return 1
    return 0


# #### Class for distinguishing polarizing parts of speech as features

# In[ ]:


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
            
            text_length = len(text)
            num_pronouns = len(prounouns_lower.intersection(tokenized_text))
            num_names = len(first_names_lower.intersection(tokenized_text))
            num_first_person = len(first_person_pronouns_lower.intersection(tokenized_text))
            num_at_symbols = text.count("@")
            num_subtweet = text.count("subtweet") + text.count("Subtweet")
            num_urls = len(re.findall(pattern, text))
            
            weighted_dict = {"useful": text_length,
                             "sentiment": simplify_polarity(TextBlob(text).sentiment.polarity),
                             "impossible": num_subtweet + num_at_symbols + num_urls,
                             "likely": num_pronouns,
                             "unlikely": num_names + num_first_person + num_at_symbols + num_subtweet + num_urls}
            final_output.append(weighted_dict)
        return final_output


# #### Build the pipeline

# In[ ]:


sentiment_pipeline = Pipeline([
    ("features", FeatureUnion([
        ("ngram_tf_idf", Pipeline([
            ("counts", CountVectorizer(tokenizer=tokenizer.tokenize)),
            ("tf_idf", TfidfTransformer())
        ])),
        ("stats_vect", Pipeline([
            ("tweet_stats", TweetStats()),
            ("vect", DictVectorizer()),
        ])),
    ])),
    ("classifier", MultinomialNB())
])


# #### Show the results

# In[ ]:


sentiment_pipeline.fit(text_train, class_train)
predictions = sentiment_pipeline.predict(text_test)


# In[ ]:


print(classification_report(class_test, predictions))


# #### Define function for visualizing confusion matrices

# In[ ]:


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

# In[ ]:


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

# In[ ]:


joblib.dump(sentiment_pipeline, "../data/other_data/subtweets_classifier.pkl") 


# #### Print tests for the classifier

# In[ ]:


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

# In[ ]:


# test_tweets = ["Some people don't know their place.", 
#                "Isn't it funny how some people don't know their place?", 
#                "How come you people act like this?", 
#                "You're such a nerd.",
#                "I love Noah, he's so cool.",
#                "Who the heck is Noah?",
#                "This is a @NoahSegalGould subtweet. Go check out https://segal-gould.com.", 
#                "This is a subtweet.", 
#                "Hey @jack!", 
#                "Hey Jack!",
#                "http://www.google.com"]


# #### Make a dataframe from the list

# In[ ]:


# test_tweets_df = pd.DataFrame({"Tweet": test_tweets, "Sentiment": [None]*len(test_tweets)})


# #### Print the tests

# In[ ]:


# tests_dataframe(test_tweets_df, text_column="Tweet", sentiment_column="Sentiment").head()


# #### Test on actual tweets

# In[ ]:


naji_df = pd.read_csv("../data/data_for_testing/other_data/naji_data.csv", error_bad_lines=False)


# #### Repair some leftover HTML

# In[ ]:


naji_df["SentimentText"] = naji_df["SentimentText"].str.replace("&quot;", "\"")
naji_df["SentimentText"] = naji_df["SentimentText"].str.replace("&amp;", "&")
naji_df["SentimentText"] = naji_df["SentimentText"].str.replace("&gt;", ">")
naji_df["SentimentText"] = naji_df["SentimentText"].str.replace("&lt;", "<")


# #### Remove rows with non-English

# In[ ]:


def is_english(s):
    return all(ord(char) < 128 for char in s)


# In[ ]:


naji_df = naji_df[naji_df["SentimentText"].map(is_english)]


# #### Show the length of the dataset

# In[ ]:


print(len(naji_df))


# #### Use randomly selected 500K rows from dataset

# In[ ]:


# naji_df = naji_df.sample(n=500000).reset_index(drop=True)


# #### Print and time the tests

# In[ ]:


# %%time
naji_df = tests_dataframe(naji_df)


# In[ ]:


naji_df.to_csv("../data/data_from_testing/other_data/naji_tests.csv")


# In[ ]:


# naji_df.head()


# #### Tests on friends' tweets

# In[ ]:


aaron_df = pd.read_csv("../data/data_for_testing/friends_data/akrapf96_tweets.csv").dropna()
aaron_df["Sentiment"] = None


# In[ ]:


# %%time
aaron_df = tests_dataframe(aaron_df, text_column="Text", sentiment_column="Sentiment")


# In[ ]:


aaron_df.to_csv("../data/data_from_testing/friends_data/akrapf96_tests.csv")


# In[ ]:


# aaron_df.head()


# In[ ]:


julia_df = pd.read_csv("../data/data_for_testing/friends_data/juliaeberry_tweets.csv").dropna()
julia_df["Sentiment"] = None


# In[ ]:


# %%time
julia_df = tests_dataframe(julia_df, text_column="Text", sentiment_column="Sentiment")


# In[ ]:


julia_df.to_csv("../data/data_from_testing/friends_data/juliaeberry_tests.csv")


# In[ ]:


# julia_df.head()


# In[ ]:


zoe_df = pd.read_csv("../data/data_for_testing/friends_data/zoeterhune_tweets.csv").dropna()
zoe_df["Sentiment"] = None


# In[ ]:


# %%time
zoe_df = tests_dataframe(zoe_df, text_column="Text", sentiment_column="Sentiment")


# In[ ]:


zoe_df.to_csv("../data/data_from_testing/friends_data/zoeterhune_tests.csv")


# In[ ]:


# zoe_df.head()


# In[ ]:


noah_df = pd.read_csv("../data/data_for_testing/friends_data/noahsegalgould_tweets.csv").dropna()
noah_df["Sentiment"] = None


# In[ ]:


# %%time
noah_df = tests_dataframe(noah_df, text_column="Text", sentiment_column="Sentiment")


# In[ ]:


noah_df.to_csv("../data/data_from_testing/friends_data/noahsegalgould_tests.csv")


# In[ ]:


# noah_df.head()


# #### Test it in realtime
# #### Define some useful variables for later

# In[ ]:


THRESHOLD = 0.95
DURATION = 3600 # 1 hour


# #### Load Twitter API credentials

# In[ ]:


consumer_key, consumer_secret, access_token, access_token_secret = open("../../credentials.txt").read().split("\n")


# #### Prepare the final dataframe

# In[ ]:


subtweets_live_list = []
non_subtweets_live_list = []


# #### Create a custom class for streaming subtweets

# In[ ]:


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
            
            subtweets_live_list.append(row)
            subtweets_df = pd.DataFrame(subtweets_live_list).sort_values(by="subtweet_probability", 
                                                                         ascending=False)
            subtweets_df.to_csv("../data/data_from_testing/live_downloaded_data/subtweets_live_data.csv")
            
            print("Subtweet:\n{}\nGeo Data:{}\nTotal tweets acquired: {}\n".format(print_list, 
                                                                                   status.place.bounding_box.coordinates, 
                                                                                   (len(subtweets_live_list) 
                                                                                    + len(non_subtweets_live_list))))
            
            return row
        else:
            non_subtweets_live_list.append(row)
            non_subtweets_df = pd.DataFrame(non_subtweets_live_list).sort_values(by="subtweet_probability", 
                                                                                 ascending=False)
            non_subtweets_df.to_csv("../data/data_from_testing/live_downloaded_data/non_subtweets_live_data.csv")
            
            # print("Not a Subtweet:\n{}\nTotal tweets acquired: {}\n".format(print_list, len(subtweets_live_list) + len(non_subtweets_live_list)))
            return row


# #### Use the API credentials to connect to the API

# In[ ]:


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, retry_delay=5, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, )


# #### Create a list of all my followers' account IDs

# In[ ]:


my_followers_int = list(set(list(tweepy.Cursor(api.followers_ids, screen_name="NoahSegalGould").pages())[0]))
my_followers_str = [str(i) for i in my_followers_int]


# #### Instantiate the listener

# In[ ]:


stream_listener = StreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)


# #### Start the stream asynchronously, and stop it after some duration of seconds

# In[ ]:


# %%time
bounding_box = [-73.920176, 42.009637,
                -73.899739, 42.033421]
stream.filter(locations=bounding_box, async=True) # Bard College
#stream.filter(follow=my_followers_str, async=True)
#stream.filter(track="some people", async=True)
print("Columns:")
print("screen_name, sentiment_polarity, sentiment_subjectivity, subtweet_probability, time, text")
sleep(DURATION)
stream.disconnect()

