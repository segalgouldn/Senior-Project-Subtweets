import tweepy
import pause
import time
import re
import pandas as pd
from datetime import time, date, datetime, timedelta
from nltk.tokenize import TweetTokenizer

consumer_key, consumer_secret, access_token, access_token_secret = open("credentials.txt").read().split("\n")

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

def first_tweet(tweet_id):
    tweet = api.get_status(tweet_id)
    try:
        return first_tweet(tweet._json["in_reply_to_status_id"])
    except tweepy.TweepError:
        return tweet
        
number_of_days = 14
for j in range(number_of_days):
    today = date.today()
    today_string = today.strftime("%Y-%m-%d")

    yesterday = today - timedelta(1)
    yesterday_string = yesterday.strftime("%Y-%m-%d")

    two_days_ago = today - timedelta(2)
    two_days_ago_string = two_days_ago.strftime("%Y-%m-%d")

    tomorrow = today + timedelta(1)
    tomorrow_string = tomorrow.strftime("%Y-%m-%d")
    
    query = "\"subtweet\" since:" + two_days_ago_string + " until:" + yesterday_string
    max_tweets = 1000000

    statuses = []
    for status in tweepy.Cursor(api.search, q=query, lang="en").items(max_tweets):
        try:
            if status._json["in_reply_to_status_id"]:
                print(status._json["text"])
                statuses.append(status)
            else:
                continue
        except tweepy.TweepError:
            continue

    print("Statuses acquired: " + str(len(statuses)))

    statuses = [status for status in statuses if "subtweet" in status._json["text"]]
    print("Statuses actually containing \"subtweet\": " + str(len(statuses)))


    df_dict = {}
    accuser_usernames = []
    subtweet_evidences = []
    subtweet_evidence_ids = []
    subtweeter_usernames = []
    alleged_subtweets = []
    alleged_subtweet_ids = []

    for i in range(len(statuses)):
        status = statuses[i]._json
    
        user = status["user"]["screen_name"]
        tweet_text = status["text"]
        tweet_id = status["id"]
        
        accuser_usernames.append(user)
        subtweet_evidences.append(tweet_text)
        subtweet_evidence_ids.append(tweet_id)
        try:
            first = first_tweet(tweet_id)._json
            first_user = first["user"]["screen_name"]
            first_text = first["text"]
            first_id = first["id"]
            if first_user != user: 
                subtweeter_usernames.append(first_user)
                alleged_subtweets.append(first_text)
                alleged_subtweet_ids.append(first_id)
            else:
                del accuser_usernames[-1]
                del subtweet_evidences[-1]
                del subtweet_evidence_ids[-1]
        except tweepy.TweepError:
            del accuser_usernames[-1]
            del subtweet_evidences[-1]
            del subtweet_evidence_ids[-1]

    df_dict = {"accuser_username": accuser_usernames, 
               "subtweet_evidence": subtweet_evidences, 
               "subtweet_evidence_id": subtweet_evidence_ids, 
               "subtweeter_username": subtweeter_usernames,
               "alleged_subtweet": alleged_subtweets,
               "alleged_subtweet_id": alleged_subtweet_ids}

    tokenizer = TweetTokenizer()

    df_dict_copy = {"accuser_username": [], 
                    "subtweet_evidence": [], 
                    "subtweet_evidence_id": [], 
                    "subtweeter_username": [],
                    "alleged_subtweet": [],
                    "alleged_subtweet_id": []}

    pattern = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')

    for i in range(len(df_dict["alleged_subtweet"])):
        if ("@" not in df_dict["alleged_subtweet"][i] and 
            not pattern.findall(df_dict["alleged_subtweet"][i]) and 
            "subtweet" not in df_dict["alleged_subtweet"][i] and 
            len(tokenizer.tokenize(df_dict["alleged_subtweet"][i])) > 5): 
            df_dict_copy["accuser_username"].append(df_dict["accuser_username"][i])
            df_dict_copy["subtweet_evidence"].append(df_dict["subtweet_evidence"][i])
            df_dict_copy["subtweet_evidence_id"].append(df_dict["subtweet_evidence_id"][i])
            df_dict_copy["subtweeter_username"].append(df_dict["subtweeter_username"][i])
            df_dict_copy["alleged_subtweet"].append(df_dict["alleged_subtweet"][i])
            df_dict_copy["alleged_subtweet_id"].append(df_dict["alleged_subtweet_id"][i])

    print("Number of accusers (usernames): " + str(len(df_dict_copy["accuser_username"])))
    print("Number of evidence Tweets (text): " + str(len(df_dict_copy["subtweet_evidence"])))
    print("Number of evidence Tweets (IDs): " + str(len(df_dict_copy["subtweet_evidence_id"])))
    print("Number of subtweeters (usernames): " + str(len(df_dict_copy["subtweeter_username"])))
    print("Number of subtweets (text): " + str(len(df_dict_copy["alleged_subtweet"])))
    print("Number of subtweets (IDs): " + str(len(df_dict_copy["alleged_subtweet_id"])))

    df = pd.DataFrame(df_dict_copy, columns=["accuser_username", 
                                             "subtweet_evidence", 
                                             "subtweet_evidence_id", 
                                             "subtweeter_username", 
                                             "alleged_subtweet", 
                                             "alleged_subtweet_id"])

    df.to_csv(two_days_ago_string + " to " + yesterday_string + ".csv")
    pause.until(datetime.combine(tomorrow, time.min))
