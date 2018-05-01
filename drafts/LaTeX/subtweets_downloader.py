
# coding: utf-8

# #### Script for downloading a ground truth subtweets dataset

# #### Import libraries for accessing the API and managing JSON data

# In[ ]:


import tweepy
import json


# #### Load the API credentials

# In[ ]:


consumer_key, consumer_secret, access_token, access_token_secret = (open("../../credentials.txt")
                                                                    .read().split("\n"))


# #### Authenticate the connection to the API using the credentials

# In[ ]:


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)


# #### Connect to the API

# In[ ]:


api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


# #### Define a function for recursively accessing parent tweets

# In[ ]:


def first_tweet(tweet_status_object):
    try:
        return first_tweet(api.get_status(tweet_status_object.in_reply_to_status_id_str, 
                                          tweet_mode="extended"))
    except tweepy.TweepError:
        return tweet_status_object


# #### Define a function for finding tweets with replies that specifically do call them subtweets

# In[ ]:


def get_subtweets(max_tweets=10000000, 
                  query=("subtweet AND @ since:2018-03-01 exclude:retweets filter:replies")):
    subtweets_ids_list = []
    subtweets_list = []
    i = 0
    for potential_subtweet_reply in tweepy.Cursor(api.search, lang="en", 
                                                  tweet_mode="extended", q=query).items(max_tweets):
        i += 1
        potential_subtweet_original = first_tweet(potential_subtweet_reply)
        if (not potential_subtweet_original.in_reply_to_status_id_str 
            and potential_subtweet_original.user.lang == "en"):
            if (potential_subtweet_original.id_str in subtweets_ids_list 
                or "subtweet" in potential_subtweet_original.full_text 
                or "Subtweet" in potential_subtweet_original.full_text 
                or "SUBTWEET" in potential_subtweet_original.full_text):
                continue
            else:
                subtweets_ids_list.append(potential_subtweet_original.id_str)
                subtweets_list.append({"tweet_data": potential_subtweet_original._json, 
                                       "reply": potential_subtweet_reply._json})
                with open("../data/other_data/subtweets.json", "w") as outfile:
                    json.dump(subtweets_list, outfile, indent=4)
                print(("Tweet #{0} was a reply to a subtweet: {1}\n"
                       .format(i, potential_subtweet_original.full_text.replace("\n", " "))))
    return subtweets_list


# #### Show the results

# In[ ]:


subtweets_list = get_subtweets()
print("Total: {}".format(len(subtweets_list)))

