
# coding: utf-8

# ## Non-subtweets Downloader Jupyter Notebook-in-Progress

# In[ ]:


import tweepy
import json


# #### Set up access to the API

# In[ ]:


consumer_key, consumer_secret, access_token, access_token_secret = open("credentials_2.txt").read().split("\n")


# In[ ]:


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)


# #### Specifically take advantage of built-in methods to handle Twitter API rate limits

# In[ ]:


api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


# #### Find tweets with replies that do not claim it is a subtweet

# In[ ]:


def get_subtweets(max_tweets=20000, max_replies=10, query=("since:2018-02-28 until:2018-03-28 exclude:retweets exclude:replies")):
    subtweets_list = []
    i = 0
    for potential_subtweet in tweepy.Cursor(api.search, lang="en", tweet_mode="extended", q=query).items(max_tweets):
        i += 1
        if (potential_subtweet.in_reply_to_status_id_str 
            or "subtweet" in potential_subtweet.full_text 
            or "Subtweet" in potential_subtweet.full_text):
            continue
        replies = [reply for reply in tweepy.Cursor(api.search, lang="en", tweet_mode="extended", 
                                                    q="to:{} exclude:retweets".format(potential_subtweet.user.screen_name), 
                                                    since_id=potential_subtweet.id_str).items(max_replies)]
        valid_replies = False
        for reply in replies:
            if "subtweet" in reply.full_text or "Subtweet" in reply.full_text:
                valid_replies = True
                break
        if valid_replies:
            print("Tweet #{0} is a subtweet: {1}\n".format(i+1, potential_subtweet.full_text.replace("\n", " ")))
            subtweets_list.append({"tweet_data": potential_subtweet._json, "replies": [reply._json for reply in replies]})
    return subtweets_list


# In[ ]:


subtweets_list = get_subtweets()
print("Total: {}".format(len(subtweets_list)))
with open("subtweets.json", "w") as outfile:
    json.dump(subtweets_list, outfile, indent=4)

