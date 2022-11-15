from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import pandas as pd
import re
import geocoder
import textblob
from textblob import TextBlob
from community import community_louvain

from os import path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from pandas import DataFrame
from PIL import Image
import base64
import io
import networkx as nx


from nltk.corpus import stopwords
global status
nltk.download('stopwords')

set(stopwords.words('english'))


import tweepy
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""


# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)


app = Flask(__name__)


@app.route('/')
def home1():
    public_tweets = api.home_timeline(tweet_mode = 'extended')
    #for tweet in public_tweets:
    #    print(tweet.full_text, tweet.created_at)
    return render_template('home.html', final=public_tweets[0].full_text,response=public_tweets)

@app.route('/hashtag.html')
def hashtag():
    global status
    return render_template('hashtag.html')

@app.route('/wordcloud.html')
def wordcloud():
    return render_template('wordcloud.html')

@app.route('/location.html')
def location():
    return render_template('location.html')


@app.route('/sentiment.html')
def sentiment():
    return render_template('sentiment.html')

@app.route('/profile.html')
def profile():
    return render_template('profile.html')

@app.route('/favorites_retweets.html')
def favorites_retweets():
    return render_template('favorites_retweets.html')

@app.route('/followers.html')
def follower_s():
    return render_template('followers.html')

@app.route('/tweets_count.html')
def tweets_count():
    return render_template('tweets_count.html')

@app.route('/profile.html', methods=['POST'])
def profile_post():
    text1 = request.form['twitter_username'].lower()
    try:
        details = api.get_user(screen_name=text1)
        tweets = api.user_timeline(screen_name=text1,include_rts=False,count=25,tweet_mode='extended')
        tweets_25=[]

        for tweet in tweets:
            tweets_25.append([tweet.full_text,tweet.id])

        return render_template('profile.html', final=1, user_details=details,len=len(tweets_25),tweets_25=tweets_25)

    except tweepy.errors.NotFound as e:
        return render_template('404.html')

@app.errorhandler(404)
def page_not_found(e):

    return render_template('404.html')

@app.route('/hashtag.html', methods=['POST'])
def hashtag_post():
    
    text1 = request.form['twitter_hashtag'].lower()
    l=[]
    for tweet in tweepy.Cursor(api.search_tweets,q=text1,count=100,
                           lang="en",tweet_mode="extended").items(100):
        if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
            l.append(tweet)

    return render_template('hashtag.html', final=1, text=l, hashtag=text1)


@app.route('/location.html', methods=['POST'])
def location_post():
    text1 = request.form['location'].lower()  # location as argument variable

    # Trends for Specific Country
    g = geocoder.osm(text1)  # getting object that has location's latitude and longitude

    closest_loc = api.closest_trends(g.lat, g.lng)
    # fetching the trends
    trends = api.get_place_trends(closest_loc[0]['woeid'], lang="en")

    loc = []
    dataframe_data=[]
    for value in trends:
        for trend in value['trends']:
            if trend['tweet_volume']:
                dataframe_data.append([trend['name'],trend['tweet_volume']])
                loc.append([trend['name'],trend['url'],trend['tweet_volume']])


    loc_data = sorted(loc, key=lambda row: (row[2]), reverse=True)
    new_data = sorted(dataframe_data, key=lambda row: (row[1]), reverse=True)


    df = pd.DataFrame(new_data,columns =['tweet', 'tweet_volume'])
    print(df)
    df.plot.barh(x='tweet', y='tweet_volume', rot=0)

    plt.title("Trending hashtags and their count")
    plt.xlabel("Hashtag Count")
    plt.ylabel("Hashtag")

    plt.savefig('static/location_trends.png', bbox_inches='tight')

    im = Image.open("static/location_trends.png")
    im = im.convert('RGB')
    data = io.BytesIO()
    im.save(data, "JPEG")

    # Then encode the saved image file.
    encoded_img_data = base64.b64encode(data.getvalue())

    return render_template('location.html', final=1, loc=text1,loc_tweets=loc_data,img_data=encoded_img_data.decode('utf-8'))

def get_tweets(query, count=10):
    tweets = []
    try:
        # call twitter api to fetch tweets
        for tweet in tweepy.Cursor(api.search_tweets, q=query, count=80, lang="en").items(80):
            if tweet.retweet_count > 0:
                # if tweet has retweets, ensure that it is appended only once
                if tweet.text not in tweets:
                    tweets.append(tweet.text)
                else:
                    tweets.append(tweet.text)
        return tweets
    except Exception as e:
        print("Error : " + str(e))

def get_tweet_sentiment(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


def clean_tweet(tweet):
    # return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    # Converting the whole text to lowercase
    Tweet_Texts_Cleaned = tweet.lower()

    # Removing the URLS from the tweet string
    Tweet_Texts_Cleaned = re.sub(r'http\S+', ' ', Tweet_Texts_Cleaned)

    # Deleting everything which is not characters
    Tweet_Texts_Cleaned = re.sub(r'[^a-z A-Z]', ' ', Tweet_Texts_Cleaned)

    return Tweet_Texts_Cleaned

@app.route('/sentiment.html', methods=['POST'])
def sentiment_post():
    # convert to lowercase
    text1 = request.form['keyword'].lower()

    tweets = []
    #for tweet in get_tweets(query=text1, count=200):
    for tweet in tweepy.Cursor(api.search_tweets, q=text1, count=200,
                               lang="en", tweet_mode="extended").items(200):
        if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
            tweets.append(tweet)

    ptweets = [tweet for tweet in tweets if get_tweet_sentiment(tweet.full_text) == 'positive']
    p_percentage = 100 * len(ptweets) / len(tweets)
    p_percentage = round(p_percentage,2)

    ntweets = [tweet for tweet in tweets if get_tweet_sentiment(tweet.full_text) == 'negative']
    n_percentage = 100 * len(ntweets) / len(tweets)
    n_percentage = round(n_percentage,2)

    neu_percentage = 100 * (len(tweets) - (len(ntweets) + len(ptweets))) / len(tweets)
    neu_percentage = round(neu_percentage,2)

    pieLabels = ["Positive", "Negative", "Neutral"]

    TweetSA = [p_percentage, n_percentage, neu_percentage]

    figureObject, axesObject = plt.subplots()

    axesObject.pie(TweetSA, labels=pieLabels, autopct='%1.2f', startangle=90)

    axesObject.axis('equal')

    plt.savefig('static/sentiment_analysis.png', bbox_inches='tight')

    im = Image.open("static/sentiment_analysis.png")
    im = im.convert('RGB')
    data = io.BytesIO()
    im.save(data, "JPEG")

    # Then encode the saved image file.
    encoded_img_data = base64.b64encode(data.getvalue())


    return render_template('sentiment.html', final=1, keyword=text1,pos_tweets=ptweets,neg_tweets=ntweets, pos_per=p_percentage, neg_per=n_percentage, neu_per=neu_percentage,img_data=encoded_img_data.decode('utf-8'))


def clear_tweet(tweet):
    # Converting the whole text to lowercase
    Tweet_Texts_Cleaned = tweet.lower()

    # Removing the twitter usernames from tweet string
    Tweet_Texts_Cleaned = re.sub(r'@\w+', ' ', Tweet_Texts_Cleaned)

    # Removing the URLS from the tweet string
    Tweet_Texts_Cleaned = re.sub(r'http\S+', ' ', Tweet_Texts_Cleaned)

    # Deleting everything which is not characters
    Tweet_Texts_Cleaned = re.sub(r'[^a-z A-Z]', ' ', Tweet_Texts_Cleaned)

    # Deleting any word which is less than 3-characters mostly those are stopwords
    Tweet_Texts_Cleaned = re.sub(r'\b\w{1,2}\b', '', Tweet_Texts_Cleaned)

    # Stripping extra spaces in the text
    Tweet_Texts_Cleaned = re.sub(r' +', ' ', Tweet_Texts_Cleaned)

    return Tweet_Texts_Cleaned

@app.route('/wordcloud.html',  methods=['POST'])
def wordcloud_post():
    query=request.form['word'].lower() + " -filter:retweets"
    text = ""
    limit = 100
    language = "en"
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=query, lang=language).items(limit):
        text += clear_tweet(tweet.text.lower())
        tweets.append(tweet)

    cloud = WordCloud(background_color="black").generate(text)
    plt.figure(figsize=(10, 5))
    plt.axis('off')
    plt.imshow(cloud, aspect='auto',interpolation='bilinear')
    plt.savefig('static/wordcloud.png', bbox_inches='tight')

    im = Image.open("static/wordcloud.png")
    im = im.convert('RGB')
    data = io.BytesIO()
    im.save(data, "JPEG")

    # Then encode the saved image file.
    encoded_img_data = base64.b64encode(data.getvalue())
    #print(len(tweets))
    return render_template('wordcloud.html',len=len(tweets),search=request.form['word'].lower(),tweets=tweets,img_data=encoded_img_data.decode('utf-8'))

@app.route('/tweets_count.html',  methods=['POST'])
def tweets_count_post():
    #query = request.form['word'].lower()
    home_cursor = tweepy.Cursor(api.home_timeline).items(100)
    authors = []
    for tweet in home_cursor:
        #print(tweet.author.screen_name, tweet.text)
        authors.append(tweet.author.screen_name)
    #tweets = [i.author.screen_name for i in home_cursor]
    unq_authors = set(authors)
    #print(unq_authors)
    freq = {uname: authors.count(uname) for uname in unq_authors}
    #print(freq.keys())
    #print(list(unq_authors))
    plt.barh(list(unq_authors), freq.values())
    plt.xlabel("Tweets count")
    plt.ylabel("Twitter user")
    plt.title("Tweets count of users from home timeline")
    plt.savefig('static/tweets_count.png', bbox_inches='tight')

    im = Image.open("static/tweets_count.png")
    im = im.convert('RGB')
    data = io.BytesIO()
    im.save(data, "JPEG")

    # Then encode the saved image file.
    encoded_img_data = base64.b64encode(data.getvalue())

    return render_template('tweets_count.html',img_data=encoded_img_data.decode('utf-8'))


@app.route('/favorites_retweets.html',  methods=['POST'])
def favorites_retweets_screen():
    user = request.form['keyword'].lower()
    tweets = api.user_timeline(screen_name=user,
                               count=200,
                               include_rts=False,
                               tweet_mode='extended'
                               )
    all_tweets = []
    all_tweets.extend(tweets)
    oldest_id = tweets[-1].id
    while True:
        tweets = api.user_timeline(screen_name=user,
                                   # 200 is the maximum allowed count
                                   count=200,
                                   include_rts=False,
                                   max_id=oldest_id - 1,
                                   tweet_mode='extended'
                                   )
        if len(tweets) == 0:
            break
        oldest_id = tweets[-1].id
        all_tweets.extend(tweets)
        outtweets = [[tweet.id_str,
                      tweet.created_at,
                      tweet.favorite_count,
                      tweet.retweet_count,
                      tweet.full_text.encode("utf-8").decode("utf-8")]
                     for idx, tweet in enumerate(all_tweets)]
        df = DataFrame(outtweets, columns=["id", "created_at", "favorite_count", "retweet_count", "text"])
        df.to_csv('%s_tweets.csv' % user, index=False)
        df.head(3)

        ylabels = ["favorite_count", "retweet_count"]

        fig = plt.figure(figsize=(12,6 ))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.tight_layout()

        n_row = len(ylabels)
        n_col = 1
        color = ["tab:blue", "tab:green"]
        for count, ylabel in enumerate(ylabels):
            ax = fig.add_subplot(n_row, n_col, count + 1)
            ax.plot(df["created_at"], df[ylabel], color[count])
            ax.set_ylabel(ylabel)
        plt.savefig('static/graph.png')
        im = Image.open("static/graph.png")
        im = im.convert('RGB')
        data = io.BytesIO()
        im.save(data, "JPEG")

        # Then encode the saved image file.
        encoded_img_data = base64.b64encode(data.getvalue())

    return render_template('favorites_retweets.html', user=user, img_data=encoded_img_data.decode('utf-8'))

@app.route('/followers.html',  methods=['POST'])
def followers_screen():
    username = request.form['users'].lower()
    me = api.get_user(screen_name=username)
    user_id = me.id
    followers = []
    followers_response_list = api.get_friend_ids(user_id=user_id, count=10)

    for index, follower_id in enumerate(followers_response_list):
        follower = api.get_user(user_id=follower_id)
        followers.append(follower.screen_name)

    df = pd.DataFrame(columns=['source', 'target'])  # Empty DataFrame
    df['target'] = followers  # Set the list of followers as the target column
    print(followers)
    df['source'] = username  # Set my user ID as the source

    user_list = list(df['target'])  # Use the list of followers we extracted in the code above
    for userID in user_list:
        print(userID)
        followers = []
        # fetching the user
        user = api.get_user(screen_name=userID)
        followers_response_list = api.get_friend_ids(user_id=user.id, count=10)
        for index, follower_id in enumerate(followers_response_list):
            follower = api.get_user(user_id=follower_id)
            followers.append(follower.screen_name)

        temp = pd.DataFrame(columns=['source', 'target'])
        print("df", df)
        temp['target'] = followers
        temp['source'] = userID
        print(temp)
        df = pd.concat([df, temp])
        df.to_csv("static/networkOfFollowers.csv")

    df = pd.read_csv("static/networkOfFollowers.csv")  # Read into a df
    G = nx.from_pandas_edgelist(df, 'source', 'target')
    G.number_of_nodes()  # Find the total number of nodes in this graph
    G_sorted = pd.DataFrame(sorted(G.degree, key=lambda x: x[1], reverse=True))
    G_sorted.columns = ['nconst', 'degree']
    G_sorted.head()

    G_tmp = nx.k_core(G, 1)  # Exclude nodes with degree less than 1
    partition = community_louvain.best_partition(G_tmp)
    # Turn partition into dataframe
    partition1 = pd.DataFrame([partition]).T
    partition1 = partition1.reset_index()
    partition1.columns = ['names', 'group']
    G_sorted = pd.DataFrame(sorted(G_tmp.degree, key=lambda x: x[1], reverse=True))
    G_sorted.columns = ['names', 'degree']
    G_sorted.head()
    dc = G_sorted
    combined = pd.merge(dc, partition1, how='left', left_on="names", right_on="names")
    pos = nx.spring_layout(G_tmp)
    f, ax = plt.subplots(figsize=(12, 12))
    plt.style.use('ggplot')
    # cc = nx.betweenness_centrality(G2)
    nodes = nx.draw_networkx_nodes(G_tmp, pos,
                                   cmap=plt.cm.Set1,
                                   node_color=combined['group'],
                                   node_size = 400,
                                   alpha=0.8)
    nodes.set_edgecolor('k')
    nx.draw_networkx_labels(G_tmp, pos, font_size=12)
    nx.draw_networkx_edges(G_tmp, pos, width=1.0, alpha=0.2)
    plt.savefig("static/followers_graph.png", format="PNG")
    im = Image.open("static/followers_graph.png")
    im = im.convert('RGB')
    data = io.BytesIO()
    im.save(data, "JPEG")

    # Then encode the saved image file.
    encoded_img_data = base64.b64encode(data.getvalue())


    return render_template('followers.html', user=followers,img_data=encoded_img_data.decode('utf-8'))



if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
