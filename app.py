from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
import geocoder
import textblob
from textblob import TextBlob
import matplotlib.pyplot as plt

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
api = tweepy.API(auth) 


app = Flask(__name__)


@app.route('/home.html')
def home1():
    public_tweets = api.home_timeline()
    for tweet in public_tweets:
        print(tweet.text, tweet.created_at)
    return render_template('home.html', final=public_tweets[0].text,response=public_tweets)

@app.route('/form.html')
def my_form():
    return render_template('form.html')

@app.route('/form2.html')
def my_form2():
    global status
    return render_template('form2.html')

@app.route('/formg.html')
def my_formg():
    return render_template('formg.html')

@app.route('/form3.html')
def my_form3():
    return render_template('form3.html')


@app.route('/form4.html')
def my_form4():
    return render_template('form4.html')


@app.route('/form5.html')
def my_form5():
    return render_template('form5.html')




@app.route('/form.html', methods=['POST'])
def my_form_post():
    global status
    text1 = request.form['text1'].lower()
    available_loc = api.available_trends()
    screen_name = text1

    count = 3

    statuses = api.user_timeline(screen_name=screen_name, count=count)
      
    # printing the statuses
    for status in statuses:
        print(status.text, end = "\n\n")
    return render_template('form.html', final=10, text1=text1,text2=status.text,text5=status.text,text4=status.text,text3=status.text)


@app.route('/form5.html', methods=['POST'])
def my_form5_post():
    text1 = request.form['twitter_username'].lower()
    details = api.get_user(screen_name=text1)
    return render_template('form5.html', final=1, user_details=details)



@app.route('/form2.html', methods=['POST'])
def my_form2_post():
    
    text1 = request.form['twitter_hashtag'].lower()
    l=[]
    for tweet in tweepy.Cursor(api.search_tweets,q=text1,count=10,
                           lang="en").items(50):
        if (not tweet.retweeted) and ('RT @' not in tweet.text):
            l.append(tweet.text)

    return render_template('form2.html', final=1, text=l, hashtag=text1)


@app.route('/form3.html', methods=['POST'])
def my_form3_post():
    text1 = request.form['location'].lower()  # location as argument variable

    # Trends for Specific Country
    g = geocoder.osm(text1)  # getting object that has location's latitude and longitude

    closest_loc = api.closest_trends(g.lat, g.lng)
    # fetching the trends
    trends = api.get_place_trends(closest_loc[0]['woeid'], lang="en")

    loc = []

    for value in trends:
        for trend in value['trends']:
            loc.append(trend['name'])

    return render_template('form3.html', final=1, loc=text1,loc_tweets=loc)

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
    # create TextBlob object of passed tweet text
    analysis = TextBlob(clean_tweet(tweet))
    # set sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


@app.route('/form4.html', methods=['POST'])
def my_form4_post():
    # convert to lowercase
    text1 = request.form['keyword'].lower()

    tweets = get_tweets(query=text1, count=200)
    print(len(tweets))
    ptweets = [tweet for tweet in tweets if get_tweet_sentiment(tweet) == 'positive']
    p_percentage = 100 * len(ptweets) / len(tweets)
    # percentage of positive tweets
    print("Positive tweets percentage: {} %".format(100 * len(ptweets) / len(tweets)))
    # picking negative tweets from tweets
    ntweets = [tweet for tweet in tweets if get_tweet_sentiment(tweet) == 'negative']
    # percentage of negative tweets
    n_percentage = 100 * len(ntweets) / len(tweets)
    print("Negative tweets percentage: {} %".format(100 * len(ntweets) / len(tweets)))
    # percentage of neutral tweets
    neu_percentage = 100 * (len(tweets) - (len(ntweets) + len(ptweets))) / len(tweets)
    print("Neutral tweets percentage: {} % \
        ".format(100 * (len(tweets) - (len(ntweets) + len(ptweets))) / len(tweets)))

    # printing first 5 positive tweets
    print("\n\nPositive tweets:")
    for tweet in ptweets[:10]:
        print(tweet)

    # printing first 5 negative tweets
    print("\n\nNegative tweets:")
    for tweet in ntweets[:10]:
        print(tweet)

    return render_template('form4.html', final=1, keyword=text1,pos_tweets=ptweets,neg_tweets=ntweets, pos_per=p_percentage, neg_per=n_percentage, neu_per=neu_percentage)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
