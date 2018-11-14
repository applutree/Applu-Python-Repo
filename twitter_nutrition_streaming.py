#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
access_token = "825551586193711104-yuTO5TrA2ZahkWRuQZky1UsxhjEdeXT"
access_token_secret = "H6QqFhnqYQgYYKL9qOpc6J1ovAPIwtO7KVImjFJGJ4xMH"
consumer_key = "gnl270YQVTHXphgw0eFdtG2rQ"
consumer_secret = "WRXC2OVa9qkXpAKEdiPxw26HDgXyhwdk9FyS6QhVdx36KUThLQ"


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'nutrition' and 'diet'
    stream.filter(track=['nutrition', 'diet'])