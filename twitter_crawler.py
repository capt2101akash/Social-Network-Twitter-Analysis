from __future__ import print_function
import getopt
import logging
import os
import sys
import csv
import tweepy


Logger = None
CONSUMER_KEY = 'nxfgPXMhpWPdxEXtE32KZe86I'
CONSUMER_SECRET = '5cQ3mvaBcTSfCcCyBiDFP5z19xYo3PXECnpew5UQxotGjDeWTm'
OAUTH_TOKEN = '761622332670640128-ryYSls34csVFsWsJX48LrCPaJYWWRkW'
OAUTH_TOKEN_SECRET = 'ho6vJy088rTs2slJVAUV43WAhrkr5ZjhrXz5kpdH8b7LM'
BASE_DIR=''

def get_tweet_id(line):
    '''
    Extracts and returns tweet ID from a line in the input.
    '''
    tweet_id = line.split('\n')
    return tweet_id[0]

def get_tweets_single(twapi, idfilepath):
    '''
    Fetches content for tweet IDs in a file one at a time,
    which means a ton of HTTPS requests, so NOT recommended.

    `twapi`: Initialized, authorized API object from Tweepy
    `idfilepath`: Path to file containing IDs
    '''
    with open(idfilepath, 'rt') as idfile:
        tweet_output = open(os.path.join(BASE_DIR, 'crawled_tweets.csv'), 'w')
        writer = csv.writer(tweet_output)
        for line in idfile:
            tweet_id = get_tweet_id(line)
            Logger.debug('Fetching tweet for ID %s', tweet_id)
            try:
                tweet = twapi.get_status(tweet_id)
                # writer.writerrow(tweet.text+)
                writer.writerow([tweet.id, tweet.text])
                # tweet_output.write(tweet.text+','+str(tweet.id)+'\n')
            #print('%s,%s' % (tweet_id, tweet.text.encode('UTF-8')))
            except tweepy.TweepError as te:
                Logger.warn('Failed to get tweet ID %s: %s', tweet_id, te.args)
        tweet_output.close()
def get_tweet_list(twapi, idlist):
    '''
    Invokes bulk lookup method.
    Raises an exception if rate limit is exceeded.
    '''

    tweets = twapi.statuses_lookup(id_=idlist, include_entities=False, trim_user=True)
    tweet_output = open(os.path.join(BASE_DIR, 'crawled_tweets.csv'), 'a')
    writer = csv.writer(tweet_output)
    for tweet in tweets:
        writer.writerow([tweet.id, tweet.text])
        print('%s, %s' % (tweet.id, tweet.text))
    tweet_output.close()
def get_tweets_bulk(twapi, idfilepath):
    '''
    Fetches content for tweet IDs in a file using bulk request method,
    which vastly reduces number of HTTPS requests compared to above;
    however, it does not warn about IDs that yield no tweet.

    `twapi`: Initialized, authorized API object from Tweepy
    `idfilepath`: Path to file containing IDs
    '''

    tweet_ids = list()
    with open(idfilepath, 'rt') as idfile:

        for line in idfile:
            tweet_id = get_tweet_id(line)
            Logger.debug('Fetching tweet for ID %s', tweet_id)
            # API limits batch size to 100
            if len(tweet_ids) < 100:
                tweet_ids.append(tweet_id)
            else:
                get_tweet_list(twapi, tweet_ids)
                tweet_ids = list()

    if len(tweet_ids) > 0:
        get_tweet_list(twapi, tweet_ids)

def main():
    logging.basicConfig(level=logging.WARN)
    global Logger
    Logger = logging.getLogger('get_tweets_by_id')
    bulk = True
    idfile = 'tweet_ids.txt'
    if not os.path.isfile(idfile):
        print('Not found or not a file: %s' % idfile, file=sys.stderr)
        usage()

    # connect to twitter
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)

    # hydrate tweet IDs
    if bulk:
        get_tweets_bulk(api, idfile)
    else:
        get_tweets_single(api, idfile)

main()
