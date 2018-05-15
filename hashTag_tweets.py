from twython import Twython

CONSUMER_KEY = 'nxfgPXMhpWPdxEXtE32KZe86I'
CONSUMER_SECRET = '5cQ3mvaBcTSfCcCyBiDFP5z19xYo3PXECnpew5UQxotGjDeWTm'
OAUTH_TOKEN = '761622332670640128-ryYSls34csVFsWsJX48LrCPaJYWWRkW'
OAUTH_TOKEN_SECRET = 'ho6vJy088rTs2slJVAUV43WAhrkr5ZjhrXz5kpdH8b7LM'

twitter = Twython(app_key=CONSUMER_KEY, app_secret=CONSUMER_SECRET, oauth_token=OAUTH_TOKEN, oauth_token_secret=OAUTH_TOKEN_SECRET)

search_results = twitter.search(q="#NaMO", count=2000)
print(search_results['statuses'][0])
with open('out.txt','a') as f:
    count=0
    for tweet in search_results['statuses']:
        count+=1
        f.write("%s %s\n" % (str(count), tweet['text']))
f.close()
