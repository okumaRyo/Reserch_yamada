import time
import tweepy
from pprint import pprint

# API情報を記入
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAOZSgQEAAAAAYjg1dRXaWz7wPCecKgHz5XVMIdc%3DVhJUsStjEGoaLBB14iZkQdWFgueHCv9pc2PoLlODRbH58IhX9X"
API_KEY = "kZ5yVh9zBId4leMIfwRI16IoQ"
API_SECRET = "NormHn62txYFCv4jocefWCvGzmHrRkkqMRWn20Og2VrbzmBhIi"
ACCESS_TOKEN = "em1ralItdGxJSDhpbS1BbWJKNEs6MTpjaQ"
ACCESS_TOKEN_SECRET = "bOS1JYY95zEXBqKB5-7lBk-33cn8lWhex6rxCwK-Bo4tic4Oh7"


# クライアント関数を作成
def ClientInfo():
    client = tweepy.Client(bearer_token=BEARER_TOKEN,
                           consumer_key=API_KEY,
                           consumer_secret=API_SECRET,
                           access_token=ACCESS_TOKEN,
                           access_token_secret=ACCESS_TOKEN_SECRET,
                           )

    return client


time_sta = time.time()
# ★必要情報入力
search = "ハフィーニャ"  # 検索対象
tweet_max = 10           # 取得したいツイート数(10〜100で設定可能)

# 関数


def SearchTweets(search, tweet_max):
    # 直近のツイート取得
    tweets = ClientInfo().search_recent_tweets(query=search, max_results=tweet_max)

    # 取得したデータ加工
    results = []
    tweets_data = tweets.data

    # tweet検索結果取得
    if tweets_data is not None:
        for tweet in tweets_data:
            obj = {}
            obj["tweet_id"] = tweet.id      # Tweet_ID
            obj["tweet_date"] = tweet.created_at
            obj["text"] = tweet.text  # Tweet Content
            results.append(obj)
    else:
        results.append('')

    # 結果出力
    return results


time_end = time.time()

tim = time_end - time_sta
# 関数実行・出力
pprint(SearchTweets(search, tweet_max))
print(f'処理時間：{tim}s')
