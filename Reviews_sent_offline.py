import pandas as pd
reviews = pd.read_csv('http://data.insideairbnb.com/united-states/ca/los-angeles/2020-10-09/data/reviews.csv.gz')

#changing the date column data type to datetime and setting as index
reviews['date']= pd.to_datetime(reviews.date)

#setting date is index
reviews.set_index('date', inplace= True)

reviews_total = reviews

data_eda = pd.read_csv('airbnb_cleaned.csv')

unique_ids = data_eda.id.unique().tolist()
len(unique_ids)

reviews_total = reviews_total[reviews_total.comments!="nan"]
reviews_total = reviews_total[reviews_total.comments!="."]

reviews_total.comments =reviews_total.comments.astype(str)

reviews_total[reviews_total.listing_id.isin(unique_ids)]

#applying the function to check the language of the different reivews
from tqdm import tqdm
tqdm.pandas()

from langdetect import detect

def detect_function (text):
  try: 
    language = detect(text)
    return language
  except:
    return "error"


reviews_total['language'] = reviews_total.progress_apply(lambda row: detect_function(row['comments']), axis=1)

reviews_total = reviews_total[reviews_total.language=="en"]

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()

reviews_total['sentiment_score'] = reviews_total.progress_apply(lambda row: sentiment.polarity_scores(row['comments'])["compound"], axis=1)

reviews_total.to_csv("sentiment.csv")

sentiment_df = reviews_total.copy()


#Calculating the average sentiment score per listing id
review_id_sentiment_score = pd.DataFrame(sentiment_df.groupby('listing_id')['sentiment_score'].mean())
data_eda = data_eda.merge(review_id_sentiment_score,left_on="id", right_on="listing_id", how='left')
data_eda[['id', 'sentiment_score']].to_csv('sentiment_score_ML.csv', index_label="index")

