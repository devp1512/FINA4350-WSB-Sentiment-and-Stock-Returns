---
Title: Sentiment Analysis and Metric Incorporation
Date: 2022-04-30 12:00
Category: Progress Report
---

Simplicity

By: Puri Dev Lalit (Dev)

## Author Introduction

I am Dev, a Year 2 student majoring in Quantitative Finance at HKU, and I am in charge of two parts, namely (1) Analyzing post sentiment and (2) Updating it according to the overall popularity of the post.

## Initial Thoughts

I was aware of the fact that there are several Python libraries for Sentiment Analysis, such as pre-built ones like TextBlob and VADER, or trainable ones like keras. I took a look at my dataset and thought a bit about the nature of the text I was trying to analyze - Reddit posts. They would most likely be full of colloquialisms, acronyms and even emojis. It would be out of my area of competence, and take an enormous amount of time if I were to train my own text analyzer, so I was down to two options: TextBlob or VADER. After [some research](https://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf) on the pros and cons of both, I found out that TextBlob was more suitable for formal writing, while VADER works better on social media content, so I chose VADER.

## Updating the VADER Dictionary

After a quick look through the [VADER documentation](https://github.com/cjhutto/vaderSentiment#python-demo-and-code-examples), I realized that I could update the VADER sentiment dictionary to fit my needs. Since r/wallstreetbets is the birthplace of many new and original phrases, I would have to not only find an exhaustive list of these terms, but also assign a non-arbitrary sentiment score to them. Though after a couple of minutes of web surfing, my work was cut short for me. Apparently, there had been others who were also interested in analyzing this subredit's sentiment, and had already created the dictionary with a long list of terms ([1](https://github.com/mdominguez2010/wsb-sentiment-analysis/blob/main/stocks_to_trade.py), [2](), [3]()). I took inspiration the sentiment scores from the three sources and created my own dictionary which is shown below.

```python
wsb_lingo = {'citron': -4.0, 'hidenburg': -4.0, 'moon': 4.0, 'highs': 2.0,
             'mooning': 4.0, 'long': 2.0, 'short': -2.0, 'call': 4.0, 'calls': 4.0,
             'put': -4.0, 'puts': -4.0, 'break': 2.0, 'tendie': 2.0, 'tendies': 2.0,
             'town': 2.0, 'overvalued': -3.0, 'undervalued': 3.0, 'buy': 4.0, "hold": 1.0,
             'sell': -4.0, 'gone': -1.0, 'gtfo': -1.7, 'paper': -1.7, 'bullish': 3.7,
             'bearish': -3.7, 'bagholder': -1.7, 'stonk': 1.9, 'green': 1.9, 'money': 1.2,
             'print': 2.2, 'bull': 2.9, 'bear': -2.9, 'pumping': -1.0, 'sus': -3.0,
             'offering': -2.3, 'rip': -4.0, 'downgrade': -3.0, 'upgrade': 3.0,
             'maintain': 1.0, 'pump': 1.9, 'hot': 1.5, 'drop': -2.5, 'rebound': 1.5,
             'crack': 2.5, "BTFD": 4.0, "FD": 4.0, "diamond hands": 0.0, "paper hands": 0.0,
             "DD": 4.0, "GUH": -4.0, "pump": 4.0, "dump": -4.0, "gem stone": 4.0, "rocket": 4.0,
             "andromeda": 0.0, "to the moon": 4.0}
```

It is worth noting that sentiment scores are given on a scale from -4 (most negative) to 4 (most positive). Also, another interesting thing about VADER is that emojis in text are automatically converted into text before analysis. To this end, it was only necessary for me to give a description of the emojis for VADER to be able to pick up on its sentiment. This can be seen above with "rocket" and "gem stone".

The actual code to initialize the VADER Sentiment Analyzer and Dictionary Updating is given below.

```python
# Import the necessary libraries
import pandas as pd 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Update the VADER Sentiment Analyzer with the above terms
sia = SentimentIntensityAnalyzer()
sia.lexicon.update(wsb_lingo)

# Import the cleaned dataframe from the previous step
cleaned_dataframe = pd.read_csv("reddit_with_ticker.csv", lineterminator = "\n") 

# Run the post content through VADER and store the output
sentiment_list = []
for content in cleaned_dataframe["all_content"]:
    sentiment_list.append([sia.polarity_scores(content)["neg"], sia.polarity_scores(content)["neu"], 
                      sia.polarity_scores(content)["pos"], sia.polarity_scores(content)["compound"]])
sentiment = pd.DataFrame(sentiment_list, columns = ["Sell Signal", "Hold Signal", "Buy Signal", "Compound Signal"])


# Combine the results with the original dataframe
df_concat = pd.concat([cleaned_dataframe, sentiment], axis=1)

# Save the new dataframe as a csv file
df_concat.to_csv('reddit_with_ticker_with_sentiment.csv')
```
