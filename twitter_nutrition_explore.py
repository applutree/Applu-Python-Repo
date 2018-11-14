import json
import pandas as pd
import matplotlib.pyplot as plt
import re

tweets_data_path = 'C:/Users/Skysus/Code/python/datasets/twitter_data.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")

for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue

print(tweets_data)
print(len(tweets_data))

tweets = pd.DataFrame()
tweets['text'] = list(map(lambda tweet: tweet['text'], tweets_data))
tweets['lang'] = list(map(lambda tweet: tweet['lang'], tweets_data))
tweets['country'] = list(map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data))

tweets_by_lang = tweets['lang'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=8)
ax.set_xlabel('Languages', fontsize=12)
ax.set_ylabel('Number of tweets' , fontsize=12)
ax.set_title('Top 5 languages', fontsize=12, fontweight='bold')
tweets_by_lang[:5].plot(ax=ax, kind='bar', color='red')

plt.show()

tweets_by_country = tweets['country'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=8)
ax.set_xlabel('Countries', fontsize=12)
ax.set_ylabel('Number of tweets' , fontsize=12)
ax.set_title('Top 5 countries', fontsize=12, fontweight='bold')
tweets_by_country[:5].plot(ax=ax, kind='bar', color='blue')

plt.show()

def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False

tweets['nutrition'] = tweets['text'].apply(lambda tweet: word_in_text('nutrition', tweet))
tweets['diet'] = tweets['text'].apply(lambda tweet: word_in_text('diet', tweet))

print(tweets['nutrition'].value_counts()[True])
print(tweets['diet'].value_counts()[True])

topics = ['nutrition', 'diet']
tweets_by_topics = [tweets['nutrition'].value_counts()[True], tweets['diet'].value_counts()[True]]

x_pos = list(range(len(topics)))
width = 0.8
fig, ax = plt.subplots()
plt.bar(x_pos, tweets_by_topics, width, alpha=1, color='g')

# Setting axis labels and ticks
ax.set_ylabel('Number of tweets', fontsize=15)
ax.set_title('Ranking: nutrition vs. diet (Raw data)', fontsize=10, fontweight='bold')
ax.set_xticks([p + 0.4 * width for p in x_pos])
ax.set_xticklabels(topics)
plt.grid()
plt.show()

from collections import Counter

json = tweets_data
total = [dic['text'] for dic in json]
# print(total)
# total = [cat for sublist in total for cat in sublist] # Flatten the list

Counter(total)

total = ''.join(total)
print(total)

def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

print(word_count(total))

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(total)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# lower max_font_size, change the maximum number of word and lighten the background:
stopwords = set(STOPWORDS)
stopwords.update(["https", "RT"])

wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=100, background_color="white", width=800, height=400).generate(total)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
