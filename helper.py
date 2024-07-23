from urlextract import URLExtract
from wordcloud import WordCloud
import emoji
import pandas as pd
from collections import Counter
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import numpy as np



extractor = URLExtract()


def fetch_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    num_message = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'] == "<Media omitted>\n"].shape[0]

    link = []

    for message in df['message']:
        link.extend(extractor.find_urls(message))

    return num_message, len(words), num_media_messages, len(link)


def most_busy_users(df):
    filtered_df = df[df['user'] != 'group_notification']

    x = filtered_df['user'].value_counts().head()

    df = round((filtered_df["user"].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={
        "index": "name", "user": "percent"})
    return x, df


def create_word_cloud(selected_user, df):
    f = open("stop_hinglish.txt", "r")
    stop_words = f.read()

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])
    emoji_counts = Counter(emojis)
    emoji_df = pd.DataFrame(emoji_counts.most_common(), columns=['emoji', 'count'])
    
    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def sentiment_analysis(df):
    def get_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    df['sentiment'] = df['message'].apply(get_sentiment)
    
    def categorize_sentiment(score):
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)
    return df

def forecast_trends(df):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    daily_data = df.resample('D').size()

    X = np.arange(len(daily_data)).reshape(-1, 1)
    y = daily_data.values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_dates = pd.date_range(start=daily_data.index[-1] + pd.Timedelta(days=1), periods=30)
    future_X = np.arange(len(daily_data), len(daily_data) + 30).reshape(-1, 1)
    future_y = model.predict(future_X)
    
    forecast_df = pd.DataFrame({'date': future_dates, 'forecast': future_y})
    
    end_date = daily_data.index[-1]
    start_date = end_date - pd.DateOffset(months=6)
    historical_subset = daily_data[start_date:end_date]
    
    return historical_subset, forecast_df
