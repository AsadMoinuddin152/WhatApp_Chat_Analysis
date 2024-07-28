import re
import pandas as pd

def preprocess(data):

    pattern_24h = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    pattern_12h = r'\[\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}:\d{2}\u202F[APMapm]{2}\]'

    # Determine which pattern is present in the data
    if re.search(pattern_24h, data):
        pattern = pattern_24h
        date_format = '%d/%m/%y, %H:%M - '
    elif re.search(pattern_12h, data):
        pattern = pattern_12h
        date_format = '[%d/%m/%y, %I:%M:%S\u202F%p]'
    else:
        raise ValueError("Date format not recognized. Please provide data with 12-hour or 24-hour time format.")


    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    df['message_date'] = pd.to_datetime(df['message_date'], format=date_format)
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    user_messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]: 
            users.append(entry[1])
            user_messages.append(entry[2])
        else:
            users.append('group_notification')
            user_messages.append(entry[0])

    df['user'] = users
    df['message'] = user_messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df
