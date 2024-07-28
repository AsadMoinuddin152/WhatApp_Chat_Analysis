import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessor, helper
import pandas as pd

def generate_gradient_colors(num_colors):
    cmap = plt.get_cmap('viridis')
    gradient_colors = [cmap(i / num_colors) for i in range(num_colors)]
    return gradient_colors


st.sidebar.title("Whatsapp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    df = helper.sentiment_analysis(df)



    user_list = df["user"].unique().tolist()
    if "group_notification" in user_list:
        user_list.remove("group_notification")
    user_list.sort()
    user_list.insert(0, "Overall")


    selected_user = st.sidebar.selectbox("Show analysis WRT", user_list)

    if st.sidebar.button("Show Analysis"):
        
        num_messages, words, num_media_message, num_link_message = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.subheader(num_messages)

        with col2:
            st.header("Message Count")
            st.subheader(words)

        with col3:
            st.header("Media Shared")
            st.subheader(num_media_message)

        with col4:
            st.header("Link Shared")
            st.subheader(num_link_message)

        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)


        if selected_user == "Overall":
            st.title("Most Busy Users")
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            gradient_colors = generate_gradient_colors(len(x))

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x, color=gradient_colors)
                plt.xticks(rotation="vertical")
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)


    st.title("Word Cloud")
    df_wc = helper.create_word_cloud(selected_user, df)
    fig, ax = plt.subplots()
    plt.imshow(df_wc)
    st.pyplot(fig)

    most_common_df = helper.most_common_words(selected_user, df)
    fig, ax = plt.subplots()

    ax.barh(most_common_df[0], most_common_df[1])
    plt.xticks(rotation="vertical")

    st.title("Most Common Words")
    st.pyplot(fig)

    # emoji analysis
    emoji_df = helper.emoji_helper(selected_user, df)
    st.title("Emoji Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(emoji_df)
    with col2:
        fig, ax = plt.subplots()
        ax.pie(emoji_df['count'].head(), labels=emoji_df['emoji'].head(), autopct="%0.2f")
        st.pyplot(fig)


    st.title("Sentiment Analysis")
    sentiment_counts = df['sentiment_category'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values, color='blue')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.title('Sentiment Distribution')
    st.pyplot(fig)


    st.title("Message Volume Forecast")
    historical_data, forecast_df = helper.forecast_trends(df)

    
    all_dates = historical_data.index.union(forecast_df['date'])
    combined_data = pd.DataFrame(index=all_dates)
    combined_data['Historical'] = historical_data
    combined_data['Forecast'] = forecast_df.set_index('date')['forecast']
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        combined_data['Historical'].plot(ax=ax, label='Historical Data', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Message Count')
        plt.title('Historical Message Volume')
        plt.legend()
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        combined_data['Forecast'].plot(ax=ax, label='Forecast', color='red')
        plt.xlabel('Date')
        plt.ylabel('Message Count')
        plt.title('Message Volume Forecast')
        plt.legend()
        plt.xticks(rotation='vertical')
        st.pyplot(fig)