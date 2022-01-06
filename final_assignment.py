import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_dataframe(file_path, remove=False):
    # cols = ['sentiment', 'text']
    df = pd.read_csv(
        file_path,
        delimiter='\t',
        header=None,
        # names=cols
    )
    if remove:
        df = df.iloc[:, :-1]
    df.columns = ['id', 'label', 'text']
    # df.set_index('id', inplace=True)
    return df


def plot_sentiment_distribution(df, title):
    sns.countplot(x='label', data=df)
    plt.title('{} set sentiment distribution'.format(title))
    plt.savefig('./png/{}_sentiment_distribution.png'.format(str.lower(title)))
    plt.show()


train13_df = read_dataframe('semeval-2017-tweets_Subtask-A/downloaded/twitter-2013train-A.tsv')
train15_df = read_dataframe('semeval-2017-tweets_Subtask-A/downloaded/twitter-2015train-A.tsv')
train16_df = read_dataframe('semeval-2017-tweets_Subtask-A/downloaded/twitter-2016train-A.tsv')
train_df = pd.concat([train13_df, train15_df, train16_df]).drop_duplicates()
plot_sentiment_distribution(train_df, title='Train')
print(train_df.shape)
train_df['label'].replace({"negative": 0, "neutral": 1, "positive": 2}, inplace=True)
test13_df = read_dataframe('semeval-2017-tweets_Subtask-A/downloaded/twitter-2013test-A.tsv')
test15_df = read_dataframe('semeval-2017-tweets_Subtask-A/downloaded/twitter-2015test-A.tsv')
test16_df = read_dataframe('semeval-2017-tweets_Subtask-A/downloaded/twitter-2016test-A.tsv', remove=True)
test_df = pd.concat([test13_df, test15_df, test16_df]).drop_duplicates()
plot_sentiment_distribution(test_df, title='Test')
print(test_df.shape)
test_df['label'].replace({"negative": 0, "neutral": 1, "positive": 2}, inplace=True)
dev_df = read_dataframe('semeval-2017-tweets_Subtask-A/downloaded/twitter-2016dev-A.tsv')
plot_sentiment_distribution(dev_df, title='Dev')
print(dev_df.shape)
devtest_df = read_dataframe('semeval-2017-tweets_Subtask-A/downloaded/twitter-2016devtest-A.tsv')
plot_sentiment_distribution(devtest_df, title='Devtest')
print(devtest_df.shape)

train_df['lenght'] = train_df.apply(lambda row: len(row['text']), axis=1)
pass
