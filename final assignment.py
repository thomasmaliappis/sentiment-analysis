import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_dataframe(file_path, remove=False):
    cols = ['sentiment', 'text']
    df = pd.read_csv(
        file_path,
        delimiter='\t',
        header=None,
        # names=cols
    )
    if remove:
        df = df.iloc[:, :-1]
    df.columns = ['id', 'sentiment', 'text']
    return df


def plot_sentiment_distribution(df, title):
    sns.countplot(x='sentiment', data=df)
    plt.title('{} set sentiment distribution'.format(title))
    plt.savefig('./png/{}_sentiment_distribution.png'.format(str.lower(title)))
    plt.show()


train_df = read_dataframe('semeval-2017-tweets_Subtask-A/downloaded/twitter-2016train-A.tsv')
plot_sentiment_distribution(train_df, title='Train')
print(train_df.shape)
test_df = read_dataframe('semeval-2017-tweets_Subtask-A/downloaded/twitter-2016test-A.tsv', remove=True)
plot_sentiment_distribution(test_df, title='Test')
print(test_df.shape)
dev_df = read_dataframe('semeval-2017-tweets_Subtask-A/downloaded/twitter-2016dev-A.tsv')
plot_sentiment_distribution(dev_df, title='Dev')
print(dev_df.shape)
devtest_df = read_dataframe('semeval-2017-tweets_Subtask-A/downloaded/twitter-2016devtest-A.tsv')
plot_sentiment_distribution(devtest_df, title='Devtest')
print(devtest_df.shape)
