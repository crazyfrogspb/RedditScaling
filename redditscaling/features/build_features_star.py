import argparse
import glob
import os.path as osp

import pandas as pd
from tqdm import tqdm

from redditscore.tokenizer import CrazyTokenizer


def clean_data(comments_folder, comments_per_subreddit):
    csv_files = glob.glob(osp.join(comments_folder, '*.csv'))
    df_comments = pd.concat((pd.read_csv(csv_file, lineterminator='\n', usecols=[
        'id', 'body', 'subreddit', 'created_utc']) for csv_file in csv_files))
    df_comments.drop_duplicates('id', inplace=True)
    df_comments['created_utc'] = pd.to_datetime(
        df_comments['created_utc'], unit='s')

    # removing things in sqare brackets
    df_comments.body = df_comments.body.str.replace(r"\[.*\]", ' ')
    # removing citations
    df_comments.body = df_comments.body.str.replace(r'&gt;[^\n]+\n', ' ')
    # removing line breaks
    df_comments.replace({r'\r': ' '}, regex=True, inplace=True)
    df_comments.replace({r'\r\n': ' '}, regex=True, inplace=True)
    df_comments.replace({r'\n': ' '}, regex=True, inplace=True)

    df_comments.body = df_comments.body.str.strip()
    df_comments = df_comments.loc[df_comments.body.str.len() > 0]

    df_comments = df_comments.groupby('subreddit').filter(
        lambda x: len(x) >= comments_per_subreddit)
    if df_comments.empty:
        raise ValueError(
            f'There are no subreddits with at least {comments_per_subreddit} comments')
    df_comments = df_comments.sample(frac=1.0, random_state=24)
    df_comments = df_comments.groupby('subreddit').head(comments_per_subreddit)

    return df_comments


def tokenize_data(df_comments, ignore_stopwords=True, keepcaps=False, decontract=True, remove_punct=True):
    if ignore_stopwords:
        ignore_stopwords = 'english'

    tokenizer = CrazyTokenizer(
        ignore_stopwords=ignore_stopwords,
        keepcaps=keepcaps,
        subreddits='',
        reddit_usernames='',
        emails='',
        urls='',
        decontract=decontract,
        remove_punct=remove_punct)

    tokenized = []
    for i in tqdm(range(df_comments.shape[0])):
        current_tokens = tokenizer.tokenize(
            df_comments.iloc[i, df_comments.columns.get_loc('body')])
        tokenized.append(' '.join(current_tokens))

    df_comments['body'] = tokenized

    return df_comments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenize Reddit data')

    parser.add_argument('comments_folder', type=str)
    parser.add_argument('output_file', type=str)

    parser.add_argument('--comments_per_subreddit', type=int, default=1000)

    parser.add_argument('--ignore_stopwords',
                        dest='ignore_stopwords', action='store_true')
    parser.set_defaults(ignore_stopwords=False)

    parser.add_argument('--keepcaps',
                        dest='keepcaps', action='store_true')
    parser.set_defaults(keepcaps=False)

    parser.add_argument('--decontract',
                        dest='decontract', action='store_true')
    parser.set_defaults(decontract=False)

    parser.add_argument('--remove_punct',
                        dest='remove_punct', action='store_true')
    parser.set_defaults(remove_punct=False)

    args = parser.parse_args()
    args_dict = vars(args)

    df_comments = clean_data(
        args_dict['comments_folder'], args_dict['comments_per_subreddit'])
    df_tokenized = tokenize_data(
        df_comments, args_dict['ignore_stopwords'], args_dict['keepcaps'],
        args_dict['decontract'], args_dict['remove_punct'])
    df_tokenized['subreddit'] = '__label__' + df_tokenized['subreddit']
    df_tokenized['body'].to_csv(args_dict['output_file'], index=False)
