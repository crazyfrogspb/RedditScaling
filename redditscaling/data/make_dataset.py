import argparse
import glob
import json
import os
import os.path as osp
from collections import defaultdict

from dotenv import find_dotenv, load_dotenv

from reddiscaling.config import config
from redditscore import get_reddit_data as grd

load_dotenv(find_dotenv())

PROJECT_ID = os.environ.get('PROJECT_ID')

CURRENT_PATH = osp.dirname(os.path.realpath(__file__))
PRIVATE_KEY = osp.join(config.data_dir, 'reddit_google_key.json')


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def BFS(self, s, limit=5000):
        visited = {}
        subreddits = []
        for key in self.graph:
            visited[key] = False
        queue = []
        queue.append(s)
        visited[s] = True
        while queue and len(subreddits) < limit:
            s = queue.pop(0)
            subreddits.append(s)
            for i in self.graph[s]:
                if i in visited and visited[i] == False:
                    queue.append(i)
                    visited[i] = True
        return subreddits


def collect_data(save_path, start_month, end_month, subreddit_number):
    sayit_dir = osp.join(config.data_dir, 'external', 'sayit-data', '1')
    jsons = glob.glob(osp.join(sayit_dir, '*.json'))

    substitutes_dict = {}
    with open(jsons[-4], 'r') as fin:
        substitutes = json.load(fin)
    for sub in substitutes:
        substitutes_dict[sub[0]] = sub[1:]

    similarity_graph = Graph()
    for file in jsons:
        if osp.basename(file) in ['count_low_threshold.json', 'substitutes.json', 'count.json']:
            continue
        with open(file, 'r') as fin:
            sim_list = json.load(fin)
        for subreddit_list in sim_list:
            node = subreddit_list[0]
            if node in substitutes_dict:
                subreddit_list = substitutes_dict[node]
            else:
                subreddit_list = subreddit_list[1:]
            for subreddit in subreddit_list:
                similarity_graph.addEdge(node, subreddit)

    subreddits = similarity_graph.BFS('politics', subreddit_number)
    grd.get_comments((start_month, end_month),
                     PROJECT_ID,
                     PRIVATE_KEY,
                     subreddits=subreddits,
                     comments_per_month=1000,
                     top_scores=True,
                     csv_directory=osp.join(config.data_dir, 'raw',
                                            'reddit_comments_new'),
                     verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect Reddit data')

    parser.add_argument('save_path', type=str)
    parser.add_argument('start_month', type=str)
    parser.add_argument('end_month', type=str)
    parser.add_argument('--subreddit_number', type=int, default=5000)

    args = parser.parse_args()
    args_dict = vars(args)

    collect_data(**args_dict)
