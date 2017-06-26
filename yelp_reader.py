"""
Utilities for parsing Yelp review files.

Originally 3G json file; breakup -->  gsplit -C 30m -a 2 -d yelp_reviews.json "file_"

TODO --> TF Records queue for managing files, batching, etc.

Test file for prototyping/ debugging --> assets/tiny.json

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import re
from nltk import tokenize
from keras.utils import np_utils
from collections import Counter
import pdb

SPLIT_RE = re.compile(r'(\W+)?')
PAD_TOKEN = '_PAD'
UNK_TOKEN = '_UNK'
PAD_ID = 0
UNK_ID = 1
max_sent_len = 30
max_doc_len = 5
max_vocab = 10000
n_labels = 5

def prepare_data(data_path, tiny=False):
    text, stars = get_text_and_stars(data_path, tiny=tiny)
    reviews = []
    for review in text:
        sents = tokenize.sent_tokenize(review)
        reviews.append([tokenize_sent(sent) for sent in sents])

    reviews = truncate_docs(reviews, max_sent_len, max_doc_len)

    vocab, token_to_id = get_vocab_ids(reviews)

    reviews_pad = pad_reviews(reviews, max_sent_len, max_doc_len)

    reivew_ids = tokenize_reviews(reviews, token_to_id)

    stars = np_utils.to_categorical([star-1 for star in stars], n_labels)

    return reivew_ids, stars, token_to_id


def get_text_and_stars(data_path, tiny=False):
    if tiny:
        data_path = "./assets/tiny.json"
    content = []
    with open(data_path, 'r') as f:
        for line in f:
            content.append(json.loads(line))
    return zip(*[[x['text'], x['stars']] for x in content])

def tokenize_reviews(reviews, token_to_id):
    review_ids = []
    for review in reviews:
        review = [[token_to_id.get(token, UNK_ID) for token in sentence] \
                    for sentence in review]
        review_ids.append(review)
    return review_ids

def get_vocab_ids(reviews):
    tokens_all = []
    for review in reviews:
        tokens_all.extend([token for sentence in review for token in sentence])
    counter = Counter(tokens_all).most_common(max_vocab-2)
    vocab = [PAD_TOKEN, UNK_TOKEN] + [x[0] for x in counter]
    token_to_id = {token: i for i, token in enumerate(vocab)}
    return vocab, token_to_id

def pad_reviews(reviews, max_sent_len, max_doc_len):
    for review in reviews:
        for sent in review:
            for _ in range(max_sent_len - len(sent)):
                sent.append(PAD_TOKEN)
            assert len(sent) == max_sent_len

        for _ in range(max_doc_len - len(review)):
            review.append([PAD_TOKEN for _ in range(max_sent_len)])
        assert len(review) == max_doc_len
    return reviews

def truncate_docs(reviews, max_sent_len, max_doc_len):
    new_docs = []
    for review in reviews:
        new_review = []
        for sent in review[-max_doc_len:]:
            new_review.append(sent[-max_sent_len:])
        new_docs.append(new_review)
    return new_docs

def tokenize_sent(sent):
    "Split on non-word characters and strip whitespace."
    return [token.strip().lower() for token in re.split(SPLIT_RE, sent) \
            if token.strip()]

def prepare_for_visual(document, token_to_id):
    doc_sents_tokens = [tokenize_sent(sent) for sent in tokenize.sent_tokenize(document)]
    docs_truncated = truncate_docs([doc_sents_tokens], max_sent_len, max_doc_len)
    doc_padded = pad_reviews(docs_truncated, max_sent_len, max_doc_len)
    doc_ids = tokenize_reviews(doc_padded, token_to_id)
    return doc_ids
