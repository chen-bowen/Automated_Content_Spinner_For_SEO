from bs4 import BeautifulSoup
from collections import defaultdict
import os
import sys
import numpy as np
from itertools import chain
import json
import nltk


class LoadTextCorpus:
    """ Utility class to load reviews """

    NONPRINT_TRANS_TABLE = {
        i: None for i in range(0, sys.maxunicode + 1) if not chr(i).isprintable()
    }

    def __init__(
        self,
        cached_path_reviews=os.path.join(os.path.dirname(__file__), "cache/reviews.json"),
    ):
        self._init_file_dir = os.path.dirname(__file__)
        self.categories = ["electronics", "dvd", "kitchen_&_housewares", "books"]
        self.cached_path_reviews = cached_path_reviews
        self.get_tokenized_text()

    @staticmethod
    def strip_non_printable(string):
        """ strip all the non printable characters in a string """
        return string.translate(LoadTextCorpus.NONPRINT_TRANS_TABLE)

    def load_reviews(self):
        """ Load all reviews from the data folder """

        self.reviews = defaultdict(dict)
        np.random.seed(7)
        # populate reviews dict
        for review_type in ["positive", "negative"]:
            for cat in self.categories:
                file_path = os.path.join(
                    self._init_file_dir,
                    "../../..",
                    "text_data_corpus/reviews/{}/{}.review".format(cat, review_type),
                )
                reviews_raw = BeautifulSoup(
                    open(file_path).read(), features="html.parser"
                )
                self.reviews[review_type][cat] = [
                    self.strip_non_printable(review.text)
                    for review in reviews_raw.find_all("review_text")
                ]

                # merge all categories into one
            self.reviews[review_type] = list(
                chain(*list(self.reviews[review_type].values()))
            )
            np.random.shuffle(self.reviews[review_type])

        # save tokenized reviews to cache to speedup build process
        with open(self.cached_path_reviews, "w") as fp:
            json.dump(self.reviews, fp)

    def load_wikipedia(self):
        """ Load wikipedia text from wikipedia folder """
        file_path = os.path.join(
            self._init_file_dir, "../../..", "text_data_corpus/wikipedia/"
        )
        # read text as string
        self.wikipedia_corpus = [
            open(os.path.join(file_path, f), "r").read() for f in os.listdir(file_path)
        ]

    def get_tokenized_text(self):
        """" Tokenize all reviews, preprocess the reviews using custom tokenizer """
        # load reviews and wikipedia
        self.load_reviews()
        self.load_wikipedia()

        # get tokenized reviews
        cached_path_tokenized = os.path.join(
            self._init_file_dir, "cache/text_tokenized.json"
        )

        # use cached file if exists
        if os.path.exists(cached_path_tokenized):
            with open(cached_path_tokenized, "r") as fp:
                self.text_tokenized = json.load(fp)
        else:
            self.text_tokenized = [
                nltk.tokenize.word_tokenize(i)
                for i in (self.reviews["positive"] + self.wikipedia_corpus)
            ]
        # save tokenized reviews to cache to speedup build process
        with open(cached_path_tokenized, "w") as fp:
            json.dump(self.text_tokenized, fp)
