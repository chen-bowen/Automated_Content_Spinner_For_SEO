import pandas as pd
import numpy as np
import os, pickle
from content_spinner.data.load_text_corpus import LoadTextCorpus
from collections import Counter, defaultdict


class PentagramModel:
    """ Builds the trigram mapping from tokenized reviews """

    def __init__(self):
        self._init_file_dir = os.path.dirname(__file__)
        self.build()

    @property
    def text_corpus(self):
        """ Get tokenized reviews """
        return LoadTextCorpus().text_tokenized

    def get_pentagrams(self):
        """ Build pentagrams on only the positive reviews """
        pentagrams = defaultdict(list)
        for text in self.text_corpus:
            for i in range(len(text) - 4):
                # extract pentagrams from the text corpus
                trigram_context = (text[i], text[i + 1], text[i + 3], text[i + 4])
                pentagrams[trigram_context].append(text[i + 2])
        return pentagrams

    def build(self):
        """ Build trigram probabilities as an attribute """
        # use cached pentagrams to speed up process
        model_path = os.path.join(
            self._init_file_dir, "model_file/pentagram_model.pickle"
        )

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.pentagrams_probabilities = pickle.load(f)
        else:
            # get pentagrams
            self.pentagrams = self.get_pentagrams()
            # get pentagrams frequency count
            self.pentagrams_frequency_count = {
                context: Counter(word_list)
                for context, word_list in self.pentagrams.items()
            }
            # get pentagrams probabilities
            self.pentagrams_probabilities = {
                context: {
                    word: count / sum(words_count.values())
                    for word, count in words_count.items()
                }
                for context, words_count in self.pentagrams_frequency_count.items()
                if len(words_count.values()) > 1
            }

            # save the pentagram to speedup the process
            with open(model_path, "wb") as f:
                pickle.dump(
                    self.pentagrams_probabilities, f, protocol=pickle.HIGHEST_PROTOCOL
                )

