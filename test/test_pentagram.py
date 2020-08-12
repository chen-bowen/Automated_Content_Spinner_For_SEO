import pytest
from content_spinner.models.pentagram import PentagramModel


def test_pentagram_construction(sample_text_corpus):
    """ Test construction of trigrams """
    pentagrams = PentagramModel().pentagrams_probabilities

    # assert length of the middle words at least 2
    for pentagram in pentagrams.values():
        assert len(pentagram.values()) > 1

    # assert keys of dict are tuple pairs
    for context in pentagrams:
        assert len(context) == 4
