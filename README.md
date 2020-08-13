# Automated Content Spinner With NLP

Using pentagram as the language model to automatically spin content for search engine optimizations

## Environment Setup

1. Clone the repo locally
2. In your terminal, navigate to the directory where you have just cloned this repo
3. Type `pip install poetry`. This will install the package management system poetry which will help you install all the required dependencies
4. Type `poetry install`
5. Type `pip install jupyter`. Now you should be able to view the notebook that generated the above analysis

## Data Download

The raw text data is being used to train a pentagram model, which are from the following two sources

1. [Product Reviews from John Hopkins University Reasearch Lab](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)
2. [Preprocessed Wikipedia Articles Sample](https://lazyprogrammer.me/course_files/enwiki-preprocessed.zip)

## Component Explanations

The main component in this project contains the `LoadTextCorpus`, `PentagramModel` and `ContentSpinner` modules.

### LoadTextCorpus

The class that reads in and tokenizes the reviews and wikipedia text data. The class also saves the cached files in the cached folder, as tokenizing the large text corpus is relatively time consuming. The `LoadTextCorpus` class is automatically used in `PentagramModel` class. 

Note: tokenizing all the texts takes approximately 2 hours to complete.

### PentagramModel

The class uses the tokenized reviews and produced the probabilities of each individual single word according to the surrounding 4 words - P(word(i) | word(i-2), word(i-1), word(i+1), word(i+2)). The results are saved in a dictionary with the surrounding 4 words as keys, the probabilities of the middle word as another dictionary as values.

```
pentagram_probabilities = {("word 1", "word 2", "word3", "word 4"):{"word 5": p1, "word 6": p2, ...}, ...}
```
