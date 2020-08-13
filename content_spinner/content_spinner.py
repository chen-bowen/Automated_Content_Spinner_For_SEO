from content_spinner.models.pentagram import PentagramModel
import numpy as np
import nltk


class ContentSpinner:
    """ Samples from the language model and generate the spinned word given surrounding context """

    SPIN_PROBABILITY = 0.4

    def __init__(self):
        self.language_model = PentagramModel().pentagrams_probabilities

    def predict_word(self, context, original_word):
        """ Generate the predicted middle word from the context tuple """
        # get all possiple words and their corresponding probability distribution
        if context not in self.language_model.keys():
            return original_word

        possible_words = list(self.language_model[context].keys())
        sample_distribution = list(self.language_model[context].values())

        # sample the predicted word according to their distribution
        predicted_word = np.random.choice(a=possible_words, p=sample_distribution)
        return predicted_word

    def generate_spinned_content(self, content_list):
        """ Use the predicted word to randomly spin the given text list """
        spinned_content = []
        for content in content_list:
            # tokenize every content string
            tokenized_content = nltk.tokenize.word_tokenize(content)
            # initialize the spinned tokenized content with the first word
            spinned_tokenized_content = [tokenized_content[0]]

            for i in range(len(tokenized_content) - 4):
                # get the context words
                context = (
                    tokenized_content[i],
                    tokenized_content[i + 1],
                    tokenized_content[i + 3],
                    tokenized_content[i + 4],
                )
                # randomly spin content token if the probability is greater than spin probability
                spinned_word = (
                    self.predict_word(context, tokenized_content[i + 2])
                    if np.random.random() > self.SPIN_PROBABILITY
                    else tokenized_content[i + 2]
                )
                spinned_tokenized_content.append(spinned_word)
            # save the spinned content
            spinned_content.append(
                " ".join(spinned_tokenized_content)
                .replace(" .", ".")
                .replace(" '", "'")
                .replace(" ,", ",")
                .replace("$ ", "$")
                .replace(" !", "!")
            )
        return spinned_content

