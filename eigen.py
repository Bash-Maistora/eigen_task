import os
import sys
import nltk
import re
import csv
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class WordExtractor():
    """A class to extract important words from a file using NLTK Lemmatizer."""
    def __init__(self):
        super(WordExtractor, self).__init__()
        self.lem = WordNetLemmatizer()
        self.words_usage = {}
        self.counter = Counter()
        self.stop_words = set(stopwords.words('english')+['us', 'dont'])
        self.text = []


    def parse_and_extract_words(self, file):
        with open(file, 'r') as f:
            for line in f.readlines():
                words = [re.sub(r'[^A-Za-z]','',word.lower()) for word in line.split()]
                lematized = [self.lem.lemmatize(word) for word in words if word and word not in self.stop_words]
                for word in lematized:
                    if self.words_usage.get(word):
                        self.words_usage[word]['documents'].add(f.name)
                        self.words_usage[word]['sentences'].add(line)
                        self.counter[word] += 1
                    else:
                        self.words_usage[word] = {'documents': {f.name}, 'sentences': {line}}
                        self.counter[word] += 1
                self.text.append(' '.join(lematized))


    def generate_word_report(self):
        with open('word_report.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Word', 'Occurances', 'Documents', 'Sentences'])
            for entry in self.counter.most_common(50):
                docs = self.words_usage.get(entry[0]).get('documents')
                sent = self.words_usage.get(entry[0]).get('sentences')
                writer.writerow([entry[0], entry[1], '\n'.join(docs), ' '.join(sent)])


    def generate_word_cloud(self):
        wordcloud = WordCloud(background_color='white',
                              stopwords=self.stop_words,
                              max_words=50,
                              max_font_size=50,
                              random_state=42).generate(str(self.text))
        fig = plt.figure(1)
        plt.imshow(wordcloud)
        plt.axis('off')
        fig.savefig("word_cloud.png", dpi=900)


if __name__ == '__main__':
    extractor = WordExtractor()
    try:
        directory = sys.argv[1]
    except IndexError:
        directory = 'test docs/'
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                extractor.parse_and_extract_words(entry)
        extractor.generate_word_cloud()
        extractor.generate_word_report()

