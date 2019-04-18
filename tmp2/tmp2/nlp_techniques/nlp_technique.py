import pickle
from nltk import word_tokenize, pos_tag, WordNetLemmatizer, FreqDist, ngrams
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd


class NLPTechnique:
    """
    Represents a NLP Technique and states that
    are maintained by the instance

    Instance Members:
    - nlp_vader: boolean to indicate if the vader sentiment analyser is used to
        calculate the numerical expression of the text data
    - nlp_bow: boolean to indicate if bag of words is used to
        calculate the numerical expression of the text data
    - pp_pos: boolean to indicate whether to use part of speech in the pre-processing stage
    - pos_stop_words: list of words used in part of speech
    - pp_lemma: boolean to indicate whether to use lemmalisation in the pre-processing stage
    """

    def __init__(self, name,
                 nlp_vader,
                 nlp_bow,
                 pp_pos=False,
                 pos_stop_words=['a', 'an', 'the', 'that', 'this', 'is', 'are', 'and'],
                 pp_lemma=False,
                 pickle_file=''):
        self.name = name
        print('Init nlp ' + name + ' instance')
        check = 0
        if nlp_vader:
            check += 1
        if nlp_bow:
            check += 1
        # if check != 1:
        #     raise ValueError
        self.nlp_vader = nlp_vader
        self.nlp_bow = nlp_bow
        self.pp_pos = pp_pos
        self.pos_stop_words = pos_stop_words
        self.pp_lemma = pp_lemma
        self.pickle_file = pickle_file
        self.lemmatizer = WordNetLemmatizer()
        self.word_net_tags = ['a', 'r', 'n', 'v']  # a = adjective, r= adverb , n=noun, v=verb
        self.bow_word_counts = None
        self.processed_data = None

    def process_data(self, data, forced=False, predict=False):
        """
        Using the instance configuration to process the given data
        :param data: the data to be processed
        :type data: pandas.DataFrame
        :param forced: whether to return the cached data
        :type forced: bool
        :return: the processed data
        """
        if not forced:
            if self.processed_data is not None:
                print("Return NLP processed data: Part-of-speech: " + str(self.pp_pos) + ", Lemma: " + str(self.pp_lemma)
                      + ", Bag: " + str(self.nlp_bow) + ", Vader: " + str(self.nlp_vader))
                return self.processed_data

        print('instance ' + self.name + ' now processing data')

        # Local methods for pre-processing
        def lower_cm_func(row):
            comment = row["comments"]
            return str(comment).lower()

        # def intify_region(row):
        #     region = row["region"]
        #     return int(region)

        def standardise(df):
            df['comments'] = df.apply(lower_cm_func, axis='columns')
            # df['region'] = df.apply(intify_region, axis='columns')
            # Convert delivery_ok and on_time_in_full features into binary
            if 'delivery_ok' in df.keys():
                df['delivery_ok'] = df['delivery_ok'].map({'yes': 1, 'no': 0})
            if 'on_time_in_full' in df.keys():
                df['on_time_in_full'] = df['on_time_in_full'].map({'yes': 1, 'no': 0})
            if 'deliveryday' in df.keys():
                df['deliveryday'] = df['deliveryday'].str.lower()
                df['deliveryday'] = df['deliveryday'].map({'monday': 1,
                                                           'tuesday': 2,
                                                           'wednesday': 3,
                                                           'thursday': 4,
                                                           'friday': 5,
                                                           'saturday': 6,
                                                           'sunday': 7})
            df = df.fillna(value=0)
            # df = df.fillna(method='ffill')
            return df

        def pos_func(row):
            comment = row["comments"]
            # Tokenize the comment
            tokens = word_tokenize(comment)

            # filter out articles, etc
            filtered_sentence = [w for w in tokens if w not in self.pos_stop_words]
            # concat the remaining tokens
            comment = ' '.join(filtered_sentence)
            return comment

        def lemma_func(row):
            comment = row["comments"]
            # Tokenize the comment
            tokens = word_tokenize(comment)
            # attaches a part of speech tag to each word
            tokens_pos = pos_tag(tokens)
            comment = ''

            for word, tag in tokens_pos:
                tag = tag[0].lower()
                if tag in self.word_net_tags:
                    lemma = self.lemmatizer.lemmatize(word, tag)
                else:
                    lemma = word
                comment += lemma + ' '
            return comment

        # Pre-processing begins
        print("instance pre-processing stage")
        data = standardise(data)
        if self.pp_pos:
            print("part of speech running")
            data['comments'] = data.apply(pos_func, axis='columns')

        if self.pp_lemma:
            print("lemmalisation running")
            data['comments'] = data.apply(lemma_func, axis='columns')
        print("pre-processing completed")
        # Pre-processing ends

        # NLP begins
        if self.nlp_vader:
            vader = SentimentIntensityAnalyzer()
            print("instance nlp-vader stage")

            # Local methods for nlp
            def score_comment(row):
                comment = str(row["comments"])
                scores = vader.polarity_scores(comment)
                row['neg'] = scores['neg']
                row['neu'] = scores['neu']
                row['pos'] = scores['pos']
                row['compound'] = scores['compound']
                return row[['neg', 'neu', 'pos', 'compound']]

            data[['neg', 'neu', 'pos', 'compound']] \
                = data.apply(score_comment, axis='columns')

        if self.nlp_bow:
            print("instance nlp-bags_of_word stage")
            MIN_OCCURRENCE = 1
            NGRAM = 1

            # constructing word set and count
            data['bag_vector'] = 0
            if self.bow_word_counts is None:
                word_counts = pd.DataFrame(columns=['count'])
                for comment in data['comments']:
                    count = FreqDist(map(' '.join, ngrams(word_tokenize(str(comment)), NGRAM)))
                    for word in count:
                        if word in set(word_counts.index):
                            word_counts['count'][word] += count[word]
                        else:
                            word_counts.loc[word] = count[word]
                
                # filter by MIN_OCCURRENCE and sort alphabetically
                word_counts.drop(word_counts[word_counts['count'] < MIN_OCCURRENCE].index, inplace=True)
                word_counts.sort_index(inplace=True)

                if not predict:
                    self.bow_word_counts = word_counts

            data['bag_vector'] = data['bag_vector'].astype(list)
            
            def bag_comment(row):
                word_counts = self.bow_word_counts
                comment = str(row['comments'])
                vector = [0] * len(set(word_counts.index))
                ngram_split = map(' '.join, ngrams(word_tokenize(str(comment)), NGRAM))
                for word in ngram_split:
                    if word in set(word_counts.index):
                        vector[word_counts.index.get_loc(word)] += 1
                row['bag_vector'] = vector
                return row[['bag_vector']]
            
            data[['bag_vector']] = data.apply(bag_comment, axis=1)

        print('nlp completed')
        # NLP ends
        if not predict:
            self.processed_data = data
            self.update_or_create_pickle()
        return data

    # DUP
    def update_or_create_pickle(self):
        if self.pickle_file:
            with open(self.pickle_file, "wb") as file:
                pickle.dump(self, file)
