from collections import Counter
import math
from string import punctuation

from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import fuzzy

import legal_control_words

# reverse map the control words to the type
term_to_type_map = dict()
for key, values in legal_control_words.terms_by_type.items():
    term_to_type_map.update({value: key.lower() for value in values})


def my_tokenizer(term, map_collection, punctuation_collection, tokenizer_function):
    my_tokens = tokenizer_function(term)
    my_tokens = [token.lower() for token in my_tokens if token not in punctuation_collection]
    my_tokens = [token for token in my_tokens if token not in stopwords.words('english')]
    for i in range(0, len(my_tokens)):
        my_tokens[i] = "".join([c for c in my_tokens[i] if c not in punctuation_collection])


    for i in range(0, len(my_tokens)):
        if my_tokens[i] in map_collection:
            my_tokens[i] = map_collection[my_tokens[i]]

    return my_tokens


class EntityName:

    def __init__(self, name: str):
        self.name = name
        self.tokens = dict()

        self.tokenize()
        self.soundex_tokenize()
        self.dmeta_tokenize()
        self.nysiis_tokenize()

        self.ngrams = dict()
        self.generate_ngrams()

    def tokenize(self):
        self.tokens["DEFAULT"] = my_tokenizer(self.name, term_to_type_map, punctuation, word_tokenize)

    def soundex_tokenize(self):
        soundex = fuzzy.Soundex(4)
        self.tokens["SOUNDEX"] = [soundex(token) for token in self.tokens["DEFAULT"]]

    def dmeta_tokenize(self):
        dmeta = fuzzy.DMetaphone()
        self.tokens["DMETA"] = [dmeta(token)[0] for token in self.tokens["DEFAULT"]]

    def nysiis_tokenize(self):
        self.tokens["NYSIIS"] = [fuzzy.nysiis(token) for token in self.tokens["DEFAULT"]]

    def generate_ngrams(self):
        for token_type, tokens in self.tokens.items():

            if len(tokens) < 2:
                self.ngrams[token_type] = list()
                return

            self.ngrams[token_type] = [x for x in ngrams(tokens, 2)]

    def jaccard_distances(self, other):
        distances = dict()
        for key in self.ngrams:
            a = set(self.ngrams[key])
            b = set(other.ngrams[key])
            distances[key] = 1.0 * len(a & b) / len (a | b)

        return distances

    def cosine_similarities(self, other):
        similarities = dict()
        for key in self.ngrams:
            vec1 = Counter(self.ngrams[key])
            vec2 = Counter(other.ngrams[key])
            intersection = set(vec1.keys()) & set(vec2.keys())
            numerator = sum([vec1[x] * vec2[x] for x in intersection])
            sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
            sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
            denominator = math.sqrt(sum1) * math.sqrt(sum2)
            if not denominator:
                similarities[key] = 0.0
            else:
                similarities[key] = float(numerator) / denominator

        return similarities

term1 = "Perpetually Mortified of Columbus, LLC."
term2 = "Perenially Ministrated of Columbia, L.L.C."
term3 = "Perpetualy Mortised at Columbus Co"

entity1 = EntityName(term1)
entity2 = EntityName(term2)
entity3 = EntityName(term3)

print(entity1.tokens)
print(entity2.tokens)
print(entity3.tokens)

print(entity1.ngrams)
print(entity2.ngrams)
print(entity3.ngrams)

print(entity1.jaccard_distances(entity2))
print(entity1.cosine_similarities(entity2))

print(entity1.jaccard_distances(entity3))
print(entity1.cosine_similarities(entity3))

print(entity2.jaccard_distances(entity3))
print(entity2.cosine_similarities(entity3))
