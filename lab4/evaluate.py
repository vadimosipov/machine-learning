#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import math
from os import listdir
from os.path import isfile, join

from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from nltk.tokenize import LineTokenizer
from nltk.tokenize import RegexpTokenizer
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json

BASE_PATH = join('resources', 'data', 'base')
TEST_PATH = join('resources', 'data', 'test')
STOP_WORDS_PATH = join('resources', 'stop-words', 'russian', 'sw.txt')

ma = MorphAnalyzer()
words_tokenizer = RegexpTokenizer('[\w\dА-Яа-яёЁ]+')


def load_stop_words(lang):
    content = read_file(STOP_WORDS_PATH)
    lt = LineTokenizer()
    return lt.tokenize(content)


def filter_out(tokens, lang):
    # unfortunately, i did not found standard out-of-box implementation of linked list
    map = dict()
    for token in tokens:
        map.setdefault(token, 0)
        map[token] += 1

    for stop_word in load_stop_words(lang):
        if stop_word in map:
            del map[stop_word]

    filtered_words = []
    for k, v in map.items():
        filtered_words.extend([k] * v)
    return filtered_words


def tokenize(text):
    return [token.lower() for token in words_tokenizer.tokenize(text) if len(token) > 1 and not token.isdigit()]


def read_file(file_path):
    with io.open(file_path) as ins:
        return ins.read()


def cleanup_markup(content):
    return BeautifulSoup(content, 'html.parser').textv


def get_file_paths(parent_dir):
    pathes = []
    for f in listdir(parent_dir):
        path = join(parent_dir, f)
        if isfile(path): pathes.append(path)
    return pathes


def lemmatize(tokens, lang):
    lemmas = []
    for token in tokens:
        parses = ma.parse(token)
        parses_len = len(parses)
        if parses_len == 0:
            lemma = token
        elif parses_len > 1:
            eligible = parses[0]
            if eligible.score == parses[1].score and eligible.tag.POS in ['PRTF', 'NOUN']:
                for i in range(1, len(parses)):
                    nxt = parses[i]
                    if eligible.score == nxt.score and eligible.tag.POS == 'PRTF' and nxt.tag.POS == 'NOUN':
                        eligible = nxt
                        break
                    elif eligible.score == nxt.score and eligible.tag.POS == 'NOUN' and nxt.tag.POS in ['NOUN', 'NPRO']:
                        eligible = nxt
                        break
            lemma = eligible.normal_form
        else:
            lemma = parses[0].normal_form
        lemmas.append(lemma)
    return lemmas


def _cosine_similarity(vec1, vec2):
    def dot_product(vec1, vec2):
        return sum([vec1[dim] * vec2[dim] for dim in vec1 if dim in vec2])

    return dot_product(vec1, vec2) / math.sqrt(dot_product(vec1, vec1)) / math.sqrt(dot_product(vec2, vec2))


def main():
    for file_path in get_file_paths(BASE_PATH):
        if (file_path == 'resources/base/base_14.txt'):
            content = read_file(file_path)
            text = cleanup_markup(content)
            tokens = tokenize(text)

            print('|'.join(tokens))
            print('--------------------------')
            lemmas = lemmatize(tokens, 'russian')
            print('|'.join(lemmas))
            print('--------------------------')
            # after lemmatize due to such words as nashey -> nash - stop word
            filtered_words = filter_out(lemmas, 'russian')
            print('|'.join(filtered_words))


def big_tokenize(text):
    tokens = tokenize(text)
    return lemmatize(tokens, 'russian')


def load_document(file_path):
    return cleanup_markup(read_file(file_path))


def load_texts():
    tests = []
    documents = []

    for file_path in get_file_paths(TEST_PATH):
        documents.append(load_document(file_path))
        id = re.search(".*_([\d]+).txt", file_path).group(1)
        tests.append(int(id))

    for file_path in get_file_paths(BASE_PATH): documents.append(load_document(file_path))
    return documents, tests


def prepare_result(tests, cos_result):
    barrier = 0.0
    results = []
    tests_len = len(tests)
    for i in range(tests_len):
        results.append((tests[i], sum(cos_result[i])))
        barrier += results[-1][1]
    barrier /= tests_len

    answer = {'defined': [], 'other': []}
    for i in range(tests_len):
        where = 'defined' if results[i][1] > barrier else 'other'
        answer[where].append(results[i][0])
    # d = json.dump(answer)
    # print(d)

    print('tests:', tests_len)
    print('defined:', len(answer['defined']))
    print('other:', len(answer['other']))

    json_data = json.dumps(answer)
    with open('lab4.json', mode='w') as ous:
        ous.write(json_data)


def main2():
    lang = 'russian'

    documents, tests = load_texts()
    cv = CountVectorizer(tokenizer=big_tokenize, stop_words=load_stop_words(lang))
    matrix = cv.fit_transform(documents)
    mx = matrix.toarray()

    tests_len = len(tests)
    cos_result = cosine_similarity(mx[:tests_len], mx[tests_len:])
    prepare_result(tests, cos_result)

    # print(cos_result)

    import seaborn
    # plt.figure(figsize=(20, 9))
    # for i in range(tests_len % 20):
    #     plt.subplot(20, 5, i + 1)
    #     plt.plot(cos_result[i])
    #
    # plt.show()


def choose(word):
    parses = ma.parse(word)
    eligible = parses[0]

    print('eligible')
    print(eligible)
    if len(parses) > 1 and eligible.score == parses[1].score and eligible.tag.POS in ['PRTF', 'NOUN']:
        for i in range(1, len(parses)):
            nxt = parses[i]
            if eligible.score == nxt.score and eligible.tag.POS == 'PRTF' and nxt.tag.POS == 'NOUN':
                eligible = nxt
                break
            elif eligible.score == nxt.score and eligible.tag.POS == 'NOUN' and nxt.tag.POS in ['NOUN', 'NPRO']:
                eligible = nxt
                break
    print(eligible.normal_form)


def test():
    words = [
        u'данных',
        u'окружающего',
        u'из',
        u'при',
        u'кому',
        u'после'
    ]
    # words = [u'данных']
    for word in words:
        parses = ma.parse(word)
        # print(''.join(word))
        for parse in parses:
            print(parse)
            print(parse.normal_form)
            print('********')
        choose(word)
        print('-------------')


if __name__ == "__main__":
    main2()
    # test()
    # map = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    # json_data = json.dumps(map, indent=4, skipkeys=True, separators=(',', ': '))
    # with open('lab4.json', mode='w') as ous:
    #     ous.write(json_data)
    # import re
    # result = re.search(".*_([\d]+).txt", TEST_PATH + "/test_4.txt")
    # if result:
    #     print(result.group(1))
