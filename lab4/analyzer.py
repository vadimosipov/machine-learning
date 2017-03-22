#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join
import re
import sys


# reload(sys)
# sys.setdefaultencoding('utf-8')

# .decode('string_escape')

def getwords(content):

    txt = re.compile(r'<[^>]+>').sub('', content)
    words = re.compile(r'[^A-Z^a-z^А-Я^а-я]+').split(txt)
    return [word.lower() for word in words if word != '']


base_path = 'resources/base'
base_files = [f for f in listdir(base_path) if isfile(join(base_path, f))]
# print(base_files)

# base_path = 'resources'
# base_files = ['my.txt']
import codecs
for base in base_files:
    with codecs.open(join(base_path, base), encoding='utf-8') as f:
        content = f.read()
        words = getwords(content)
        print('print words:', words)
        sys.stdout.write('write words: ' + ' '.join(words) + '\n')


# text = '♠'
# print(text)

#
# for feedurl in file('resources/test_945.txt'):
#     try:
#         title, wc = getwordcounts(feedurl)
#     except:
#         continue
#     wordcounts[title] = wc
#     for word, count in wc.items():
#         apcount.setdefault(word, 0)
#         if count > 1:
#             apcount[word] += 1