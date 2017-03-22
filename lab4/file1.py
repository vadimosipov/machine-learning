#!/usr/bin/env python
import re
import sys

from HTMLParser import HTMLParser


class MyHTMLParser(HTMLParser):

    def handle_data(self, data):
        words = re.compile(r'[^A-Z^a-z^А-Я^а-я]+', re.U).split(data)
        # words = (w for w in data.split() if w)
        for w in words:
            sys.stdout.write(w + "\n")

    def getwords(self, data):
        txt = re.compile(r'<[^>]+>').sub('', data)
        words = re.compile(r'[^A-Z^a-z^А-Я^а-я]+', re.U).split(data)
        return [word.lower() for word in words if word != '']

if __name__ == "__main__":
    with open("resources/base/base_20.txt", "r") as src:
        MyHTMLParser().feed(src.read())
