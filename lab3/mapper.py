#!/usr/bin/env python
import re
import sys
import urllib
import urlparse

categories = {
    'cars.ru': 0, 'avto-russia.ru': 0, 'bmwclub.ru': 0,
    'zakon.kz': 1, 'egov.kz': 1, 'makler.md': 1,
    'russianfood.com': 2, 'psychologies.ru': 2, 'gotovim.ru': 2,
    'books.imhonet.ru': 3, 'zhurnaly.biz': 3, 'zvukobook.ru': 3
}


def url2domain(url):
    try:
        a = urlparse.urlparse(urllib.unquote(url.strip()))
        if (a.scheme in ['http', 'https']):
            b = re.search("(?:www\.)?(.*)", a.netloc).group(1)
            if b is not None:
                return str(b).strip()
    except:
        return ''
    return ''


def map(line):
    data = line.split('\t')
    if len(data) != 3:
        return
    uid, timestamp, url = data

    domain = url2domain(url)
    if domain in categories:
        result = uid + '\t' + str(categories[domain])
        print(result)


def read_input(file):
    for line in file:
        yield line


def main():
    for line in sys.stdin:
        map(line)


if __name__ == '__main__':
    main()
