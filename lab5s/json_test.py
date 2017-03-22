#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from urlparse import urlparse

s = '{"visits": [{"url": "http://www.xvideos.com/video9141605/ariel_rebel_solo", "timestamp": 1418907499604}, {"url": "http://www.xvideos.com/profiles/ariel-rebel-1", "timestamp": 1418907497534}, {"url": "http://alfabank.ru/retail/tariff_plans/sravnenie/", "timestamp": 1418820223309}, {"url": "http://alfabank.ru/retail/tariff_plans/comfort/", "timestamp": 1418820150939}, {"url": "http://alfabank.ru/peterburg/press/news/2014/12/17/34087.html", "timestamp": 1418819995271}, {"url": "http://alfabank.ru/press/news/2014/12/17/34085.html", "timestamp": 1418819950561}, {"url": "http://www.xvideos.com/video7679470/teen_fucks_step-daddy_for_intern_job", "timestamp": 1418648720990}, {"url": "http://www.xvideos.com/video8013396/nubile_films_-_cum_swallowing_teen_cutie", "timestamp": 1418648342550}, {"url": "http://www.xvideos.com/video5827807/blonde_teen_does_yoga_and_gets_a_big_cock", "timestamp": 1418648338601}, {"url": "http://www.xvideos.com/profiles/ariel-rebel-1", "timestamp": 1417527977655}]}'
from nltk.tokenize import RegexpTokenizer
import re

regexp = re.compile(r'^www\.', re.U)
# re.compile(r'^www\.|(\.ru|\.com)$', re.U)
# words_tokenizer = RegexpTokenizer('[A-Za-zА-Яа-яёЁ]+\.?[A-Za-zА-Яа-яёЁ]+')

words_tokenizer = RegexpTokenizer(u'[A-Za-zА-Яа-яёЁ]+', flags=re.UNICODE)
def extract_user_data(s):
    udata = json.loads(s)
    text = ''
    for info in udata['visits']:
        parsed = urlparse(info['url'])

        domain = regexp.sub('', parsed.netloc)
        text += ' ' + domain + ' ' + parsed.path
    return text

w = extract_user_data(s)
# print ' '.join(w)

def tokenize(text):
    print 'text:', text
    s3 = [token.lower() for token in words_tokenizer.tokenize(text) if len(token) > 2]
    print 's3', s3
    return s3

# print w
w = u'q=\u0431\u043b\u043e\u043d\u0434\u0438\u043d\u043a\u0443 \u043d\u0430 \u0434\u0438\u0432\u0430\u043d\u0435&s=t'
# print words_tokenizer.tokenize(w)

from urlparse import parse_qs, urlparse

s = u'http://google.com?q=\u0431\u043b\u043e\u043d\u0434\u0438\u043d\u043a\u0443 \u043d\u0430 \u0434\u0438\u0432\u0430\u043d\u0435&s=t&text=&etext=636.q0fp4ghy6tmjbrbv1d6rovslsykuhmhrfk1a2kzt-dk0sa-96xxknbsiuvlhoagxlvrqgcpduf2haijcb37gmq.beafa0dc49fd64e7bb460ddd9748ad4db092067f&uuid=&state=aiuy0dbwfj4epaese6rgeajgs2pi3dw99kudgowt9xsmcv5tmmn9utqsqbnfqxrfy1qnk6_no62ibjtslsxvv4q-xmi6yys3irhhvscvkchho6m6coyipju8iyoe1qjcuyaidkqyyo39n94sl2fhue3qleqabeszttrvckz3blxphfvhtqrsnacinb-bl96x4_viyjnevdtdpj9tx2sjd0vnlhyuvv6gvysbfce2ox2jvddm-4pkwua_io5xl1bqhw7pr8gxd9r-dxdepuxy9szelocmzodvm4nvgp5bjsywrikgrixbbdjqqcywfqh-sav7-i-8oazfcoho7lrdsng5tprwwbfh7wcyijv9omuf55izphynezryhls0hliq7_am2qkptwvxnrm08zgaloupefrxg1_1coh3vzdogsocqgp8vqouqjvxnoypan94eonu9y5dz-8&data=ulnrnmk5wktyejr0ewjfyk1ldmtxcxdiqthhazdvykexd0pts2pybxfvrm84wurxsgzwrgi0de1eumjxvmfrunbbnwtedkdltlzeohlzn1fwwk5tcmpit1rwdmptv2o4qwvimfliz0zlyvi1awhfcvyxcefszw&b64e=2&sign=90ff483dfdd10a94f35dbc2780e22f80&keyno=0&ref=cm777e4smoaycdzhdubyhtkuseoilu3m4cvndcrmfnuguixi6uhfki9bhwwohzbjt3szlamw7dsvsxox-txezpwoaau0n3-hzamnsfwxjqfae8na0hrllwndxmmdrwoz7sg5yl0arwf5rdfa87fcyd_j0k6yofc9hmmzdkdb-wk8lktyo7yr1qzljv06_-rztkik0fn_iowefarbzmaonuytbeblh0yyyark5nnaayrhamarpuv7c8yghlzkqqcqfxrzumnnquvzzkea80e3owcxyuujmmdj1m2qb0prvssrwzxvou9bx2rxni7io53qcgkja3fsevh77lpoyiyux3zceeirqf2dgpxfjxvudn-twrfuppvsqugihlafyoufmocs6g0kmciscyc1sbowd0tdw_shmsdrwcqsitljgxxhvxho9em7-j0bjjvxinr28ubiqqsrboelwcy6dabgp8qr9grzaijo4zbcxrgeorj-prorpolpd27ee5o7n-cw5tsdtauh-5xjxysd0kqmjavg0bgqsuhgki7e8fvpj4cf9xusqj4qi0loc8xfqftyvxxmdwzh2dxbennln7bjpashq7ebcvguydwsn-h7smwdbpkuhlg5t4dekpynnyqtmw_xmupbzrasf9ztze-smq_goumxvki3hd6gawo15tm2lqyluzcahqw4zy35ciouz9okhfijdokr7_6j3likwwvafcq5pusdco66uqog21kwfpkxeeu7mf52baa7tlg5gkesupbmvykwyermsfdepdxrg4i920c8ln0ignhlvv1yd0aaui62elvdm_kbeayalx2pi2bv10dlgqvhzs3qgbioo69xpdlrajm-hjt3ltnieyms4ms7q60_qdi1cdof24j2bron0u6wc4zpe2dqosgxiqz1qgq-nv9zptcnogsvmyfmsrnfthwqwwiflfdtdx5_uddorajghmujtx4rpnaxtprymi6ztgm3rzadstl939dvn9imyeakd6dgz6s4st_jjdp6jmobq_wf6dpkxewobjqlg4ulktzyeqctbswwkadksgjvqsnokikt5bthejvigu884fk9y-aksiopcleml4avsgty_8o07byrj3yqxovqcahj4ed5_nun0ok0ldmtzjf8pjglrtirllsecwr6wqafunxbdt_ih6uwgjoxfqordfvem2vpq_2ktmivyytuxnlnxgt7ltecrprm_ozbzygx9yucpju_mx-xnqjnkyy940d6mkjxwy_lxha2vozvdd_k8jsoelty0ycsgqohbkul5tu5pl81jvitmx47urt-th3hkvu-rcskplu8ntrit_v0jvyoljmt7y4mjvdpszz-0-ceiwkgpn4xripmzhbxtcbrenuae2pw_6dlfbldwyjwdbfha5pxdviaembaeciaxdt1eyrtrnckp1xpn70y6xfuuyq1l5frrgngmbhwqnrkq7hziqjo5j9ubsmsnmlobaeetdwph7mspeo&l10n=ru&cts=1427124879438&mc=4.573010557409424'
# parsed = urlparse(s)
# domain = regexp.sub('', parsed.netloc)
# text = ' ' + domain
# print text
# print parsed.query

params = parse_qs(urlparse(s).query)
print 1, params
w = [pars[0] for pars in params.values()]
print w

# w = ''
f = re.compile(u'[a-zA-Z]+=([a-zA-Zа-яА-Я]+)', re.U).findall(urlparse(s).query)
print type(f)
print ' '.join(f)

# print words_tokenizer.tokenize(w)
# print f
# w = 'cat slon nosorog'
from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(tokenizer=tokenize)
# features = vectorizer.fit_transform([w])
# words = vectorizer.get_feature_names()
# print words

# print features
# print words
# print features.shape

print '1887724'.isalnum()


l = set([1,4,5])
print l.difference([1,4,5])
print len(set([1123, 123, 12]))
b= l.update(set([1,2,3]))
print l
print b

s = u' \u0437\u043e\u0440\u044f\u043d\u0430  \u0436\u0435\u043d\u0441\u043a\u043e\u0435 \u0438\u043c\u044f'
# print [s.split(), 'abc', 'dgh']
# d = 'abc'
# print sum([d.split()], [])
print '490.9mpzz5ustvpwdllvsb5pf4fztl5xzclulrllm'.isalnum()
print '4,463280517810812'.isdigit()
print '490.9mpzz5ustvpwdllvsb5pf4fztl5xzclulrllm'.isalpha()
RE_D = re.compile('\d')
print RE_D.search('sdfsd') is None

regexp2 = re.compile(r'(\?|&)[a-zA-Z_]+=([ a-zA-Zа-яА-Я0-9\.:-=_+/]+)', re.U)
j = 'www.zoobrilka.com/homework/books/8/5/10088/ывааыв'
g = regexp2.sub('http://', j)
print g
print urlparse(g).netloc

#
# [u'alfabank.ru', u'and', u'ariel', u'big', u'blonde', u'cock', u'comfort', u'cum', u'cutie', u'daddy', u'does', u'films', u'for', u'fucks', u'gets', u'html', u'intern', u'job', u'news', u'nubile', u'peterburg', u'plans', u'press', u'profiles', u'rebel', u'retail', u'solo', u'sravnenie', u'step', u'swallowing', u'tariff', u'teen', u'video', u'xvideos.com', u'yoga']
# [u'.html', u'alfabank.ru', u'and', u'ariel', u'big', u'blonde', u'cock', u'comfort', u'cum', u'cutie', u'daddy', u'does', u'films', u'for', u'fucks', u'gets', u'intern', u'job', u'news', u'nubile', u'peterburg', u'plans', u'press', u'profiles', u'rebel', u'retail', u'solo', u'sravnenie', u'step', u'swallowing', u'tariff', u'teen', u'video', u'xvideos.com', u'yoga']

# -*- coding: utf-8 -*-
from os.path import join
from nltk.tokenize import LineTokenizer
import io

def read_file(file_path):
    with io.open(file_path, encoding='utf-8') as ins:
        return ins.read()
#         return .decode('utf-8')


def load_stop_words():
    STOP_WORDS_PATH = join('sw.txt')
    content = read_file(STOP_WORDS_PATH)
    lt = LineTokenizer()
    return lt.tokenize(content)
# print load_stop_words()
