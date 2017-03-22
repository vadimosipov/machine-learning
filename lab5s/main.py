from datetime import datetime

import math

from numpy.random.mtrand import permutation

time = '13.08.2015 9:33'
# str_date, str_time = time.split(' ')

def to_datetime(time):
    formatted_time = time + ':00' if time.count(':') == 1 else time
    return datetime.strptime(formatted_time, '%d.%m.%Y %H:%M:%S')

d = to_datetime(time)
print d.timetuple()


def from_timestamp(timestamp):
    d = datetime.utcfromtimestamp(timestamp / 1000)
    minute_of_day = d.hour * 60 + d.minute
    day_of_year = d.timetuple().tm_yday
    day_of_week = d.timetuple().tm_wday
    return minute_of_day, day_of_week, day_of_year

minute_of_day, day_of_week, day_of_year = from_timestamp(1417460427000)
print minute_of_day
print day_of_week
print day_of_year

ar = [
    1417462736000,
    1417462735000,
    1417462733000,
    1417462625000,
    1417462540000,
    1417462539000,
    1417462517000,
    1417462515000,
    1417462467000,
    1417462443000,
    1417462401000,
    1417462366000,
    1417462356000,
    1417462355000
]
for a in ar:
    print from_timestamp(a)

print dict(zip([i for i in range(0, 7)], [i for i in range(0, 7)]))

'''
{"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/drugoe", "timestamp": 1417462736000},
{"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/drugoe", "timestamp": 1417462735000},
{"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/drugoe", "timestamp": 1417462733000},
{"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/drugoe", "timestamp": 1417462625000},
{"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/drugoe", "timestamp": 1417462540000},
{"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/drugoe", "timestamp": 1417462539000},
{"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/novogodnie_podarki_469765802", "timestamp": 1417462517000},
{"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/novogodnie_podarki_469765802", "timestamp": 1417462515000},
{"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/spalnya_palermo_396222486", "timestamp": 1417462467000},
{"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/drugoe", "timestamp": 1417462443000},
{"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/elki_sosny_iskusstvennye_469979238", "timestamp": 1417462401000},
 {"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/drugoe", "timestamp": 1417462366000},
{"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/drugoe", "timestamp": 1417462356000},
 {"url": "http://https://www.avito.ru/volgograd/mebel_i_interer/drugoe", "timestamp": 1417462355000},

'''

# a = datetime.utcfromtimestamp(1417460427000 / 1000)
# print a
# print
