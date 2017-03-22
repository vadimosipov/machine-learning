#!/usr/bin/env python
import sys


def try_print_result(uid, user_cat_flags):
    categories = ['0'] * CATEGORIES_COUNT
    one_cat = False
    for k, v in user_cat_flags.items():
        if v >= DEFINITE:
            categories[k] = '1'
            one_cat = True
    if one_cat:
        result = uid + '\t' + '\t'.join(categories)
        print(result)


CATEGORIES_COUNT = 4
DEFINITE = 10


def main():
    current_uid = None
    user_cat_flags = {}

    for line in sys.stdin:
        uid, cat_id = line.strip().split('\t')

        if current_uid != uid:
            if current_uid is not None:
                try_print_result(current_uid, user_cat_flags)
            current_uid = uid
            user_cat_flags = {}
        cat_id = int(cat_id)
        user_cat_flags[cat_id] = user_cat_flags.get(cat_id, 0) + 1

    try_print_result(current_uid, user_cat_flags)


if __name__ == '__main__':
    main()
