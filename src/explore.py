from __future__ import absolute_import, division, print_function

from collections import defaultdict


def explore(file_name):
    '''

    :param file_name:
    :return:
    '''
    print(file_name)

    seg_dict = defaultdict(int)  # segment to count
    seg_tags = defaultdict(lambda: defaultdict(int))  # segment to unique set of tags  { 'A':{ 'NN','Verb'}, 'B':{'NN"}}

    lines = None
    with open(file_name, 'r') as f:
        lines = f.read().splitlines()

    for i, line in enumerate(lines):
        # skip empty lines (between sentences kept empty on purpose)
        if len(line) == 0:
            continue

        splitted = line.split('\t')
        seg = splitted[0]
        tag = splitted[1]

        seg_dict[seg] += 1
        # unique tag set per segment
        seg_tags[seg][tag] += 1

    print('unigram segments instancecount=', sum(seg_dict.itervalues()))
    print('unigram segments unique types=', len(seg_dict))
    # value:{NN:3, VB:2} --map a dict to it's sum --> 5
    print('seg-tag pairs instance count=', sum(map(lambda dic: sum(dic.itervalues()), seg_tags.values())))
    # value:{NN:3, VB:2} --map a dict to number of keys --> 5
    segtag_pairs_uniques = sum((map(len, seg_tags.values())))
    print('seg-tag pairs unique count=', segtag_pairs_uniques)
    # cool usage of map : [{NN, VB},{VB}] -- map to length of sets --> [2,1] , then just sum
    print('ambiguousness = average different tag types per segment={:.4}'.format(segtag_pairs_uniques / len(seg_tags)))


if __name__ == "__main__":
    files = ['../exps/isha/isha.gold', '../exps/heb-pos.train', '../exps/heb-pos.gold']
    for f in files:
        print('\n')
        explore(f)
