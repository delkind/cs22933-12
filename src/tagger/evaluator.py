import numpy as np
from tagger import util

from functools import reduce
from collections import defaultdict


def file_to_tags(path):
    sentences = util.parse(path)
    tags = []

    for sentence in sentences:
        tags += [[tag for (seg, tag) in sentence]]

    return tags

def seg_accuracy(tagged_sentence, gold_sentence):
    """
    segment(word) accuracy of one single sentence
    :param tagged_sentence:
    :param gold_sentence:
    :return: proportion of correct tags in a sentence (1 for each pair match, 0 otherwise)
    >>> seg_accuracy (['NN','VB'],['NN','VB'])
    1.0

    >>> seg_accuracy (['NN','NN'],['NN','VB'])
    0.5
    """
    assert len(tagged_sentence) == len(gold_sentence)

    m = map(lambda tup: tup[0] == tup[1], zip(tagged_sentence, gold_sentence))
    return sum(m) / len(tagged_sentence)


def seg_accuracy_all(tagged_list, gold_list):
    """
    macro-avg of segment accuracy (list of sentences)
    :param tagged_list:
    :param gold_list:
    :return:

    example: (1.0*2 words + 0.66*3words )/5words = 0.8
    >>> tagged=[['NN', 'VB'], ['NN', 'NN','NN']]
    >>> gold = [['NN', 'VB'], ['NN', 'NN','VB']]
    >>> seg_accuracy_all(tagged,gold)
    0.8
    """
    assert len(tagged_list) == len(gold_list)

    weighted_sum = 0
    seg_cumsum = 0
    for i in range(0, len(tagged_list)):
        seg_count = len(tagged_list[i])
        weighted_sum += seg_count * seg_accuracy(tagged_list[i], gold_list[i])
        seg_cumsum += seg_count
    return weighted_sum / seg_cumsum


def sen_accuracy(tagged_sentence, gold_sentence):
    """
    :param tagged_sentence:
    :param gold_sentence:
    :return:
    >>> sen_accuracy (['NN','VB'],['NN','VB'])
    1
    >>> sen_accuracy (['NN','NN'],['NN','VB'])
    0
    """
    return 1 == seg_accuracy(tagged_sentence, gold_sentence)


def sent_accuracy_all(tagged_list, gold_list):
    """

    :param tagged_list:
    :param gold_list:
    :return: macro-avg of sentence accuracy (if any segment mismatch, the whole sentence is 0, otherwise 1)
    example: the 1st sentence is a match, the second fails in at least one segment, thus 1+0/2=0.5
    >>> tagged=[['NN', 'VB'], ['NN', 'NN','NN']]
    >>> gold = [['NN', 'VB'], ['NN', 'NN','VB']]
    >>> sent_accuracy_all(tagged,gold)
    0.5
    """
    return np.mean([x for x in map(lambda tup: sen_accuracy(tup[0], tup[1]), zip(tagged_list, gold_list))])


def _eval_report(gold_lines, predict_lines, model, smooth_str, test_file, gold_file):
    """
    :return: a string with report
    """
    by_sentence_lines = '\n'.join('# {} {:.2f} {:.2f}'.format(i + 1, seg_accuracy(pair[0], pair[1]),
                                                              sen_accuracy(pair[0], pair[1])) for i, pair in
                                  enumerate(zip(predict_lines, gold_lines)))
    all_seg_accuracy = seg_accuracy_all(predict_lines, gold_lines)
    all_sent_accuracy = sent_accuracy_all(predict_lines, gold_lines)
    print('macro-avg {all_seg_accuracy:.4f} {all_sent_accuracy:.4f}'.format(**vars()))
    return \
        '''#-----------------------
# Part-of-Speech Tagging Evaluation
#-----------------------
#
# Model: {model}
# Smoothing: {smooth_str}
# Test File: :{test_file}
# Gold File: :{gold_file}
#
#-----------------------
# sent-num word-accuracy sent-accuracy
#-----------------------
{by_sentence_lines}
#-----------------------
macro-avg {all_seg_accuracy:.4f} {all_sent_accuracy:.4f}
        '''.format(**vars())


def confusion_matrix(predicted_list, gold_list):
    """
    :param gold_list: list of tags
    :param predicted_list: list of tags
    :return:
    """
    cm = defaultdict(int)
    for gold, pred in zip(gold_list, predicted_list):
        cm[gold, pred] += 1

    filtered = filter(lambda t: t[0][0] != t[0][1], cm.items())
    # sort from big to small
    largest = sorted(filtered, key=lambda tup: tup[1], reverse=True)
    return largest[:3]


def confusion_matrix_on_files(tagged_file, gold_file):
    tagged = reduce(lambda x, y: x + y, file_to_tags(tagged_file))
    gold = reduce(lambda x, y: x + y, file_to_tags(gold_file))
    print('largest 3 errors in confusion_matrix', confusion_matrix(tagged, gold))


def eval(tagged_file, gold_file, model, smooth_str, out_file=None):
    """
    :param tagged_file:
    :param gold_file:
    :param model:
    :param smooth:
    :param out_file:
    :return:
    """

    result = _eval_report(file_to_tags(tagged_file),
                          file_to_tags(gold_file),
                          model, smooth_str, tagged_file, gold_file)
    if out_file:
        print('evaluate: writing:', out_file)
        with open(out_file, 'w') as f:
            f.write(result)
        confusion_matrix_on_files(tagged_file, gold_file)
    else:
        print(result)
