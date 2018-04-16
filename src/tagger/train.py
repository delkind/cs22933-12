import math
import numpy as np
from tagger import util


def tokenize(sentences):
    tokenized_sentences = list()

    segments = {k: v for v, k in enumerate(list(set([segment for sentence in sentences for (segment, _) in sentence]))
                                           + [util.UNKNOWN_SEG, util.START_TAG, util.END_TAG])}
    tags = {k: v for v, k in enumerate(list(set([tag for sentence in sentences for (_, tag) in sentence])) +
                                       [util.START_TAG, util.END_TAG])}

    for sentence in sentences:
        tokenized_sentence = list()
        for (segment, tag) in sentence:
            tokenized_sentence.append((segments[segment], tags[tag]))
        tokenized_sentences.append(tokenized_sentence)

    return segments, tags, tokenized_sentences


def train(raw_sentences, order, smoothing=False):
    """
    Train the decoder on a given corpus
    :param smoothing: whether to smooth probabilities
    :param sentences: tagged corpus
    :param model: model (should contain tags, itags, segments)
    :param order: model order (0 - unigram, 1 - bigram, etc)
    :return: trained model
    All the probabilities are log probabilities as requested
    """

    isegments, itags, sentences = tokenize(raw_sentences)

    segments = {v: k for (k, v) in isegments.items()}
    tags = {v: k for (k, v) in itags.items()}

    start_tag = itags[util.START_TAG]
    end_tag = itags[util.END_TAG]
    unknown_seg = isegments[util.UNKNOWN_SEG]

    model = {'segments': segments, 'tags': tags, 'start_tag': start_tag, 'end_tag': end_tag, 'unknown_seg': unknown_seg}

    if order < 0:
        raise ValueError("Order cannot be less than 0")

    lex = np.zeros((len(segments), len(tags)))

    for sentence in sentences:
        for (segment, tag) in sentence:
            lex[segment, tag] += 1

    if order == 0:
        lex[unknown_seg, itags['NNP']] = 1
        model['lex'] = count_to_log_prob(lex, 1)
        return model

    if not smoothing:
        lex[unknown_seg, itags['NNP']] = 1
    else:
        # add one smoothing
        all_combs = np.sum(lex > 0)
        lex[lex > 0] += 1
        lex[unknown_seg] = np.sum(lex, axis=0) + 1
        lex[unknown_seg, start_tag] = 0
        lex[unknown_seg, end_tag] = 0

    ngrams = np.zeros((len(tags),) * (order + 1), "int")

    lex[isegments[util.START_TAG], start_tag] = 1
    lex[isegments[util.END_TAG], end_tag] = 1

    for sentence in sentences:
        # extract tag sequence from the the sentence
        tag_sequence = [x for (_, x) in sentence]
        # amend tag sequence with the start and end symbols as appropriate
        tag_sequence = [start_tag] * order + tag_sequence + [end_tag] * order
        # iterate over all the n-grams in the sentence and increase their respective count
        for ngram in zip(*[tag_sequence[i:] for i in range(order + 1)]):
            ngrams[ngram] += 1
    ngrams = ngrams.reshape(pow(len(tags), order), len(tags))

    # we might smooth for unseen transitions, but this does not seem to give much gain

    gram = {order: count_to_log_prob(ngrams, 1),
            order - 1: count_to_log_prob(ngrams.sum(axis=1), 0).reshape(-1, len(model['tags']))}

    model['gram'] = gram
    model['lex'] = count_to_log_prob(lex, 0)

    return model


def argnonzero(array):
    """
    Return indices for the non-zero cells in the log probability matrix
    :param array:
    :return:
    """
    return np.transpose(np.nonzero(array > util.LOG_EPSILON))


def count_to_log_prob(matrix, axis):
    """
    Turn counts into log probabilities (zero counts assigned EPSILON to avoid division by zero)
    :param matrix: counts matrix
    :param axis: axis to normalize by
    :return: log probabilities matrix of the same dimensions
    """
    matrix_sum = matrix.sum(axis=axis, keepdims=True)
    # wherever sum is 0, each count is also 0, so we can safely assign anything we want to the sum (avoids div by zero)
    matrix_sum[matrix_sum == 0] = 1
    matrix = matrix / matrix_sum
    matrix[matrix == 0] = util.EPSILON
    matrix = np.log(matrix)
    return matrix


def history_to_str(history, corpus_tags, separator=util.SEPARATOR):
    """
    Use token mappings to emit tag sequences for n-1-grams
    :param history: n-1-gram tuple
    :param corpus_tags: tag mappings
    :param separator: separator for the tag strings in the sequence
    :return: tag sequence string
    """
    history_str = ""
    for j in range(len(history)):
        history_str += corpus_tags[history[j]] + separator

    return history_str


def emit_param_files(lex_file_name, gram_file_name, model):
    """
    Emit gram and lex parameter files
    :param model: trained model
    :param lex_file_name: path for lex file
    :param gram_file_name: path for gram file
    """
    if 'gram' in model:
        emit_gram_file(gram_file_name, model)
    emit_lex_file(lex_file_name, model)


def emit_lex_file(file_name, model):
    """
    Emit lex file
    :param model: trained model
    :param file_name: lex file path
    """

    lex = model['lex']

    with open(file_name, "w") as f:
        nz_segs = np.transpose(np.nonzero(lex > util.LOG_EPSILON))
        grouped = {}
        for i in range(nz_segs.shape[0]):
            if nz_segs[i, 0] not in grouped:
                grouped[nz_segs[i, 0]] = list()
            grouped[nz_segs[i, 0]] += [nz_segs[i, 1]]

        for (segment, tags) in grouped.items():
            f.write(model['segments'][segment])
            for tag in tags:
                f.write(util.SEPARATOR + model["tags"][tag] + util.SEPARATOR + fl_format(lex[segment, tag]))
            f.write("\n")

        # f.write("<s> <s> 0.0\n")
        # f.write("<e> <e> 0.0\n")
        # f.write("<UNKNOWN> NNP 0.0\n")


def fl_format(num):
    """
    Float formatter for parameter files
    :param num: floating point number
    :return: formatted string
    """
    return "{:.5}".format(num)


def emit_gram_file(file_name, model):
    """
    Emit gram file name
    :param model: trained model
    :param file_name: gram file path
    """
    with open(file_name, "w") as f:
        # write data section
        print("\\data\\", file=f)
        nonzeros = {}

        for (order, matrix) in sorted(model['gram'].items()):
            nonzeros[order] = argnonzero(matrix)
            print("ngram " + str(order + 1) + " = " + str(nonzeros[order].shape[0]), file=f)
        print("", file=f)

        # write ngram sections
        for (order, matrix) in sorted(model['gram'].items()):
            print("\\" + str(order + 1) + "-grams\\", file=f)

            nz_grams = nonzeros[order]

            for i in range(nz_grams.shape[0]):
                coord = (nz_grams[i, 0], nz_grams[i, 1])
                print(fl_format(matrix[coord]) + util.SEPARATOR +
                      history_to_str(util.index_to_tuple(coord[0], len(model['tags']), order) + (coord[1],),
                      model['tags'])[:-1], file=f)

            print("", file=f)
