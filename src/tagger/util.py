import math
import time
import numpy as np

EPSILON = 1e-100
SEPARATOR = " "
END_TAG = "<e>"
START_TAG = "<s>"
# UNKNOWN_SEG = "<UNKNOWN>"
UNKNOWN_SEG = '<unkown_seg>'
LOG_EPSILON = math.log(EPSILON)


def parse(file_path, tagged=True, percent=100):
    """
    Parse the train file and build corpus. All the strings (both tags and segments) are tokenized for efficiency
    :param tagged: whether the file is tagged (gold/train) or not
    :param file_path: Path to the train file
    :return: corpus consisting of mapping for segments and tags and list of the tokenized sentences from the corpus
    """
    sentences = list()

    with open(file_path, "r") as infile:
        sentence = list()
        for line in infile.readlines():
            # we skip comment lines
            if line[0] != '#':
                params = line.split()
                if len(params) == 2 and tagged:
                    # we are parsing tagged file
                    sentence.append((params[0], params[1]))
                elif len(params) == 1 and not tagged:
                    # we are parsing untagged file
                    sentence.append(params[0])
                elif len(params) == 0:
                    # sentence ends here
                    if len(sentence) > 0:
                        sentences.append(sentence)
                    sentence = list()
                else:
                    # invalid format
                    raise SyntaxError("File format is unknown")

        if len(sentence) > 0:
            # append last sentence (in case there was no empty line at the end of the file)
            sentences.append(sentence)

        return sentences[:int(len(sentences) * percent / 100)]


def parse0(file_path, model, tagged=True):
    """
    Parse the train file and build corpus. All the strings (both tags and segments) are tokenized for efficiency
    :param model: model to be trained
    :param tagged: whether the file is tagged (gold/train) or not
    :param file_path: Path to the train file
    :return: corpus consisting of mapping for segments and tags and list of the tokenized sentences from the corpus
    """
    sentences = list()

    if 'segments' not in model:
        segments = {}
    else:
        segments = {v: k for (k, v) in model['segments'].items()}

    if 'tags' not in model:
        tags = {}
    else:
        tags = {v: k for (k, v) in model['tags'].items()}

    seg_count = len(segments)
    tag_count = len(tags)

    with open(file_path, "r") as infile:
        sentence = list()
        for line in infile.readlines():
            # we skip comment lines
            if line[0] != '#':
                params = line.split()
                if len(params) == 2 and tagged:
                    # we are parsing tagged file
                    segment, tag = params[0], params[1]
                    if segment not in segments:
                        # segment we haven't seen, add it
                        segments[segment] = seg_count
                        seg_count += 1
                    if tag not in tags:
                        # tag we haven't seen, add it
                        tags[tag] = tag_count
                        tag_count += 1
                    sentence.append((segments[segment], tags[tag]))
                elif len(params) == 1 and not tagged:
                    # we are parsing untagged file
                    if params[0] not in segments:
                        segments[params[0]] = seg_count
                        seg_count += 1
                    sentence.append((segments[params[0]], None))
                elif len(params) == 0:
                    # sentence ends here
                    if len(sentence) > 0:
                        sentences.append(sentence)
                    sentence = list()
                else:
                    # invalid format
                    raise SyntaxError("File format is unknown")

        if len(sentence) > 0:
            # append last sentence (in case there was no empty line at the end of the file)
            sentences.append(sentence)

    if END_TAG not in tags:
        tags[END_TAG] = tag_count + 1
    if START_TAG not in tags:
        tags[START_TAG] = tag_count

    model['tags'] = {v: k for (k, v) in tags.items()}
    model['itags'] = tags
    model['segments'] = {v: k for (k, v) in segments.items()}

    return model, sentences


def tuple_to_index(tup, base):
    index = 0
    for i in tup:
        index = index * base + i

    return index


def index_to_tuple(index, base, order):
    tup = tuple()
    while index > 0:
        tup = (index % base,) + tup
        index //= base
        order -= 1
    for i in range(order):
        tup = (0,) + tup

    return tup


def print_progress_bar(iteration, total, prefix ='', suffix ='', decimals = 1, length = 100, fill ='*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def current_time_millis():
    return int(round(time.time() * 1000))


def argmax(array):
    argmax_values = np.zeros_like(array)
    for i in range(array.shape[0]):
        maxval = -math.inf
        for j in range(array.shape[1]):
            if maxval < array[i, j]:
                argmax_values[i] = j
                maxval = array[i, j]
    return argmax_values
