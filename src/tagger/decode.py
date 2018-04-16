import numpy as np
import math
from tagger import util


def tokenize(sentence, model):
    isegments = model['isegments']

    tokenized_sentence = list()
    for segment in sentence:
        if segment in isegments:
            tokenized_sentence.append(isegments[segment])
        else:
            tokenized_sentence.append(model['isegments'][util.UNKNOWN_SEG])

    return tokenized_sentence


def emit_tagged_file(file_path, model, sentences):
    with open(file_path, "w") as f:
        for sentence in sentences:
            for (segment, tag) in sentence:
                print(segment + '\t' + model['tags'][tag], file=f)
            print("", file=f)


def majority_vote_decoder(model, sentence):
    """
    Majority vote decoder
    :param model: trained model
    :param sentence: sentence to tag
    :return:
    """
    tag_sequence = []
    for segment in sentence:
        tag_sequence.append(np.argmax(model['lex'][segment]))
    return tag_sequence


def viterbi_decoder(model, order, sentence):
    ngrams = model['gram'][order + 1]
    base = len(model['tags'])
    order = int(math.floor(math.log(ngrams.shape[0], base) + 0.5))
    start_tag = model['itags'][util.START_TAG]
    end_tag = model['itags'][util.END_TAG]
    lex = model['lex']

    viterbi = np.zeros((ngrams.shape[0], len(sentence) + 1))
    backpointer = np.zeros_like(viterbi).astype("int")

    # Initialize. Complexity: O(T^Order)
    complete_start = util.tuple_to_index((start_tag,) * order, base)
    for s in range(viterbi.shape[0]):
        viterbi[s, 0] = ngrams[complete_start, s % base] + lex[sentence[0], s % base]

    step = 1
    # Main loop. Complexity: O(S*T^(Order*2))
    for segment in sentence[1:]:
        # TODO: figure out better way than just looping (straightforward np.tile() is much slower due to dimensions)
        weighted_prev_step = np.zeros((viterbi.shape[0],) * 2)
        for s in range(viterbi.shape[0]):
            weighted_prev_step[:, s] = viterbi[:, step - 1] + ngrams[:, s % base] + lex[segment, s % base]

        argmax = np.argmax(weighted_prev_step, axis=0)
        viterbi[:, step] = weighted_prev_step[argmax, np.arange(weighted_prev_step.shape[0])]
        backpointer[:, step] = argmax
        step += 1

    weighted_prev_step = viterbi[:, step - 1] + ngrams[:, end_tag]
    argmax = np.argmax(weighted_prev_step)
    backpointer_current = argmax

    tag_sequence = []

    for i in range(len(sentence)):
        tag_sequence.append(backpointer_current % base)
        backpointer_current = backpointer[backpointer_current, step - 1]
        step -= 1

    return reversed(tag_sequence)


def decode(sentences, model, order, smoothing=False, display_progress=True):
    tagged_sentences = []
    progress = 0

    start = util.current_time_millis()

    for sentence in sentences:
        util.print_progress_bar(progress, len(sentences),
                                suffix="{} elapsed".format((util.current_time_millis() - start) // 1000))
        progress += 1
        if order == 0:
            tag_sequence = majority_vote_decoder(model, tokenize(sentence, model))
        else:
            tag_sequence = viterbi_decoder(model, order, tokenize(sentence, model))
        tagged_sentences.append(zip(sentence, tag_sequence))

    print("{} sentences processed in {} seconds.".format(len(sentences), (util.current_time_millis() - start) // 1000))

    return tagged_sentences


def parse_lex_file(file_name):
    lex_builder = {}
    segments = {}
    tags = {}

    segment_count = 0
    tag_count = 0

    with open(file_name) as f:
        # read line by line
        for line in f.readlines():
            # split line into tokens
            tokens = line.split(' ')
            # first token is segment
            seg = tokens[0]
            # add segment to segments if we haven't yet
            if seg not in segments:
                segments[seg] = segment_count
                segment_count += 1
            # the rest are <tags, probability> pairs
            rest = tokens[1:]
            for i in range(0, len(rest), 2):
                # add tag if we haven't yet
                if rest[i] not in tags:
                    tags[rest[i]] = tag_count
                    tag = tag_count
                    tag_count += 1
                else:
                    tag = tags[rest[i]]
                # add segment with corresponding tags/probs to the list that will be turned into numpy array
                if seg not in lex_builder:
                    lex_builder[seg] = []
                lex_builder[seg].append((tag, float(rest[i + 1])))

    if util.START_TAG not in tags:
        tags[util.START_TAG] = tag_count
    if util.END_TAG not in tags:
        tags[util.END_TAG] = tag_count + 1

    # now that we know all the tags and segments in the model we can create an array
    lex = np.full((len(segments), len(tags)), util.LOG_EPSILON, float)

    smoothing = False

    # and populate it with probabilities
    for (seg, taglist) in lex_builder.items():
        for (tag, logprob) in taglist:
            lex[segments[seg], tag] = logprob
            # can we determine whether smoothing was enabled? Yes, we can! If it's disabled, P(UNKNOWN_SEG | tag) is 0
            # unless the tag is NNP.
            if seg == util.UNKNOWN_SEG and tag != tags['NNP']:
                smoothing = True

    return \
        {'lex': lex,
         'segments': {v: k for (k, v) in segments.items()},
         'tags': {v: k for (k, v) in tags.items()},
         'isegments': segments,
         'itags': tags,
         'smoothing': smoothing
         }


def parse_gram_file(file_name, model):
    # TODO: Replace with regex matching
    # sections of the file, will be used for state transitions
    sections = set(['data'] + ['{}-grams'.format(i) for i in range(1, 6)])

    gram = {}

    tags = model['tags']
    with open(file_name) as f:
        curr_section = None
        for i, line in enumerate(f.read().splitlines()):
            # which section are we in?
            if curr_section is None:
                curr_section = line.replace('\\', '')
                if curr_section not in sections:
                    raise ValueError('unkown section header: ' + str(curr_section))
                continue
            elif len(line.strip()) < 1:
                curr_section = None
                continue

            # here we must be inside a section
            tokens = line.split(' ')
            if curr_section is None:
                raise ValueError(
                    'got a line outside a section, this is invalid case. check if you have 2 line breaks')
            elif curr_section == 'data':
                # 'ngram 1 = 127884'
                if tokens[0] != 'ngram':
                    raise ValueError('row %d must start with ngram' % i)
                ngram_type = int(tokens[1])  # 1 or 2
                # params.ngrams_to_count[ngram_type] = int(tokens[3])
            else:
                # 0.000105 AUX RB
                order = int(curr_section.split('-')[0])
                ngram_tags = tuple(model['itags'][tag] for tag in tokens[1:])
                logprob = float(tokens[0])
                # we keep the transition matrix per order in the dictionary
                if order not in gram:
                    gram[order] = np.full((pow(len(tags), order - 1), len(tags)), util.LOG_EPSILON,
                                          float)
                gram[order][util.tuple_to_index(ngram_tags[:-1], len(tags)), ngram_tags[-1]] = logprob

    if model['smoothing']:
        # interpolate the transition probabilities using the lower order ones
        real_order = max([key for key in gram.keys()])
        for prev in range(pow(len(tags), real_order - 1)):
            for cur in range(len(tags)):
                if gram[real_order][prev, cur] <= util.LOG_EPSILON:
                    total = 0
                    prob = 0
                    for i in reversed(range(1, real_order)):
                        if i in gram:
                            prob += math.exp(gram[i][prev % pow(len(tags), i - 1), cur])
                            total += 1
                    if prob > 0:
                        # P = 0.9*P(current order)+sum(P(lower orders))/count(lower orders).
                        # It should be lower than the probability of the order-grams so that the transition
                        # of the real order fire with higher probability
                        prob /= total * 5
                        gram[real_order][prev, cur] = math.log(prob)

    model['gram'] = gram

    return model


def parse_params(lex_file_name, gram_file_name):
    model = parse_lex_file(lex_file_name)
    if gram_file_name is not None:
        model = parse_gram_file(gram_file_name, model)
    return model
