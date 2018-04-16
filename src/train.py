from tagger import util
from tagger import train
import sys

if __name__ == '__main__':
    print('\ntrain args:', sys.argv)
    # train('../exps/heb-pos.train') ; exit(1)
    if len(sys.argv) < 4:
        print('expected params< model > < file = heb-pos.train > < smoothing(y / n) > ')
        print('model can be: 1 indicate the baseline (majority vote) tagger ')
        print('            : 2 indicates a bi-gram tagger ')
        print('            : 3 indicates a tri-g ')
    model = int(sys.argv[1])
    file_name = sys.argv[2]

    if sys.argv[3].lower() == 'y':
        smoothing = True
    elif sys.argv[3].lower() == 'n':
        smoothing = False
    else:
        raise ValueError("illegal value for smoothing y/n expected. was: " + sys.argv[3])

    parts_of_10 = 10
    if len(sys.argv) == 5:
        parts_of_10 = int(sys.argv[4])
        print('warning: reading parts_of_10', parts_of_10)

    out_file_prefix = file_name.split('/')[-1]
    if model == 1:
        sentences = util.parse(file_name)
        model = train.train(sentences, 0, smoothing=smoothing)
        train.emit_param_files(out_file_prefix + ".param", None, model)
    else:
        sentences = util.parse(file_name, percent=parts_of_10 * 10)
        model = train.train(sentences, model - 1, smoothing=smoothing)
        train.emit_param_files(out_file_prefix + ".lex", out_file_prefix + ".gram", model)
