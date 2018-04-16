from tagger import util
from tagger import decode
import sys

if __name__ == '__main__':
    print('\nrunning', sys.argv)
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print('expected params < model > < file = heb-pos.test > < param-file1 > [< param-file2 >] ')
        print('model can be: 1 indicate the baseline (majority vote) tagger ')
        print('            : 2 indicates a bi-gram tagger ')
        print('            : 3 indicates a tri-g ')
    order = int(sys.argv[1])
    test_file = sys.argv[2]
    lex_file = sys.argv[3]

    out_file = test_file.split('/')[-1] + '.tagged'

    if order == 1:
        param_file = sys.argv[3]
        model = decode.parse_params(param_file, None)
        sentences = util.parse(test_file, False)
        sentences = decode.decode(sentences, model, 0)
        decode.emit_tagged_file(out_file, model, sentences)
    else:
        lex_file = sys.argv[3]
        gram_file = sys.argv[4].strip()
        model = decode.parse_params(lex_file, gram_file)
        sentences = util.parse(test_file, False)
        sentences = decode.decode(sentences, model, order - 1)
        decode.emit_tagged_file(out_file, model, sentences)
