import sys
from tagger import evaluator

if __name__ == '__main__':
    print('\nevaluate called', sys.argv)
    if len(sys.argv) != 5:
        print('expected: train < *.tagged >< heb-pos.gold >< model >< smoothing(y/n) > ')
        print('model can be: 1 indicate the baseline (majority vote) tagger ')
        print('            : 2 indicates a bi-gram tagger ')
        print('            : 3 indicates a tri-g ')
    out_file = sys.argv[1].split('/')[-1].replace('.tagged', '.eval')
    evaluator.eval(tagged_file=sys.argv[1],
                   gold_file=sys.argv[2],
                   model=int(sys.argv[3]),
                   smooth_str=sys.argv[4],
                   out_file=out_file)
