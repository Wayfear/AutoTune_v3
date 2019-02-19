import argparse
import sys
import os
from os.path import join
import itertools
import random


def main(args):
    base_path = os.path.expanduser(args.data_dir)
    people_paths = os.listdir(base_path)
    pics = {}
    for p in people_paths:
        if os.path.isdir(join(base_path, p)):
            pics[p] = os.listdir(join(base_path, p))
    pair_peo = list(itertools.combinations(pics.keys(), 2))
    classes = len(pics) + len(pair_peo)
    per_class = args.test_case_num//classes
    # result_file = os.path.expanduser(args.data_dir)
    with open(args.result_file, "w") as f:
        f.write(str(per_class*classes)+'\n')
        for p in people_paths:
            if os.path.isdir(join(base_path, p)):
                for paths in random.sample(list(itertools.combinations(pics[p], 2)), per_class):
                    f.write("%s\t%s\t%s\n"%(p, paths[0], paths[1]))
        for p in pair_peo:
            for paths in random.sample(list(itertools.product(pics[p[0]], pics[p[1]])), per_class):
                f.write("%s\t%s\t%s\t%s\n"%(p[0], paths[0], p[1], paths[1]))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        help='Directory where to place validation data.', default='/home/chris/Documents/2017/test/self')
    parser.add_argument('--result_file', type=str,
                       help='Directory where to place validation data.', default='/home/chris/Documents/2017/test/self/pairs.txt')
    parser.add_argument('--test_case_num', type=int,
                        help='Number of test cases to use for cross validation. Mainly used for testing.', default=2000)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))