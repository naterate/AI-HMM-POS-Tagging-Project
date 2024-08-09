import argparse
import sys

from base_viterbi import base_viterbi
from optimized_viterbi import optimized_viterbi

import utilities

"""
This file contains the main application that is run.
"""


def main(args):
    print("Loading dataset...")
    train_set = utilities.load_dataset(args.training_file)
    test_set = utilities.load_dataset(args.test_file)
    print("Loaded dataset")
    print()

    algorithms = {"base_viterbi": base_viterbi, "optimized_viterbi": optimized_viterbi}
    algorithm = algorithms[args.algorithm]
    
    print("Running {}...".format(args.algorithm))
    testtag_predictions = algorithm(train_set, utilities.strip_tags(test_set))
    baseline_acc, correct_wordtagcounter, wrong_wordtagcounter = utilities.evaluate_accuracies(testtag_predictions, test_set)
    multitags_acc, unseen_acc, = utilities.specialword_accuracies(train_set, testtag_predictions, test_set)

    print("Accuracy: {:.2f}%".format(baseline_acc * 100))
    print("\tMultitags Accuracy: {:.2f}%".format(multitags_acc * 100))
    print("\tUnseen words Accuracy: {:.2f}%".format(unseen_acc * 100))
    print("\tTop 4 Wrong Word-Tag Predictions: {}".format(utilities.topk_wordtagcounter(wrong_wordtagcounter, k=4)))
    print("\tTop 4 Correct Word-Tag Predictions: {}".format(utilities.topk_wordtagcounter(correct_wordtagcounter, k=4)))
    
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHPE HMM AI Project')
    parser.add_argument('--train', dest='training_file', type=str, help='the file of the training data')
    parser.add_argument('--test', dest='test_file', type=str, help='the file of the testing data')
    parser.add_argument('--algorithm', dest='algorithm', type=str, default="baseline", help='which algorithm to run: base_viterbi, optimized_viterbi')
    args = parser.parse_args()
    
    if args.training_file == None or args.test_file == None:
        sys.exit('You must specify training file and testing file!')

    main(args)
