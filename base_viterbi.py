# import math
from collections import defaultdict, Counter
from math import log

epsilon_for_pt = 1e-5
emit_epsilon = 1e-10   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    param: sentences
    return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob_known = defaultdict(lambda: defaultdict(lambda: 0))  # {tag: {word: # }} for known words
    # emit_prob_unknown = defaultdict(lambda: 0)  # {tag: # } for unknown words
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # Input the training set, output the formatted probabilities according to data statistics.
    tag_count = defaultdict(int)
    tag_pair_count = defaultdict(lambda: defaultdict(int))
    tag_word_count = defaultdict(lambda: defaultdict(int))

    for sentence in sentences:
        prev_tag = None
        for word, tag in sentence:
            tag_count[tag] += 1
            tag_pair_count[prev_tag][tag] += 1
            tag_word_count[tag][word] += 1
            prev_tag = tag

    alpha = 1e-7  # smoothing constant
    total_tags = len(tag_count)

    # initial probabilities
    for tag, count in tag_count.items():
        init_prob[tag] = (count + alpha) / (len(sentences) + alpha * total_tags)

    # known emission probabilities
    for tag, words in tag_word_count.items():
        total_words = len(words)
        for word, count in words.items():
            emit_prob_known[tag][word] = (tag_word_count[tag][word] + alpha) / (tag_count[tag] + alpha * (total_words + 1))

    # unknown emission probabilities
    # for tag in tag_count:
    #     total_words = len(words)
    #     emit_prob_unknown[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # transition probabilities
    for tag_i in tag_count:
        for tag_j in tag_count:
            trans_prob[tag_i][tag_j] = (tag_pair_count[tag_i][tag_j] + alpha) / (tag_count[tag_i] + alpha * total_tags)

    # hapax
    hapax_tag_probs = defaultdict(lambda: 0)
    hapax_words = {}

    for tag, words in tag_word_count.items():
        for word, count in words.items():
            if count == 1:
                hapax_words[word] = tag
    
    hapax_tag_count = defaultdict(int)
    hapax_total_words = 0
    for word in hapax_words:
        for tag in tag_count:
            if hapax_words[word] == tag:
                hapax_tag_count[tag] += 1
                hapax_total_words += 1
    hapax_total_tags = len(hapax_tag_count)

    for tag in tag_count:
        hapax_smoothing = hapax_tag_count[tag] / hapax_total_words
        total_words = 1000
        if hapax_smoothing != 0:
            hapax_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (hapax_total_tags + 1))
        else:
            hapax_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (hapax_total_tags + 1))

    # print(tag_count)
    # print(hapax_tag_count)
    # print(hapax_words)
    # print(emit_prob_known)
    # print(emit_prob_unknown)
    # print(hapax_tag_probs)
    # print(hapax_total_words)
    return init_prob, emit_prob_known, trans_prob, hapax_tag_probs

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob_known, trans_prob, hapax_tag_probs):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob_known: Emission probabilities
    :param trans_prob: Transition probabilities
    :param hapax_tag_probs: Hapax probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # implement one step of trellis computation at column (i)

    # first column has a special case
    if i == 0:
        for tag in emit_prob_known:
            if emit_prob_known[tag][word] == 0:
                log_prob_emit_known = log(hapax_tag_probs[tag])
            else:
                log_prob_emit_known = log(emit_prob_known[tag][word])
            log_prob_trans = log(trans_prob["START"][tag])
            log_prob[tag] = log_prob_emit_known + log_prob_trans
            predict_tag_seq[tag] = ["START"]

    # all other columns
    else:
        for tag in emit_prob_known:
            max_log_prob = float('-inf')
            best_prev_tag = None
            for prev_tag in emit_prob_known:
                # CRUCIAL
                if emit_prob_known[tag][word] == 0:
                    log_prob_emit_known = log(hapax_tag_probs[tag])
                else:
                    log_prob_emit_known = log(emit_prob_known[tag][word])

                log_prob_trans = log(trans_prob[prev_tag][tag])
                prev_log_prob = prev_prob[prev_tag]
                total_log_prob = prev_log_prob + log_prob_emit_known + log_prob_trans

                if total_log_prob > max_log_prob:
                    max_log_prob = total_log_prob
                    best_prev_tag = prev_tag
            log_prob[tag] = max_log_prob
            predict_tag_seq[tag] = prev_predict_tag_seq[best_prev_tag] + [tag]
    # print(emit_prob_known)
    # print(hapax_tag_probs)
    
    return log_prob, predict_tag_seq

def base_viterbi(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob_known, trans_prob, hapax_tag_probs = training(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob_known:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob_known, trans_prob, hapax_tag_probs)
        # print(hapax_tag_probs)

        # according to the storage of probabilities and sequences, get the final prediction.
        best_tag = max(log_prob, key=log_prob.get)
        best_tag_seq = predict_tag_seq[best_tag]
        predicted_tags = [(word, tag) for word, tag in zip(sentence, best_tag_seq)]
        predicts.append(predicted_tags)
        
    return predicts
