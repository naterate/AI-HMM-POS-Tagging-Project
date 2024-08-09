from collections import defaultdict, Counter
from math import log

epsilon_for_pt = 1e-5
emit_epsilon = 1e-10   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob_known = defaultdict(lambda: defaultdict(lambda: 0))  # {tag: {word: # }} for known words
    emit_prob_unknown = defaultdict(lambda: 0)  # {tag: # } for unknown words
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
    for tag in tag_count:
        total_words = len(words)
        emit_prob_unknown[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # transition probabilities
    for tag_i in tag_count:
        for tag_j in tag_count:
            trans_prob[tag_i][tag_j] = (tag_pair_count[tag_i][tag_j] + alpha) / (tag_count[tag_i] + alpha * total_tags)

    # hapax
    hapax_tag_probs = defaultdict(lambda: 0)
    hapax_words = {}
    hapax_words_temp = {}
    word_count = {}

    for tag, words in tag_word_count.items():
        for word, count in words.items():
            if count == 1:
                hapax_words_temp[word] = tag
                word_count[word] = 0

    for tag, words in tag_word_count.items():
        for word, count in words.items():
            if count == 1:
                word_count[word] += 1
    
    for word in word_count:
        if word_count[word] == 1:
            hapax_words[word] = hapax_words_temp[word]
    
    # -ing section
    ing_words = {}
    for word in hapax_words:
        if word.endswith("ing"):
            ing_words[word] = hapax_words[word]
    
    ing_tag_count = defaultdict(int)
    ing_total_words = 0 
    for word in ing_words:
        for tag in tag_count:
            if ing_words[word] == tag:
                ing_tag_count[tag] += 1
                ing_total_words += 1

    ing_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = ing_tag_count[tag] / max(ing_total_words, 1)
        total_words = 500
        for word in ing_words:
            if hapax_smoothing != 0:
                ing_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                ing_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -ly section
    ly_words = {}
    for word in hapax_words:
        if word.endswith("ly"):
            ly_words[word] = hapax_words[word]
    
    ly_tag_count = defaultdict(int)
    ly_total_words = 0 
    for word in ly_words:
        for tag in tag_count:
            if ly_words[word] == tag:
                ly_tag_count[tag] += 100
                ly_total_words += 1

    ly_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = ly_tag_count[tag] / max(ly_total_words, 1)
        total_words = 500
        for word in ly_words:
            if hapax_smoothing != 0:
                ly_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                ly_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -ion section
    ion_words = {}
    for word in hapax_words:
        if word.endswith("ion"):
            ion_words[word] = hapax_words[word]
    
    ion_tag_count = defaultdict(int)
    ion_total_words = 0 
    for word in ion_words:
        for tag in tag_count:
            if ion_words[word] == tag:
                ion_tag_count[tag] += 1
                ion_total_words += 1

    ion_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = ion_tag_count[tag] / max(ion_total_words, 1)
        total_words = 500
        for word in ion_words:
            if hapax_smoothing != 0:
                ion_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                ion_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -er section
    er_words = {}
    for word in hapax_words:
        if word.endswith("er"):
            er_words[word] = hapax_words[word]
    
    er_tag_count = defaultdict(int)
    er_total_words = 0 
    for word in er_words:
        for tag in tag_count:
            if er_words[word] == tag:
                er_tag_count[tag] += 1
                er_total_words += 1

    er_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = er_tag_count[tag] / max(er_total_words, 1)
        total_words = 500
        for word in er_words:
            if hapax_smoothing != 0:
                er_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                er_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -en section
    en_words = {}
    for word in hapax_words:
        if word.endswith("en"):
            en_words[word] = hapax_words[word]
    
    en_tag_count = defaultdict(int)
    en_total_words = 0 
    for word in en_words:
        for tag in tag_count:
            if en_words[word] == tag:
                en_tag_count[tag] += 1
                en_total_words += 1

    en_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = en_tag_count[tag] / max(en_total_words, 1)
        total_words = 500
        for word in en_words:
            if hapax_smoothing != 0:
                en_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                en_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -ity section
    ity_words = {}
    for word in hapax_words:
        if word.endswith("ity"):
            ity_words[word] = hapax_words[word]
    
    ity_tag_count = defaultdict(int)
    ity_total_words = 0 
    for word in ity_words:
        for tag in tag_count:
            if ity_words[word] == tag:
                ity_tag_count[tag] += 1
                ity_total_words += 1

    ity_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = ity_tag_count[tag] / max(ity_total_words, 1)
        total_words = 500
        for word in ity_words:
            if hapax_smoothing != 0:
                ity_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                ity_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -ness section
    ness_words = {}
    for word in hapax_words:
        if word.endswith("ness"):
            ness_words[word] = hapax_words[word]
    
    ness_tag_count = defaultdict(int)
    ness_total_words = 0 
    for word in ness_words:
        for tag in tag_count:
            if ness_words[word] == tag:
                ness_tag_count[tag] += 1
                ness_total_words += 1

    ness_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = ness_tag_count[tag] / max(ness_total_words, 1)
        total_words = 500
        for word in ness_words:
            if hapax_smoothing != 0:
                ness_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                ness_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -ed section
    ed_words = {}
    for word in hapax_words:
        if word.endswith("ed"):
            ed_words[word] = hapax_words[word]
    
    ed_tag_count = defaultdict(int)
    ed_total_words = 0 
    for word in ed_words:
        for tag in tag_count:
            if ed_words[word] == tag:
                ed_tag_count[tag] += 1
                ed_total_words += 1

    ed_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = ed_tag_count[tag] / max(ed_total_words, 1)
        total_words = 500
        for word in ed_words:
            if hapax_smoothing != 0:
                ed_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                ed_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -es section
    es_words = {}
    for word in hapax_words:
        if word.endswith("es"):
            es_words[word] = hapax_words[word]
    
    es_tag_count = defaultdict(int)
    es_total_words = 0 
    for word in es_words:
        for tag in tag_count:
            if es_words[word] == tag:
                es_tag_count[tag] += 1
                es_total_words += 1

    es_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = es_tag_count[tag] / max(es_total_words, 1)
        total_words = 500
        for word in es_words:
            if hapax_smoothing != 0:
                es_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                es_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -al section
    al_words = {}
    for word in hapax_words:
        if word.endswith("al"):
            al_words[word] = hapax_words[word]
    
    al_tag_count = defaultdict(int)
    al_total_words = 0 
    for word in al_words:
        for tag in tag_count:
            if al_words[word] == tag:
                al_tag_count[tag] += 1
                al_total_words += 1

    al_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = al_tag_count[tag] / max(al_total_words, 1)
        total_words = 500
        for word in al_words:
            if hapax_smoothing != 0:
                al_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                al_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -ive section
    ive_words = {}
    for word in hapax_words:
        if word.endswith("ive"):
            ive_words[word] = hapax_words[word]
    
    ive_tag_count = defaultdict(int)
    ive_total_words = 0 
    for word in ive_words:
        for tag in tag_count:
            if ive_words[word] == tag:
                ive_tag_count[tag] += 1
                ive_total_words += 1

    ive_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = ive_tag_count[tag] / max(ive_total_words, 1)
        total_words = 500
        for word in ive_words:
            if hapax_smoothing != 0:
                ive_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                ive_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -ent section
    ent_words = {}
    for word in hapax_words:
        if word.endswith("ent"):
            ent_words[word] = hapax_words[word]
    
    ent_tag_count = defaultdict(int)
    ent_total_words = 0 
    for word in ent_words:
        for tag in tag_count:
            if ent_words[word] == tag:
                ent_tag_count[tag] += 1
                ent_total_words += 1

    ent_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = ent_tag_count[tag] / max(ent_total_words, 1)
        total_words = 500
        for word in ent_words:
            if hapax_smoothing != 0:
                ent_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                ent_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -ic section
    ic_words = {}
    for word in hapax_words:
        if word.endswith("ic"):
            ic_words[word] = hapax_words[word]
    
    ic_tag_count = defaultdict(int)
    ic_total_words = 0 
    for word in ic_words:
        for tag in tag_count:
            if ic_words[word] == tag:
                ic_tag_count[tag] += 1
                ic_total_words += 1

    ic_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = ic_tag_count[tag] / max(ic_total_words, 1)
        total_words = 500
        for word in ic_words:
            if hapax_smoothing != 0:
                ic_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                ic_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -ous section
    ous_words = {}
    for word in hapax_words:
        if word.endswith("ous"):
            ous_words[word] = hapax_words[word]
    
    ous_tag_count = defaultdict(int)
    ous_total_words = 0 
    for word in ous_words:
        for tag in tag_count:
            if ous_words[word] == tag:
                ous_tag_count[tag] += 1
                ous_total_words += 1

    ous_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = ous_tag_count[tag] / max(ous_total_words, 1)
        total_words = 500
        for word in ous_words:
            if hapax_smoothing != 0:
                ous_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                ous_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -able section
    able_words = {}
    for word in hapax_words:
        if word.endswith("able"):
            able_words[word] = hapax_words[word]
    
    able_tag_count = defaultdict(int)
    able_total_words = 0 
    for word in able_words:
        for tag in tag_count:
            if able_words[word] == tag:
                able_tag_count[tag] += 1
                able_total_words += 1

    able_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = able_tag_count[tag] / max(able_total_words, 1)
        total_words = 500
        for word in able_words:
            if hapax_smoothing != 0:
                able_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                able_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))
    
    # inter- section
    inter_words = {}
    for word in hapax_words:
        if word.startswith("inter"):
            inter_words[word] = hapax_words[word]
    
    inter_tag_count = defaultdict(int)
    inter_total_words = 0 
    for word in inter_words:
        for tag in tag_count:
            if inter_words[word] == tag:
                inter_tag_count[tag] += 1
                inter_total_words += 1

    inter_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = inter_tag_count[tag] / max(inter_total_words, 1)
        total_words = 500
        for word in inter_words:
            if hapax_smoothing != 0:
                inter_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                inter_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -co section
    co_words = {}
    for word in hapax_words:
        if word.endswith("co"):
            co_words[word] = hapax_words[word]
    
    co_tag_count = defaultdict(int)
    co_total_words = 0 
    for word in co_words:
        for tag in tag_count:
            if co_words[word] == tag:
                co_tag_count[tag] += 1
                co_total_words += 1

    co_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = co_tag_count[tag] / max(co_total_words, 1)
        total_words = 500
        for word in co_words:
            if hapax_smoothing != 0:
                co_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                co_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -at section
    at_words = {}
    for word in hapax_words:
        if word.endswith("at"):
            at_words[word] = hapax_words[word]
    
    at_tag_count = defaultdict(int)
    at_total_words = 0 
    for word in at_words:
        for tag in tag_count:
            if at_words[word] == tag:
                at_tag_count[tag] += 1
                at_total_words += 1

    at_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = at_tag_count[tag] / max(at_total_words, 1)
        total_words = 500
        for word in at_words:
            if hapax_smoothing != 0:
                at_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                at_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -ful section
    ful_words = {}
    for word in hapax_words:
        if word.endswith("ful"):
            ful_words[word] = hapax_words[word]
    
    ful_tag_count = defaultdict(int)
    ful_total_words = 0 
    for word in ful_words:
        for tag in tag_count:
            if ful_words[word] == tag:
                ful_tag_count[tag] += 1
                ful_total_words += 1

    ful_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = ful_tag_count[tag] / max(ful_total_words, 1)
        total_words = 500
        for word in ful_words:
            if hapax_smoothing != 0:
                ful_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                ful_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -a section
    a_words = {}
    for word in hapax_words:
        if word.endswith("a"):
            a_words[word] = hapax_words[word]
    
    a_tag_count = defaultdict(int)
    a_total_words = 0 
    for word in a_words:
        for tag in tag_count:
            if a_words[word] == tag:
                a_tag_count[tag] += 1
                a_total_words += 1

    a_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = a_tag_count[tag] / max(a_total_words, 1)
        total_words = 500
        for word in a_words:
            if hapax_smoothing != 0:
                a_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                a_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -i section
    i_words = {}
    for word in hapax_words:
        if word.endswith("i"):
            i_words[word] = hapax_words[word]
    
    i_tag_count = defaultdict(int)
    i_total_words = 0 
    for word in i_words:
        for tag in tag_count:
            if i_words[word] == tag:
                i_tag_count[tag] += 1
                i_total_words += 1

    i_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = i_tag_count[tag] / max(i_total_words, 1)
        total_words = 500
        for word in i_words:
            if hapax_smoothing != 0:
                i_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                i_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    # -s section
    s_words = {}
    for word in hapax_words:
        if word.endswith("s"):
            s_words[word] = hapax_words[word]
    
    s_tag_count = defaultdict(int)
    s_total_words = 0 
    for word in s_words:
        for tag in tag_count:
            if s_words[word] == tag:
                s_tag_count[tag] += 1
                s_total_words += 1

    s_tag_probs = defaultdict(lambda: 0)
    for tag in tag_count:
        hapax_smoothing = s_tag_count[tag] / max(s_total_words, 1)
        total_words = 500
        for word in s_words:
            if hapax_smoothing != 0:
                s_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (total_tags + 1))
            else:
                s_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (total_tags + 1))

    hapax_tag_count = defaultdict(int)
    hapax_total_words = 0
    for word in hapax_words:
        for tag in tag_count:
            if hapax_words[word] == tag:
                hapax_tag_count[tag] += 1
                hapax_total_words += 1
    hapax_total_tags = len(hapax_tag_count)

    for tag in tag_count:
        hapax_smoothing = hapax_tag_count[tag] / max(hapax_total_words, 1)
        total_words = 500
        for word in hapax_words:
            if hapax_smoothing != 0:
                hapax_tag_probs[tag] = (hapax_smoothing*total_words * alpha) / (tag_count[tag] + (hapax_smoothing*total_words * alpha) * (hapax_total_tags + 1))
            else:
                hapax_tag_probs[tag] = alpha / (tag_count[tag] + alpha * (hapax_total_tags + 1))

    # print(tag_count)
    # print(hapax_tag_count)
    # print(ing_tag_count)
    # print(hapax_words)
    # print(emit_prob_known)
    # print(emit_prob_unknown)
    # print(ing_words)
    # print(hapax_tag_probs)
    # print(hapax_total_words)
    # print(ing_tag_probs)
    return (init_prob, emit_prob_known, emit_prob_unknown, trans_prob, hapax_tag_probs, 
                        ing_tag_probs, ly_tag_probs, ion_tag_probs, er_tag_probs, en_tag_probs, ity_tag_probs, ness_tag_probs, ed_tag_probs, es_tag_probs, al_tag_probs,
                        ive_tag_probs, ic_tag_probs, ous_tag_probs, able_tag_probs, inter_tag_probs, co_tag_probs, at_tag_probs, ful_tag_probs, a_tag_probs,
                        i_tag_probs, s_tag_probs)

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob_known, emit_prob_unknown, trans_prob, hapax_tag_probs, 
                        ing_tag_probs, ly_tag_probs, ion_tag_probs, er_tag_probs, en_tag_probs, ity_tag_probs, ness_tag_probs, ed_tag_probs, es_tag_probs, al_tag_probs,
                        ive_tag_probs, ic_tag_probs, ous_tag_probs, able_tag_probs, inter_tag_probs, co_tag_probs, at_tag_probs, ful_tag_probs, a_tag_probs,
                        i_tag_probs, s_tag_probs):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :params for each special case word beginning and ending
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.

    # first column has a special case
    if i == 0:
        for tag in emit_prob_known:
            if emit_prob_known[tag][word] == 0:
                if word.endswith("ing"):
                    log_prob_emit_known = log(ing_tag_probs[tag])
                elif word.endswith("ly"):
                    log_prob_emit_known = log(ly_tag_probs[tag])
                elif word.endswith("ion"):
                    log_prob_emit_known = log(ion_tag_probs[tag])
                elif word.endswith("er"):
                    log_prob_emit_known = log(er_tag_probs[tag])
                elif word.endswith("en"):
                    log_prob_emit_known = log(en_tag_probs[tag])
                elif word.endswith("ity"):
                    log_prob_emit_known = log(ity_tag_probs[tag])
                elif word.endswith("ness"):
                    log_prob_emit_known = log(ness_tag_probs[tag])
                elif word.endswith("ed"):
                    log_prob_emit_known = log(ed_tag_probs[tag])
                elif word.endswith("es"):
                    log_prob_emit_known = log(es_tag_probs[tag])
                elif word.endswith("al"):
                    log_prob_emit_known = log(al_tag_probs[tag])
                elif word.endswith("ive"):
                    log_prob_emit_known = log(ive_tag_probs[tag])
                else:
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
                if emit_prob_known[tag][word] == 0:
                    if word.endswith("ing"):
                        log_prob_emit_known = log(ing_tag_probs[tag])
                    elif word.endswith("ly"):
                        log_prob_emit_known = log(ly_tag_probs[tag])
                    elif word.endswith("ion"):
                        log_prob_emit_known = log(ion_tag_probs[tag])
                    elif word.endswith("er"):
                        log_prob_emit_known = log(er_tag_probs[tag])
                    elif word.endswith("en"):
                        log_prob_emit_known = log(en_tag_probs[tag])
                    elif word.endswith("ity"):
                        log_prob_emit_known = log(ity_tag_probs[tag])
                    elif word.endswith("ness"):
                        log_prob_emit_known = log(ness_tag_probs[tag])
                    elif word.endswith("ed"):
                        log_prob_emit_known = log(ed_tag_probs[tag])
                    elif word.endswith("es"):
                        log_prob_emit_known = log(es_tag_probs[tag])
                    elif word.endswith("al"):
                        log_prob_emit_known = log(al_tag_probs[tag])
                    elif word.endswith("ive"):
                        log_prob_emit_known = log(ive_tag_probs[tag])
                    elif word.endswith("ic"):
                        log_prob_emit_known = log(ic_tag_probs[tag])
                    elif word.endswith("ous"):
                        log_prob_emit_known = log(ous_tag_probs[tag])
                    elif word.endswith("able"):
                        log_prob_emit_known = log(able_tag_probs[tag])
                    elif word.startswith("inter"):
                        log_prob_emit_known = log(inter_tag_probs[tag])
                    elif word.endswith("co"):
                        log_prob_emit_known = log(co_tag_probs[tag])
                    elif word.endswith("at"):
                        log_prob_emit_known = log(at_tag_probs[tag])
                    elif word.endswith("ful"):
                        log_prob_emit_known = log(ful_tag_probs[tag])
                    elif word.endswith("a"):
                        log_prob_emit_known = log(a_tag_probs[tag])
                    elif word.endswith("i"):
                        log_prob_emit_known = log(i_tag_probs[tag])
                    elif word.endswith("s"):
                        log_prob_emit_known = log(s_tag_probs[tag])
                    else:
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

def optimized_viterbi(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    (init_prob, emit_prob_known, emit_prob_unknown, trans_prob, hapax_tag_probs, 
            ing_tag_probs, ly_tag_probs, ion_tag_probs, er_tag_probs, en_tag_probs, ity_tag_probs, ness_tag_probs, ed_tag_probs, es_tag_probs,
            al_tag_probs, ive_tag_probs, ic_tag_probs, ous_tag_probs, able_tag_probs, inter_tag_probs, co_tag_probs, at_tag_probs, ful_tag_probs,
            a_tag_probs, i_tag_probs, s_tag_probs) = training(train)
    
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
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob_known, emit_prob_unknown, trans_prob, hapax_tag_probs, 
                                ing_tag_probs, ly_tag_probs, ion_tag_probs, er_tag_probs, en_tag_probs, ity_tag_probs, ness_tag_probs, ed_tag_probs, es_tag_probs,
                                al_tag_probs, ive_tag_probs, ic_tag_probs, ous_tag_probs, able_tag_probs, inter_tag_probs, co_tag_probs, at_tag_probs, ful_tag_probs,
                                a_tag_probs, i_tag_probs, s_tag_probs)
        # print(hapax_tag_probs)

        # according to the storage of probabilities and sequences, get the final prediction.
        best_tag = max(log_prob, key=log_prob.get)
        best_tag_seq = predict_tag_seq[best_tag]
        predicted_tags = [(word, tag) for word, tag in zip(sentence, best_tag_seq)]
        predicts.append(predicted_tags)
        
    return predicts
