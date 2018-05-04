import re


def _current_word_features(word_freqdist, sentence, i, rare_word_cutoff,
                           features):
    if word_freqdist[sentence[i]] >= rare_word_cutoff:
        # $1dditional features for 'non-rare' words
        features["w"] = sentence[i]


def _contains_number_features(number, sentence, i, features):
    if number.search(sentence[i]) is not None:
        features["contains-number"] = True


def _suffix_prefix_features(features,
                            sentence,
                            i,
                            with_suffix=True,
                            with_prefix=True):
    suffix = {"suffix(2)": sentence[i][-2:], "suffix(3)": sentence[i][-3:]}
    prefix = {"prefix(2)": sentence[i][:2], "prefix(3)": sentence[i][:3]}

    if with_prefix:
        for k, v in prefix.items():
            features.update({k: v})
    if with_suffix:
        for k, v in suffix.items():
            features.update({k: v})


def _w1_w2_t1_t2_features(i, features, sentence, history):
    if i == 0:  # $1irst word of sentence
        features.update({
            "w-1": "<START>",
            "t-1": "<START>",
            "w-2": "<START>",
            "t-2 t-1": "<START> <START>"
        })
    elif i == 1:  # $1econd word of sentence
        features.update({
            "w-1": sentence[i - 1],
            "t-1": history[i - 1],
            "w-2": "<START>",
            "t-2 t-1": "<START> %s" % (history[i - 1])
        })
    else:
        features.update({
            "w-1": sentence[i - 1],
            "t-1": history[i - 1],
            "w-2": sentence[i - 2],
            "t-2 t-1": "%s %s" % (history[i - 2], history[i - 1])
        })


def _w1w2_features(word_freqdist, sentence, i, rare_word_cutoff, features):
    for inc in [1, 2]:
        try:
            if word_freqdist[sentence[i + inc]] < rare_word_cutoff:
                features["w+%i" % (inc)] = "UNK"
            else:
                features["w+%i" % (inc)] = sentence[i + inc]
        except IndexError:
            features["w+%i" % (inc)] = "<END>"


def extract_features_all(sentence,
                         i,
                         history,
                         word_freqdist,
                         rare_word_cutoff=5):
    features = {}
    number = re.compile("\d")

    _w1_w2_t1_t2_features(i, features, sentence, history)

    _w1w2_features(word_freqdist, sentence, i, rare_word_cutoff, features)

    _current_word_features(word_freqdist, sentence, i, rare_word_cutoff,
                           features)

    _suffix_prefix_features(features, sentence, i)
    _contains_number_features(number, sentence, i, features)

    return features


# $1EATURE ABLATION
def extract_features_suffix_prefix_removed(sentence,
                                           i,
                                           history,
                                           word_freqdist,
                                           rare_word_cutoff=5):
    """Remove suffix and prefix features"""
    features = {}
    number = re.compile("\d")

    _w1_w2_t1_t2_features(i, features, sentence, history)

    _w1w2_features(word_freqdist, sentence, i, rare_word_cutoff, features)

    _current_word_features(word_freqdist, sentence, i, rare_word_cutoff,
                           features)

    _contains_number_features(number, sentence, i, features)

    return features


def extract_features_suffix_removed(sentence,
                                    i,
                                    history,
                                    word_freqdist,
                                    rare_word_cutoff=5):
    """Remove suffix"""
    features = {}
    number = re.compile("\d")

    _w1_w2_t1_t2_features(i, features, sentence, history)

    _w1w2_features(word_freqdist, sentence, i, rare_word_cutoff, features)

    _current_word_features(word_freqdist, sentence, i, rare_word_cutoff,
                           features)

    _suffix_prefix_features(features, sentence, i, with_suffix=False)

    _contains_number_features(number, sentence, i, features)

    return features


def extract_features_w2_removed(sentence,
                                i,
                                history,
                                word_freqdist,
                                rare_word_cutoff=5):
    """Remove w+2"""
    features = {}
    number = re.compile("\d")

    _w1_w2_t1_t2_features(i, features, sentence, history)

    for inc in [1]:
        try:
            if word_freqdist[sentence[i + inc]] < rare_word_cutoff:
                features["w+%i" % (inc)] = "UNK"
            else:
                features["w+%i" % (inc)] = sentence[i + inc]
        except IndexError:
            features["w+%i" % (inc)] = "<END>"

    _current_word_features(word_freqdist, sentence, i, rare_word_cutoff,
                           features)

    _suffix_prefix_features(features, sentence, i)

    _contains_number_features(number, sentence, i, features)

    return features


def extract_features_w1w2_removed(sentence,
                                  i,
                                  history,
                                  word_freqdist,
                                  rare_word_cutoff=5):
    """Remove w+1 and w+2"""
    features = {}
    number = re.compile("\d")

    _w1_w2_t1_t2_features(i, features, sentence, history)

    _current_word_features(word_freqdist, sentence, i, rare_word_cutoff,
                           features)

    _suffix_prefix_features(features, sentence, i)

    _contains_number_features(number, sentence, i, features)

    return features


def extract_features_contains_number_removed(sentence,
                                             i,
                                             history,
                                             word_freqdist,
                                             rare_word_cutoff=5):
    """Remove contains number"""
    features = {}

    _w1_w2_t1_t2_features(i, features, sentence, history)

    _w1w2_features(word_freqdist, sentence, i, rare_word_cutoff, features)

    _current_word_features(word_freqdist, sentence, i, rare_word_cutoff,
                           features)

    _suffix_prefix_features(features, sentence, i)

    return features


def extract_features_prefix_removed(sentence,
                                    i,
                                    history,
                                    word_freqdist,
                                    rare_word_cutoff=5):
    features = {}
    number = re.compile("\d")

    _w1_w2_t1_t2_features(i, features, sentence, history)

    _w1w2_features(word_freqdist, sentence, i, rare_word_cutoff, features)

    _current_word_features(word_freqdist, sentence, i, rare_word_cutoff,
                           features)

    _suffix_prefix_features(features, sentence, i, with_preffix=False)
    _contains_number_features(number, sentence, i, features)

    return features
