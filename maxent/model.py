from nltk import FreqDist, untag
from nltk.classify.maxent import MaxentClassifier


class MaxentPosTagger(object):
    """
    MaxentPosTagger is a part-of-speech tagger based on Maximum Entropy models.
    """

    def __init__(self,
                 feature_extraction_fn,
                 algorithm='megam',
                 rare_word_cutoff=5,
                 rare_feat_cutoff=5,
                 trace=3):
        self._feature_extraction_fn = feature_extraction_fn
        self._algorithm = algorithm
        self._rare_word_cutoff = rare_word_cutoff
        self._rare_feat_cutoff = rare_feat_cutoff
        self._trace = trace

    def train(self, train_sents, **cutoffs):
        self.word_freqdist = self.gen_word_freqs(train_sents)
        featuresets = self.gen_featsets(train_sents, self._rare_word_cutoff)

        print("Start training maxent...")
        self.classifier = MaxentClassifier.train(featuresets, self._algorithm,
                                                 self._trace, **cutoffs)
        print("Finish training maxent!")

    def gen_word_freqs(self, train_sents):
        word_freqdist = FreqDist()
        for tagged_sent in train_sents:
            for (word, _tag) in tagged_sent:
                word_freqdist[word] += 1
        return word_freqdist

    def gen_featsets(self, train_sents, rare_word_cutoff):
        featuresets = []
        for tagged_sent in train_sents:
            history = []
            untagged_sent = untag(tagged_sent)
            for (i, (_word, tag)) in enumerate(tagged_sent):
                featuresets.append((self._feature_extraction_fn(
                    untagged_sent, i, history, self.word_freqdist,
                    rare_word_cutoff), tag))
                history.append(tag)
        return featuresets

    def tag(self, sentence, rare_word_cutoff=5):
        history = []
        for i in range(len(sentence)):
            featureset = self._feature_extraction_fn(
                sentence, i, history, self.word_freqdist, rare_word_cutoff)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)
