from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def test(tagger, test_data):
    pred = []
    real_labels = []

    for td in test_data:
        words, tags = zip(*td)

        result = tagger.tag(words)
        for idx, r in enumerate(result):
            pred.append(r[1])
            real_labels.append(tags[idx])

    print("==============================")
    print("Accuracy: ", accuracy_score(real_labels, pred), "\n")
    print(confusion_matrix(real_labels, pred))
    print(classification_report(real_labels, pred))
    print("==============================")