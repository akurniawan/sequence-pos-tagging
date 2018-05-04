import csv


def get_data(path):
    result = []
    with open(path, "rt") as csvfile:
        train_reader = csv.reader(csvfile)
        next(train_reader)
        for row in train_reader:
            text = row[0].split()
            tags = row[1].split()
            pair = []
            for te, ta in zip(text, tags):
                pair.append((te, ta))
            result.append(pair)

    return result
