import argparse
import feature_extraction

from argparse import RawTextHelpFormatter
from data import get_data
from model import MaxentPosTagger
from helper import test

TRAIN_DATASET = "../data/postags_train.csv"
TEST_DATASET = "../data/postags_test.csv"

PARSER = argparse.ArgumentParser(
    description="MaxEnt Trainer", formatter_class=RawTextHelpFormatter)
PARSER.add_argument(
    "--feature_extraction_fn",
    type=str,
    default="extract_features_all",
    help="""The name of feature extraction function.
    You can choose one of the following:
    1. extract_features_all
    2. extract_features_suffix_prefix_removed
    3. extract_features_suffix_removed
    4. extract_features_w2_removed
    5. extract_features_w1w2_removed
    6. extract_features_contains_number_removed
    7. extract_features_prefix_removed
    """)
ARGS = PARSER.parse_args()


def main():
    train_data = get_data(TRAIN_DATASET)
    test_data = get_data(TEST_DATASET)
    feature_extraction_fn_name = ARGS.feature_extraction_fn
    tagger = MaxentPosTagger(
        feature_extraction_fn=getattr(feature_extraction,
                                      feature_extraction_fn_name))
    tagger.train(train_data)

    test(tagger, test_data)


if __name__ == '__main__':
    main()