import argparse
import textwrap

project_argparser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# CONFIGURATION
project_argparser.add_argument(
    "--out_directory",
    dest="out_directory",
    default=".",
)
'''
project_argparser.add_argument(
    "--cross_val",
    dest="cross_val",
    default=False,
)
'''
project_argparser.add_argument(
    "--train_full_model",
    dest="train_full_model",
    default=False,
)
project_argparser.add_argument(
    "--input-file",
    dest="input_file",
    default="train_technical_test.csv"
)

project_argparser.add_argument(
    "--previous-accuracy",
    dest="previous_accuracy",
    default=0,
)

project_argparser.add_argument(
    "--predict",
    dest="predict",
    default=False,
)

project_argparser.add_argument(
    "--test-file",
    dest="test_file",
    default="test_technical_test.csv",
)
