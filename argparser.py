import argparse
import textwrap

project_argparser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# CONFIGURATION
project_argparser.add_argument(
    "--out_directory",
    dest="out_directory",
    default=".",
)

project_argparser.add_argument(
    "--train_full_model",
    dest="train_full_model",
    action="store_true",
    default=False,
)
project_argparser.add_argument(
    "--input_file",
    dest="input_file",
    default="train_technical_test.csv"
)

project_argparser.add_argument(
    "--previous_accuracy",
    dest="previous_accuracy",
    default=0,
)

project_argparser.add_argument(
    "--predict",
    dest="predict",
    action="store_true",
    default=False,
)

project_argparser.add_argument(
    "--test_file",
    dest="test_file",
    default="test_technical_test.csv",
)

project_argparser.add_argument(
    "--model_file",
    dest="model_file",
)
