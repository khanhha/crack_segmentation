import numpy as np
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input_dir', help='input annotated directory')
    parser.add_argument('-output_dir', help='output dataset directory')
    args = parser.parse_args()

    for path in Path(args.input_dir).glob('*.jpg'):
