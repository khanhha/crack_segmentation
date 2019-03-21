import json
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir')
    args = parser.parse_args()
    for path in Path(args.in_dir).glob('**/*.json'):
        with open(str(path), 'rb') as file:
            data = json.load(file)
            for shape in data['shapes']:
                type = shape['shape_type']
                if type == 'polygon':
                    print(f'update file {path}')
                    shape['shape_type'] = 'linestrip'

                label = shape['label']
                if label != 'c':
                    print(f'update file {path}')
                    shape['label'] = 'c'

            for shape in data['shapes']:
                type = shape['shape_type']
                assert type == 'linestrip'

        with open(str(path), 'w') as file:
            json.dump(data, file)


