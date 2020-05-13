"""
Ad-hoc script for extracting values from lidbox config yamls.
The yq program might be a better alternative in general.
"""
from argparse import ArgumentParser
from pprint import pformat
from yaml import safe_load

def main(yaml_path, keys):
    with open(yaml_path) as f:
        value = safe_load(f.read())
    for key in keys.split('.'):
        index = None
        if key.endswith(']'):
            key, index = key.split('[', 1)
            index = index.split(']', 1)[0]
        if key not in value:
            value = None
            break
        value = value[key]
        if index is not None:
            value = value[int(index)]
    if type(value) is list:
        value_str = '\n'.join(value)
    elif type(value) is dict:
        value_str = pformat(value, indent=1)
    else:
        value_str = str(value)
    return value_str

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("yaml_file", type=str)
    parser.add_argument("yaml_keys", type=str, nargs='*')
    args = parser.parse_args()
    value_str = str(None)
    for keys in args.yaml_keys:
        value_str = main(args.yaml_file, keys)
        if value_str != 'None':
            break
    print(value_str)
