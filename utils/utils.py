import argparse
import json


def parse_dict(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid dictionary format. Use JSON format.")


def pretty_list(list):
    text = str(["{:.2f}".format(i) for i in list])
    return text