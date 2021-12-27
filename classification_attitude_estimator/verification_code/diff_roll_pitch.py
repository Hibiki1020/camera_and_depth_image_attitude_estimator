import csv
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./diff_roll_pitch.py")

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default='../pyyaml/diff_roll_pitch.yaml',
    )

    FLAGS, unparsed = parser.parse_known_args()

    #Load .yaml file
    try:
        print("Opening config file %s", FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening config file %s",FLAGS.config)
        quit()