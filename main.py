import argparse
import os
import torch
from torch import nn
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    args = parser.parse_args()