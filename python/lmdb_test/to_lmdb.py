import bson
import numpy as np
import pandas as pd
import os
import argparse
import struct
import lmdb
from dataset import find_inputs

parser = argparse.ArgumentParser(description='Process cdiscount datasets')
parser.add_argument('data', metavar='DIR',
                    help='dir of images')
parser.add_argument('-t', default='train', type=str, metavar='TYPE',
                    help='Type of dataset, "train" or "test"')
parser.add_argument('-o', '--output', default='./output', type=str, metavar='DIR',
                    help='Output directory')


def main():
    args = parser.parse_args()

    output_dir = os.path.join(args.output, args.t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inputs = find_inputs(args.data)

    inputs = sorted(inputs, key=lambda x: (x[0], x[1]))
    if not inputs:
        exit(1)

    chunk_size = 10000
    num_inputs = len(inputs)
    print(num_inputs)
    env = lmdb.open(output_dir, map_size=2**38, sync=False)

    for chunk_idx in range(0, len(inputs), chunk_size):
        with env.begin(write=True, buffers=True) as txn:
            print(chunk_idx, min(chunk_idx + chunk_size, num_inputs))
            for i in inputs[chunk_idx:min(chunk_idx + chunk_size, num_inputs)]:
                #print(i)
                key = struct.pack('>IB', i[0], i[1])
                with open(i[2], 'rb') as f:
                    value = f.read()
                    txn.put(key, value)


if __name__ == '__main__':
    main()