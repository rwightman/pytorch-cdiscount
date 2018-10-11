import os
import argparse
import struct
import lmdb
from dataset import find_inputs
import time

parser = argparse.ArgumentParser(description='Process cdiscount datasets')
parser.add_argument('data', metavar='DIR',
                    help='dir of images')

def main():
    args = parser.parse_args()

    start = time.time()
    inputs = find_inputs(args.data)
    inputs = sorted(inputs, key=lambda x: (x[0], x[1]))
    print('Inputs via FS took', time.time() - start)

    start = time.time()
    env = lmdb.open(args.data, sync=False, readahead=False, readonly=True, lock=False)
    entries = []
    with env.begin() as txn:
        cursor = txn.cursor()
        k, v = cursor.first()
        print(k, len(v))
        k, v = cursor.next()
        print(k, len(v))
        k, v = cursor.last()
        print(k, len(v))
        cursor.first()
        for k in cursor.iternext(keys=True, values=False):
            kk = struct.unpack('>IB', k)
            entries.append(kk)
    print(len(entries))
    print('Inputs via LMDB took', time.time() - start)


if __name__ == '__main__':
    main()
