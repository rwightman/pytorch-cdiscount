import os
import argparse
import struct
import lmdb
import csv
from dataset import find_inputs
import time
import numpy as np
import pandas as pd
import zlib

parser = argparse.ArgumentParser(description='Process cdiscount datasets')
parser.add_argument('data', metavar='DIR',
                    help='dir of images')
parser.add_argument('--categories', default='', type=str, metavar='PATH',
                    help='path to category map file')

def main():
    args = parser.parse_args()
    start = time.time()

    categories = []
    if os.path.isfile(args.categories):
        category_df = pd.read_csv(args.categories)
        categories = category_df.category_id
        categories = sorted(categories)
    else:
        print('WARNIGN: No category mapping found, writing raw label indices into output.')

    cf = open(os.path.join('./', 'results-ensemble.csv'), mode='w')
    writer = csv.writer(cf)
    writer.writerow(['_id', 'category_id'])

    dirs = args.data.split(',')
    envs = [lmdb.open(d, sync=False, readahead=False, readonly=True, lock=False) for d in dirs]
    txns = [e.begin() for e in envs]
    cursors = [t.cursor() for t in txns]
    num_sources = len(envs)
    probs = []
    prev_prod_id = -1
    written = 0

    def _write_result():
        if num_sources == len(probs):
            result = np.mean(probs, axis=0)
        else:
            probs_arr = np.array(probs)
            for i in range(num_sources):
                probs_arr[-i] *= 2.0
            result = np.sum(probs_arr, axis=0)
        top1_label = np.argmax(result)
        writer.writerow([prev_prod_id, categories[top1_label] if categories else top1_label])

    try:
        iters = [c.iternext(keys=True, values=True) for c in cursors]
        while True:
            fetches = [i.__next__() for i in iters]
            for i, (k, v) in enumerate(fetches):
                prod_id, img_id = struct.unpack('>IB', k)
                if written % 1000 == 0 and i == 0:
                    print(prod_id, img_id)
                if prev_prod_id > 0 and (i == 0 and prod_id != prev_prod_id):
                    _write_result()
                    written += 1
                    probs = []
                va = np.frombuffer(zlib.decompress(v), dtype=np.float32)
                prev_prod_id = prod_id
                probs.append(va)

    except StopIteration:
        print('STOP')
        pass

    if probs:
        _write_result()
        written += 1

    print(written)
    print('Inputs via LMDB took', time.time() - start)


if __name__ == '__main__':
    main()