import bson
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Process cdiscount datasets')
parser.add_argument('data', metavar='DIR',
                    help='dir of dataset BSON files')
parser.add_argument('-t', default='train', type=str, metavar='TYPE',
                    help='Type of dataset, "train" or "test"')
parser.add_argument('-c', '--categories', default='category_names.csv', type=str, metavar='PATH',
                    help='category csv file path')
parser.add_argument('-o', '--output', default='./output', type=str, metavar='DIR',
                    help='Output directory')


def main():
    args = parser.parse_args()

    output_dir = os.path.join(args.output, args.t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train = False if args.t == 'test' else True
    num_products = 7069896 if train else 1768182

    if train:
        categories = pd.read_csv(os.path.join(args.data, args.categories), index_col='category_id')
        for category in tqdm(categories.index):
            os.mkdir(os.path.join(output_dir, str(category)))

    bson_file = args.t + '.bson'
    bar = tqdm(total=num_products)
    with open(os.path.join(args.data, bson_file), 'rb') as fbson:
        data = bson.decode_file_iter(fbson)

        prod_metadata = []
        for c, d in enumerate(data):
            if train:
                category_id = d['category_id']
            prod_id = d['_id']
            for i, img in enumerate(d['imgs']):
                if train:
                    fname = os.path.join(output_dir, str(category_id), '{}-{}.jpg'.format(prod_id, i))
                else:
                    fname = os.path.join(output_dir, '{}-{}.jpg'.format(prod_id, i))
                with open(fname, 'wb') as f:
                    f.write(img['picture'])
            if train:
                prod_metadata.append((prod_id, category_id, len(d['imgs'])))
            else:
                prod_metadata.append((prod_id, len(d['imgs'])))
            bar.update()

        cols = ['prod_id', 'category_id', 'num_images'] if train else ['prod_id', 'num_images']
        prodd = pd.DataFrame(prod_metadata, columns=cols)
        prodd.to_csv(os.path.join(output_dir, 'prod_metadata.csv'))


if __name__ == '__main__':
    main()