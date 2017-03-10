#! /usr/bin/env python3

import os
import pandas as pd
import argparse
import numpy as np

def main():
  parser = argparse.ArgumentParser(
      description='combine downloaded data into one .npy file',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--results-folder', help='folder to read .npy files from', default='result')
  parser.add_argument('--ds-size', help='desired size of the dataset', default=100_000, type=int)
  parser.add_argument('-c', '--csv', nargs='+', help='csv files to read', default=['DR12_spec_phot_sample.csv'])
  parser.add_argument('--out-csv', help='how many threads to use for downloading', default='result/SDSS_DR12_100K.csv')
  parser.add_argument('--out-npy', help='file name to store resulting dataset in', default='result/dataset.npy')
  args = parser.parse_args()


  assert args.out_npy.endswith('.npy')
  assert args.out_csv.endswith('.csv')


  dfs = [(fname, pd.read_csv(fname, dtype={"objID": "object"})) for fname in args.csv]
  
  df = dfs[0][1][:0]

  count = 0
  done = set(d.split(".")[2] for d in os.listdir(args.results_folder) if d.count('.') >= 2)
  ds = np.zeros((args.ds_size, 5, 48, 48), dtype=np.float64)

  for fname, dfi in dfs:
    print(fname)
    for i, r in dfi.iterrows():
        outname = 'result/{}.48x48.{}.npy'.format(r['class'], r['objID'])
        if r['objID'] not in done:
          print('X', end='', flush=True)
          continue
        df.loc[count] = r
        ds[count] = np.load(outname)
        count += 1
        if count == args.ds_size:
            print('DONE')
            break
        if count % 100 == 99:
            print('.', end='', flush=True)
        if count % 10_000 == 9_999:
            print(count)
    else:  # "nobreak" else
      # means the inner loop did NOT break, i.e. we still have work to do
      continue
    # this means we broke the last loop, i.e. we have enough data, OR we ran out of csv files
    break


  if count < args.ds_size:
    print("WARNING: only found {} data points when {} were asked for")
    ds = df[:count]

  df.to_csv(args.out_csv, index=False)
  np.save(args.out_npy, ds)

if __name__ == '__main__':
  main()

