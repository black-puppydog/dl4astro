#!/usr/bin/env python

'''
Author: Edward J Kim <edward.junhyung.kim@gmail.com>

This script

- Makes an SQL query to the SDSS DR12 database (using its API at
  http://skyserver.sdss.org/dr12/en/help/docs/api.aspx) to create a catalog,

- Downloads the FITS files,

- Uses Montage (http://montage.ipac.caltech.edu/) and
  montage wrapper (http://www.astropy.org/montage-wrapper/) to align each image
  to the image in the r-band, and

- Uses Sextractor (http://www.astromatic.net/software/sextractor) to find the
  pixel position of objects, and

- Converts the fluxes in FITS files to luptitudes
  (http://www.sdss.org/dr12/algorithms/magnitudes/#asinh).

See Dockerfile at https://github.com/EdwardJKim/deeplearning4astro/tree/master/docker.
It has all packages necessary to run this notebook.

To use this script with CasJobs, see
https://github.com/EdwardJKim/dl4astro/blob/master/scripts/README.md.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
from typing import Dict

import os
import shutil

import logging

import aiohttp as aiohttp
import requests
import tempfile
import bz2
import re
import subprocess
from time import sleep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpi4py import MPI

import montage_wrapper as mw
import sys
from astropy.io import fits
from astropy import wcs

from contextlib import closing
import urllib.request
import asyncio

from multiprocessing import Pool

BANDS = 'ugriz'


def download_and_extract(source, target):
    response = urllib.request.urlopen(source)
    data = response.read()  # a `bytes` object
    with urllib.request.urlopen(source) as response, \
            open(target, 'wb') as out_file:
        data = response.read()  # a `bytes` object
        out_file.write(bz2.decompress(data))
    # print('{} --> {}'.format(source, target))

def fetch_fits(r, dirname):
    url_folder = "http://data.sdss3.org/sas/dr12/boss/photoObj/frames/{rerun}/{run}/{camcol}/".format(**r)
    filenames = [get_fits_filename(r, band) for band in BANDS]

    for fname in filenames:
      url = url_folder+fname+'.bz2'
      download_and_extract(url, os.path.join(dirname, fname))

def get_fits_filename(r, band, prefix='frame', ending='.fits'):
    return "{pref}-{band}-{run:06d}-{camcol}-{field:04d}{end}".format(band=band, pref=prefix, end=ending, **r)

def align_images(r, tmp_dir):

  frame_path = [ os.path.join(tmp_dir, get_fits_filename(r, b)) for b in BANDS]
  registered_path = [os.path.join(tmp_dir, get_fits_filename(r, b, 'registered')) for b in BANDS]

  header = os.path.join(tmp_dir, get_fits_filename(r, 'r', 'header', '.hdr'))

  mw.commands.mGetHdr(os.path.join(tmp_dir, get_fits_filename(r, 'r')), header)
  mw.reproject(frame_path, registered_path,
               header=header, exact_size=True,
               silent_cleanup=True, common=True)

def convert_catalog_to_pixels(r, dirname):
  fits_file = get_fits_filename(r, 'r', 'registered')
  fits_path = os.path.join(dirname, fits_file)

  hdulist = fits.open(fits_path)

  w = wcs.WCS(hdulist[0].header, relax=False)

  px, py = w.all_world2pix(r["ra"], r["dec"], 1)

  pixel_list = get_fits_filename(r, 'r', 'registered', '.list')
  pixel_path = os.path.join(dirname, pixel_list)
  with open(pixel_path, "w") as fout:
    fout.write("0 {} {} {}\n".format(px, py, r["class"]))

def run_sex(r, dirname="temp"):

    f = get_fits_filename(r, 'r', 'registered')
    fpath = os.path.join(dirname, f)

    list_file = get_fits_filename(r, 'r', 'registered', '.list')
    list_path = os.path.join(dirname, list_file)

    cat_file = get_fits_filename(r, 'r', 'registered', '.cat')
    cat_path = os.path.join(dirname, cat_file)

    config_file = f.replace(".fits", ".sex")
    shutil.copy('default.conv', dirname)
    shutil.copy('default.param', dirname)

    # write config file for this run
    # TODO: please get rid of this abomination!
    with open("default.sex", "r") as default:
        with open(os.path.join(dirname, config_file), "w") as temp:
            for line in default:
                line = re.sub(r"^ASSOC_NAME\s+sky.list",
                              "ASSOC_NAME       {}".format(list_file),
                              line)
                line = re.sub(r"^CATALOG_NAME\s+test.cat",
                              "CATALOG_NAME       {}".format(cat_file),
                              line)
                temp.write(line)

    cwd = os.getcwd()
    os.chdir(dirname)

    # shutil.copy(list_path, os.getcwd())

    subprocess.call(["sextractor", "-c", config_file, f])

    os.remove(config_file)

    os.chdir(cwd)

    cat = pd.read_csv(
        cat_path,
        skiprows=5,
        sep="\s+",
        names=["xmin", "ymin", "xmax", "ymax", "match"])
    # print(cat_path)
    # print(cat)
    cat["file"] = f
    # print(len(cat))
    assert len(cat) == 1, 'i assumed one output per sex call...'+str(cat)

    # os.remove(os.path.join(os.getcwd(), list_file))

    if len(cat) > 0:
         cat["class"] = r["class"]
         cat["objID"] = r["objID"]
    ##cat = cat.reset_index(drop=True)

    return cat

def nanomaggie_to_luptitude(array, band):
    '''
    Converts nanomaggies (flux) to luptitudes (magnitude).

    http://www.sdss.org/dr12/algorithms/magnitudes/#asinh
    http://arxiv.org/abs/astro-ph/9903081
    '''
    b = {
        'u': 1.4e-10,
        'g': 0.9e-10,
        'r': 1.2e-10,
        'i': 1.8e-10,
        'z': 7.4e-10
    }
    nanomaggie = array * 1.0e-9 # fluxes are in nanomaggies

    luptitude = -2.5 / np.log(10) * (np.arcsinh((nanomaggie / (2 * b[band]))) + np.log(b[band]))
    
    return luptitude

def save_cutout(df, row, image_dir, save_dir, size=48):

    def find_position(xmin, xmax, cut_size, frame_size):
        diff = 0.5 * ((xmax - xmin) - cut_size)
        if xmin + diff < 0:
            r = 0
            l = r + cut_size
        elif xmax + diff >= frame_size:
            l = frame_size
            r = l - cut_size
        else:
            r = int(xmin + diff)
            l = r + cut_size
        return r, l

    array = np.zeros((5, size, size))

    y0, x0, y1, x1 = row[["xmin", "ymin", "xmax", "ymax"]].values[0]

    for j, b in enumerate(BANDS):

        fpath = os.path.join(image_dir, row["file"].item())
        image_data = fits.getdata(fpath.replace("-r-", "-{}-".format(b)))

        extinction = df["extinction_{}".format(b)].item()

        right, left = find_position(x0, x1, size, image_data.shape[0])
        down, up = find_position(y0, y1, size, image_data.shape[1])

        cut_out = image_data[right: left, down: up]

        if cut_out.shape[0] == size and cut_out.shape[1] == size:
            cut_out = nanomaggie_to_luptitude(cut_out, b) - extinction
            array[j, :, :] = cut_out

    if np.isnan(array).sum() == 0 and array.sum() > 0:
        save_path = os.path.join(save_dir, "{0}.{1}x{1}.{2}.npy".format(row["class"].item(), size, row["objID"].item()))
        np.save(save_path, array)

def run_online_mode(filename, output_folder, threads, tmproot):
    os.makedirs(tmproot, exist_ok=True)

    df = pd.read_csv(filename, dtype={"objID": "object"})

    if os.path.exists(output_folder):
        done = os.listdir(output_folder)
        done = [d.split(".")[2] for d in done if d.count('.') >= 2]
        # check existing results and skip
        df = df[~df.objID.isin(done)]

    N = len(df)
    # N = 1
    if threads == 1:
      for i in range(N):
          download_record(df.iloc[i], i, output_folder, tmproot)
          print(' {:6}/{:6}'.format(i+1, N))
    else:
      with Pool(threads) as pool:
          pool.starmap(download_record, ((df.iloc[i], i, output_folder, tmproot) for i in range(N)))

def download_record(record: pd.DataFrame, i, output_folder, tmproot):
  tmp_dir = tempfile.mkdtemp(dir=tmproot)
  try:
    # tmp_dir = '/home/clear/dwynen/tmp/par/tmpq75dvebx'
    os.makedirs(tmp_dir, exist_ok=True)
    fetch_fits(record, tmp_dir)
    align_images(record, tmp_dir)
    convert_catalog_to_pixels(record, tmp_dir)
    cat = run_sex(record, tmp_dir)
    saved = save_cutout(record, cat, size=48, image_dir=tmp_dir, save_dir=output_folder)
  except Exception as e:
    print('failed to fetch entry {}'.format(i), file=sys.stderr)
    print(e)
  finally:
    shutil.rmtree(tmp_dir)

def main():
  parser = argparse.ArgumentParser(
      description='Download data from sdss3.org',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-i', '--input-csv', help='csv file obtained from CasJobs', default='DR12_spec_phot_sample.csv')
  parser.add_argument('-o', '--output-folder', help='where to store the results', default='result')
  parser.add_argument('-td', '--download-threads', help='how many threads to use for downloading', type=int, default=1)
  parser.add_argument('--tmpdir', help='root folder for intermediate results.', default='/tmp/sdss')
  args = parser.parse_args()
  print(args)

  os.makedirs(args.output_folder, exist_ok=True)

  run_online_mode(args.input_csv, args.output_folder, args.download_threads, tmproot=args.tmpdir)



if __name__ == "__main__":
    main()
