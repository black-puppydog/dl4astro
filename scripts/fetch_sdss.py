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


def fetch_fits(df, dirname):

    bands = 'ugriz'

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i, r in df.iterrows():

        url = "http://data.sdss3.org/sas/dr12/boss/photoObj/frames/{0}/{1}/{2}/".format(
            r["rerun"], r["run"], r["camcol"], r["field"])

        for band in bands:

            filename = "frame-{4}-{1:06d}-{2}-{3:04d}.fits".format(
                r["rerun"], r["run"], r["camcol"], r["field"], band)
            filepath = os.path.join(dirname, filename)

            for _ in range(10):
                try:
                    print(url+filename+'.bz2')
                    resp = requests.get(url + filename + ".bz2")
                except:
                    sleep(1)
                    continue
                
                if resp.status_code == 200:
                    with open(filepath, "wb") as f:
                        img = bz2.decompress(resp.content)
                        f.write(img)
                    #print("Downloaded {}".format(filename))
                    break
                else:
                    sleep(1)
                    continue

            if not os.path.exists(filepath):
                raise Exception

def get_ref_list(df):

    ref_images = []
    
    for row in df.iterrows():
        r = row[1]
        filename = "frame-r-{1:06d}-{2}-{3:04d}.fits".format(r["rerun"], r["run"], r["camcol"], r["field"])
        ref_images.append(filename)

    return ref_images

def align_images(images, tmp_dir):
    '''
    '''

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    for image in images:
        
        #print("Processing {}...".format(image))
    
        frame_path = [
            os.path.join(tmp_dir, image.replace("frame-r-", "frame-{}-").format(b))
            for b in "ugriz"
            ]
        registered_path = [
            os.path.join(tmp_dir, image.replace("frame-r-", "registered-{}-").format(b))
            for b in "ugriz"
            ]

        header = os.path.join(
            tmp_dir,
            image.replace("frame", "header").replace(".fits", ".hdr")
            )

        mw.commands.mGetHdr(os.path.join(tmp_dir, image), header)
        mw.reproject(
            frame_path, registered_path,
            header=header, exact_size=True, silent_cleanup=True, common=True
            )

    return None


def convert_catalog_to_pixels(df, dirname):

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    pixels = []
    fits_list = []

    for i, r in df.iterrows():

        fits_file = "registered-r-{1:06d}-{2}-{3:04d}.fits".format(
            r["rerun"], r["run"], r["camcol"], r["field"])
        fits_path = os.path.join(dirname, fits_file)
            
        hdulist = fits.open(fits_path)

        w = wcs.WCS(hdulist[0].header, relax=False)
        
        px, py = w.all_world2pix(r["ra"], r["dec"], 1)

        fits_list.append(fits_file)
        pixels.append((i, px, py, r["class"]))

    for i, fits_file in enumerate(fits_list):
        ix, px, py, c = pixels[i]
        pixel_list = fits_file.replace(".fits", ".list")
        pixel_path = os.path.join(dirname, pixel_list)
        with open(pixel_path, "a") as fout:
            fout.write("{} {} {} {}\n".format(ix, px, py, c))

    return None

def write_default_conv():

    default_conv = (
        "CONV NORM\n"
        "# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.\n"
        "1 2 1\n"
        "2 4 2\n"
        "1 2 1\n"
    ).format()

    with open("default.conv", "w") as f:
        f.write(default_conv)

    return None

def write_default_param():

    default_param = (
        "XMIN_IMAGE               Minimum x-coordinate among detected pixels                [pixel]\n"
        "YMIN_IMAGE               Minimum y-coordinate among detected pixels                [pixel]\n"
        "XMAX_IMAGE               Maximum x-coordinate among detected pixels                [pixel]\n"
        "YMAX_IMAGE               Maximum y-coordinate among detected pixels                [pixel]\n"
        "VECTOR_ASSOC(1)          #ASSOCiated parameter vector"
    ).format()

    with open("default.param", "w") as f:
        f.write(default_param)

    return None

def write_default_sex():

    default_sex = (
        "#-------------------------------- Catalog ------------------------------------\n"
        "\n"
        "CATALOG_NAME     test.cat       # name of the output catalog\n"
        "CATALOG_TYPE     ASCII_HEAD     # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,\n"
        "                                # ASCII_VOTABLE, FITS_1.0 or FITS_LDAC\n"
        "PARAMETERS_NAME  default.param  # name of the file containing catalog contents\n"
        " \n"
        "#------------------------------- Extraction ----------------------------------\n"
        " \n"
        "DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)\n"
        "DETECT_MINAREA   3              # min. # of pixels above threshold\n"
        "DETECT_THRESH    1.5            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n"
        "ANALYSIS_THRESH  1.5            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n"
        " \n"
        "FILTER           Y              # apply filter for detection (Y or N)?\n"
        "FILTER_NAME      default.conv   # name of the file containing the filter\n"
        " \n"
        "DEBLEND_NTHRESH  32             # Number of deblending sub-thresholds\n"
        "DEBLEND_MINCONT  0.005          # Minimum contrast parameter for deblending\n"
        " \n"
        "CLEAN            Y              # Clean spurious detections? (Y or N)?\n"
        "CLEAN_PARAM      1.0            # Cleaning efficiency\n"
        " \n"
        "MASK_TYPE        CORRECT        # type of detection MASKing: can be one of\n"
        "                                # NONE, BLANK or CORRECT\n"
        "\n"
        "#------------------------------ Photometry -----------------------------------\n"
        " \n"
        "PHOT_APERTURES   5              # MAG_APER aperture diameter(s) in pixels\n"
        "PHOT_AUTOPARAMS  2.5, 3.5       # MAG_AUTO parameters: <Kron_fact>,<min_radius>\n"
        "PHOT_PETROPARAMS 2.0, 3.5       # MAG_PETRO parameters: <Petrosian_fact>,\n"
        "                                # <min_radius>\n"
        "\n"
        "SATUR_LEVEL      50000.0        # level (in ADUs) at which arises saturation\n"
        "SATUR_KEY        SATURATE       # keyword for saturation level (in ADUs)\n"
        " \n"
        "MAG_ZEROPOINT    0.0            # magnitude zero-point\n"
        "MAG_GAMMA        4.0            # gamma of emulsion (for photographic scans)\n"
        "GAIN             0.0            # detector gain in e-/ADU\n"
        "GAIN_KEY         GAIN           # keyword for detector gain in e-/ADU\n"
        "PIXEL_SCALE      1.0            # size of pixel in arcsec (0=use FITS WCS info)\n"
        " \n"
        "#------------------------- Star/Galaxy Separation ----------------------------\n"
        " \n"
        "SEEING_FWHM      1.2            # stellar FWHM in arcsec\n"
        "STARNNW_NAME     default.nnw    # Neural-Network_Weight table filename\n"
        " \n"
        "#------------------------------ Background -----------------------------------\n"
        " \n"
        "BACK_SIZE        64             # Background mesh: <size> or <width>,<height>\n"
        "BACK_FILTERSIZE  3              # Background filter: <size> or <width>,<height>\n"
        " \n"
        "BACKPHOTO_TYPE   GLOBAL         # can be GLOBAL or LOCAL\n"
        " \n"
        "#------------------------------ Check Image ----------------------------------\n"
        " \n"
        "CHECKIMAGE_TYPE  SEGMENTATION   # can be NONE, BACKGROUND, BACKGROUND_RMS,\n"
        "                                # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,\n"
        "                                # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,\n"
        "                                # or APERTURES\n"
        "CHECKIMAGE_NAME  check.fits     # Filename for the check-image\n"
        " \n"
        "#--------------------- Memory (change with caution!) -------------------------\n"
        " \n"
        "MEMORY_OBJSTACK  3000           # number of objects in stack\n"
        "MEMORY_PIXSTACK  300000         # number of pixels in stack\n"
        "MEMORY_BUFSIZE   1024           # number of lines in buffer\n"
        " \n"
        "#----------------------------- Miscellaneous ---------------------------------\n"
        " \n"
        "VERBOSE_TYPE     QUIET          # can be QUIET, NORMAL or FULL\n"
        "HEADER_SUFFIX    .head          # Filename extension for additional headers\n"
        "WRITE_XML        N              # Write XML file (Y/N)?\n"
        "XML_NAME         sex.xml        # Filename for XML output\n"
        "\n"
        "#----------------------------- ASSOC parameters ---------------------------------\n"
        "\n"
        "ASSOC_NAME       sky.list       # name of the ASCII file to ASSOCiate, the expected pixel \n"
        "                                # coordinates list given as [id, xpos, ypos]\n"
        "ASSOC_DATA       1              # columns of the data to replicate (0=all), replicate id\n"
        "                                # of the object in the SExtractor output file\n"
        "ASSOC_PARAMS     2,3            # columns of xpos,ypos[,mag] in the expected pixel\n"
        "                                # coordinates list\n"
        "ASSOC_RADIUS     2.0            # cross-matching radius (pixels)\n"
        "ASSOC_TYPE       NEAREST        # ASSOCiation method: FIRST, NEAREST, MEAN,\n"
        "                                # MAG_MEAN, SUM, MAG_SUM, MIN or MAX\n"
        "ASSOCSELEC_TYPE  MATCHED        # ASSOC selection type: ALL, MATCHED or -MATCHED\n"
    ).format()

    with open("default.sex", "w") as f:
        f.write(default_sex)

def run_sex(df, dirname="temp"):
    """
    """

    cat = pd.DataFrame()

    ref_images = get_ref_list(df) 
    registered_all = [f.replace("frame-", "registered-") for f in ref_images]
    
    for f in registered_all:
        
        fpath = os.path.join(dirname, f)
        
        list_file = f.replace(".fits", ".list")
        list_path = os.path.join(dirname, list_file)

        config_file = f.replace(".fits", ".sex")

        with open("default.sex", "r") as default:
            with open(config_file, "w") as temp:
                for line in default:
                    line = re.sub(
                        r"^ASSOC_NAME\s+sky.list",
                        "ASSOC_NAME       {}".format(list_file),
                        line
                    )
                    temp.write(line)
    
        shutil.copy(list_path, os.getcwd())
    
        subprocess.call(["sextractor", "-c", config_file, fpath])

        os.remove(config_file)
    
        try:
            assoc = pd.read_csv(
                "test.cat",
                skiprows=5,
                sep="\s+",
                names=["xmin", "ymin", "xmax", "ymax", "match"]
            )
            assoc["file"] = f
            cat = cat.append(assoc)
        except:
            pass
        
        os.remove(os.path.join(os.getcwd(), list_file))
    
    if len(cat) > 0:
         cat["class"] = df.ix[cat["match"], "class"].values
         cat["objID"] = df.ix[cat["match"], "objID"].values
    #cat = cat.reset_index(drop=True)

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

def save_cutout(df, cat, image_dir, save_dir, size=48):

    saved = pd.DataFrame()

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

    for i, row in cat.iterrows():

        array = np.zeros((5, size, size))
        
        y0, x0, y1, x1 = row[["xmin", "ymin", "xmax", "ymax"]].values
        matched = df[df["objID"] == row.astype("object")["objID"]]
        assert len(matched) == 1

        for j, b in enumerate("ugriz"):

            fpath = os.path.join(image_dir, row["file"])
            image_data = fits.getdata(fpath.replace("-r-", "-{}-".format(b)))
            
            extinction = matched["extinction_{}".format(b)].values[0]

            right, left = find_position(x0, x1, size, image_data.shape[0])
            down, up = find_position(y0, y1, size, image_data.shape[1])

            cut_out = image_data[right: left, down: up]
        
            if cut_out.shape[0] == size and cut_out.shape[1] == size:
                cut_out = nanomaggie_to_luptitude(cut_out, b) - extinction
                array[j, :, :] = cut_out
                
        if np.isnan(array).sum() == 0 and array.sum() > 0:
            save_path = os.path.join(save_dir, "{0}.{1}x{1}.{2}.npy".format(row["class"], size, row["objID"]))
            np.save(save_path, array)

def run_online_mode(filename, output_folder, threads):

    df = pd.read_csv(filename, dtype={"objID": "object"})

    if os.path.exists(output_folder):
        done = os.listdir(output_folder)
        done = [d.split(".")[2] for d in done]
        # check existing results and skip
        df = df[~df.objID.isin(done)]

    write_default_conv()
    write_default_param()
    write_default_sex()

    N = len(df)
    if threads == 1:
      for i in range(N):
          download_record(df, i, output_folder)
          print(' {:6}/{:6}'.format(i+1, N))
    else:
      with Pool(threads) as pool:
          pool.starmap(download_record, ((df, i, output_folder) for i in range(N)))

async def run_online_mode_parallel(filename, output_folder):

    df = pd.read_csv(filename, dtype={"objID": "object"})

    if os.path.exists(output_folder):
        done = os.listdir(output_folder)
        done = [d.split(".")[2] for d in done]
        # check existing results and skip
        df = df[~df.objID.isin(done)]

    write_default_conv()
    write_default_param()
    write_default_sex()

    N = len(df)
    tasks = list()
    for i in range(0, 2):
        tasks.append(download_record_parallel(df[i: i+1], i, output_folder))
    print(tasks)
    await asyncio.wait(tasks)


async def download(url, filename, session, chunk_size=1 << 15, num_try=3):
    # with (yield from semaphore_download):  # limit number of concurrent downloads
        for t in range(num_try):
            try:
                logging.info('downloading %s', filename+'.bz2')
                with session.get(filename+'.bz2') as resp:
                    with open(filename, "wb") as f:
                        img = bz2.decompress(await resp.text())
                        f.write(img)
                logging.info('done %s', filename)
                success = True
            except:
                success = False
                if t == num_try - 1:
                    logging.error('failed to download {}'.format(url))
                    raise
            if success:
                break

def download_parallel(urls_filenames, output_folder):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    with closing(asyncio.get_event_loop()) as loop, \
            closing(aiohttp.ClientSession()) as session:
        download_tasks = (download(url+fname, output_folder+fname, session) for url, fname in urls_filenames)
        result = loop.run_until_complete(asyncio.gather(*download_tasks))

async def download_record_parallel(chunk, i, output_folder):
    tmp_dir = tempfile.mkdtemp(dir='/tmp/dldl4a')

    bands = 'ugriz'

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    print(tmp_dir)
    urls_filenames = (("http://data.sdss3.org/sas/dr12/boss/photoObj/frames/{r.rerun}/{r.run}/{r.camcol}/".format(r=r, band=b),
                      "frame-{band}-{r.run:06d}-{r.camcol}-{r.field:04d}.fits".format(r=r, band=b))
                      for (_, r) in chunk.iterrows() for b in bands)
    # download image fits files
    download_parallel(urls_filenames, tmp_dir)
    ref_images = get_ref_list(chunk)
    align_images(ref_images, tmp_dir)
    convert_catalog_to_pixels(chunk, tmp_dir)
    cat = run_sex(chunk, tmp_dir)
    try:
        saved = save_cutout(chunk, cat, size=48, image_dir=tmp_dir, save_dir=output_folder)
    except:
        print('failed to fetch entry {}'.format(i), file=sys.stderr)
    # shutil.rmtree(tmp_dir)

def download_record(df, i, output_folder):
    chunk = df[i: i+1]
    # download image fits files
    tmp_dir = tempfile.mkdtemp(dir='/tmp')
    print(tmp_dir)
    fetch_fits(chunk, tmp_dir)
    ref_images = get_ref_list(chunk)
    align_images(ref_images, tmp_dir)
    convert_catalog_to_pixels(chunk, tmp_dir)
    cat = run_sex(chunk, tmp_dir)
    try:
        saved = save_cutout(chunk, cat, size=48, image_dir=tmp_dir, save_dir=output_folder)
    except:
        print('failed to fetch entry {}'.format(i), file=sys.stderr)
    finally:
      shutil.rmtree(tmp_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Download data from sdss3.org',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--parallel', action='store_true', help='use multiple threads for downloading and processing')
    parser.add_argument('-i', '--input-csv', help='csv file obtained from CasJobs', default='DR12_spec_phot_sample.csv')
    parser.add_argument('-o', '--output-folder', help='where to store the results', default='result')
    parser.add_argument('-td', '--download-threads', help='how many threads to use for downloading', type=int, default=1)
    parser.add_argument('-ts', '--sextractor-threads', help='how many threads to use for sextractor calls', type=int, default=2)
    parser.add_argument('-tm', '---montage-threads', help='how many threads to use for montage calls', type=int, default=2)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if not args.parallel:
        run_online_mode(args.input_csv, args.output_folder, args.download_threads)
        sys.exit()
    print('using parallel processing')

    global semaphore_download
    semaphore_download = asyncio.Semaphore(args.download_threads)
    global semaphore_sextractor
    semaphore_sextractor = asyncio.Semaphore(args.sextractor_threads)
    global semaphore_montage
    semaphore_montage = asyncio.Semaphore(args.montage_threads)

    tasks = run_online_mode_parallel(args.input_csv, args.output_folder)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(tasks)


if __name__ == "__main__":
    main()
