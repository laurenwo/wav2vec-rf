#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import os
import random


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.01,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--dest", default="./data/rf/", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--savename", default="manifest"
    )
    parser.add_argument(
        "--ext", default=".sigmf-data", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    return parser


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "*" + args.ext)

    with open(os.path.join(args.dest, args.savename + ".tsv"), "w") as _f:
        print(dir_path, file=_f)
        files = glob.glob(search_path)

        for fname in files:
            file_path = os.path.realpath(fname)

            if args.path_must_contain and args.path_must_contain not in file_path:
                continue

            ## TODO: get this from meta file
            n_samples = 512

            print(
                "{}\t{}".format(os.path.relpath(file_path, dir_path), n_samples), file=_f
            )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
