# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, OrderedDict
from fairseq.data.multi_corpus_dataset import MultiCorpusDataset
from omegaconf import MISSING, II, OmegaConf

from fairseq.data import SubsampleDataset
from dataload import FileRFDataset
from dataload import Dictionary, AddTargetDataset
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.data.text_compressor import TextCompressionLevel

from . import FairseqTask, register_task


logger = logging.getLogger(__name__)

class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )

@register_task("rf_pretraining")
class RFPretrainingTask(FairseqTask):
    """ """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="if set, normalizes input to have 0 mean and unit variance",
        )
        parser.add_argument(
            "--max-sample-size",
            default=512,
            type=int,
            help="max sample size to crop to for batching. default = min sample length",
        )
        parser.add_argument(
            "--min-sample-size",
            default=512,
            type=int,
            help="min sample size to crop to for batching. default = same as --max-sample-size",
        )

        parser.add_argument(
            "--enable-padding",
            action="store_true",
            help="pad shorter samples instead of cropping",
        )

        parser.add_argument(
            "--labels",
            type=str,
            default=None,
            help="extension of the label file to load, if any",
        )

    def __init__(self, args, source_dictionary=None):
        super().__init__(args)
        self._target_dictionary = None
        self._source_dictionary = source_dictionary
        self.is_ctc = args.criterion == "ctc"

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (RFPretrainingConfig): configuration of this task
        """

        return cls(args)

    def load_dataset(self, split: str, **kwargs):
        data_path = self.args.data

        manifest_path = os.path.join(data_path, "{}.tsv".format(split))                

        self.datasets[split] = FileRFDataset(
            manifest_path=manifest_path,
            max_sample_size=self.args.max_sample_size,
            min_sample_size=self.args.min_sample_size,
            pad=self.args.labels is not None or self.args.enable_padding,
            normalize=self.args.normalize,
        )
            
        if self.args.labels:
            dict_path = os.path.join(self.args.data, f"dict.{self.args.labels}.txt")
            self._target_dictionary = Dictionary.load(dict_path)
            label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
            labels = []
            with open(label_path, "r") as f:
                for line in f:
                    labels.append(line)

            process_label = LabelEncoder(self.target_dictionary)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                bos=None if self.is_ctc else self.target_dictionary.bos(),
                eos=None if self.is_ctc else self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label
            )

    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self._target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize

    def filter_indices_by_size(
            self,
            indices,
            dataset,
            max_positions=None,
            ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices