# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys
import time
import io

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from . import FairseqDataset
from .rf_transforms import *
from .data_utils import compute_block_mask_1d, get_buckets, get_bucketed_sizes

logger = logging.getLogger(__name__)


class RawRFDataset(FairseqDataset):
    def __init__(
        self,
        # sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
    ):
        super().__init__()

        # self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize
        self.transforms = transforms.Compose([
              transforms.ToTensor()
            ])

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, iq):
        # inline_transforms = transforms.Compose([
        #     Crop(capture_start=self.capture_start, sample_len=self.example_len),
        #     IQ_Merge(type=self.merge_type, axis=self.merge_axis)
        # ])
        # iq = inline_transforms((i,q))
        # iq = self.transforms(iq)

        assert iq.dim() == 1, iq.dim()

        if self.normalize:
            with torch.no_grad():
                iqq = F.layer_norm(iq, iq.shape)
        return iq

    def crop_to_max_size(self, t, target_size, dim=0):
        size = t.size(dim)
        diff = size - target_size
        if diff <= 0:
            return t

        start = np.random.randint(0, diff + 1)
        end = size - diff + start

        slices = []
        for d in range(dim):
            slices.append(slice(None))
        slices.append(slice(start, end))

        return t[slices]

    # @staticmethod
    # def _bucket_tensor(tensor, num_pad, value):
    #     return F.pad(tensor, (0, num_pad), value=value)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        # if self.corpus_key is not None:
        #     input["corpus_key"] = [self.corpus_key] * len(sources)
        # out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            order.append(
                np.minimum(
                    np.array(self.sizes),
                    self.max_sample_size,
                )
            )
            return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))


class FileRFDataset(RawRFDataset):
    def __init__(
        self,
        manifest_path,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
    ):
        super().__init__(
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
        )

        # self.text_compressor = TextCompressor(level=text_compression_level)

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(items[0])
                # self.fnames.append(self.text_compressor.compress(items[0]))
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

    def __getitem__(self, index):
        fname = os.path.join(self.root_dir, self.fnames[index])
        with open(fname, "rb") as _f:
            f_data = np.load(_f)
        iq = torch.from_numpy(f_data).float()
        iq = self.postprocess(iq)
        return {"id": index, "source": iq}
        