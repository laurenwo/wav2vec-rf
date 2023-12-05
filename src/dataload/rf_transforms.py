from typing import Any
import numpy as np

class IQ_Merge:
    def __init__(self, type='stack', axis=None):
        if type not in ['stack']:
            print(f"IQ_Merge: type {type} not valid, using type 'stack' instead")
            type = 'stack'
        self.type = type
        self.axis = axis
    def __call__(self, x):
        (i, q) = x
        if self.type == 'stack':
            if self.axis:
                return np.stack((i,q), axis=self.axis)
            else:
                return np.stack((i,q))

class Crop:
    def __init__(self, capture_start, sample_len):
        self.capture_start = capture_start
        self.sample_len = sample_len
    def __call__(self, x):
        (i, q) = x
        return i[self.capture_start:self.capture_start + self.sample_len], \
               q[self.capture_start:self.capture_start + self.sample_len]

class Jitter:
    def __init__(self, sigma=0.8):
        self.sigma = sigma
    def __call__(self, x):
        # https://arxiv.org/pdf/1706.00527.pdf
        return x + np.random.normal(loc=0., scale=self.sigma, size=x.shape)

class Scaling:
    def __init__(self, sigma=1.1):
        self.sigma = sigma
    def __call__(self, x):
        # https://arxiv.org/pdf/1706.00527.pdf
        factor = np.random.normal(loc=0., scale=self.sigma, size=(x.shape[0]))
        return (x.T * factor).T

class Permutation:
    def __init__(self, max_segments=5, seg_mode="random"):
        self.max_segments = max_segments
        self.seg_mode = seg_mode
    def __call__(self, x):
        orig_steps = np.arange(x.shape[1])
        num_segs = np.random.randint(1, self.max_segments)

        if num_segs > 1:
            if self.seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs)
            import warnings
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)     
            warp = np.concatenate(np.random.permutation(splits), dtype=np.int).ravel()
            return x[:, warp]
        else:
            return x

'''
class Scaling:
    def __init__(self, sigma=1.1):
        self.sigma = sigma
    def __call__(self, x):
        # https://arxiv.org/pdf/1706.00527.pdf
        factor = np.random.normal(loc=2., scale=self.sigma, size=(x.shape[0], x.shape[2]))
        ai = []
        for i in range(x.shape[1]):
            xi = x[:, i, :]
            ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
        return np.concatenate((ai), axis=1)


class Permutation:
    def __init__(self, max_segments=5, seg_mode="random"):
        self.max_segments = max_segments
        self.seg_mode = seg_mode
    def __call__(self, x):
        orig_steps = np.arange(x.shape[2])

        num_segs = np.random.randint(1, self.max_segments, size=(x.shape[0]))

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            if num_segs[i] > 1:
                if self.seg_mode == "random":
                    split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                    split_points.sort()
                    splits = np.split(orig_steps, split_points)
                else:
                    splits = np.array_split(orig_steps, num_segs[i])
                warp = np.concatenate(np.random.permutation(splits)).ravel()
                ret[i] = pat[0,warp]
            else:
                ret[i] = pat
        return torch.from_numpy(ret)
'''