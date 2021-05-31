"""
module containing utilities to load
the dataset for the training
of the siamese recurrent network.
"""
import json
import copy
import itertools
import functools
import logging
import lz4.frame

from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, Subset
from skimage import io
import torch

import recsiam.utils as utils

# entry
# { "id" :  int,
#   "paths" : str,
#   "metadata" : str}

def nextid():
    cid = 0
    while True:
        yield cid
        cid += 1

def descriptor_from_filesystem(root_path):
    desc = []
    root_path = Path(root_path)

    id_gen = nextid()

    embedded = False
    sample = next(root_path.glob("*/*/*")).name
    if sample.endswith("npy") or sample.endswith("lz4"):
        embedded = True

    for subd in sorted(root_path.iterdir()):

        obj_desc = {"id":  next(id_gen), "name": str(subd)}

        if not embedded:
            seqs_dir = sorted(str(subsub) for subsub in subd.iterdir()if subsub.is_dir())
            seqs = [sorted(str(frame) for frame in Path(d).iterdir()) for d in seqs_dir]
        else:
            seqs = sorted(str(subsub / sample) for subsub in subd.iterdir()if subsub.is_dir())

        obj_desc["paths"] = seqs

        desc.append(obj_desc)


    return desc


def ambiguity_descriptor_from_filesystem(root_path, discr="differentia", amb="genus"):
    desc = []
    root_path = Path(root_path)

    id_gen = nextid()

    embedded = False
    sample = next(root_path.glob("*/*/*/*/*")).name
    if sample.endswith("npy") or sample.endswith("lz4"):
        embedded = True

    for class_dir in sorted(root_path.iterdir()):

        #obj_desc = {"id":  next(id_gen), "class": str(class_dir)}

        discr_obj = set(subd.name for subd in (class_dir / discr).iterdir() if subd.is_dir())
        amb_obj = set(subd.name for subd in (class_dir / amb).iterdir() if subd.is_dir())

        all_obj = discr_obj | amb_obj

        if not embedded:
            raise NotImplementedError("Not yet implemented")
#            seqs_dir = sorted(str(subsub) for subsub in class_dir.iterdir()if subsub.is_dir())
#            seqs = [sorted(str(frame) for frame in Path(d).iterdir()) for d in seqs_dir]
        else:
            for obj in all_obj:
                seqs = []
                is_amb = []
                if obj in discr_obj:
                    d_seqs = list(str(subd / sample)
                                  for subd in (class_dir / discr / obj).iterdir()
                                  if subd.is_dir())
                    is_amb += [False] * len(d_seqs)
                    seqs += d_seqs
                if obj in amb_obj:
                    a_seqs = list(str(subd / sample)
                                  for subd in (class_dir / amb / obj).iterdir()
                                  if subd.is_dir())
                    is_amb += [True] * len(a_seqs)
                    seqs += a_seqs

                seqs = np.asarray(seqs)
                is_amb = np.asarray(is_amb)

                order = np.argsort(seqs)

                seqs = seqs[order]
                is_amb = is_amb[order]

                obj_desc = {"id":  next(id_gen), "class": str(class_dir)}
                obj_desc["paths"] = seqs.tolist()
                obj_desc["ambiguous"] = is_amb.tolist()
                desc.append(obj_desc)

    return desc


def mask_from_filesystem(root_path, mask_name="unary.png"):
    desc = []
    root_path = Path(root_path)

    id_gen = nextid()

    for subd in sorted(root_path.iterdir()):

        obj_desc = {"id":  next(id_gen), "name": str(subd)}

        seqs_dir = sorted(str(subsub) for subsub in subd.iterdir()if subsub.is_dir())
        seqs = [sorted(str(frame_d / mask_name) for frame_d in Path(d).iterdir() if frame_d.is_dir()) for d in seqs_dir]


        obj_desc["paths"] = seqs

        desc.append(obj_desc)


    return desc


class VideoDataSet(Dataset):
    """
    Class that implements the pythorch Dataset
    of sequences of frames
    """
    def __init__(self, descriptor):

        self.logger = logging.getLogger("recsiam.data.VideoDataSet")

        self.descriptor = descriptor
        if isinstance(self.descriptor, (list, tuple, np.ndarray)):
            self.data = np.asarray(self.descriptor)
        else:
            with Path(self.descriptor).open("r") as ifile:
                self.data = np.array(json.load(ifile))

        self.paths = np.array([d["paths"] for d in self.data])
        self.seq_number = np.array([len(path) for path in self.paths])

        def get_id_entry(elem_id):
            return self.data[elem_id]["id"]

        self.id_table = np.vectorize(get_id_entry)

        self.embedded = False
        self.compressed = False
        try:
            np.load(self.paths[0][0])
            self.embedded = True
        except Exception:
            pass

        try:
            with lz4.frame.open(self.paths[0][0], mode="rb") as f:
                np.load(f)
                self.embedded = True
                self.compressed = True
        except Exception:
            pass

        self.n_elems = len(self.paths)

    @property
    def is_embed(self):
        return self.embedded

    def get_metadata(self, key, elem_ind, object_level=True):
        elem_ind = np.asarray(elem_ind)

        obj_d = self.data[elem_ind[0]]

        if object_level:
            val = obj_d[key]
        else:
            val = obj_d[key][elem_ind[1]]
        l = logging.getLogger(self.logger.name + ".get_metadata")
        l.debug("key = {}\telem_ind ={}\tval = {}".format(key, elem_ind, val))

        return val

    def load_array(self, path):
        if not self.compressed:
            loaded = np.load(str(path))

        else:
            with lz4.frame.open(str(path), mode="rb") as f:
                loaded = np.load(f)

        return loaded

    def __len__(self):
        return self.n_elems

    def __getitem__(self, value):
        return self._getitem(value)

    def _getitem(self, value):

        if isinstance(value, (list, tuple, np.ndarray)):
            if self._valid_t(value):
                return self._get_single_item(*value)
            elif np.all([self._valid_t(val) for val in value]):
                return np.array([self._get_single_item(*val) for val in value])
            else:
                raise TypeError("Invalid argument type: {}.".format(value))
        else:
            raise TypeError("Invalid argument type: {}.".format(value))

    @staticmethod
    def _valid_t(value):
        return isinstance(value, (tuple, list, np.ndarray)) and \
                len(value) == 3 and \
                isinstance(value[0], (int, np.integer)) and \
                isinstance(value[1], (int, np.integer)) and \
                isinstance(value[2], (int, np.integer,
                                      slice, list, np.ndarray))

    def sample_size(self):
        return self._get_single_item(0, 0, slice(0, 1)).shape[1:]

    def _get_single_item(self, idx1, idx2, idx3):

        path = self.paths[idx1]
        seq_path = path[idx2]
        if not self.is_embed:
            p_list = np.array(seq_path)[idx3]
        else:
            p_list = seq_path, idx3

        l = logging.getLogger(self.logger.name + "._get_single_item")
        l.debug("path_ind = {}\tseq_path ={}\telem_ind = {}".format(idx1, seq_path, idx3))
        sequences = self._load_sequence(p_list)

        return sequences

    def _load_sequence(self, paths_list):

        if not self.is_embed:
            sequence = np.array(io.imread_collection(paths_list,
                                                     conserve_memory=False))
            # if gray, add bogus dimension
            if len(sequence.shape) == 2 + int(len(paths_list) > 1):
                sequence = sequence[..., None]
            sequence = np.transpose(sequence, (0, 3, 1, 2))

        else:
            sequence = self.load_array(paths_list[0])[paths_list[1]]

        return sequence

    def gen_embed_dataset(self):
        for obj in range(self.n_elems):
            for seq in range(len(self.paths[obj])):
                yield self[obj, seq, :], self.paths[obj][seq]


def dataset_from_filesystem(root_path):
    descriptor = descriptor_from_filesystem(root_path)
    return VideoDataSet(descriptor)


class TrainSeqDataSet(VideoDataSet):

    def __getitem__(self, value):
        if isinstance(value, (list, tuple, np.ndarray)) and \
           len(value) == 2 and \
           np.all([self._valid_t(val) for val in value]):

            items = self._getitem(value)
            seq_len = np.array([len(val)
                                for val in items])

            return items, seq_len, (value[0][0], value[1][0])
        else:
            error_str = "The input must be in the form " +\
                        "((int, int, slice), (int, int,  slice)). " +\
                        "Found {}"

            raise ValueError(error_str.format(value))


class FlattenedDataSet(VideoDataSet):

    def __init__(self, *args, preload=False, pre_embed=None):
        super().__init__(*args)

        self.val_map = []
        for itx in range(len(self.seq_number)):
            self.val_map.extend([(itx, i) for i in range(self.seq_number[itx])])

        self.val_map = np.array(self.val_map)

        self.flen = len(self.val_map)
        self.pre_embed = pre_embed

        self.preloaded = None
        if preload:
            self.preloaded = []
            for i in range(len(self)):
                self.preloaded.append(self.real_getitem(i))

            self.preloaded = np.array(self.preloaded)

    def map_value(self, value):
        return self.val_map[value]

    def __len__(self):
        return self.flen

    def get_metadata(self, key, elem_ind, **kwargs):
        return super().get_metadata(key, self.val_map[elem_ind].squeeze(), **kwargs)

    def get_label(self, value):
        ndim = np.ndim(value)
        if ndim == 0:
            return self.map_value(value)[0]
        elif ndim == 1:
            return self.map_value(value)[:, 0]
        else:
            raise ValueError("np.ndim(value) > 1")

    def __getitem__(self, i):
        if self.preloaded is not None:
            return self.preloaded[i]
        else:
            return self.getitems(i)

    def getitems(self, ind):
        if isinstance(ind, slice):
            ind = np.arange(*ind.indices(len(self)))

        if isinstance(ind, (list, tuple, np.ndarray)):
            return np.array([self.real_getitem(i) for i in ind])
        else:
            return self.real_getitem(ind)

    def real_getitem(self, value):
        t = tuple(self.map_value(value)) + (slice(None),)
        items = super().__getitem__(t)
        if self.pre_embed is not None:
            items = self.pre_embed([utils.a2t(items)])[0]
        return items, t[0]

    def balanced_sample(self, elem_per_class, rnd, separate=False, ind_subset=None):
        if ind_subset is None:
            p_ind = rnd.permutation(len(self.val_map))
        else:
            assert np.unique(ind_subset).size == ind_subset.size
            assert (ind_subset >= 0).all() and (ind_subset < len(self.val_map)).all()
            p_ind = rnd.permutation(ind_subset)
        perm = self.val_map[p_ind]
        cls = perm[:, 0]

        _, indices = np.unique(cls, return_index=True)

        remaining_ind = np.delete(np.arange(len(cls)), indices)

        ind_sets = [indices]

        for i in range(elem_per_class - 1):
            p = cls[remaining_ind]
            _, ind = np.unique(p, return_index=True)

            ind_sets.append(ind)
            indices = np.concatenate([indices, remaining_ind[ind]])
            remaining_ind = np.delete(remaining_ind, ind)

        if not separate:
            return p_ind[indices]
        else:
            return tuple(p_ind[i] for i in ind_sets)

    def get_n_objects(self, number, rnd, ind_subset=None):
        if ind_subset is None:
            elems = len(self.seq_number)
        else:
            elems = np.unique(self.get_label(ind_subset))
        obj_ind = rnd.choice(elems, size=number, replace=False)

        class_ind = np.where(np.isin(self.val_map[:, 0], obj_ind))[0]
        if ind_subset is not None:
            class_ind = np.intersect1d(class_ind, ind_subset)

        return class_ind


class FramesDataset(torch.utils.data.Dataset):

    def __init__(self, seq_dataset):
        super().__init__()

        self.s_data = seq_dataset
        l = [list(zip(itertools.repeat(e), range(len[i[0]]))) for e, i in enumerate(self.s_data)]

        self.c_list = np.array(sum(l, []))

        def __len__(self):
            return len(self.c_list)

        def __getitem__(self, items):
            coord = self.c_list[items]

            data = self.s_data[coord[0]]

            return data[0][coord[1]], data[1]



class SiameseDatasetView(torch.utils.data.Dataset):

    def __init__(self, x1, x2, lab):
        self.x1 = x1
        self.x2 = x2
        self.lab = lab

        assert x1.size == x2.size
        assert lab.size == x2.size

    def __len__(self):
        return self.x1.size

    def __getitem__(self, items):
        return self.x1[items], self.x2[items], self.lab[items]


def gen_siamese_dataset(dataset, couples, rnd):
    labs = dataset.get_label(slice(None))
    u, counts = np.unique(labs, return_counts=True)
    d_class = {}
    for l in u:
        d_class[l] = np.where(labs == l)[0]

    def sample(N):
        return rnd.choice(len(dataset), size=(N, 2))

    def accept(p):
        return labs[p[:, 0]] != labs[p[:, 1]]

    while True:
        same_cp = np.array([rnd.choice(d_class[c], size=2, replace=False)
                            for c in rnd.choice(u, size=couples // 2)])
        diff_cp = utils.rejection_sampling(couples // 2 + couples % 2, sample, accept)

        concat_cp = np.concatenate([same_cp, diff_cp])
        lab_cp = np.concatenate([np.ones(same_cp.shape[0]), np.zeros(diff_cp.shape[0])]).astype(np.uint8)

        yield SiameseDatasetView(dataset[concat_cp[:, 0]][:, 0], dataset[concat_cp[:, 1]][:, 0], lab_cp)


class TripletDatasetView(torch.utils.data.Dataset):

    def __init__(self, *data):
        super().__init()
        self.data = data

        assert len(data) == 3

    def __len__(self):
        return len(self.datai[0])

    def __getitem_(self, item):
        return tuple(d[item] for d in self.data)


def gen_triplet_dataset(dataset, elements, rnd):
    labs = dataset.get_label(slice(None))
    u, counts = np.unique(labs, return_counts=True)
    d_class = {}
    for l in u:
        d_class[l] = np.where(labs == l)[0]

    def sample(rnd):
        labels = rnd.choice(u, size=2, replace=False)
        same_objs = rnd.choice(d_class[labels[0]], size=2, replace=False)
        diff_obj  = rnd.choice(d_class[labels[0]], size=1)
        return np.append(same_objs, diff_obj)

    while True:
        concat_cp = np.array([sample(rnd) for _ in range(elements)])
        yield TripletDatasetView(dataset[concat_cp[:, 0]][:, 0],
                                 dataset[concat_cp[:, 1]][:, 0],
                                 dataset[concat_cp[:, 2]][:, 0])


def list_collate(data):
    emb = [utils.astensor(d[0]) for d in data]
    lab = np.array([d[1] for d in data])

    return emb, lab


class ExtendedSubset(Subset):

    def __init__(self, dataset, indices=None):
        if indices is None:
            indices = np.arange(len(dataset))
        if isinstance(dataset, Subset):
            indices = dataset.indices[indices]
            dataset = dataset.dataset

        super().__init__(dataset, indices)

    def get_label(self, value):
        return self.dataset.get_label(self.indices[value])

    def get_metadata(self, key, elem_ind, **kwargs):
        return self.dataset.get_metadata(key, self.indices[elem_ind], **kwargs)

    def split_balanced(self, elem_per_class, rnd):
        ind = self.dataset.balanced_sample(elem_per_class, rnd, False, self.indices)

        other_ind = np.setdiff1d(self.indices, ind)

        return (ExtendedSubset(self.dataset, ind),
                ExtendedSubset(self.dataset, other_ind))

    def split_n_objects(self, number, rnd):
        ind = self.dataset.get_n_objects(number, rnd, self.indices)
        other_ind = np.setdiff1d(self.indices, ind)

        return (ExtendedSubset(self.dataset, ind),
                ExtendedSubset(self.dataset, other_ind))

    def same_subset(self, dataset):
        assert len(dataset) == len(self.dataset)
        return ExtendedSubset(dataset, self.indices)




class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mask):

        self.dataset = dataset
        self.mask = mask

        super().__init__()

    def get_label(self, value):
        return self.dataset.get_label(value)

    def __getitem__(self, value):
        data, labels = self.dataset[value]
        data_mask, _ = self.mask[value]

        masked = self.apply_mask(data_mask.astype(np.bool), data)
        #assert False
        return masked, labels

    @staticmethod
    def apply_mask(data_mask, data):
        masked = np.array([np.where(m, d, 127) for m, d in zip(data_mask, data)])
        return masked

    def gen_embed_dataset(self, return_mask_paths=False):
        assert isinstance(self.dataset, VideoDataSet)
        assert isinstance(self.mask, VideoDataSet)

        gens = (self.dataset.gen_embed_dataset(), self.mask.gen_embed_dataset())
        for (d_seq, d_paths), (m_seq, m_paths) in zip(*gens):
            masked = self.apply_mask(m_seq.astype(np.bool), d_seq)
            if not return_mask_paths:
                yield masked, d_paths
            else:
                yield masked, d_paths, m_paths


def train_val_split(dataset, seed, dl_arg={},
                    incremental_evaluation=None, prob_new=None):
    rs = np.random.RandomState
    rnd_s, rnd_e, rnd_i = [rs(s) for s in rs(seed).randint(2**32 - 1, size=3)]

    ordered_train_ind = np.arange(len(dataset))

    if prob_new is None:
        train_ind = rnd_s.permutation(ordered_train_ind)
    else:
        new_order = utils.shuffle_with_probablity(
                                        dataset.get_label(ordered_train_ind),
                                        prob_new,
                                        rnd_s)
        train_ind = ordered_train_ind[new_order]

    train_ds = ExtendedSubset(dataset, train_ind)

    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=False, collate_fn=list_collate, **dl_arg)

    return train_dl, None, None


def train_test_shuf(train_ds, test_ds, seed, dl_arg={}, prob_new=None):
    rnd = np.random.RandomState(seed)
    ordered_train_ind = np.arange(len(train_ds))

    if prob_new is None:
        train_ind = rnd.permutation(ordered_train_ind)
    else:
        new_order = utils.shuffle_with_probablity(
                                        dataset.get_label(ordered_train_ind),
                                        prob_new,
                                        rnd)
        train_ind = ordered_train_ind[new_order]

    shuf_train_ds = ExtendedSubset(train_ds, train_ind)

    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False,collate_fn=list_collate, **dl_arg)
    train_dl = torch.utils.data.DataLoader(shuf_train_ds, shuffle=False,collate_fn=list_collate, **dl_arg)

    return train_dl, test_dl


def train_test_desc_split(descriptor, test_seed):
    rnd = np.random.RandomState(test_seed)

    test_desc = copy.deepcopy(descriptor)
    train_desc = copy.deepcopy(descriptor)

    for test, train in zip(test_desc, train_desc):
        ind = rnd.randint(len(train["paths"]))
        test["paths"] = [train["paths"][ind]]
        del train["paths"][ind]

    return train_desc, test_desc


def train_test_factory(descriptor, test_seed, dl_arg, prob_new=None, pre_embed=None):

    train_desc, test_desc = train_test_desc_split(descriptor, test_seed)

    test_ds = FlattenedDataSet(test_desc, pre_embed=pre_embed)
    train_ds = FlattenedDataSet(train_desc, pre_embed=pre_embed)

    return functools.partial(train_test_shuf, train_ds, test_ds, dl_arg=dl_arg, prob_new=prob_new)


def train_val_factory(descriptor, test_seed, dl_arg,
                      remove_test=True, incremental_evaluation=None,
                      prob_new=None, pre_embed=None):
    if remove_test:
        train_desc, test_desc = train_test_desc_split(descriptor, test_seed)
    else:
        train_desc = descriptor

    train_ds = FlattenedDataSet(train_desc, pre_embed=pre_embed)

    return functools.partial(train_val_split, train_ds,
                             dl_arg=dl_arg, incremental_evaluation=incremental_evaluation,
                             prob_new=prob_new)
