from __future__ import division

import torch
import numpy as np
import networkx as nx


from . import utils
from .utils import a_app, t_app


class ObjectsMemory(object):

    def __init__(self):
        self.M = torch.tensor([])

        self.G = nx.Graph()
        self.seq_ids = np.array([])

        self.ambiguous = np.array([], dtype=np.bool)
        self.last = -1

    @property
    def empty(self):
        return len(self.M) == 0

    def __len__(self):
        return len(self.M)

    @property
    def sequences(self):
        return len(self.G.nodes)

    @property
    def clusters(self):
        return cc2clusters(self.G)

    def get_embed(self, indices):
        return self.M[indices, ...]

    def get_mask_from_sid(self, sids):
        return np.isin(self.seq_ids, sids)

    def get_sid(self, indices):
        return self.seq_ids[indices]

    def get_distances(self, element):
        if len(element.shape) == 1:
            element = element[None, ...]
        distances = cart_euclidean_using_matmul(self.M, element)

        return distances

    def get_knn(self, element, k=1):
        dist = self.get_distances(element).t()

        t_k = torch.topk(dist, k=k, largest=False, sorted=True)

        if t_k[0].shape == ():
            raise ValueError

        return t_k

    def add_new_element(self, element, s_id, ambiguous=None):

        assert self.last + 1 == s_id
        self.last = s_id

        if ambiguous is None:
            ambiguous = np.zeros(element.shape[0], dtype=np.bool)
        assert len(ambiguous) == element.shape[0]

        self.ambiguous = a_app(self.ambiguous, ambiguous)

        a_id = np.tile(s_id, element.shape[0])

        self.seq_ids = a_app(self.seq_ids, a_id, ndim=1)

        self.M = t_app(self.M, element, ndim=2)
        self.G.add_node(s_id)

    def update_ambiguous(self, s_id, distances, threshold):
        cl = cc2clusters(self.G)
        nearmat  = distances < threshold

        emb_mask = self.get_mask_from_sid(np.where(cl == cl[s_id])[0])
        new_mask = self.get_mask_from_sid(s_id)
        emb_mask = emb_mask[~new_mask]
        new_amb = ~((nearmat <= emb_mask[None, :]).all(axis=1))
        old_amb = (nearmat.T.any(axis=1) & ~ emb_mask)

        self.ambiguous[new_mask] |= new_amb
        self.ambiguous[~new_mask] |= old_amb


    def add_neighbors(self, s_id, targets, by_sid=False):

        if not by_sid:
            self.G.add_edges_from([(s_id, self.seq_ids[t]) for t in targets])
        else:
            self.G.add_edges_from([(s_id, t) for t in targets])

    def remove_targets(self, targets):
        keep = _t(self.M.shape[0])
        keep[targets] = False

        self.seq_ids = self.seq_ids[keep]
        self.M = self.M[torch.ByteTensor(keep.astype(np.uint8))]
 

class SupervisionMemory(torch.utils.data.Dataset):

    def __init__(self):
        self.couples = []
        self.labels = np.array([], dtype=np.int32)
        self.distances = np.array([])

        self.ins_cnt = 0
        self.insertion_orders = np.array([])

    def __len__(self):
        return len(self.distances)

    def __getitem__(self, idx):
        return self.couples[idx], self.labels[idx]

    def add_entry(self, new_data, labels, distance):
        pos = np.searchsorted(self.distances, distance)

        self.labels = np.insert(self.labels, pos, labels, axis=0)
        self.distances = np.insert(self.distances, pos, distance, axis=0)

        self.insertion_orders = np.insert(self.insertion_orders, pos, self.ins_cnt, axis=0)
        self.ins_cnt += 1

    def del_entry(self, pos=None):

        if pos is None:
            pos = np.argmin(self.insertion_orders)

        self.labels = np.delete(self.labels, pos, axis=0)
        self.distances = np.delete(self.distances, pos, axis=0)

        self.insertion_orders = np.delete(self.insertion_orders, pos, axis=0)

        del self.couples[pos]



def recompute_supervision_memory(agent):
    old_sup_mem = agent.sup_mem
    new_sup_mem = SupervisionMemory()

    with torch.no_grad():
        for i in np.argsort(old_sup_mem.insertion_orders):
            couple, lab = old_sup_mem[i]
            d = torch.norm(agent.model.forward([couple[0]])[0] - couple[1])

            new_sup_mem.add_entry([couple], [lab], [utils.t2a(d)])

    return new_sup_mem

def compute_linear_threshold(gt, dgt):
    t_cs = gt.cumsum() + np.logical_not(gt)[::-1].cumsum()[::-1]

    t_indexes = np.where(t_cs == t_cs.max())[0]

    t_ind = t_indexes[len(t_indexes) // 2]

    overflowing = ((t_ind == 0) and not gt[t_ind]) or \
                  ((t_ind == (len(t_cs) - 1)) and gt[t_ind])

    if not overflowing:
        other_ind = t_ind + gt[t_ind]*2 - 1
        threshold = (dgt[t_ind] + dgt[other_ind]) / 2.0

    else:
        threshold = dgt[t_ind] / 1.05 if t_ind == 0 else dgt[t_ind] * 1.05

    return threshold


def compute_thresolds_from_indexes(gt, dgt, indexes, w_sz):
    l_ind = indexes[len(indexes) // 2]
    u_ind = l_ind + w_sz - 1

    assert l_ind >= 0
    assert u_ind <= len(dgt) - 1

    l_thr = dgt[l_ind-1:l_ind+1].mean() if l_ind > 0 else dgt[l_ind] / 2. 
    u_thr = dgt[u_ind:u_ind+2].mean() if u_ind < len(dgt) - 1 else dgt[u_ind] * 2. 

    return l_thr, u_thr, l_ind, u_ind


def binary_entropy(p):
    eps = 1e-7
    corr_p = p + np.where(p < eps, eps, 0)
    corr_p = corr_p - np.where(corr_p > (1 - eps), eps, 0)
    p = corr_p
    entropy =  -( p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps)  )

    return entropy

def _compute_subtract_entropy_thresholds(gt, dgt, w_sz):

    gt = gt.astype(np.bool)
    c_win = np.ones(w_sz)

    ara = np.arange(1, gt.size + 1, dtype=np.float64)
    w_ent = binary_entropy(np.convolve(gt, c_win, mode='valid') / w_sz)

    eps_ent = binary_entropy(0.0)
    lb_entropy = binary_entropy(gt[:-w_sz].cumsum() / ara[:-w_sz])
    lb_entropy = np.insert(lb_entropy, 0, eps_ent)

    ub_entropy = binary_entropy((~ gt[w_sz:])[::-1].cumsum() / ara[:-w_sz] )[::-1]
    ub_entropy = np.append(ub_entropy, eps_ent)

    w_div_b = w_ent - lb_entropy - ub_entropy

    indexes = np.where(w_div_b == w_div_b.max())[0]

    res = compute_thresolds_from_indexes(gt, dgt, indexes, w_sz)

    return res +  (w_div_b[res[2]],)


def compute_subtract_entropy_thresholds(gt, dgt, fraction):
    w_sz = max(np.round(len(gt) / fraction**(-1)).astype(int), 1)

    return _compute_subtract_entropy_thresholds(gt, dgt, w_sz)


def cc2clusters(G):

    cl = np.arange(len(G.nodes))
    cc_id = 0
    for cc in nx.connected_components(G):
        for node in cc:
            cl[node] = cc_id

        cc_id += 1

    return cl


def cart_euclidean_using_matmul(e0, e1, use_cuda=False):
    if use_cuda:
        e0 = e0.cuda()
        e1 = e1.cuda()

    e0t2 = (e0**2).sum(dim=1).expand(e1.shape[0], e0.shape[0]).t()
    e1t2 = (e1**2).sum(dim=1).expand(e0.shape[0], e1.shape[0])

    e0e1 = torch.matmul(e0, e1.t())

    _EPSILON = 1e-7

    l2norm = torch.sqrt(torch.clamp(e0t2 + e1t2 - (2 * e0e1), min=_EPSILON))

    return l2norm.cpu()


# Versatile but inefficient implementation.
# Given that the threshold does not vary in this phase,
# this could be done incrementally (as in the pseudocode of the paper)
def compute_ambiguous_old(memory, threshold):
    cl = cc2clusters(memory.G)
    
    nearmat = cart_euclidean_using_matmul(memory.M, memory.M) < threshold
    nearmat = nearmat.numpy()

    amb_mask = np.zeros(nearmat.shape[0], dtype=np.bool)
    for c in np.unique(cl):
        emb_mask = memory.get_mask_from_sid(np.where(cl == c)[0])

        # implication: neramat -> emb_mat
        new_amb = ~((nearmat <= emb_mask[:, None]).all(axis=0) | (~emb_mask))
        
        amb_mask = amb_mask | new_amb

    return amb_mask

def compute_incremental_ambiguous(memory, elem,  threshold):
    cl = cc2clusters(memory.G)
    
    nearmat = cart_euclidean_using_matmul(elem, memory.M) < threshold
    nearmat = nearmat.numpy()

    amb_mask = np.zeros(nearmat.shape[0], dtype=np.bool)
    for c in np.unique(cl):
        emb_mask = memory.get_mask_from_sid(np.where(cl == c)[0])

        # implication: neramat -> emb_mat
        new_amb = ~((nearmat <= emb_mask[:, None]).all(axis=0) | (~emb_mask))
        
        amb_mask = amb_mask | new_amb

    return amb_mask

class RandomRemover():

    def __init__(self, seed, fraction=None, number=None):
        f_none = fraction is None 
        n_none = number is None 

        assert not (f_none and n_none)
        assert f_none or n_none

        self.fraction = fraction
        self.number = number
        self.seed = seed
        self.rnd = np.random.RandomState(seed)

    def __call__(self, agent, new, old, same_obj, *args, **kwargs):
        cat = np.concatenate([new, old]) if same_obj else new
        samples = None
        if self.number is not None and cat.size > self.number:
            samples = self.rnd.choice(cat, size=cat.size - self.number, replace=False)

        elif self.fraction is not None:
            samples = self.rnd.choice(cat, size=round(cat.size * (1. - self.fraction)), replace=False)

        if samples is not None:
            agent.obj_mem.remove_targets(samples)


class GlobalRandomRemover(RandomRemover):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cnt = 0

    def __call__(self, agent, *args, **kwargs):
        seq_n = agent.obj_mem.sequences
        samples = None
        if self.number is not None and seq_n > self.number:
            self.cnt += 1
            samples = self.rnd.choice(seq_n, size=seq_n - self.number * self.cnt, replace=False)

        elif self.fraction is not None:
            samples = self.rnd.choice(seq_n, size=round(seq_n * (1. - self.fraction)), replace=False)

        if samples is not None:
            agent.obj_mem.remove_targets(samples)


class SparsityBasedRemover(RandomRemover):

    def __call__(self, agent, new, old, same_obj, **kwargs):

        number = None
        shapesum = new.shape[0] + old.shape[0] if same_obj else new.shape[0]
        if self.number is not None and shapesum > self.number:
            number = shapesum - self.number
        elif self.fraction is not None:
            number = round(shapesum * (1. - self.fraction))

        if number is not None:
            cat = np.concatenate([new, old]) if same_obj else new
            concatenation = agent.obj_mem.get_embed(cat)

            sp_likelihood = sparsity_likelihood(concatenation, probability=False)
            indices = choice_with_likelihood(sp_likelihood, number, self.rnd)

            agent.obj_mem.remove_targets(cat[indices])


def sparsity_likelihood(embeds, probability=False):
    distmat = utils.t2a(cart_euclidean_using_matmul(embeds, embeds)) ** 2
    distmat = distmat.mean(axis=1) ** .5

    distmat = distmat ** -1
    if not probability:
        return distmat
    else:
        return distmat / distmat.sum()


def choice_with_likelihood(likelihoods, size, rnd):
    assert len(likelihoods) >= size

    norm = - (np.asarray(likelihoods) * rnd.uniform(size=len(likelihoods)))

    indices = np.argpartition(norm, np.arange(size))[:size]

    return indices


def confusion_likelihood(embeds, others, probability=False):
    distmat = utils.t2a(cart_euclidean_using_matmul(embeds, others))
    distmat = distmat.min(axis=1)

    if not probability:
        return - distmat
    else:
        distmat = distmat ** -1 
        return distmat / distmat.sum()



class ConfusionBasedRemover(RandomRemover):

    def __call__(self, agent, new, old, same_obj,  **kwargs):
        number = None
        shapesum = new.shape[0] + old.shape[0] if same_obj else new.shape[0]
        if self.number is not None and shapesum > self.number:
            number = shapesum - self.number
        elif self.fraction is not None:
            number = round(shapesum * (1. - self.fraction))

        if number is not None:
            if not same_obj:

                c_likelihood = confusion_likelihood(agent.obj_mem.get_embed(new),
                                                    agent.obj_mem.get_embed(old))

                indices = choice_with_likelihood(c_likelihood, number, self.rnd)
                agent.obj_mem.remove_targets(new[indices])
            else:
                cat = np.concatenate([new, old])
                indices = self.rnd.choice(cat, size=number, replace=False)
                agent.obj_mem.remove_targets(indices)


_REMOVERS = {"random": RandomRemover,
             "global_random": GlobalRandomRemover,
             "sparsity": SparsityBasedRemover,
             "confusion": ConfusionBasedRemover}


def get_remover(key):
    return _REMOVERS[key]
