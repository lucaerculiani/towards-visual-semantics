from __future__ import division

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx

from functools import partial

from ignite.engine import Engine, Events

from . import utils
from .utils import a_app, t_app
from .sampling import SeadableRandomSampler

import logging

from . import memory as mem


def online_agent_template(model_factory, seed, supervisor,
                          bootstrap, max_neigh_check, **kwargs):
    o_mem = mem.ObjectsMemory()
    s_mem = mem.SupervisionMemory()
    model = model_factory()
    ag = Agent(seed, o_mem, s_mem, model, supervisor, bootstrap, max_neigh_check, 
               **kwargs)

    return ag


def online_agent_factory(model_factory, **kwargs):
    return partial(online_agent_template, model_factory, **kwargs)


def _t(shape):
    return np.ones(shape, dtype=np.bool)


_T = _t(1)


def _f(shape):
    return np.zeros(shape, dtype=np.bool)


_F = _f(1)


class Agent(object):

    def __init__(self, seed, obj_mem, sup_mem, model,
                 supervisior, bootstrap, max_neigh_check,
                 dynamic_model=False,
                 add_seen_element=utils.default_notimplemented,
                 propagate_genus=False,
                 supervision_rate=1.0,
                 th_scale=1.0,
                 th_gen_scale=1.0,
                 remove=utils.default_ignore,
                 random=False):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.random = random
        self.random_rng = np.random.RandomState(np.random.RandomState(seed).randint(0, 2**32 - 1))
        self.obj_mem = obj_mem
        self.sup_mem = sup_mem
        self.model = model
        self.dynamic = dynamic_model
        self.supervisor = supervisior


        self.bootstrap = bootstrap
        self.max_neigh_check = max_neigh_check

        # functions
        self.add_seen_element = add_seen_element
        self.add_new_element = self.obj_mem.add_new_element

        self.propagate_genus = propagate_genus
        self._remove = remove
        self.supervision_rate = supervision_rate

        self.on_cuda = False
        self.ambiguity_detection = False
        self.th_scale = th_scale
        self.th_gen_scale = th_gen_scale

    def tocuda(self):
        self.model.cuda()
        self.on_cuda = True

    def tocpu(self):
        self.model.cpu()
        self.on_cuda = False

    def in_bootstrap(self):
        return self.obj_mem.sequences < self.bootstrap

    def process_next(self, data, s_id, disable_supervision=_F):
        return self._process_next(data, s_id, disable_supervision=disable_supervision)

    def forward(self, data):
        self.model.eval()

        if self.on_cuda:
            data = [d.cuda() for d in data]

        with torch.no_grad():
            if self.dynamic:
                embed = self.model(data, agent=self)[0]
            else:
                embed = self.model(data)

        return embed

    def ask_for_supervision(self, size):
        return self.rng.uniform(size=size) < self.supervision_rate

    def _process_next(self, data, s_id, disable_supervision=_F):
        embed = self.forward(data)[0]

        if self.obj_mem.empty:
            self.obj_mem.add_new_element(embed, s_id)
            return _F, np.array([s_id]), _F, -3  # bogus values

        best, disc_best = self.get_neighbors(embed)
        if  self.random:
            value = self.random_rng.randint(-1, 2) 
            if value == -1:
                return _F, _F, best[2], False
            elif value == 0:
                return _T, _F, best[2], False
            else:
                return _T, _T, disc_best[2], False

        if len(self.sup_mem) > 0:
            inst_is_known = self.decide(disc_best[0], False)
            genus_is_known = self.decide(best[0], True) | inst_is_known
        else:
            inst_is_known = _f(disc_best[1].size)
            genus_is_known = _f(best[1].size)

        same_inst = inst_is_known.copy()
        same_genus = genus_is_known.copy()

        all_dists = best[3]
        if same_inst[0]:
            best_k_dist, best_k, nearest_s_id = disc_best[:3]
        else:
            best_k_dist, best_k, nearest_s_id = best[:3]

        ask_supervision = self.ask_for_supervision(same_inst.shape)
        ask_supervision &= ~ disable_supervision
        ask_supervision |= self.in_bootstrap()

        sup = None
        if ask_supervision[0]:
            ask_indices = 0
            sup = self.supervisor.ask(nearest_s_id[ask_indices], s_id)

        new_amb = None

        if (not self.in_bootstrap()) and sup == 0 and self.th_gen_scale > 0:
            if not self.propagate_genus:
                neighbors = nearest_s_id
            else:
                neighbors = list(self.obj_mem.G[nearest_s_id[0]]) + [nearest_s_id[0]]

            targets = self.obj_mem.get_mask_from_sid(neighbors)

            amb = all_dists[:, targets].min(axis=0) < self.linear_threshold
            #amb = all_dists[:, targets].min(axis=0) < 10e10
            #amb = all_dists.min(axis=0) < self.linear_threshold
            new_amb = all_dists[:, targets].min(axis=1) < self.linear_threshold
#            new_amb = _t(new_amb.size)

            self.obj_mem.ambiguous[targets] |= amb
            #self.obj_mem.ambiguous |= amb

        self.obj_mem.add_new_element(embed, s_id, ambiguous=new_amb)

        if same_inst.any():
            self.obj_mem.add_neighbors(s_id, disc_best[1][same_inst])

        if ask_supervision.any() and (self.th_gen_scale > 0 or sup != 0):

            sup_data = None
            self.sup_mem.add_entry(sup_data, sup >= 0, best_k_dist[ask_indices])

        return genus_is_known, inst_is_known, nearest_s_id, sup is not None
        #return genus_is_known | True, inst_is_known , best[2], sup is not None

    def predict(self, *args, **kwargs):
        thr_pred = self.predict_distance_learning(*args, **kwargs)

        if not self.ambiguity_detection:
            return thr_pred
        else:
            amb_pred = self.predict_ambiguity(*args, **kwargs)
            return thr_pred + amb_pred

    def predict_distance_learning(self, data, lengths=None, skip_error=False):

        if lengths is None:
            lengths = [None for d in data]

        if self.obj_mem.empty:
            if not skip_error:
                raise Exception("the object's memory is empty!")
            else:
                z = np.zeros(len(lengths))
                return z.astype(np.bool), z.astype(int)

        if len(self.sup_mem) == 0:
            if not skip_error:
                raise Exception("the supervision memory is empty!")
            else:
                z = np.zeros(len(lengths))
                return z.astype(np.bool), z.astype(int)

        fded = self.forward(data)

        if isinstance(fded, list):
            lengths = [len(d) for d in fded]
            fded = torch.cat(fded)
        else:
            raise ValueError("output of model is not list")

        lengths = np.cumsum(lengths)

        dist = utils.t2a(self.obj_mem.get_distances(fded).t())

        all_nn = dist.argmin(axis=1)
        all_nn_d = dist.min(axis=1)

        agg_nn = utils.reduce_packed_array(all_nn_d, lengths)

        real_nn = agg_nn.copy()
        real_nn[1:] += lengths[:-1]

        real_nn_dist = all_nn_d[real_nn]

        is_known, ask_supervision = self.decide(real_nn_dist)
        neighbors = all_nn[real_nn]

        return is_known, self.obj_mem.get_sid(neighbors)

    def predict_ambiguity(self, data, lengths=None, skip_error=False):

        if lengths is None:
            lengths = [None for d in data]

        fded = self.forward(data)

        if isinstance(fded, list):
            lengths = [len(d) for d in fded]
            fded = torch.cat(fded)
        else:
            raise ValueError("output of model is not list")

        lengths = np.cumsum(lengths)

        dist = utils.t2a(self.obj_mem.get_distances(fded).t())
        discr = ~self.obj_mem.ambiguous

        dist[:, discr] = dist.max() + 1

        all_nn = dist.argmin(axis=1)
        all_nn_d = dist.min(axis=1)

        agg_nn = utils.reduce_packed_array(all_nn_d, lengths)

        real_nn = agg_nn.copy()
        real_nn[1:] += lengths[:-1]

        real_nn_dist = all_nn_d[real_nn]

        is_known, ask_supervision = self.decide(real_nn_dist)
        neighbors = all_nn[real_nn]

        return is_known, self.obj_mem.get_sid(neighbors)

    def get_neighbors(self, new_elem, ret_distances=True):
        dist = utils.t2a(self.obj_mem.get_distances(new_elem).t())
        assert dist.shape == (new_elem.shape[0],
                              self.obj_mem.ambiguous.shape[0])

        if self.max_neigh_check == 1:
            ind = np.unravel_index(np.argmin(dist),
                                   dist.shape)
            min_dist = dist[ind][None, ...]
            s_ids = self.obj_mem.get_sid(ind[1][None, ...])

            disc_dist = dist[:, ~ self.obj_mem.ambiguous]
            if disc_dist.size > 0:
                disc_ind = np.unravel_index(np.argmin(disc_dist),
                                            disc_dist.shape)
                disc_min_dist = disc_dist[disc_ind][None, ...]
            else:
                disc_ind = np.unravel_index(np.argmin(dist),
                                            dist.shape) 
                disc_min_dist = np.array((10e100,))
            disc_s_ids = self.obj_mem.get_sid(disc_ind[1][None, ...])

            if not ret_distances:
                return ((min_dist, ind[1][None, ...], s_ids),
                        (disc_min_dist, disc_ind[1][None, ...], disc_s_ids))
            else:
                return ((min_dist, ind[1][None, ...], s_ids, dist),
                        (disc_min_dist, disc_ind[1][None, ...], disc_s_ids,
                         disc_dist))

        else:
            raise NotImplementedError()

    def decide(self, distance, genus=None):
        distance = np.asarray(distance)
        thr = self.linear_threshold

        if genus is not None:
            thr = thr * self.th_gen_scale if genus else thr * self.th_scale
        return thr > distance

    @property
    def linear_threshold(self):
        if len(self.sup_mem) > 0:
            thr = mem.compute_linear_threshold(self.sup_mem.labels, self.sup_mem.distances)
        else:
            thr = 1e10

        return thr


def add_seen_separate(elem, s_id, same_as, agent):
    agent.obj_mem.add_new_element(elem, s_id)
    agent.obj_mem.add_neighbors(s_id, same_as)
