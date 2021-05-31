from __future__ import division

import logging
import itertools
import numpy as np
import torch
from .memory import cc2clusters
from . import utils
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score 


_OBJ_ID = "obj_id"
_SEQ_ID  = "s_id"

class Supervisor(object):
    def __init__(self, knowledge, k_range=None):

        self.knowledge = utils.as_list(knowledge)
        self.k_range = k_range
        if self.k_range is None:
            self.k_range = [range(len(knowledge))]

        self.mapping = self.compute_mapping()
        self.get_label = np.vectorize(self.get_one_label)

    def compute_mapping(self):
        base = np.array((), dtype=np.int32)
        k_dict = {}

        for itx, r in enumerate(self.k_range):
            a = np.arange(r.start, r.stop, r.step)
            assert np.intersect1d(base, a).size == 0
            base = utils.a_app(base, a)
            k_dict.update(zip(a, zip([itx] * len(a), range(len(a)))))

        assert base.size == np.unique(base).size
        return k_dict

    def get_one_label(self, s_id):
        coord = self.mapping[s_id]
        label = self.knowledge[coord[0]].get_label(coord[1])
        return label

    def get_one_meta(self, s_id, key, object_level=True):
        coord = self.mapping[s_id]
        meta = self.knowledge[coord[0]].get_metadata(key, coord[1],
                                                     object_level=object_level)
        return meta

    def ask(self, s_id1, s_id2):
        id1_amb = self.get_one_meta(s_id1, "ambiguous", False)
        id2_amb = self.get_one_meta(s_id2, "ambiguous", False)

        id1_class = self.get_one_meta(s_id1, "class", True)
        id2_class = self.get_one_meta(s_id2, "class", True)

        id1_lab = self.get_label(s_id1)
        id2_lab = self.get_label(s_id2)

        if id1_class != id2_class:
            return -1

        elif id1_amb or id2_amb:
            return 0

        else:
            return 1 if id1_lab == id2_lab else 0


def supervisor_factory(dataset, ranges=None):
    return Supervisor(dataset, ranges)


class OpenWorld(object):

    def __init__(self, agent_factory, dataset_factory, supervisor_factory, seed,
                 amb_dataset_factory=None):
        self.agent_factory = agent_factory
        self.dataset_factory = dataset_factory
        self.supervisor_factory = supervisor_factory
        self.seed = seed

        self.rnd = np.random.RandomState(self.seed)
        self.exp_seed, self.env_seed = self.rnd.randint(2**32, size=2)
        self.amb_seed = self.rnd.randint(2**32)

    def gen_experiments(self, n_exp):

        exp_seeds = get_exp_seeds(self.exp_seed, n_exp)
        env_seeds = get_exp_seeds(self.env_seed, n_exp)

        for exp_s, env_s in zip(exp_seeds, env_seeds):

            session_ds, eval_ds, inc_eval_ds = self.dataset_factory(env_s)
            supervisor = self.supervisor_factory(session_ds.dataset)
            agent = self.agent_factory(exp_s, supervisor)

            s_range = range(len(session_ds))
            len_inc = 0 if inc_eval_ds is None else len(inc_eval_ds)
            inc_range = range(s_range.stop, s_range.stop + len_inc)

            yield agent, ((session_ds, s_range), eval_ds, (inc_eval_ds,
                                                           inc_range))


def get_exp_seeds(seed, n_exp):
    n_exp = np.asarray(n_exp)

    if n_exp.shape == ():
        n_exp = np.array([0, n_exp])

    elif n_exp.shape == (2,):
        pass

    else:
        raise ValueError("shape of n_exp is {}".format(n_exp.shape))

    rnd = np.random.RandomState(seed)
    seeds = rnd.randint(2**32, size=n_exp[1])[n_exp[0]:]

    return seeds


def counter(start=0):
    while True:
        yield start
        start += 1


def do_experiment(agent, session_seqs, eval_seqs, inc_eval_seqs, do_eval,
                  metadata=[], meta_args=[{}]):
    logger = logging.getLogger("recsiam.openworld.do_experiment")

    logger.debug("session = {}\tsession len = {}\tsession range".format(session_seqs[0], len(session_seqs[0]), session_seqs[1]))

    session_pred = []
    session_id = []
    session_neigh_id = []
    session_class = []
    session_sup = []
    session_n_ask = []
    session_n_pred = []

    session_embeds = []

    session_metadata = [[] for _ in metadata]

    eval_pred = []
    eval_class = []
    eval_neigh_id = []
    eval_amb_pred = []
    eval_amb_neigh_id = []

    eval_metadata = [[] for _ in metadata]

    ev_data = []
    ev_data_len = []

    if do_eval:
        logger.debug("started evaluation data preload")
        for ds_ind, (data, obj_id) in enumerate(eval_seqs):
            eval_class.append(obj_id)

            ev_data.append(data[0])
            ev_data_len.append(len(data[0]))
            for i, (m, m_a) in enumerate(zip(metadata, meta_args)):
                meta = eval_seqs.dataset.get_metadata(m, ds_ind, **m_a)
                eval_metadata[i].append(meta)
                logger.debug("obj_id={}\tmet_keys = {}\tmeta_values = {}".format(obj_id, m, meta))

    logger.debug("started session")
    for ds_ind, ((data, obj_id), s_id) in enumerate(zip(*session_seqs)):

        for i, (m, m_a) in enumerate(zip(metadata, meta_args)):
            meta = session_seqs[0].dataset.get_metadata(m, ds_ind, **m_a)
            session_metadata[i].append(meta)
            logger.debug("ds_ind ={}\tobj_id={}\tmet_keys = {}\tmeta_values = {}".format(ds_ind, obj_id, m, meta))

        # process next video
        genus_pred, inst_pred, all_n_s_id, sup = agent.process_next(data, s_id)

        pred = -1 + genus_pred + inst_pred
        n_s_id = all_n_s_id[0]

        session_pred.append(pred)
        session_neigh_id.append(n_s_id)
        session_id.append(s_id)
        session_class.append(obj_id)
        session_sup.append(sup)

        # validate / test
        if do_eval:
            e = np.array(agent.predict(ev_data, skip_error=True))
            eval_pred.append(e[0, :].astype(np.bool))
            eval_neigh_id.append(e[1, :])

            if e.shape[0] > 2:
                eval_amb_pred.append(e[2, :].astype(np.bool))
                eval_amb_neigh_id.append(e[3, :])
            else:
                eval_amb_pred.append(e[0, :].astype(np.bool))
                eval_amb_neigh_id.append(e[1, :])

        session_embeds.append(agent.obj_mem.sequences)


    session_embeds = np.array(session_embeds)
    session_embeds[1:] -= session_embeds[:-1]

    s_d = {"pred": np.squeeze(session_pred), "neigh": np.squeeze(session_neigh_id),
           _SEQ_ID: np.squeeze(session_id),   "sup": np.squeeze(session_sup),
           _OBJ_ID: np.squeeze(session_class),
           "n_ask": np.squeeze(session_n_ask), "n_pred": np.squeeze(session_n_pred),
           "n_embed": np.array([agent.obj_mem.sequences]),
           "embeds": session_embeds,
           "famb": agent.obj_mem.ambiguous.mean(), 
           **{m: np.asarray(v) for m, v in zip(metadata, session_metadata)}
           }

    e_d = {"pred": np.squeeze(eval_pred), "neigh": np.squeeze(eval_neigh_id),
            _OBJ_ID: np.squeeze(eval_class), "ambpred": np.squeeze(eval_amb_pred),
            "ambneigh": np.squeeze(eval_amb_neigh_id),
           **{m: np.asarray(v) for m, v in zip(metadata, eval_metadata)}
           }

    return s_d, e_d


def stack_results(res_l):

    stacked = {}
    for key in res_l[0]:
        if len(res_l) > 1:
            stacked[key] = np.array([r[key] for r in res_l])
        else:
            stacked[key] = res_l[0][key][None, ...]

    return stacked


def maybe_unsqueeze_seq(seq):
    assert seq.ndim <= 2 and seq.ndim > 0

    if (seq.ndim) == 1:
        seq = seq[None, ...]

    return seq


def new_obj_in_seq(seq):

    seq = maybe_unsqueeze_seq(seq)

    new_obj = np.zeros(seq.shape, dtype=np.bool)

    for s_ind in range(seq.shape[0]):
        s = set()

        for e, i in zip(seq[s_ind], range(len(seq[s_ind]))):
            new_obj[s_ind, i] = e not in s
            s.add(e)

    return new_obj


def known_class_mat(seq, new_obj):
    seq = maybe_unsqueeze_seq(seq)
    new_obj = maybe_unsqueeze_seq(new_obj)

    return np.array([known_class_mat_onerow(seq[itx], new_obj[itx]) for itx in range(seq.shape[0])])


def known_class_mat_onerow(seq, new_obj):
    uniq = np.unique(seq)
    uniq.sort()
    assert (uniq == np.arange(len(uniq))).all()

    k_m = np.zeros((uniq.size, seq.size)).astype(np.bool)
    for i in np.where(new_obj)[0]:
        k_m[seq[i], i] = True

    return k_m



def is_same_class(neigh, classes, eval_classes=None):

    neigh = maybe_unsqueeze_seq(neigh)
    classes = maybe_unsqueeze_seq(classes)

    if eval_classes is None:
        return np.array([c_r == c_r[n_r]
                         for n_r, c_r in zip(neigh, classes)])
    else:
        eval_classes = maybe_unsqueeze_seq(eval_classes)
        assert len(eval_classes.shape) == len(neigh.shape)
        assert classes.shape[0] == 1
        assert eval_classes.shape[0] == 1
        resa = np.array([eval_classes[0] == classes[0][n_r]
                         for n_r in  neigh])
    return resa


def session_accuracy(s_d, by_step=False):
    same_obj = is_same_class(s_d["neigh"], s_d[_OBJ_ID])
    same_genus = s_d["class"] == np.take_along_axis(s_d["class"], s_d["neigh"], axis=1)  

    is_discr = ~s_d["ambiguous"]

    new_objs = np.concatenate([new_obj_in_seq(s) for s in s_d[_OBJ_ID]])
    fake_obj_id = s_d[_OBJ_ID].max() + 1
    discr_objs = np.where(is_discr, s_d[_OBJ_ID], fake_obj_id)
    new_discr_objs = np.concatenate([new_obj_in_seq(s) for s in discr_objs])

    tmp_discr_obj_mat = known_class_mat(discr_objs, new_discr_objs)[:, :-1, :].cumsum(axis=2).astype(np.bool)
    discr_obj_mat = np.zeros(tmp_discr_obj_mat.shape, dtype=np.bool)
    discr_obj_mat[:, :, 1:] = tmp_discr_obj_mat[:, :, :-1]

    ax = 0 if by_step else 1 

    seen_discr = np.array([d[s, np.arange(d.shape[1])]
                           for d, s in zip(discr_obj_mat, s_d[_OBJ_ID])])
    true_unk = (new_objs & (s_d["pred"] < 0)).sum(axis=ax)

    true_inst_known = ((~ new_objs) & (s_d["pred"] > 0) & same_obj & is_discr & seen_discr).sum(axis=ax)


    valid_gen = ~(is_discr & seen_discr)
    true_gen_known = ((~ new_objs) & (s_d["pred"] == 0) & same_genus & valid_gen).sum(axis=ax)

    return (true_unk + true_inst_known + true_gen_known) / float(s_d[_OBJ_ID].shape[ax])


def _get_cl_gt(s_d):
    cl = s_d["cc"][s_d["is_eval"]]
    gt = s_d[_OBJ_ID][s_d["is_eval"]]
    return cl, gt


def evaluation_clustering(s_d, fn):
    cl, gt = s_d["cc"], s_d[_OBJ_ID]

    cl = cl[:, -1, -gt.shape[1]:]

    metric_l = []
    for c, g in zip(cl, gt):
        metric = fn(c, g)
        metric_l.append(metric)

    return np.array(metric_l)


def eval_ari(s_d):
    return evaluation_clustering(s_d, adjusted_rand_score)


def eval_ami(s_d):
    return evaluation_clustering(s_d, adjusted_mutual_info_score)


def session_ari(s_d):
    session_ari = []
    for cc, cl in zip(s_d["cc"], s_d[_OBJ_ID]):
        ari_l = [1.0]
        for i in range(1, cc.shape[0]):
            ari = adjusted_rand_score(cc[i, :i], cl[:i])
            ari_l.append(ari)

        session_ari.append(ari_l)

    return np.array(session_ari)


def session_ami(s_d):
    session_ari = []
    for cc, cl in zip(s_d["cc"], s_d[_OBJ_ID]):
        ari_l = [1.0]
        for i in range(1, cc.shape[0]):
            ari = adjusted_mutual_info_score(cc[i, :i], cl[:i])
            ari_l.append(ari)

        session_ari.append(ari_l)

    return np.array(session_ari)


def pad_to_dense(M):

    maxlen = max(len(r) for r in M)

    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        Z[enu, :len(row)] += row
    return Z
