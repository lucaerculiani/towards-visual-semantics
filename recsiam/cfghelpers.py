from pathlib import Path
import json

from collections import OrderedDict

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import recsiam.agent as ag
import recsiam.data as data
import recsiam.embeddings as emb
import recsiam.loss as loss
import recsiam.openworld as ow
import recsiam.utils as utils
import recsiam.models as models

from functools import partial

import torch

_EXP_DICT = {
        "seed": None,
        "remove_test" : True,
        "validation": True,
        "evaluation": False,
        "incremental_evaluation": {"number" : 5, "setting": 0.3},
        "clustering": False,
        "n_exp": 1,
        "exp_amb": False,
        "setting" : None, 
        "dataset": {"split_seed": None, "descriptor": None, "dl_args": {}, "pre_embedded": False, "metadata": [], "meta_args": [{}]},
        "ambiguity_dataset": {"split_seed": None, "descriptor": None, "dl_args": {}, "pre_embedded": False, "metadata": [], "meta_args": [{}]},
        "agent": {
                "bootstrap": 2,
                "max_neigh_check": 1,
                "fn": {"add_seen_element": "separate"},
                "remove" : {"name": "random",
                            "args" : {},
                            "seed": 2},
                "name": "online",
                "ag_args": {},
                 },
        "model": {
                "embedding": "squeezenet1_1",
                "emb_train": False,
                "pretrained": True,
                "aggregator": "mean",
                "ag_dynamic": False,
                "ag_args": {},
                "pre_embed" : True
                },
        "refine": {
                "optimizer": {"name":  "adam", "params": {}},
                "loss": {"name": "contrastive", "params": {}},
                "epochs": 1,
                "dl_args": {},
                "couples": 200
                }

}


# DATASETS

def load_dataset_descriptor(path):
    path = Path(path)

    with path.open("r") as ifile:
        return json.load(ifile)


def prep_dataset(params, base_key="dataset"):

    if isinstance(params[base_key]["descriptor"], (str, Path)):
        desc = load_dataset_descriptor(params[base_key]["descriptor"])
    else:
        desc = params[base_key]["descriptor"]

    pre_embed = None
    if params["validation"]:
        fac = data.train_val_factory(desc,
                                     params[base_key]["split_seed"],
                                     params[base_key]["dl_args"],
                                     remove_test=params["remove_test"],
                                     incremental_evaluation=params["incremental_evaluation"],
                                     prob_new=params["setting"],
                                     pre_embed=pre_embed)

    else:
        fac = data.train_test_factory(desc,
                                      params[base_key]["split_seed"],
                                      params[base_key]["dl_args"],
                                      prob_new=params["setting"],
                                      pre_embed=pre_embed)

    return fac

# MODELS


def is_dynamic(params):
    return np.any(params["model"]["ag_dynamic"])


def prep_model(params):
    
    if not is_dynamic(params):
        sseq  = torch.nn.Sequential
    else:
        sseq = models.KwargsSequential

    def instance_model():
        module_list = []

        if not params["dataset"]["pre_embedded"]:

            emb_model = emb.get_embedding(params["model"]["embedding"])

            seq_module_list = [utils.default_image_normalizer(),
                               emb_model(pretrained=params["model"]["pretrained"]),
                               models.BatchFlattener()]
            emb_m = models.SequenceSequential(*seq_module_list)
            if is_dynamic(params):
                emb_m =  models.ParamForward(emb_m)

        else:
            emb_m = sseq()
        module_list.append(("embed", emb_m))

        p_aggr = utils.as_list(params["model"]["aggregator"])
        p_ag_args = utils.as_list(params["model"]["ag_args"])
        p_is_dyn = utils.as_list(params["model"]["ag_dynamic"])
        assert len(p_aggr) == len(p_ag_args) and len(p_aggr) == len(p_is_dyn)
        if len(p_aggr) > 1:
            aggr = [models.get_aggregator(a)(**p) for a, p in zip(p_aggr, p_ag_args)]
            if is_dynamic(params):
                aggr = [m if d else models.ParamForward(m) for d, m in zip(p_is_dyn, aggr)]
            aggr = sseq(*aggr)

        else:
            aggr = models.get_aggregator(p_aggr[0])(**p_ag_args[0])
            if is_dynamic(params) and not p_is_dyn[0]:
                aggr = models.ParamForward(aggr)

        module_list.append(("aggr", aggr))

        model = sseq(OrderedDict(module_list))

        return model

    return instance_model



def get_optimizer(key):
    return getattr(torch.optim, key)


def prep_optimizer(params):

    def instance_optimizer(model):
        opt = get_optimizer(params["refine"]["optimizer"]["name"])
        m_p = (p for p in model.parameters() if p.requires_grad)
        return opt(m_p, **params["refine"]["optimizer"]["params"])

    return instance_optimizer


def prep_loss(params):

    def instance_loss(agent):
        l = loss.get_loss(params["refine"]["loss"]["name"])
        thr = ag.compute_linear_threshold(agent.sup_mem.labels, agent.sup_mem.distances)
        return l(thr, **params["refine"]["loss"]["params"])

    return instance_loss


def prep_refinement(params):
    opt_fac = prep_optimizer(params)
    loss_fac = prep_loss(params)

    return partial(ag.refine_agent,
                   opt_fac=opt_fac, loss_fac=loss_fac,
                   epochs=params["refine"]["epochs"],
                   dl_args=params["refine"]["dl_args"])


_AGENT_FACT = {"online": ag.online_agent_factory}


def get_agent_factory(key):
    return _AGENT_FACT[key]


_SEEN_FN = {"separate": ag.add_seen_separate}
get_seen_policy = _SEEN_FN.get


_AG_FN = {
        "add_seen_element" : _SEEN_FN
        }

def get_ag_fn(params):
    ag_fn_par  = params["agent"]["fn"]

    res = {}
    for k, v in ag_fn_par.items():
        res[k] = _AG_FN[k][v]

    return res


def prep_agent(params):
    ag_f = get_agent_factory(params["agent"]["name"])

    assert (not params["dataset"]["pre_embedded"]) or (not params["model"]["emb_train"])

    m_f = prep_model(params)

    kwargs = params["agent"]["ag_args"]
    kwargs = kwargs if kwargs is not None else {}

    kwargs = {**kwargs, **get_ag_fn(params)}


    if params["refine"] is not None:

        r_f = prep_refinement(params)

        kwargs["refine"] = r_f

    kwargs["dynamic_model"] = is_dynamic(params)

    return ag_f(m_f, bootstrap=params["agent"]["bootstrap"], **kwargs)


def instance_ow_exp(params):
    a_f = prep_agent(params)
    d_f = prep_dataset(params)
    ad_f = None
    s_f = ow.supervisor_factory

    return ow.OpenWorld(a_f, d_f, s_f, params["seed"], amb_dataset_factory=ad_f)


def do_experiment(agent, ds_set1, args_exp1={}, **kwargs):
    res1 = ow.do_experiment(agent, *ds_set1, **args_exp1, **kwargs)
#    if exp2_amb:
#        agent.ambiguity_detection = True
#    res2 = ow.do_experiment(agent, *ds_set2, **args_exp2, **kwargs)

    return res1#, res2

def run_ow_exp(params, workers, quiet=False, torch_threads=1):
    exp = instance_ow_exp(params)
    gen = tqdm(exp.gen_experiments(params["n_exp"]), total=params["n_exp"], smoothing=0, disable=quiet)
    if torch_threads != -1:
        torch.set_num_threads(torch_threads)

    pool = Parallel(n_jobs=workers, batch_size=1)
    keys = ["metadata", "meta_args"]
    args_exp1 = {k: params["dataset"][k] for k in keys}

    results = pool(delayed(do_experiment)(*args,
                                             do_eval=params["evaluation"],
                                             args_exp1=args_exp1)
                                             for args in gen)
    sess_res1 = [r[0] for r in results]
    eval_res1 = [r[1] for r in results]

    return (ow.stack_results(sess_res1), ow.stack_results(eval_res1))
