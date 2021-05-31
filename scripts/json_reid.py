import argparse
import logging
import itertools
import json
import tempfile
import torch
import numpy as np
from pathlib import Path
import recsiam.cfghelpers as cfg
import recsiam.agent as ag

import copy

import ignite
import recsiam

from torch.utils.data import DataLoader, ConcatDataset
from recsiam.data import FlattenedDataSet, ExtendedSubset

import recsiam.loss as loss


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class FunctionalModule(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def prep_net():
    m = [recsiam.models.GlobalMean(), FunctionalModule(torch.cat), MLP(2048, 2048, 2048)]

    return torch.nn.Sequential(*m)


class ProtorypeDistancesLoss(torch.nn.Module):
    def __init__(self, prototype):
        super().__init__()
        self.prototype = prototype


    def forward(self, x1, x2):
        pass


def set_torch_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def instance_loss(params):

    l = loss.get_loss(params["refine"]["loss"]["name"])
    return l(**params["refine"]["loss"]["params"])


def divide_dataset(params):
    if isinstance(params["dataset"]["descriptor"], (str, Path)):
        desc = cfg.load_dataset_descriptor(params["dataset"]["descriptor"])
    else:
        desc = params["dataset"]["descriptor"]

    pre_embed = None
    if params["model"]["pre_embed"]:
        pre_embed = cfg.prep_model(params)()

    dataset = FlattenedDataSet(desc, pre_embed=pre_embed)

    all_datasets = dataset_split(dataset, params["test_seed"], params["incremental_evaluation"]["number"])

    return all_datasets


def dataset_split(dataset, seed, ow_obj):
    rs = np.random.RandomState
    rnd_s, rnd_e, rnd_i = [rs(s) for s in rs(seed).randint(2**32 - 1, size=3)]

    base_subset = ExtendedSubset(dataset, rnd_e.permutation(len(dataset)))
    ow_test, cw_subset = base_subset.split_n_objects(ow_obj, rnd_s)

    cw_test, cw_train = cw_subset.split_balanced(2, rnd_s)

    cw_test_repr, cw_test_emb = cw_test.split_balanced(1, rnd_s)
    ow_test_repr, ow_test_emb = ow_test.split_balanced(1, rnd_s)

    return cw_train, cw_test_repr, cw_test_emb, ow_test_repr, ow_test_emb



def reid_collate(data):
    emb = [recsiam.utils.astensor(d[0]).cuda() for d in data]
    lab = recsiam.utils.astensor(np.array([d[1] for d in data])).cuda()

    return emb, lab


def train_collate(data):
    emb1 = [recsiam.utils.astensor(d[0]).cuda() for d in data]
    emb2 = [recsiam.utils.astensor(d[1]).cuda() for d in data]
    lab = recsiam.utils.astensor(np.array([d[2] for d in data])).cuda()

    return emb1, emb2, lab


def make_evaluation_engine(model, repr_dataset):

    class_repr = []
    labels = []
    labels_onehot = []

    def reid_eval(engine, batch):

        model.eval()

        inputs1, targets = batch
        with torch.no_grad():
            outputs1 = model(inputs1)

        if isinstance(outputs1, (list, tuple, np.ndarray)):
            outputs1 = torch.cat(outputs1)
        distances = ag.cart_euclidean_using_matmul(class_repr, outputs1)
        t_k = torch.topk(distances.t(), k=1, largest=False)

        y_pred = labels[t_k[1].t()[0]]

        return labels_onehot[y_pred], labels_onehot[targets]

    engine = ignite.engine.Engine(reid_eval)

    repr_dl = DataLoader(repr_dataset, collate_fn=reid_collate)

    @engine.on(ignite.engine.Events.STARTED)
    def embed_and_store(engine):
        nonlocal class_repr, labels, labels_onehot
        model.eval()

        class_repr = []
        labels = []
        labels_onehot = []

        with torch.no_grad():
            for batch in repr_dl:
                inputs1, targets = batch
                outputs1 = model(inputs1)
                if isinstance(outputs1, (list, tuple, np.ndarray)):
                    outputs1 = torch.cat(outputs1)

                labels.append(targets)
                class_repr.append(outputs1)

        class_repr = torch.cat(class_repr)
        labels = recsiam.utils.astensor(labels)
        labels_onehot = torch.eye(torch.unique(labels).max() + 1)

    return engine


def make_train_engine(model, loss_fn, optimizer, unsupervised=False):

    def train_and_store_loss(engine, batch):

        model.train()

        optimizer.zero_grad()
        if not unsupervised:
            inputs1, inputs2, targets = batch
            outputs1 = model(inputs1)
            outputs2 = model(inputs2)
            loss, *_ = loss_fn(outputs1, outputs2, targets)
        else:
            inputs1, *_ = batch
            outputs1 = model(inputs1)
            loss = loss_fn(torch.cat(outputs1), torch.cat(inputs1))
        loss.backward()

        total_norm = 0.
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
#        clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        return loss.item(), total_norm

    engine = ignite.engine.Engine(train_and_store_loss)

    @engine.on(ignite.engine.Events.EPOCH_STARTED)
    def on_epoch_started(engine):
        engine.state.losses = []
        engine.state.grad_norms = []

    @engine.on(ignite.engine.Events.ITERATION_COMPLETED)
    def on_it_finished(engine):
        engine.state.losses.append(engine.state.output[0])
        engine.state.grad_norms.append(engine.state.output[1])

    return engine


def reference_params(params):
    cp = copy.deepcopy(params)
    mod = cp["model"]
    mod["aggregator"] = "mean"
    mod["ag_args"] = {}
    
    return cp

def main(cmdline):

    params = json.loads(Path(cmdline.json).read_text())
    set_torch_seeds(params["seed"])

    model = cfg.prep_model(params)()
    ref_params = reference_params(params)
    ref_model = cfg.prep_model(ref_params)()

    model.cuda()
    loss_fn = instance_loss(params)
    optimizer = cfg.prep_optimizer(params)(model)

    cw_train, cw_test_repr, cw_test_emb, ow_test_repr, ow_test_emb = divide_dataset(params)

    rnd = np.random.RandomState(params["test_seed"])
    cw_train_repr, cw_train_emb = cw_train.split_balanced(1, rnd)

    repr_cat = ConcatDataset([cw_test_repr, ow_test_repr])

    train_engine = make_train_engine(model, loss_fn, optimizer,
                                     unsupervised=params["refine"]["setting"]["unsupervised"])

    trainval_engine = make_evaluation_engine(model, cw_train_repr)
    eval_engine = make_evaluation_engine(model, repr_cat)
    ref_eval_engine = make_evaluation_engine(ref_model, repr_cat)

    acc = ignite.metrics.Accuracy(is_multilabel=True)

    acc.attach(eval_engine, "acc")
    acc.attach(trainval_engine, "acc")
    acc.attach(ref_eval_engine, "acc")

    ow_eval_dl = DataLoader(ow_test_emb, collate_fn=reid_collate)
    cw_eval_dl = DataLoader(cw_test_emb, collate_fn=reid_collate)
    trainval_dl = DataLoader(cw_train_emb, collate_fn=reid_collate)

    rnd = np.random.RandomState(params["seed"])
    if params["refine"]["setting"]["unsupervised"]:
        gen_dataset = itertools.repeat(cw_train)
    else:
        gen_dataset = recsiam.data.gen_siamese_dataset(cw_train, params["refine"]["samples"], rnd) 

    def evaluate(reference=False):
        if not reference:
            eng = eval_engine
        else:
            eng = ref_eval_engine
        eng.run(cw_eval_dl)
        cw_value = eng.state.metrics["acc"]

        eng.run(ow_eval_dl)
        ow_value = eng.state.metrics["acc"]

        acc_value = cw_value * len(cw_test_emb) + ow_value * len(ow_test_emb)
        acc_value /= len(cw_test_emb) + len(ow_test_emb)
        values = [np.round(v, decimals=3) for v in (cw_value, ow_value, acc_value)]
        res_str = "cw_acc: {}\tow_acc: {}\ttot_acc: {}".format(*values)
        if reference:
            res_str = res_str + "\treference"
        logging.getLogger().info(res_str)

    def val():
        eng = trainval_engine
        eng.run(trainval_dl)
        cw_value = eng.state.metrics["acc"]

        values = [np.round(v, decimals=3) for v in [cw_value]]
        res_str = "trainval_ acc: {}".format(*values)
        logging.getLogger().info(res_str)

    def train():
        if params["refine"]["setting"]["unsupervised"]:
            dl = DataLoader(next(gen_dataset), shuffle=True, collate_fn=reid_collate, **params["refine"]["dl_args"])
        else:
            dl = DataLoader(next(gen_dataset), shuffle=True, collate_fn=train_collate, **params["refine"]["dl_args"])

        m_epochs = np.ceil(params["refine"]["samples"] / len(dl.dataset)).astype(int)
        train_engine.run(dl, max_epochs=m_epochs)
        s = train_engine.state

        values = (np.mean(s.losses), np.std(s.losses), np.mean(s.grad_norms), np.std(s.grad_norms))
        values = (np.round(v, decimals=3) for v in values)
        res_str = "loss mean: {}\tstd: {}, grad_norm mean: {}\tstd: {}".format(*values)
        logging.getLogger().info(res_str)

    evaluate(reference=True)
    evaluate()
    for i in range(params["refine"]["epochs"]):
        train()
        val()
        evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=str,
                        help="path containing the json to use")
    parser.add_argument("--results", type=str, default=None,
                        help="output file")
    parser.add_argument("-w", "--workers", type=int, default=-1,
                        help="number of joblib workers")
    parser.add_argument("-t", "--threads", type=int, default=1,
                        help="number of pytorch threads")

#       verbosity
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="triggers verbose mode")
    parser.add_argument("-q", "--quite", action='store_true',
                        help="do not output warnings")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quite:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.results is not None:
        assert Path(args.results).parent.exists()
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARN)
    main(args)
