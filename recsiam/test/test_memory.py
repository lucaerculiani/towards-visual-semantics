"""
HH this is 4 U
Unit test module for functions and classes defined in  recsiam.evaluation
"""
from collections import OrderedDict
import unittest
import numpy as np
import numpy.testing as npt

import recsiam.memory as mem
import recsiam.utils as utils
import recsiam.sampling as samp
import recsiam.models as models
import recsiam.loss as loss
import torch


def do_mean(ab):
    return (ab[0] + ab[1]) / 2


class TestObjectMemory(unittest.TestCase):
    """Unit tests for class evaluation.PaiwiseEvaluator"""

    def test_add_new_element(self):

        om = mem.ObjectsMemory()

        t1 = torch.arange(10, 20)[None, ...]

        om.add_new_element(t1, 0)

        npt.assert_equal(utils.t2a(om.M), np.arange(10, 20).reshape((1, 10)))
        npt.assert_equal(om.seq_ids, np.arange(0, 1))

        g_set = set([0])

        self.assertEqual(set(om.G.nodes), g_set)

        om = mem.ObjectsMemory()

        t1 = torch.arange(10, 20)[None, ...]
        t2 = torch.arange(20, 30)[None, ...]
        t3 = torch.arange(30, 40)[None, ...]

        om.add_new_element(t1, 0)
        om.add_new_element(t2, 1)
        om.add_new_element(t3, 2)

        npt.assert_equal(utils.t2a(om.M), np.arange(10, 40).reshape((3, 10)))
        npt.assert_equal(om.seq_ids, np.arange(0, 3))

        g_set = set([0, 1, 2])

        self.assertEqual(set(om.G.nodes), g_set)

    def test_add_neighbors(self):
        om = mem.ObjectsMemory()

        t1 = torch.arange(10, 20)[None, ...]
        t2 = torch.arange(20, 30)[None, ...]
        t3 = torch.arange(30, 40)[None, ...]

        om.add_new_element(t1, 0)
        om.add_new_element(t2, 1)
        om.add_new_element(t3, 2)
        om.add_neighbors(2, [0])

        npt.assert_equal(utils.t2a(om.M), np.arange(10, 40).reshape((3, 10)))
        npt.assert_equal(om.seq_ids, np.arange(0, 3))

        g_set = set([0, 1, 2])

        self.assertEqual(set(om.G.nodes), g_set)

        e_set = set([(0, 2)])

        self.assertEqual(set([tuple(sorted(e)) for e in om.G.edges]), e_set)

    def test_get_knn(self):
        om = mem.ObjectsMemory()

        t1 = torch.arange(10, 20).float()
        t2 = torch.arange(20, 30).float()
        t3 = torch.arange(40, 50).float()

        om.add_new_element(t1, 0)
        om.add_new_element(t2, 1)
        om.add_new_element(t3, 2)

        npt.assert_equal(utils.t2a(om.get_knn(t1, k=1)[1]), 0)
        npt.assert_equal(utils.t2a(om.get_knn(t1, k=2)[1]), np.array([[0, 1]]))

        t12 = torch.stack([t1, t2])

        npt.assert_equal(utils.t2a(om.get_knn(t12, k=1)[1]), np.array([[0], [1]]))
        npt.assert_equal(utils.t2a(om.get_knn(t12, k=2)[1]), np.array([[0, 1],[1, 0]]))


    def tets_len(self):
        om = mem.ObjectsMemory()

        t1 = torch.arange(10, 20)
        t2 = torch.arange(20, 30)
        t3 = torch.arange(30, 40)

        om.add_new_element(t1, 0)
        om.add_new_element(t2, 1)
        om.add_new_element(t3, 2)

        self.assertEqual(len(om), 3)
        self.assertEqual(om.sequences, 3)


    def test_get_something(self):
        om = mem.ObjectsMemory()

        t1 = torch.arange(10, 20)
        t2 = torch.arange(20, 30)
        t3 = torch.arange(30, 40)

        om.add_new_element(t1, 0)
        om.add_new_element(t2, 1)
        om.add_new_element(t3, 2)

        npt.assert_equal(om.get_sid(0), 0)
        npt.assert_equal(utils.t2a(om.get_embed(0)), utils.t2a(t1))


def bogus_data():
    e = np.random.randn(10)
    s = np.random.randn(5, 3, 10, 10)

    return [(e, s)]


class TestSupervisionMemory(unittest.TestCase):

    def test_add_entry(self):
        m = mem.SupervisionMemory()

        d1, l1 = bogus_data(), ([0], [3.0])
        d2, l2 = bogus_data(), ([1], [0.5])
        d3, l3 = bogus_data(), ([0], [1.5])
        d4, l4 = bogus_data(), ([1], [1.0])

        m.add_entry(d1, *l1)
        m.add_entry(d2, *l2)
        m.add_entry(d3, *l3)
        m.add_entry(d4, *l4)

        npt.assert_equal(m.labels, np.array([1, 1, 0, 0]))
        npt.assert_equal(m.distances, np.array([0.5, 1.0, 1.5, 3.0]))
        npt.assert_equal(m.insertion_orders, np.array([1, 3, 2, 0]))

        npt.assert_equal(m.couples, d2 + d4 + d3 + d1)

        self.assertEqual(len(m), 4)

    def test_del_entry(self):
        m = mem.SupervisionMemory()

        d1, l1 = bogus_data(), ([0], [3.0])
        d2, l2 = bogus_data(), ([1], [0.5])
        d3, l3 = bogus_data(), ([0], [1.5])
        d4, l4 = bogus_data(), ([1], [1.0])

        m.add_entry(d1, *l1)
        m.add_entry(d2, *l2)
        m.add_entry(d3, *l3)
        m.add_entry(d4, *l4)

        m.del_entry()

        npt.assert_equal(m.labels, np.array([1, 1, 0]))
        npt.assert_equal(m.distances, np.array([0.5, 1.0, 1.5]))
        npt.assert_equal(m.insertion_orders, np.array([1, 3, 2]))

        npt.assert_equal(m.couples, d2 + d4 + d3)
        self.assertEqual(len(m), 3)

        m.del_entry(1)

        npt.assert_equal(m.labels, np.array([1, 0]))
        npt.assert_equal(m.distances, np.array([0.5, 1.5]))
        npt.assert_equal(m.insertion_orders, np.array([1, 2]))

        npt.assert_equal(m.couples, d2 + d3)
        self.assertEqual(len(m), 2)

    def test_getitem(self):
        m = mem.SupervisionMemory()

        d1, l1 = bogus_data(), ([0], [3.0])
        d2, l2 = bogus_data(), ([1], [0.5])
        d3, l3 = bogus_data(), ([0], [1.5])
        d4, l4 = bogus_data(), ([1], [1.0])

        m.add_entry(d1, *l1)
        m.add_entry(d2, *l2)
        m.add_entry(d3, *l3)
        m.add_entry(d4, *l4)

        item = m[2]

        npt.assert_equal(item, (d3[0], l3[0][0]))

        m.del_entry(1)

        item = m[1]

        npt.assert_equal(item, (d3[0], l3[0][0]))


class simplesupervisor():

    def __init__(self, labels):
        self.labels = labels

    def ask_pairwise_supervision(self, l1, l2):
        return self.labels[l1] == self.labels[l2]


def simpledataset():
    data = np.array([
        [[1., 0, 0], [1., 0, 0]],
        [[2., 0, 0], [2., 0, 0]],
        [[5., 0, 0], [5., 0, 0]],
        [[6., 0, 0], [6., 0, 0]],
        [[17., 0, 0], [17., 0, 0]],
        [[18., 0, 0], [18., 0, 0]],
        ])

    lab = np.array([0, 0, 1, 1, 2, 2])
    s_id = np.arange(len(lab))

    return utils.a2t(data).float(), lab, s_id


def simplemodel():
    module_list = []
    module_list.append(("embed", models.SequenceSequential(torch.nn.Linear(3, 3))))

    module_list.append(("memgr", models.GlobalMean()))
    model = torch.nn.Sequential(OrderedDict(module_list))

    return model

class TestDistances(unittest.TestCase):

    def test_euclidean(self):

        e0 = torch.from_numpy(np.tile((0, 0, 1), (10, 1))).float()
        e1 = torch.from_numpy(np.tile((0, 0, 1), (15, 1))).float()

        e1 = torch.from_numpy(np.tile((0, 1, 1), (15, 1))).float()
        dmat = mem.cart_euclidean_using_matmul(e0, e1).numpy()
        self.assertTrue((dmat.round(decimals=3) == 1.0).all())
