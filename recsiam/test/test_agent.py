"""
HH this is 4 U
Unit test module for functions and classes defined in  recsiam.evaluation
"""
from collections import OrderedDict
import unittest
import numpy as np
import numpy.testing as npt

import recsiam.memory as mem
import recsiam.agent as ag
import recsiam.utils as utils
import recsiam.sampling as samp
import recsiam.models as models
import recsiam.loss as loss
import torch


def do_mean(ab):
    return (ab[0] + ab[1]) / 2




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

    module_list.append(("aggr", models.GlobalMean()))
    model = torch.nn.Sequential(OrderedDict(module_list))

    return model

class TestAgent(unittest.TestCase):

    def test_process_next_out(self):

        data = simpledataset()
        sup = simplesupervisor(data[1])

        agent = ag.Agent(1, mem.ObjectsMemory(), mem.SupervisionMemory(),
                         simplemodel(), sup, bootstrap=2,
                         max_neigh_check=1,
                         add_seen_element=ag.add_seen_separate)

        output = [agent.process_next([data[0][itx]], data[2][itx]) for itx in range(len(data[0]))]

        for itx in range(1,len(output)):
            with self.subTest(n=itx):
                self.assertTrue(output[itx][1] < itx)
                self.assertTrue(output[itx][2])

    def test_process_next_internals(self):

        data = simpledataset()
        sup = simplesupervisor(data[1])

        agent = ag.Agent(1, mem.ObjectsMemory(), mem.SupervisionMemory(),
                         simplemodel(), sup, bootstrap=2,
                         max_neigh_check=1,
                         add_seen_element=ag.add_seen_separate)

        for itx in range(len(data[0])):
            with self.subTest(n=itx):
                out = agent.process_next([data[0][itx]], data[2][itx])

                if itx < 2 and itx != 0:
                    self.assertTrue(out[2])
                self.assertEqual(len(agent.obj_mem), itx +1)
                self.assertEqual(len(agent.sup_mem), itx)

        data = simpledataset()
        sup = simplesupervisor(data[1])


    def test_refine(self):
        data = simpledataset()
        sup = simplesupervisor(data[1])


        def refine(agent):
            optim = torch.optim.sgd()
            l = loss.ContrastiveLoss()
            e = ag.create_siamese_trainer(agent, optim, l)
            sampler = samp.SeadableRandomSampler(agent.sup_mem, 1)
            data_loader = torch.utils.data.DataLoader(agent.sup_mem, sampler=sampler)

            e.run(data_loader, max_epochs=2)

        agent = ag.Agent(1, mem.ObjectsMemory(), mem.SupervisionMemory(),
                         simplemodel(), sup, bootstrap=2,
                         max_neigh_check=1,
                         add_seen_element=ag.add_seen_separate,
                         refine=refine)

        for itx in range(len(data[0])):
            with self.subTest(n=itx):
                out = agent.process_next([data[0][itx]], data[2][itx])

                if itx < 2 and itx != 0:
                    self.assertTrue(out[2])
                self.assertEqual(len(agent.obj_mem), itx + 1)
                self.assertEqual(len(agent.sup_mem), itx)


    def test_process_next_active(self):

        data = simpledataset()
        sup = simplesupervisor(data[1])

        agent = ag.ActiveAgent(0.5, 1, mem.ObjectsMemory(), mem.SupervisionMemory(),
                               simplemodel(), sup, bootstrap=2,
                               max_neigh_check=1,
                               add_seen_element=ag.add_seen_separate)

        output = [agent.process_next([data[0][itx]], data[2][itx]) for itx in range(len(data[0]))]

        asked_sup = np.array([o[2] for o in output])

        for itx in range(len(output)):
            with self.subTest(n=itx):
                if itx > 0:
                    self.assertTrue(output[itx][1] < itx)

        self.assertTrue(len(agent.sup_mem), asked_sup.sum())



class TestActiveAgent(unittest.TestCase):

    def test_predict(self):
        data = simpledataset()
        sup = simplesupervisor(data[1])

        agent = ag.ActiveAgent(0.5, 1, mem.ObjectsMemory(), mem.SupervisionMemory(),
                               simplemodel(), sup, bootstrap=2,
                               max_neigh_check=1,
                               add_seen_element=ag.add_seen_separate)

        output = [agent.process_next([data[0][itx]], data[2][itx]) for itx in range(len(data[0]))]

        predictions = [agent.predict([d])[1] for d in data[0]] 
        is_known = [agent.predict([d])[0] for d in data[0]] 

        npt.assert_equal(np.concatenate(predictions), data[2])

        all_pred = agent.predict(list(data[0]))

        npt.assert_equal(np.concatenate(is_known), all_pred[0])
        npt.assert_equal(np.concatenate(predictions), all_pred[1])


        npt.assert_equal(agent.predict(list(data[0])), all_pred)


    def test_supervision(self):
        data = simpledataset()
        sup = simplesupervisor(data[1])

        agent = ag.ActiveAgent(1.0, 1, mem.ObjectsMemory(), mem.SupervisionMemory(),
                               simplemodel(), sup, bootstrap=2,
                               max_neigh_check=1,
                               add_seen_element=ag.add_seen_separate)

        output = np.array([agent.process_next([data[0][itx]], data[2][itx])
                           for itx in range(len(data[0]))])

        self.assertTrue(output[1:, 2].any())

        agent = ag.ActiveAgent(0.01, 1, mem.ObjectsMemory(), mem.SupervisionMemory(),
                               simplemodel(), sup, bootstrap=2,
                               max_neigh_check=1,
                               add_seen_element=ag.add_seen_separate)

        output = np.array([agent.process_next([data[0][itx]], data[2][itx])
                           for itx in range(len(data[0]))])

        self.assertFalse(output[1:, 2].all())

