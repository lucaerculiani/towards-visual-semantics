"""
HH this is 4 U
Unit test module for function and classes defined in  recsiam.data
"""

import unittest
import pathlib
from pathlib import Path

import numpy as np
import numpy.testing as npt

import recsiam.data as data
import skimage

TEST_DATASET_PATH = pathlib.Path(__file__).parent.parent / "testdata" / "dataset"
TEST_MASK_DATASET_PATH = pathlib.Path(__file__).parent.parent / "testdata" / "dataset_seg"

TEST_IMAGE_PATH = pathlib.Path(__file__).parent.parent / "testdata/" / "fake_can" / "can_0001.jpg" 
TEST_IMAGE_SEG_PATH = TEST_IMAGE_PATH.parent / "can_seg.png"

TEST_DATASET = data.descriptor_from_filesystem(TEST_DATASET_PATH)

TEST_DATASET_LEN = 15

TEST_SEQ_PER_OBJ = 3

TEST_SEQ_LEN = 30

class TestVideoDataSet(unittest.TestCase):
    """
    Test class for data.VideoDataSet
    """

    def test_sample_size(self):
        new_dataset = data.VideoDataSet(TEST_DATASET)
        s_size = new_dataset.sample_size()

        self.assertEqual((3, 120, 120), s_size)

    def test_len(self):
        dataset = data.VideoDataSet(TEST_DATASET)
        self.assertEqual(len(dataset), TEST_DATASET_LEN)

    def test_getitem(self):
        dataset = data.VideoDataSet(TEST_DATASET)
        seq_shape = (5, 3, 120, 120)

        seq = dataset[(0, 0, slice(0, 5))]
        self.assertEqual(seq.shape, seq_shape)

        seq = dataset[(0, 0, slice(0, 10, 2))]
        self.assertEqual(seq.shape, (5,) + seq_shape[1:])

        seq = dataset[(0, 0, np.arange(5).tolist())]
        self.assertEqual(seq.shape, seq_shape)

        seq = dataset[((0, 0, np.arange(5)), (0, 0, np.arange(5)))]
        self.assertEqual(seq.shape, (2,) + seq_shape)

        seq = dataset[((0, 0, slice(0, 5)), (0, 0, slice(0, 5)))]
        self.assertEqual(seq.shape, (2,) + seq_shape)

        with self.assertRaises(TypeError):
            dataset[(5, 2)]

        with self.assertRaises(TypeError):
            dataset[(1, slice(None), 2)]

        with self.assertRaises(TypeError):
            dataset[1]

        with self.assertRaises(TypeError):
            dataset[slice(None)]

        with self.assertRaises(TypeError):
            dataset[(1, 1)]


class TrainSeqDataSet(unittest.TestCase):
    """
    Test class for data.TrainSeqDataSet
    """

    def test_getitem(self):

        dataset = data.TrainSeqDataSet(TEST_DATASET)
        value = ((0, 0, slice(0, 20)), (0, 2, slice(10, 30)))

        res = dataset[value]

        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].shape, (2, 20, 3, 120, 120))
        self.assertTrue(np.array_equal(res[1], np.array((20, 20))))
        self.assertEqual(res[2][0], res[2][1])

        value = ((0, 0, slice(0, 20)), (1, 0, slice(10, 30)))
        res = dataset[value]

        self.assertNotEqual(res[2][0], res[2][1])

        with self.assertRaises(ValueError):
            dataset[(0, np.arange(90))]

        with self.assertRaises(ValueError):
            dataset[0]

        with self.assertRaises(ValueError):
            wrong_value = ((1, slice(0, 20)), (0, slice(20, 40)))
            dataset[wrong_value + wrong_value]


class TestFlattenedDataSet(unittest.TestCase):
    """
    Test class for data.FlattenedDataSet
    """

    def test_getitem(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)
        seq_shape = (30, 3, 120, 120)

        seq = dataset[0]
        self.assertEqual(seq[0].shape, seq_shape)

        seq2 = dataset[15]
        self.assertNotEqual(seq[1],  seq2[1])

        all_seq = dataset[slice(None)]
        self.assertEqual(len(all_seq), len(dataset))

    def test_get_label(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)

        seq = dataset[0]

        self.assertEqual(seq[1], dataset.get_label(0))

        all_lab = dataset.get_label(np.arange(len(dataset)))
        self.assertEqual(len(all_lab), len(dataset))


    def test_preload(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)
        preloaded = data.FlattenedDataSet(TEST_DATASET, preload=True)

        def test_single(a, b):
            for i, j in zip(a, b):
                npt.assert_array_equal(i, j)

        test_single(dataset[3], preloaded[3])

        for a, b in zip(dataset[:], preloaded[:]):
            test_single(a, b)

        for a, b in zip(dataset[1:5], preloaded[1:5]):
            test_single(a, b)

    def test_len(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)
        self.assertEqual(len(dataset), TEST_DATASET_LEN * 3)

    def test_get_n_objects(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)
        all_lab = dataset.get_label(np.arange(len(dataset)))

        N_OBJ = 5

        rnd = np.random.RandomState(0)
        selected_ind = dataset.get_n_objects(N_OBJ, rnd)

        selected_lab = all_lab[selected_ind]
        remaining_lab = np.delete(all_lab, selected_ind)

        self.assertTrue(np.unique(selected_lab).size, N_OBJ)
        self.assertTrue(np.unique(remaining_lab).size, np.unique(all_lab.size) - N_OBJ)

        _, uniq_ind = np.unique(all_lab, return_index=True)

        subset_ind = np.delete(np.arange(all_lab.size), uniq_ind)

        selected_ind = dataset.get_n_objects(N_OBJ, rnd, ind_subset=subset_ind)

        selected_lab = all_lab[selected_ind]
        remaining_lab = all_lab[np.setdiff1d(subset_ind, selected_ind)]

        self.assertEqual(np.unique(selected_ind).size, selected_ind.size)
        self.assertTrue(np.isin(selected_ind, subset_ind).all())

        self.assertEqual(np.unique(selected_lab).size, N_OBJ)
        self.assertEqual(np.unique(remaining_lab).size, np.unique(all_lab).size - N_OBJ)

        

    def test_balanced_sample(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)

        rnd = np.random.RandomState(0)
        ind = dataset.balanced_sample(1, rnd)

        lab = np.array([dataset.get_label(i) for i in ind])

        self.assertEqual(lab.shape[0], TEST_DATASET_LEN)
        self.assertEqual(np.unique(lab).shape[0], TEST_DATASET_LEN)

        rnd = np.random.RandomState(0)
        ind2 = dataset.balanced_sample(1, rnd)

        npt.assert_equal(ind, ind2)

        def check_sampling(num):
            rnd = np.random.RandomState(num)
            ind = dataset.balanced_sample(num, rnd)
            lab = np.array([dataset.get_label(i) for i in ind])

            self.assertEqual(lab.shape[0], TEST_DATASET_LEN * num)
            self.assertEqual(np.unique(ind).shape[0], TEST_DATASET_LEN * num)

            uniq, cnt = np.unique(lab, return_counts=True)

            self.assertEqual(uniq.shape[0], TEST_DATASET_LEN)
            npt.assert_equal(cnt, np.tile(num, len(cnt)))

        check_sampling(2)
        check_sampling(3)


        all_lab = dataset.get_label(np.arange(len(dataset)))

        rnd = np.random.RandomState(0)
        del_ind = dataset.get_n_objects(5, rnd)
        subset_ind = np.delete(np.arange(all_lab.size), del_ind)

        remaining_lab = all_lab[subset_ind]

        def check_balanced_sampling(num):
            sampled_ind = dataset.balanced_sample(num, rnd, ind_subset=subset_ind)
            self.assertTrue(np.isin(sampled_ind, subset_ind).all())

            npt.assert_array_equal(np.unique(all_lab[sampled_ind], return_counts=True)[1],
                             np.tile(num, np.unique(remaining_lab).size))

        for i in range(1,4):
            check_balanced_sampling(i)

class TestExtendedSubset(unittest.TestCase):

    def test_get_label(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)
        subset = data.ExtendedSubset(dataset)
        all_indices = np.arange(len(dataset))
        npt.assert_array_equal(dataset.get_label(all_indices),
                               subset.get_label(all_indices))

        npt.assert_array_equal(dataset.get_label(5),
                               subset.get_label(5))
        rnd = np.random.RandomState(0)

        random_selection = rnd.choice(all_indices, replace=False)
        npt.assert_array_equal(dataset.get_label(random_selection), 
                               subset.get_label(random_selection))

        permuted = rnd.permutation(all_indices)
        subset = data.ExtendedSubset(dataset, permuted)
        npt.assert_array_equal(dataset.get_label(all_indices),
                               subset.get_label(all_indices)[np.argsort(permuted)])

    def test_split_balanced(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)
        subset = data.ExtendedSubset(dataset)
        
        N_ELEM = 15
        N_SEQ = 3

        def test_for_num(num, dset):
            rnd = np.random.RandomState(0)
            ssmall, sbig = subset.split_balanced(num, rnd)

            self.assertTrue(np.isin(ssmall.indices, dset.indices).all())
            self.assertTrue(np.isin(sbig.indices, dset.indices).all())
            self.assertEqual(len(ssmall), N_ELEM * num)
            self.assertEqual(len(sbig), len(dset) - len(ssmall))

            self.assertEqual(np.intersect1d(ssmall.indices, sbig.indices).size, 0)

            uniq_l, count_l = np.unique(ssmall.get_label(np.arange(len(ssmall))),
                                        return_counts=True)

            self.assertEqual(uniq_l.size, N_ELEM)
            npt.assert_array_equal(count_l, np.tile(num, uniq_l.size))


        for i in range(1,3):
            with self.subTest(i=i):
                test_for_num(i, subset)

        subset = data.ExtendedSubset(dataset, np.arange(len(dataset))[1:])
        for i in range(1,3):
            with self.subTest(i=i):
                test_for_num(i, subset)


        subset = data.ExtendedSubset(dataset)
        subset = data.ExtendedSubset(subset, np.arange(len(dataset))[1:])
        for i in range(1,3):
            with self.subTest(i=i):
                test_for_num(i, subset)

    def test_split_n_objects(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)
        subset = data.ExtendedSubset(dataset)
        
        N_ELEM = 15
        N_SEQ = 3

        def test_for_num(num, dset):
            rnd = np.random.RandomState(0)
            ssmall, sbig = subset.split_n_objects(num, rnd)

            self.assertTrue(np.isin(ssmall.indices, dset.indices).all())
            self.assertTrue(np.isin(sbig.indices, dset.indices).all())
            self.assertEqual(len(sbig), len(dset) - len(ssmall))

            self.assertEqual(np.intersect1d(ssmall.indices, sbig.indices).size, 0)

            uniq_ssmall = np.unique(ssmall.get_label(np.arange(len(ssmall))))
            uniq_sbig = np.unique(sbig.get_label(np.arange(len(sbig))))

            self.assertEqual(uniq_ssmall.size, num)
            self.assertEqual(np.intersect1d(uniq_ssmall, uniq_sbig).size, 0)

        subset = data.ExtendedSubset(dataset, np.arange(len(dataset))[1:])
        for i in range(1, 5):
            with self.subTest(i=i):
                test_for_num(i, subset)


        subset = data.ExtendedSubset(dataset)
        subset = data.ExtendedSubset(subset, np.arange(len(dataset))[1:])
        for i in range(1, 5):
            with self.subTest(i=i):
                test_for_num(i, subset)

class TestMasking(unittest.TestCase):

    def test_mask_from_filesystem(self):
        mask_desc = data.mask_from_filesystem(TEST_MASK_DATASET_PATH)
        
        self.assertEqual(len(mask_desc), TEST_DATASET_LEN)
        for d in mask_desc:
            self.assertEqual(len(d["paths"]), TEST_SEQ_PER_OBJ)
            for p in d["paths"]:
                self.assertEqual(len(p), TEST_SEQ_LEN)

        desc = data.descriptor_from_filesystem(TEST_DATASET_PATH)

        for i, d, m in zip(range(len(desc)), desc, mask_desc):
            with self.subTest(i=i):
                d_names = [[Path(frame_p).name for frame_p in seq_p]
                        for seq_p in d["paths"]]

                m_names = [[Path(frame_p).parent.name for frame_p in seq_p]
                        for seq_p in m["paths"]]

                self.assertEqual(d_names, m_names)


def get_masked_dataset(flattened):

    desc = data.descriptor_from_filesystem(TEST_DATASET_PATH)
    mask_desc = data.mask_from_filesystem(TEST_MASK_DATASET_PATH)
    if flattened:
        dataclass = data.FlattenedDataSet
    else:
        dataclass = data.VideoDataSet

    return data.MaskedDataset(dataclass(desc), dataclass(mask_desc))

def get_segmented_image():
    image = skimage.io.imread(TEST_IMAGE_PATH)
    image_seg = np.where(skimage.io.imread(TEST_IMAGE_SEG_PATH).astype(np.bool)[:, :, None], image, 127)

    return image_seg

class TestMaskedDataset(unittest.TestCase):

    def test_getitem(self):
        m_dataset = get_masked_dataset(True)

        image = get_segmented_image().transpose(2, 0, 1)
        self.assertTrue((m_dataset[1][0] == image[None, ...]).all())


    def test_gen_embed_dataset(self):
        m_dataset = get_masked_dataset(False)

        image = get_segmented_image().transpose(2, 0, 1)
        for i, g_ret in zip(range(5), m_dataset.gen_embed_dataset(return_mask_paths=True)):
            self.assertTrue((g_ret[0] == image[None, ...]).all())
            d_names = [Path(frame_p).relative_to(Path(frame_p).parents[1])
                      for frame_p in g_ret[1]]

            m_names = [Path(frame_p).relative_to(Path(frame_p).parents[2]).parent
                      for frame_p in g_ret[2]]

            self.assertEqual(d_names, m_names)



class TestSplitFunctions(unittest.TestCase):

    def test_train_val(self):
        dataset = data.FlattenedDataSet(TEST_DATASET)
        train_dl, val_dl, _ = data.train_val_split(dataset, 0)

        indices = np.concatenate((train_dl.dataset.indices, val_dl.dataset.indices))

        self.assertEqual(val_dl.dataset.indices.shape[0], TEST_DATASET_LEN)
        self.assertEqual(indices.shape[0], TEST_DATASET_LEN * 3)
        self.assertEqual(indices.shape[0], np.unique(indices).shape[0])

        lab = np.array([val_dl.dataset.dataset.get_label(i) for i in indices])
        self.assertEqual(lab.shape[0], TEST_DATASET_LEN * 3)
        self.assertEqual(np.unique(lab).shape[0], TEST_DATASET_LEN)

    def test_train_test_desc_split(self):
        tr, te = data.train_test_desc_split(TEST_DATASET, 0)

        tr_paths = set(np.concatenate([p["paths"] for p in tr]).flatten())
        te_paths = set(np.concatenate([p["paths"] for p in te]).flatten())

        self.assertEqual(len(tr_paths | te_paths), TEST_DATASET_LEN * TEST_SEQ_PER_OBJ * TEST_SEQ_LEN)
        self.assertEqual(len(tr_paths & te_paths), 0)

        train_d = data.FlattenedDataSet(tr)
        test_d = data.FlattenedDataSet(te)

        self.assertEqual(len(train_d), TEST_DATASET_LEN * 2)
        self.assertEqual(len(test_d), TEST_DATASET_LEN)
