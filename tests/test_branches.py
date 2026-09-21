import unittest

import torch
from transformers import DynamicCache

from jiffy.branches import fork_cache


class CacheBranchTests(unittest.TestCase):
    def test_forks_share_storage_but_not_updates(self):
        for sliding in (False, True):
            with self.subTest(sliding=sliding):
                keys = torch.randn(1, 2, 10, 4)
                entry = (keys, keys.clone(), torch.tensor(4)) if sliding else (keys, keys.clone())
                parent = DynamicCache([entry])
                before = parent.layers[0].keys.clone()
                length = parent.get_seq_length()
                first, second = fork_cache(parent, 2), fork_cache(parent, 1)
                self.assertEqual(first.layers[0].keys.data_ptr(), parent.layers[0].keys.data_ptr())
                added = torch.ones(2, 2, 3, 4)
                first.update(added, added, 0)
                torch.testing.assert_close(parent.layers[0].keys, before)
                torch.testing.assert_close(second.layers[0].keys, before)
                self.assertEqual(parent.get_seq_length(), length)
                self.assertEqual(first.get_seq_length(), length + 3)

    def test_foreign_cache_is_rejected(self):
        with self.assertRaises(ValueError):
            fork_cache(object(), 1)
