import numpy as np
import torch
from crar.losses import compute_disambiguation


class TestDisambiguation:
    def test_with_zero_1d(self):
        a, b = torch.tensor([[0.0]]), torch.tensor([[0.0]])
        expected = torch.exp(-5.0 * torch.tensor([[1e-6]]))
        actual = compute_disambiguation(a, b)
        assert torch.isclose(expected.float(), actual.float())

    def test_with_zero_2d(self):
        a, b = torch.tensor([[0.0, 0.0]]), torch.tensor([[0.0, 0.0]])
        expected = torch.exp(-5.0 * torch.tensor([[1e-6]]))
        actual = compute_disambiguation(a, b)
        assert torch.isclose(expected.float(), actual.float())

    def test_with_constant_difference(self):
        a, b = torch.tensor([[0.0]]), torch.tensor([[2.0]])
        expected = torch.exp(-5.0 * torch.tensor([[2.0]]))
        actual = compute_disambiguation(a, b)
        assert torch.isclose(expected.float(), actual.float())

    def test_large_value(self):
        a, b = torch.tensor([[0.0]]), torch.tensor([[100.0]])
        expected = torch.exp(-5.0 * torch.tensor([[3.2]]))
        actual = compute_disambiguation(a, b)
        assert torch.isclose(expected.float(), actual.float())

    def test_2d_tensor(self):
        a = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        b = np.roll(a, 1, axis=0)
        c = (a - b) ** 2
        c = np.sqrt(np.clip(np.sum(c, axis=1), 1e-6, 10))
        expected = torch.exp(-5.0 * torch.tensor(c)).sum()

        a, b = torch.tensor(a), torch.tensor(b)
        actual = compute_disambiguation(a, b)
        assert torch.isclose(expected.float(), actual.float())
