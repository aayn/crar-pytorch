from crar.utils import compute_eps


class TestEps:
    def test_constant_eps(self):
        actual = [
            compute_eps(k, eps_start=0.5, eps_end=0.5, eps_last_frame=k * 10)
            for k in range(10, 5000, 100)
        ]
        expected = [0.5] * len(actual)
        assert actual == expected
