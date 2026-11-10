"""Small distributed helpers shared by training and smoke tests."""

from __future__ import annotations

from torch.utils.data import Sampler


class DistributedEvalSampler(Sampler[int]):
    """Partition evaluation data without padding or repeated samples.

    PyTorch's standard DistributedSampler pads indices when the dataset size is
    not divisible by the world size. Padding is useful for synchronized
    training, but it biases per-image validation metrics. This sampler assigns
    every validation index to exactly one rank.
    """

    def __init__(self, dataset, num_replicas: int, rank: int):
        if num_replicas <= 0:
            raise ValueError("num_replicas必须大于0")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank必须位于[0,num_replicas)范围")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        return len(range(self.rank, len(self.dataset), self.num_replicas))
