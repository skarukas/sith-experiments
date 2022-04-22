## TopKPool.py - A generalized maxpooling layer invariant to rotation

from torch import nn
import torch

class TopKPool(nn.Module):
    def __init__(self, k, dim=-1, order="max-aligned"):
        """
        A generalized maxpooling layer that returns the top k values 
            across an entire dimension. The returned values can be ordered 
            in one of three ways:
            - "sorted": In order by value
            - "unsorted": The order in which they appear in the tensor
            - "max-aligned": Rotate the values along the dimension so the maximum
                is first, then find the unsorted top-K values.
        Note: Both "sorted" and "max-aligned" are invariant to rotation (`torch.roll`), 
            and "sorted" is invariant to permutation.
        """
        super().__init__()
        self.dim = dim
        self.k = k
        self.max_aligned = (order == "max-aligned")
        self.sorted = (order == "sorted")

    def forward(self, tens):
        if self.max_aligned:
            # align so that maximum is first
            idx = tens.max(self.dim).indices.unsqueeze(-1)
            n = tens.shape[self.dim]
            idx = (torch.arange(n).unsqueeze(0) + idx) % n
            tens = tens.gather(self.dim, idx)

        # get top k values
        topk, i = tens.topk(self.k, self.dim)

        if not self.sorted:
            # sort by appearance in the tensor (torch.topk returns an arbitrary ordering)
            i = torch.sort(i, self.dim).values
            topk = tens.gather(self.dim, i)
        return topk


if __name__ == "__main__":
    # test that the "rolled" (cyclicly translated) version of a tensor
    #  results in the same pool output
    x = torch.randn(4, 5, 10, requires_grad=True)
    for dim in x.shape:
        pool = TopKPool(5, order="max-aligned", dim=dim)
        px = pool(x)
        for i in range(x.shape[dim]):
            xr = x.roll(i, dim)
            assert (px == pool(xr)).all()

    # confirm that the operation is "differentiable"
    px.sum().backward()