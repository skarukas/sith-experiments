import torch

def bilinear(x, y):
    x = torch.tensor(x)
    y = torch.tensor(y)
    dist = (1 - x.abs()) * (1 - y.abs())
    return torch.maximum(dist, torch.zeros_like(dist))

class Interpolator:
    def __init__(self, interpolation_fn=bilinear, kernel_size=2):
        self.interpolation_fn = interpolation_fn
        self.kernel_size = kernel_size

    def get_value(self, tens, i, j):
        if not isinstance(i, torch.Tensor):
            i = torch.tensor(i).unsqueeze(0)
        if not isinstance(j, torch.Tensor):
            j = torch.tensor(j).unsqueeze(0)
        j = j.unsqueeze(1).float()
        i = i.unsqueeze(1).float()
        assert (i >= 0).all()
        assert (j >= 0).all()
        
        # assume in last channel
        fl1 = i.floor().long()
        fl2 = j.floor().long()
        res_i = i - fl1
        res_j = j - fl2
        w = min(self.kernel_size, tens.shape[-2] - fl1)
        h = min(self.kernel_size, tens.shape[-1] - fl2)
        sampling_points = torch.tensor([
            [s1, s2]
            for s1 in range(w) 
            for s2 in range(h)
        ])
        # TODO: center points in kernel... for bilinear 2x2 the center point is on top left,
        #   but for 3x3 kernels etc. the center should be in the center of the kernel
        x = sampling_points[:, 0].unsqueeze(0)
        y = sampling_points[:, 1].unsqueeze(0)
        weights = self.interpolation_fn(x - res_i, y - res_j)
        return (tens[..., x+fl1, y+fl2] @ weights.T).squeeze()


def create_bilinear_filterbank(coords):
    """
    Coords: torch.Tensor of shape (N_f x 2)
    Create a ShiftedConv2D layer, sc, using the series of 
        coordinates supplied, such that 
        sc(tens)[i, j] = [
            tens[i+coords[0][0], j+coords[0][1]], 
            tens[i+coords[1][0], j+coords[1][1]],
            ...
            tens[i+coords[-1][0], j+coords[-1][1]]
        ]
    """
    shifts = coords.ceil().long()
    residual = coords - shifts
    filters = get_bilinear_filter(residual).unsqueeze(1)
    return ShiftedConv2d(shifts, filters)


def get_bilinear_filter(residual):
    x, y = residual[..., 0], residual[..., 1]
    return torch.stack([
            torch.stack([bilinear(x+i, y+j) for i in range(2)], dim=-1) 
            for j in range(2)
    ], dim=-1)


if __name__ == "__main__":
    #to do : doesn't work with multiple values because of index out of bound issues
    inter = Interpolator()
    tens = torch.tensor([
        [1, 0],
        [0, 0]
    ]).float()
    x = torch.tensor([0.5, 0, 1])
    y = torch.tensor([0.5, 1, 0])
    print(inter.get_value(tens, x, y))