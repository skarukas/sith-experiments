import torch
import math
import numpy as np
from .util import TWO_PI, unsqueeze_except

## TODO: implement sparse convolution (via matrix multiplication). The existing 
##     convolution kernels are sparse but very large (we'll run out of memory)
## TODO: filtered images are sometimes blank...why?


class LogPolarTransform(torch.nn.Module):
    def __init__(self, tau_min=.1, tau_max=100., buff_max=None, k=50, 
                 ntau=50, dt=1, g=0.0, num_angles=10,
                 window_shape="arc", localization="step", gaussian_sharpness=1, 
                 focus_points=None, stride=1, device='cpu', 
                 **kwargs):
        """
        A module for computing compressed log-polar transforms centered at various
        locations in an image, based on SITH.

        
        Parameters
        ----------
        
            - tau_min: float
                The distance to the center of the spatial receptive field for the first taustar produced. 
            - tau_max: float
                The distance to the center of the spatial receptive field for the last taustar produced. 
            - buff_max: int
                The maximum time in which the filters go into the past. NOTE: In order to 
                achieve as few edge effects as possible, buff_max needs to be bigger than
                tau_max, and dependent on k, such that the filters have enough time to reach 
                very close to 0.0. Plot the filters and you will see them go to 0. 
            - k: int
                Spatial Specificity of the taustars. If this number is high, then taustars
                will always be more narrow.
            - ntau: int
                Number of taustars produced, spread out logarithmically.
            - dt: float
                The space delta of the model. The there will be int(buff_max/dt) filters per
                taustar. Essentially this is the base rate of information being presented to the model
            - g: float
                Typically between 0 and 1. This parameter is the scaling factor of the output
                of the module. If set to 1, the output amplitude for a delta function will be
                identical through time. If set to 0, the amplitude will decay into the past, 
                getting smaller and smaller. This value should be picked on an application to 
                application basis.
            - num_angles: int
                How many angles to compute the SITH across.
            - stride: int
                The stride at which the focus points will be chosen. If greater than 1,
                the output will be effectively downsampled.
            - focus_points: Iterable[Tuple[int, int]]
                The specific points at which to perform the transformation.
            - device: str | torch.Device
                What device the computation is taking place on.
            - window_shape: 'arc' | 'line'
                Determines the shape of the receptive field as we move away 
                from a tau_star axis.
            - localization: 'gaussian' | 'step'
                How to consolidate the information from all the 
                pixels in the receptive field of a certain (tau_star, theta) pair
                    - 'step' : equal-weighting over the area after SITH impulse response
                    - 'gaussian': give a higher weight to the information 'on' the axis after SITH impulse response
                NOTE: For both choices, the shape of the window in the tau_star direction is the SITH impulse response.
            - gaussian_sharpness: int
                The sharpness of the receptive field when localization is 'gaussian'. This is 
                specifically the number of standard deviations between each tau_star axis.           
        """
        super(LogPolarTransform, self).__init__()

        self.k = k
        self.tau_min = tau_min
        self.tau_max = tau_max
        if buff_max is None:
            buff_max = 3*tau_max
        self.buff_max = buff_max
        self.ntau = ntau
        self.dt = dt
        self.g = g
        self.stride = (stride, stride)
        self.num_angles = num_angles
        self.sd_scale = gaussian_sharpness
        
        assert focus_points is None, "Not implemented yet."    


        self.c = (tau_max/tau_min)**(1./(ntau-1))-1
        
        x = torch.arange(dt, buff_max+dt, dt).type(torch.DoubleTensor)
        y = torch.arange(dt, buff_max+dt, dt).type(torch.DoubleTensor)
        filter_width = len(x)
        self.padding = filter_width // 2
        # the center point for each filter
        c_x = c_y = (buff_max+dt) // 2

        dtheta = TWO_PI / num_angles
        theta = torch.arange(num_angles) * dtheta - np.pi
        tau_star = tau_min*(1+self.c)**torch.arange(ntau).type(torch.DoubleTensor)

        # we'll need ALL combinations of (x, y, theta, tau_star), so make 
        #   all tensors broadcastable to that shape
        ndim = 4
        x = unsqueeze_except(x, ndim, dim=0)
        y = unsqueeze_except(y, ndim, dim=1)
        centered_x = x - c_x
        centered_y = y - c_y
        theta = unsqueeze_except(theta, ndim, dim=2)
        theta_orth = theta + np.pi / 2
        tau_star = unsqueeze_except(tau_star, ndim, dim=3)

        a = math.log(k)*k
        b = torch.log(torch.arange(2,k).type(torch.DoubleTensor)).sum()
        
        A = ((1/tau_star)*(torch.exp(a-b))*(tau_star**self.g))
        A = unsqueeze_except(A, ndim, dim=0)
        
        # The 'orthogonal' axis is the line or arc that stretches to the edge 
        #   of the receptive field for a given (tau_star, theta) pair
        if window_shape == "arc":
            # tau axis : Euclidian distance from center
            # 'orthogonal' axis : arc length from a certain (tau_star, theta) center point

            # the arc length between each tau_star axis
            axis_distance = tau_star * dtheta/2
            tau = np.sqrt(centered_x**2 + centered_y**2)
            
            # arc length
            grid_theta = torch.atan2(centered_y, centered_x)
            clockwise_dist = (theta - grid_theta).abs()
            cclockwise_dist = (theta - (grid_theta-2*np.pi)).abs()
            unit_circle_dist = torch.minimum(clockwise_dist, cclockwise_dist)
            tau_orth = unit_circle_dist * tau

            tau, tau_orth = torch.broadcast_tensors(tau, tau_orth)
        else: # window_shape == line
            # tau axis : rotated straight axis pointing away from center
            # 'orthogonal' axis : orthogonal direction of above

            # the distance traveled in the orthogonal direction between each tau_star axis
            axis_distance = tau_star * math.tan(dtheta/2) 

            tau = torch.cos(theta)*centered_x - torch.sin(theta)*centered_y
            tau_orth = torch.cos(theta_orth)*centered_x - torch.sin(theta_orth)*centered_y
        

        if localization == "gaussian":
            # put 'self.sd_scale' standard deviations between each 
            sd = (axis_distance / self.sd_scale)**2 
            gaussian_scale_factor = 1/np.sqrt(np.pi*2*sd)
            orthogonal_window = gaussian_scale_factor * np.exp(-tau_orth**2 / sd)
        elif localization == "step":
            radius = np.sqrt(centered_x**2 + centered_y**2)
            orthogonal_window = tau_orth.abs() / (radius / tau_star) <= axis_distance
  
        tau[tau <= 0] = 0
        tau_prime = tau / tau_star

        self.filters = A*torch.exp(torch.log(tau_prime)*(k+1) - k*tau_prime)
        self.filters = self.filters * orthogonal_window
        self.filters[tau_prime <= 0] = 0

        # reshape to filter with ntau*num_angles channels
        self.filters = self.filters.permute(2, 3, 0, 1) 
        self.filters = self.filters.reshape((ntau*num_angles, filter_width, filter_width))

        # normalize so each sums to 1
        eps = 1e-8
        filter_sum = (self.filters.sum((1, 2)) + eps)
        self.filters = self.filters / unsqueeze_except(filter_sum, n_dim=3, dim=0)
        self.filters = self.filters.unsqueeze(1).to(device).float()
    

    def extra_repr(self):
        s = "ntau={ntau}, tau_min={tau_min}, tau_max={tau_max}, buff_max={buff_max}, dt={dt}, k={k}, g={g}"
        s = s.format(**self.__dict__)
        return s    
    

    def forward(self, inp):
        """
        Takes in (Batch, features, x, y) and returns (Batch, features, Taustar, Theta, x', y')
        x' and y' will be smaller than x and y if self.stride != 1
        """
        assert(len(inp.shape) >= 4)
        # Reshape to (Batch*Features, 1, x, y)
        inp_reshaped = inp.reshape((inp.shape[0]*inp.shape[1], 1, *inp.shape[2:]))
        out = torch.conv2d(inp_reshaped, self.filters, 
                           stride=self.stride, padding=self.padding)

        return out.reshape((inp.shape[0], inp.shape[1], self.ntau, self.num_angles, *out.shape[-2:]))