import torch
import math
import numpy as np
import util

## TODO: implement sparse convolution (via matrix multiplication). The existing 
##     convolution kernels are sparse but very large (we'll run out of memory)
## TODO: filtered images are sometimes blank...why?


class LogPolarTransform(torch.nn.Module):
    def __init__(self, tau_min=.1, tau_max=100., buff_max=None, k=50, 
                 ntau=50, dt=1, g=0.0, stride=1, num_angles=10,
                 window_shape="concentric", localization="step", 
                 gaussian_sharpness=1, focus_points=None, device='cpu', 
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
            - device: str | torch.Device
                What device the computation is taking place on.
            - window_shape: 'concentric' | 'straight'
                How the log-polar should expand outward from the center
            - localization: 'gaussian' | 'step'
                The shape of the window with a focus at a certain taustar. 'step' means the window
                entends with equal strength until the surrounding angles.
            - gaussian_sharpness: int
                The sharpness of the orthogonal window when localization is 'gaussian'. This is 
                specifically the number of standard deviations between each consecutive angle.
            -// ttype: Torch Tensor
                This is the type we set the internal mechanism of the model to before running. 
                In order to calculate the filters, we must use a DoubleTensor, but this is no 
                longer necessary after they are calculated. By default we set the filters to 
                be FloatTensors. NOTE: If you plan to use CUDA, you need to pass in a 
                cuda.FloatTensor as the ttype, as using .cuda() will not put these filters on 
                the gpu.                 
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

        dtheta = 2*torch.pi / num_angles
        theta = torch.arange(num_angles) * dtheta - np.pi
        tau_star = tau_min*(1+self.c)**torch.arange(ntau).type(torch.DoubleTensor)

        # we'll need ALL combinations of (x, y, theta, tau_star), so make 
        #   all tensors broadcastable to that shape
        ndim = 4
        x = util.unsqueeze_except(x, ndim, dim=0)
        y = util.unsqueeze_except(y, ndim, dim=1)
        centered_x = x - c_x
        centered_y = y - c_y
        theta = util.unsqueeze_except(theta, ndim, dim=2)
        theta_orth = theta + torch.pi / 2
        tau_star = util.unsqueeze_except(tau_star, ndim, dim=3)

        a = math.log(k)*k
        b = torch.log(torch.arange(2,k).type(torch.DoubleTensor)).sum()
        
        A = ((1/tau_star)*(torch.exp(a-b))*(tau_star**self.g))
        A = util.unsqueeze_except(A, ndim, dim=0)
        
        # how to window in the main axis
        if window_shape == "concentric":
          # tau axis : Euclidian distance from center
          # 'orthogonal' axis : geodesic distance from a certain 
          #    point on the surface of a circle

          # the geodesic distance traveled before meeting another window
          lp_window_width = tau_star * dtheta/2
          tau = np.sqrt(centered_x**2 + centered_y**2)
          
          # geodesic distances
          grid_theta = torch.atan2(centered_y, centered_x)
          d1 = (theta - grid_theta).abs()
          d2 = (theta - (grid_theta-2*np.pi)).abs()
          theta_dist = torch.minimum(d1, d2)
          tau_orth = theta_dist * tau

          tau, tau_orth = torch.broadcast_tensors(tau, tau_orth)
        else:
          # tau axis : rotated straight axis pointing away from center
          # 'orthogonal' axis : orthogonal direction of above

          # the distance traveled in the orthogonal direction before meeting another window
          lp_window_width = tau_star * math.tan(dtheta/2) 

          tau = torch.cos(theta)*centered_x - torch.sin(theta)*centered_y
          tau_orth = torch.cos(theta_orth)*centered_x - torch.sin(theta_orth)*centered_y
        

        ## window in the opposite direction
        if localization == "gaussian":
          sd = (lp_window_width / self.sd_scale)**2 # put this many sd's in this space
          gaussian_scale_factor = 1/np.sqrt(np.pi*2*sd)
          orthogonal_window = gaussian_scale_factor * np.exp(-tau_orth**2 / sd)
        elif localization == "step":
          orthogonal_window = tau_orth.abs() <= lp_window_width
  
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
        self.filters = self.filters / util.unsqueeze_except(filter_sum, n_dim=3, dim=0)
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