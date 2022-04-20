### Testing (true) angle invariance using a subset of MNIST
- Models are trained on a subset of digits from MNIST that are distinguishable over any rotation: `[1, 2, 3, 8, 0, 6]`
- They are trained with full pooling over the theta dimension, which either has axes every 90 degrees or every 30 degrees (`num_angles=4`, `num_angles=12`).
- They are trained on only non-rotated images, and we expect that when shifted 90 degrees / 30 degrees their performance should be exactly the same as the performance on the non-rotated images, as the LP transforms in each case should be exactly shifted versions of each other.
  - Note: For the 30 degree cases, due to resampling there will likely be a slight degradation in accuracy

**Results: The tests using the fixed version of the LP transform ("xxxx_xxxx_v2") show invariance over angles.**
- Also, it's worth mentioning that when using full maxpooling, `num_angles` has no effect on the number of parameters, so a higher value should tend to lead to invariance over a larger number of angles.
