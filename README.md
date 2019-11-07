# sift-flow-gpu

Implementation of the SIFT Flow descriptor [[1]](#references) on GPU using PyTorch.

This implementation is a port of the original implementation available at
[https://people.csail.mit.edu/celiu/SIFTflow/](https://people.csail.mit.edu/celiu/SIFTflow/).

This code is able to process a batch of images simultaneously for better
performance. The most expensive operation when running in GPU mode is the
allocation of the space for the descriptors on the GPU. However, this step
is only performed when the shape of the input batch changes. Subsequent
calls using batches with the same shape as before will reuse the memory and
will, therefore, be much faster.

Code for DAISY descriptors on GPU is also available at [https://github.com/hmorimitsu/daisy-gpu](https://github.com/hmorimitsu/daisy-gpu).

## Requirements

- [Python 3](https://www.python.org/) (Tested on 3.7)
- [Numpy](https://www.numpy.org/)
- [PyTorch](https://pytorch.org/) >= 1.0.0 (Tested on 1.3.0)

## Usage

A simple example is shown below. A more complete practical usage is available as a [Jupyter demo notebook](demo_notebook_torch.ipynb)

```python
from sift_flow_torch import SiftFlowTorch

sift_flow = SiftFlowTorch()
imgs = [
    read_some_image,
    read_another_image
]
descs = sift_flow.extract_descriptor(imgs) # This first call can be
                                           # slower, due to memory allocation
imgs2 = [
    read_yet_another_image,
    read_even_one_more_image
]
descs2 = sift_flow.extract_descriptor(imgs2) # Subsequent calls are faster,
                                             # if images retain same shape

# descs[0] is the descriptor of imgs[0] and so on.
```

## Benchmark

- Machine configuration:
  - Intel i7 8750H
  - NVIDIA GeForce GTX1070
  - Images 1024 x 436
  - Descriptor size 128

Batch Size|FP16|Memory usage(GB)<sup>1</sup>|Time GPU(ms)<sup>2</sup>|Time GPU(ms)<sup>3</sup>|Time CPU(ms)
-|------------------|---|------|------|------
1|                  |0.9|  19.0| 128.0| 660.6
2|                  |1.3|  35.3| 257.1|1275.1 
4|                  |2.1|  70.7| 516.2|2559.3 
8|                  |3.7| 142.5| 969.4|5773.9 
1|:heavy_check_mark:|0.7|  14.7|      |
2|:heavy_check_mark:|0.9|  27.2|      |
4|:heavy_check_mark:|1.3|  54.8|      |
8|:heavy_check_mark:|2.1| 110.9|      |

<sup>1</sup> Maximum value reported by `nvidia-smi` during the respective tests.

<sup>2</sup> NOT including time to transfer the result from GPU to CPU.

<sup>3</sup> Including time to transfer the result from GPU to CPU.

These times are the median of 5 runs measured after a warm up run to allocate the descriptor space in memory
(read the [introduction](#sift-flow-gpu)).

## References

[1] C. Liu; Jenny Yuen; Antonio Torralba. "SIFT Flow: Dense correspondence across scenes and its applications." IEEE Transactions on Pattern Analysis and Machine Intelligence 33.5 (2010): 978-994.
