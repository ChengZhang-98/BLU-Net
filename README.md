# BLU-Net


An end-to-end software framework to compress U-Net for cell segmentation while maintaining accuracy. 

- Our compressed model, BLU-Net, has an 82.7× smaller size and a remarkable segmentation accuracy of 85.4% IoU (3.4% lower than full-precision vanilla U-Net)) on our custom dataset.

    <img src="https://s2.loli.net/2022/08/15/LSWGn6eCwb53jdu.jpg" width="700" />

- We demonstrate that fine-grained hint learning and binarisation can transfer segmentation knowledge from full-precision vanilla U-Net to BLU-Net though they have different architectures. 

    <img src="https://s2.loli.net/2022/08/15/hSdlRogYxTNGiPp.jpg" width="700" />

- To our knowledge, this is the first work in binary neural networks for biomedical image segmentation.

## Network Compression Framework
<img src="./compression-framework.png" width="700" />

```text
data.py: 
    dataset class
data_augmentation.py:
    data augmentation classes and functions
knowledge_distillation.py: 
    class and functions to transfer knowledge from vanilla unet to lightweight unet and retraining
model.py:
    functions to define unet, lightweight unet (lw_unet), and blu_net
residual_binarization.py:
    functions and class to implement residual binary network
training_utils:
    custom training callbacks, learning rate schedular, loss functions, metrics, and dataset split
```

## Segmentation Examples

<img src="https://s2.loli.net/2022/08/15/4LMBpfPrgn3mTSD.jpg" width="400" />

<img src="https://s2.loli.net/2022/08/15/hHNlWo2IMynPCcF.jpg" width="400" />