# BP - bachelor's work
Generating realistic images from natural language with artificial neural net

## My setup
### PC
OS: Fedora33 with KDE

GPU: NVIDIA 1080TI with 11GB of global memory

CUDA Version: 11.2

### ENV
Python version: 3.

Tensorflow version: 1.15


## Generating from pretrained network
As provided in [stylegan2](https://github.com/NVlabs/stylegan2) repo, to generate images from [paper]() figure 12 use run_generator.py:

`python run_generator.py generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
  --seeds=66,230,389,1518 --truncation-psi=1.0`

This didn't work for me on the first try, we were getting an error - undefined symbol: \_ZN10tensorflow12OpDefBuilder6OutputESs

I found a simple hotfix at this [blog](https://blog.csdn.net/zaf0516/article/details/103618601). Turns out that the only thing we had to do is to set this nvcc compiler option `-fPIC -D_GLIBCXX_USE_CXX11_ABI=0` to `-fPIC -D_GLIBCXX_USE_CXX11_ABI=1` in **dnnlib/tflib/ops/custom_ops.py** (I assume that all of this is happening because the dnnlib/tflib/ops has a .cu file to compile to add fused bias activation function to tensorflow). 

After this run the run_generator.py again as mentioned and see the results.

## Custom generation test
If we want to generate condtitonaly, first we have to generate from random noise vector W.

As mention in stylegan2 readme, to add pretrained weights using pickle we need to use dnnlib. To import dnnlib from anywhere, we have added path to stylegan to our PYTHONPATH. Before picle load, we need to use `dnnlib.tflib.init_tf()` to initialize Tensorfow. 