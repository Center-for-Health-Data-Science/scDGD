# scDGD

scDGD is an application of our encoder-less generative model, the Deep Generative Decoder (DGD), to single-cell transcriptomics data. 

It learns low-dimensional representations of full transcriptomics matrices without feature selection. The low-dimensional embeddings are of higher quality than comparable methods such as scVI and the data reconstruction is highly data efficient, outperforming scVI and scVAE, especially on very small data sets.

For more information about the underlying method and our results, check out our [manuscript](https://arxiv.org/abs/2110.06672).

<img src="https://github.com/Center-for-Health-Data-Science/scDGD/DGD_schematic.png" width="500">

## Installation

You can install the package via
```
pip install git+https://github.com/Center-for-Health-Data-Science/scDGD
```

## How to use it

From our experience, scDGD can be applied to data sets with as few as 500 cells and as many as one million.

Check out the notebook showing an example of how to use scDGD:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Center-for-Health-Data-Science/scDGD/blob/HEAD/examples/scDGD_training_mousebrain5k.ipynb)

We have also uploaded pre-trained models for the PBMC (Zheng et al. 2017) and the 10X 1 million mouse brain data sets, along with notebooks showing how to load them [in pretrained_models](https://github/Center-for-Health-Data-Science/scDGD/pretrained_models/).

## Reference

If you use scDGD in your research, please consider citing

```
@misc{schusterkrogh2022,
      title={The Deep Generative Decoder: MAP estimation of representations improves modeling of single-cell RNA data}, 
      author={Viktoria Schuster and Anders Krogh},
      year={2022},
      eprint={2110.06672},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```