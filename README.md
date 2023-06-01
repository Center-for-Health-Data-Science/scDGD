# scDGD

scDGD is an application of our encoder-less generative model, the Deep Generative Decoder (DGD), to single-cell transcriptomics data. 

It learns low-dimensional representations of full transcriptomics matrices without feature selection. The low-dimensional embeddings are of higher quality than comparable methods such as scVI and the data reconstruction is highly data efficient, outperforming scVI and scVAE, especially on very small data sets.

For more information about the underlying method and our results, check out our [manuscript](https://arxiv.org/abs/2110.06672).

## Installation

You can install the package via
```
pip install git+https://github.com/Center-for-Health-Data-Science/scDGD
```

## How to use it

From our experience, scDGD can be applied to data sets with as few as 500 cells and as many as one million.

Check out the notebook showing an example of how to use scDGD:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Center-for-Health-Data-Science/scDGD/blob/HEAD/examples/scDGD_training_mousebrain5k.ipynb)

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