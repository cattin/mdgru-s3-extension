# mdgru-s3-extension
MD-GRU with the additional S3 extension and with some changes to the loss function.

Multi-dimensional Gated Recurrent Units
=======================================

This repository is a fork of the original MD-GRU code in git+https://github.com/zubata88/mdgru.git.

The most prominent changes to this fork is the ability of loading data from S3 storage. In addition a minor bug was corrected to avoid doing data augmentation also during the validation phase. Furthermore, the loss function was entangled and made simpler to use.

The code has been developed in Python==3.5.2. It is best to set up a **virtual environment** (e.g. with `conda <https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/>`_) with the mentioned properties in order to develop the deep learning model. For this purpose, follow the instructions in the `docs <https://mdgru.readthedocs.io/en/latest/index.html>`_, and install mdgru (together with mvloader) using pip. In addition, make sure you have `CUDA <https://developer.nvidia.com/cuda-90-download-archive>`_/`cuDNN <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html>`_ installed.

::

    pip install git+https://github.com/zubata88/mdgru.git
    pip install git+https://github.com/spezold/mvloader.git

Papers
''''''

Reference implementation (and based on former Caffe version):

::

    @inproceedings{andermatt2016multi,
      title={Multi-dimensional gated recurrent units for the segmentation of biomedical 3D-data},
      author={Andermatt, Simon and Pezold, Simon and Cattin, Philippe},
      booktitle={International Workshop on Large-Scale Annotation of Biomedical Data and Expert Label Synthesis},
      pages={142--151},
      year={2016},
      organization={Springer}
    }

Code also used for (with modifications):

::

    @inproceedings{andermatt2017a,
      title = {{{Automated Segmentation of Multiple Sclerosis Lesions}} using {{Multi-Dimensional Gated Recurrent Units}}},
      timestamp = {2017-08-09T07:27:10Z},
      journal = {Lecture Notes in Computer Science},
      author = {Andermatt, Simon and Pezold, Simon and Cattin, Philippe},
      year = {2017},
      booktitle={International Workshop on Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries},
      note = {{{[accepted]}}},
      organization={Springer}
    }
    
    @article{andermatt2017b,
      title={Multi-dimensional Gated Recurrent Units for Automated Anatomical Landmark Localization},
      author={Andermatt, Simon and Pezold, Simon and Amann, Michael and Cattin, Philippe C},
      journal={arXiv preprint arXiv:1708.02766},
      year={2017}
    }
    
    @article{andermatt2017wmh,
      title={Multi-dimensional Gated Recurrent Units for the Segmentation of White Matter Hyperintensites},
      author={Andermatt, Simon and Pezold, Simon and Cattin, Philippe}
    }
    
    @inproceedings{andermatt2017brats,
    title = {Multi-dimensional Gated Recurrent Units for
    Brain Tumor Segmentation},
    author = {Simon Andermatt and Simon Pezold and Philippe C. Cattin},
    year = 2017,
    booktitle = {2017 International {{MICCAI}} BraTS Challenge}
    }

When using this code, please cite at least *andermatt2016multi*, since
it is the foundation of this work. Furthermore, feel free to cite the
publication matching your use-case from above. E.g. if you're using the
code for pathology segmentation, it would be adequate to cite
*andermatt2017a* as well.

Acknowledgements
''''''''''''''''

We thank the Medical Image Analysis Center for funding this work. |MIAC
Logo|

.. |MIAC Logo| image:: http://miac.swiss/gallery/normal/116/miaclogo@2x.png

