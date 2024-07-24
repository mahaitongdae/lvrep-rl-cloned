Installation
=====


.. _installation:

.. Installation
.. ------------

To use repr-control, 

1. first install anaconda (if you haven't) and create new environment,

   .. code-block:: console

      $ conda create -n repr-control python=3.10
      $ conda activate repr-control

2. Install pytorch
   
   **Windows or Linux**: If you have CUDA-compatible GPUs,

   .. code-block:: console

      $ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

   If you don't have CUDA-compatible GPUs,

   .. code-block:: console

      $ conda install pytorch torchvision torchaudio cpuonly -c pytorch

   **Mac**:

   .. code-block:: console

      $ conda install pytorch::pytorch torchvision torchaudio -c pytorch

   Ohter helpful resources for creating python environments:

   - `Anaconda environment <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_
   - `Installing Pytorch <https://pytorch.org/get-started/locally/>`_