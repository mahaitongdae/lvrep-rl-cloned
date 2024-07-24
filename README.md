# repr-control: A Toolbox to solve stochastic nonlinear control

repr-control is a toolbox to solve nonlinear stochastic control via representation learning. 
User can simply input the **dynamics, rewards, initial distributions** [sample_files](repr_control/define_problem.py) of the nonlinear control problem
and get the optimal controller parametrized by a neural network.

The optimal controller is trained via Spectral Dynamics Embedding Control (SDEC) algorithm based on representation learning and reinforcement learning.
For those interested in the details of SDEC algorithm, please check our [papers](https://arxiv.org/abs/2304.03907).

## Installation
1. install anaconda (if you haven't) and create new environment,
    ```shell
    conda create -n repr-control python=3.10
    conda activate repr-control
    ```
2. Install PyTorch dependencies. 
  
    **Windows or Linux**: 

    If you have CUDA-compatible GPUs,
    ```shell
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
    If you don't have CUDA-compatible GPUs,
    ```shell
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```
   **Mac**:
    ```shell
    conda install pytorch::pytorch torchvision torchaudio -c pytorch
    ```
3. install the toolbox
    ```shell
   git clone https://github.com/CoNGHarvard/repr_control.git
   cd repr_control
   pip install -e .
   ```

Helpful resources: 
- [Anaconda environment](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)
- [PyTorch installation](https://pytorch.org/get-started/locally/)

## Usage
1. Define the nonlinear control problem in [define_problem.py](repr_control/define_problem.py). Following items needs to be defined:
   - Dynamics
   - Reward function
   - Initial distributions
   - State and action bounds
   - Maximum rollout steps
   - Noise level
   
   The current file is an example of inverted pendulum.

2. Testing your model
   ```shell
   pytest tests/test_custom_model.py
   ```
3. Run the training scripts
   ```shell
   cd repr_control
   python solve.py 
   # If you have CUDA-compatible GPUs
   python solve.py --device cuda
   # If on Apple mac, not sure whether it will be faster than using cpu.
   python solve.py --device mps 
   ```
   The training results will be saved in the `repr_control/log/` directory.  

4. Evaluate solving results
   ```shell
   cd repr_control/scripts
   python eval.py $LOG_PATH
   ```
   where `$LOG_PATH` is the path of training folder. For example,
   ```shell
   cd repr_control/scripts
   python eval.py ../examples/example_results/rfsac/Pendulum/seed_0_2024-07-18-14-50-35
   ```   

5. Get controllers to use elsewhere
   ```python
   import numpy as np
   from repr_control.scripts.eval import get_controller
   log_path = '$LOG_PATH'
   agent = get_controller(log_path)
   state = np.zeros([3]) # taking the pendulum as example
   action = agent.select_action(state, explore=False)
   ```
   

## Advanced Usage

### Define training hyperparameters
You can define training hyperparameters via adding command line arguments when running `solve.py`. For example,
- setting max training steps:
   ```shell
   python solve.py --max_step 2e5
   ```

### inspect the training results using tensorboard

```shell
# during/after training
tensorboard --logdir $LOG_PATH
```

## Citations
```
@article{ren2023stochastic,
      title={Stochastic Nonlinear Control via Finite-dimensional Spectral Dynamic Embedding}, 
      author={Tongzheng Ren and Zhaolin Ren and Haitong Ma and Na Li and Bo Dai},
      year={2023},
      eprint={2304.03907},
      archivePrefix={arXiv}
}
```
