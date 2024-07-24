Usage
=====

.. _run_samples:

Running Experiments
----------------

1. Define the nonlinear control problem in ``repr_control/define_problem.py``. Following items needs to be defined:
   
   - Dynamics
   - Reward function
   - Initial distributions
   - State and action bounds
   - Maximum rollout steps
   - Noise level
   
   The current file is an example of inverted pendulum.

   .. literalinclude:: define_problem.py
      :language: python
      :linenos:


Advanced Usage
----------------

Define training hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


You can define training hyperparameters via adding command line arguments when running `solve.py`. 

For example,

- setting max training steps:
  
.. code-block:: console

   $ python solve.py --max_step 2e5
   

inspect the training results using tensorboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

   $ # during/after training
   $ tensorboard --logdir $LOG_PATH