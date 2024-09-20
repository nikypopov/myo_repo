# MyoSuite Challenge Setup
## quick to-dos
[x] setup and pretrained baselines

[ ] train agents with depRL
  - updates: https://github.com/martius-lab/depRL/issues/6 https://github.com/MyoHub/myosuite/issues/185
  - currently training
    
[ ] train vanilla agents
  - PPO, but train for much longer

## Setting Up the Conda Environment

To set up the conda environment for MyoSuite, follow the steps below:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/nikypopov/myo_repo.git
    cd myo_repo
    ```

2. **Create the Conda Environment**:
    Use the `MyoSuite.yml` file to create the conda environment.
    ```bash
    conda env create -f environment_config.yaml
    ```

3. **Activate the Environment**:
    After creating the environment, activate it with the following command:
    ```bash
    conda activate Myo
    ```

4. **Verify the Installation**:
    Ensure that the environment is set up correctly by running:
    ```bash
    conda list
    ```
## simulate.py

Script to render locomotion challenge environment and test policy

## examples

- run ```pretrained_deprl_walk.py``` or ```test_myosuite_baselines.py``` to visualize some of the pre trained baselines (at the end of script be sure to uncomment environment/task you want to render)

## Notes
- The `environment_config.yaml` file should be in the root directory of the repository. If it's located in a different directory, adjust the file path accordingly in the command.

## Resources and Info

[MyoSuite Docs](https://myosuite.readthedocs.io/en/latest/index.html).
[DepRL Docs](https://deprl.readthedocs.io/en/latest/index.html)
[DepRL Paper](https://arxiv.org/pdf/2206.00484)

---
