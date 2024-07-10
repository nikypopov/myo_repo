# MyoSuite Setup


[x] files and instructions to set up the MyoSuite environment 

[x] run pretrained baselines
  - run ```test_myosuite_baselines.py``` or ```pretrained_deprl_walk.py``` to visualize some of the pre trained baselines (at the end of script be sure to uncomment environment/task you want to render)

[ ] train agents with depRL
  - updates: https://github.com/martius-lab/depRL/issues/6 https://github.com/MyoHub/myosuite/issues/185

## Setting Up the Conda Environment

To set up the conda environment for MyoSuite, follow the steps below:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/nikypopov/rl_myosuite.git
    cd rl_myosuite
    ```

2. **Create the Conda Environment**:
    Use the `MyoSuite.yml` file to create the conda environment.
    ```bash
    conda env create -f MyoSuite.yml
    ```

3. **Activate the Environment**:
    After creating the environment, activate it with the following command:
    ```bash
    conda activate MyoSuite
    ```

4. **Verify the Installation**:
    Ensure that the environment is set up correctly by running:
    ```bash
    conda list
    ```

## Notes
- Conda [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
- The `MyoSuite.yml` file should be in the root directory of the repository. If it's located in a different directory, adjust the file path accordingly in the command.

## Additional Information

[MyoSuite Docs](https://myosuite.readthedocs.io/en/latest/index.html).
[DepRL Docs](https://deprl.readthedocs.io/en/latest/index.html)

---
