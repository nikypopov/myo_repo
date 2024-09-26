import os
import torch
import torch.multiprocessing as mp
from datetime import datetime
import myosuite
from myosuite.utils import gym
from stable_baselines3 import PPO, A2C, DQN, DDPG, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.rewards.append(reward)
        return True

    def get_rewards(self):
        return self.rewards

class Agent:
    def __init__(self, algo, env, device, log_dir=None):
        self.model = None  # Initialize model
        self.reward_logger = RewardLogger()

        # Initialize model with selected device (GPU/CPU)
        if algo == "PPO":
            self.model = PPO("MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir)
        elif algo == "A2C":
            self.model = A2C("MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir)
        elif algo == "DQN":
            self.model = DQN("MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir)
        elif algo == "DDPG":
            self.model = DDPG("MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir)
        elif algo == "SAC":
            self.model = SAC("MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir)
        else:
            print("Invalid algorithm chosen. Please choose from PPO, A2C, DQN, DDPG, or SAC.")
            return

        # If log_dir is provided, configure the logger for Tensorboard and CSV
        if log_dir:
            new_logger = configure(log_dir, ["tensorboard", "csv"])
            self.model.set_logger(new_logger)

    def train(self, train_steps, algo):
        # Start the progress bar using tqdm
        with tqdm(total=train_steps, desc=f"Training {algo}", unit="steps") as pbar:
            start_time = time.time()

            def callback(_locals, _globals):
                pbar.update(1)  # Update the progress bar by 1 step
                elapsed_time = time.time() - start_time
                steps_done = _locals["self"].num_timesteps  # Number of timesteps completed
                # Estimate remaining time
                if steps_done > 0:
                    estimated_total_time = (elapsed_time / steps_done) * train_steps
                    remaining_time = estimated_total_time - elapsed_time
                    pbar.set_postfix({"Elapsed": time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
                                      "Remaining": time.strftime("%H:%M:%S", time.gmtime(remaining_time))})
                return True  # Continue training

            # Train the model with progress bar callback
            self.model.learn(total_timesteps=train_steps, callback=callback)

        self.model.save(f"myo_challenge_{algo}_agent")
        print(f"{algo} agent trained and saved.")

    def get_training_rewards(self):
        return self.reward_logger.get_rewards()

    def evaluate(self, env, timesteps=10000):
        rewards = []
        observation, info = env.reset()

        for _ in range(timesteps):
            action, _ = self.model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()
        return np.sum(rewards)  # Return total reward over evaluation steps

def create_env(env_name, reset):
    env = gym.make(env_name, reset_type=reset)
    return env

def detect_device():
    # Detect device: Apple Metal (M1/M2) or Nvidia CUDA
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (Metal)")
        return torch.device("mps")  # Metal on macOS
    elif torch.cuda.is_available():
        print(f"Using Nvidia GPU (CUDA): {torch.cuda.device_count()} GPU(s) available")
        return torch.device("cuda")  # CUDA on Nvidia GPUs
    else:
        print("Using CPU")
        return torch.device("cpu")

# helper function for parallel_training with multiple GPUs
def train_on_single_gpu(env_name, reset, train_steps, algo, gpu_id, reward_logs, log_dir):
    device = torch.device(f"cuda:{gpu_id}")
    env = create_env(env_name, reset)
    locomotion_model = Agent(algo, env, device, log_dir=log_dir)
    locomotion_model.train(train_steps, algo)

    # Store reward logs after training is completed
    reward_logs[algo] = locomotion_model.get_training_rewards()

def sequential_training(env, train_steps, device, log_dir=None):
    algorithms = ["PPO", "A2C", "DQN", "DDPG", "SAC"]
    total_rewards = {}
    reward_logs = {}  # To store reward logs for each algorithm

    for algo in algorithms:
        print(f"Training with {algo} on {device}...")
        locomotion_model = Agent(algo, env, device, log_dir=log_dir)
        locomotion_model.train(train_steps, algo)

        # Store the reward logs for plotting
        reward_logs[algo] = locomotion_model.get_training_rewards()

        total_reward = locomotion_model.evaluate(env)
        total_rewards[algo] = total_reward
        print(f"Total reward for {algo}: {total_reward}")

    # Plot and save learning curves for all algorithms
    plot_learning_curves(reward_logs, algorithms)

    return total_rewards

def parallel_training(env_name, reset, train_steps, num_gpus, log_dir=None):
    algorithms = ["PPO", "A2C", "DQN", "DDPG", "SAC"]
    mp.set_start_method('spawn', force=True)  # For CUDA compatibility
    reward_logs = mp.Manager().dict()  # Shared dictionary for reward logs across processes

    processes = []
    for i, algo in enumerate(algorithms):
        gpu_id = i % num_gpus  # Cycle through GPUs
        p = mp.Process(target=train_on_single_gpu, args=(env_name, reset, train_steps, algo, gpu_id, reward_logs, log_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Plot and save learning curves for all algorithms after all processes finish
    plot_learning_curves(dict(reward_logs), algorithms)

def create_output_dir():
    # Create a directory with the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"training_data_{current_time}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_learning_curves(reward_logs, algorithms):
    # Create output directory
    output_dir = create_output_dir()

    for algo, rewards in reward_logs.items():
        plt.figure(figsize=(12, 6))
        plt.plot(rewards, label=algo)
        plt.title(f'Learning Curve for {algo}')
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.legend()

        # Save the plot to the output directory
        plot_filename = os.path.join(output_dir, f"{algo}_learning_curve.png")
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to avoid displaying it and free memory

        print(f"Saved plot for {algo} at {plot_filename}")

    print(f"All plots saved in {output_dir}")

if __name__ == "__main__":
    # environment and reset parameters
    env_name = 'myoChallengeRunTrackP1-v0'
    reset = 'init'
    train_steps = 5000000  # Train for 5,000,000 steps for each algorithm
    log_dir = create_output_dir()  # Log directory for both Tensorboard and CSV logging

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(f"Multiple Nvidia GPUs detected: {num_gpus}. Running in parallel mode.")
                parallel_training(env_name, reset, train_steps, num_gpus, log_dir=log_dir)
            else:
                print("Single Nvidia GPU detected, proceeding with sequential training.")
                device = torch.device("cuda:0")
                env = create_env(env_name, reset)
                sequential_training(env, train_steps, device, log_dir=log_dir)
        elif torch.backends.mps.is_available():
            print("Apple Silicon GPU (MPS) detected, proceeding with sequential training.")
            device = torch.device("mps")
            env = create_env(env_name, reset)
            sequential_training(env, train_steps, device, log_dir=log_dir)
    else:
        print("No GPU detected, using CPU.")
        device = torch.device("cpu")
        env = create_env(env_name, reset)
        sequential_training(env, train_steps, device, log_dir=log_dir)

    print("Training complete for all policies.")
