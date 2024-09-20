import myosuite
from myosuite.utils import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
# import deprl

# Step 1: Initialize the MyoSuite Environment
def create_env():
    env = gym.make('myoChallengeRunTrackP1-v0', reset_type='init')  # Use appropriate environment ID
    #check_env(env)  # Optional: Check environment for compatibility, need to fix warnings later
    return env


if __name__ == "__main__":
    env = create_env()

    # Step 2: Initialize the PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Step 3: Train the agent
    model.learn(total_timesteps=10000000)  # Adjust as needed for your training duration #COMMENT out after you've trained model

    # # Save the model
    # model.save("ppo_myo_challenge")

    # Load the model
    # model = PPO.load("ppo_myo_challenge") #PPO #UNCOMMENT after you've trained the model to use it

    # After training, you can evaluate the model:
    observation, info = env.reset()

    # Repeat 1000 time steps
    for _ in range(10000):
        # Activate mujoco rendering window
        env.mj_render()
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        # Reset env on fail
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print("script success")
