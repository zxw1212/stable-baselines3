# single train
# import gymnasium as gym
# from stable_baselines3 import PPO
# from six_dof_arm_env import SixDoFPosControlEnv

# six_dof_arm_env = SixDoFPosControlEnv()
# import time

# # Train
# model = PPO("MlpPolicy", six_dof_arm_env, verbose=1, device="cpu")
# model.learn(total_timesteps=500000)
# model.save("cartesian_move_ppo")
# del model # remove to demonstrate saving and loading

# model = PPO.load("cartesian_move_ppo", device="cpu")

# obs, _ = six_dof_arm_env.reset()
# dones = False
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, _, info = six_dof_arm_env.step(action)
#     six_dof_arm_env.render()
#     print(f"Reward: {rewards}, EE_pos: {obs[-3:]}")
#     time.sleep(0.001)




# parallel train
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from six_dof_arm_env import SixDoFPosControlEnv
from stable_baselines3.common.env_util import make_vec_env
import time

six_dof_arm_env = make_vec_env("SixDoFArm-v0", n_envs=5) # support parallel env: sac Not, PPO Yes

# Train from scratch
model = PPO("MlpPolicy", six_dof_arm_env, verbose=1, device="cpu")
# model = SAC("MlpPolicy", six_dof_arm_env, verbose=1, device="cpu")
model.learn(total_timesteps=500000)

# continue train
# model = PPO.load("cartesian_move_ppo", env=six_dof_arm_env, device="cpu")
# model.learn(total_timesteps=, reset_num_timesteps=False)

model.save("cartesian_move_ppo")
del model # remove to demonstrate saving and loading
six_dof_arm_env.close()

# test on single env
model = PPO.load("cartesian_move_ppo", device="cpu")
# model = SAC.load("cartesian_move_ppo", device="cpu")
six_dof_arm_env = SixDoFPosControlEnv(use_passive_viewer=True)
obs, _ = six_dof_arm_env.reset()
dones = False
while True:
    start_time = time.time()
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _, info = six_dof_arm_env.step(action)
    six_dof_arm_env.render()
    print(f"Reward: {rewards}, EE_pos: {obs[-3:]}")
    time.sleep(max(0, six_dof_arm_env.get_dt() - (time.time() - start_time)))

