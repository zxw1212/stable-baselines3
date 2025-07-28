import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import os
import time
from gymnasium.envs.registration import register

class SixDoFPosControlEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self, render_mode=None, use_passive_viewer=False):
        super().__init__()
        self.render_mode = render_mode
        xml_path = os.path.join(os.path.dirname(__file__), "/home/zxw/software/stable-baselines3/examle/6dof_arm.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.viewer = None

        dof = 6
        self.dof = dof
        self.last_action = np.zeros(dof, dtype=np.float32)

        # 观测空间: qpos(6), qvel(6), ee_pos(x,y,z)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6 + 6 + 3,), dtype=np.float32)
        obs_low = np.array([-np.pi] * 6 + [-10] * 6 + [-5.0, -5.0, -5.0], dtype=np.float32)  # qpos, qvel, ee_pos
        obs_high = np.array([np.pi] * 6 + [10] * 6 + [5.0, 5.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # 关节范围
        self.joint_ranges = np.array([self.model.jnt_range[i] for i in range(dof)])  # shape (6,2)
        # self.action_space = spaces.Box(low=self.joint_ranges[:, 0], high=self.joint_ranges[:, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dof,), dtype=np.float32)

        self.max_delta = 1 #0.0001 # max_v = max_delta / self.get_dt() # 0.0001 / 0.001 = 0.1 rad/s
        # 本意是为了防止动作太快, 但太小会导致探索不到位, random的action执行时一直在某个区域打转 Note: xml中的joint PD参数也会用这方面的影响

        if use_passive_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        self.max_steps = 200
        self.step_count = 0

        self.reset()
    
    def get_dt(self):
        """获取模型的时间步长"""
        return self.model.opt.timestep

    def get_ee_pos(self):
        mujoco.mj_forward(self.model, self.data)
        site_id = self.model.site("ee_site").id
        return self.data.site_xpos[site_id].copy()  # x,y,z 三维坐标

    def step(self, normalized_action):
        self.step_count += 1
        normalized_action = np.clip(normalized_action, self.action_space.low, self.action_space.high)

        action = ((normalized_action - (-1.0)) / 2.0) * (self.joint_ranges[:, 1] - self.joint_ranges[:, 0]) + self.joint_ranges[:, 0]
        action = np.clip(action - self.last_action, -self.max_delta, self.max_delta) + self.last_action
        self.last_action = action.copy()


        self.data.ctrl[:] = action.astype(np.float64)
        mujoco.mj_step(self.model, self.data)

        ee_pos = self.get_ee_pos()
        # 简单奖励：末端距离目标点(0.5, 0.0, 0.0)的负距离
        target_pose = np.array([0.205, 0,  1.7858])
        distance = np.linalg.norm(ee_pos - target_pose)

        reward = -distance
        # if distance < 0.01:
        #     reward += 10.0
        if distance < 0.1:
            alpha = 50.0
            reward += 10.0 * np.exp(-alpha * distance)
        # alpha = 50.0
        # reward = np.exp(-alpha * distance)

        # reward -= 0.001 * np.linalg.norm(self.data.qvel[:6])
        reward -= 0.01 * np.linalg.norm(self.last_action - action)

        terminated = distance < 0.01  # 如果末端位置接近目标位置则终止
        truncated = self.step_count >= self.max_steps
        info = {}
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # def reset(self):
    #     self.data.qpos[:] = 0.0
    #     self.data.qvel[:] = 0.0
    #     mujoco.mj_forward(self.model, self.data)
    #     return self._get_obs()
    def reset(self, seed=None, **kwargs):  # 添加 seed 参数
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)  # 或者你可以选择其他方式设置随机种子

        # 继续原来的重置代码
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(self.dof, dtype=np.float32)  # 重置 last_action

        self.step_count = 0
        
        info = {}
        obs = self._get_obs()
        assert self.observation_space.contains(obs)
        return obs, info

    def _get_obs(self):
        qpos = self.data.qpos[:self.dof]
        qvel = self.data.qvel[:self.dof]
        ee_pos = self.get_ee_pos()
        return np.concatenate([qpos, qvel, ee_pos]).astype(np.float32)

    def render(self):
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


register(
    id="SixDoFArm-v0",
    entry_point="six_dof_arm_env:SixDoFPosControlEnv",
    max_episode_steps=200,
)


# check env
# from gymnasium.utils.env_checker import check_env
# if __name__ == "__main__":
#     env = SixDoFPosControlEnv()
#     check_env(env) 


if __name__ == "__main__":
    env = SixDoFPosControlEnv(use_passive_viewer=True)

    obs, _ = env.reset()
    print("初始观测:", obs)
    i = 0
    delta = 0.0001
    normalized_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    for step in range(1000000):
        time_start = time.time()
        # normalized_action = env.action_space.sample()  # 随机动作, 验证当前env max delta和joint pd设置下, agent是否能覆盖工作空间
        normalized_action += np.array([delta, delta, delta, delta, delta, delta], dtype=np.float32)
        obs, reward, done, _, info = env.step(normalized_action)
        print(f"Step {step} → Reward: {reward}, EE_pos: {obs[-3:]}")
        env.render()
        time.sleep(max(0, env.get_dt() - (time.time() - time_start)))
