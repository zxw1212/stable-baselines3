# import mujoco
# import mujoco.viewer
# from mujoco import MjModel, MjData

# # 加载模型
# model = mujoco.MjModel.from_xml_path("/home/zxw/software/stable-baselines3/examle/6dof_arm.xml")
# data = mujoco.MjData(model)

# # 启动交互式 viewer，可用鼠标旋转、拖动、缩放场景
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     print("MuJoCo viewer is running... Close the window to end the program.")

#     # 每帧仿真
#     while viewer.is_running():
#         mujoco.mj_step(model, data)  # 推进一步仿真
#         viewer.sync()                # 同步 viewer 状态


import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData
import numpy as np
import time

# 加载模型
model = mujoco.MjModel.from_xml_path("/home/zxw/software/stable-baselines3/examle/6dof_arm.xml")
data = mujoco.MjData(model)
timestep = model.opt.timestep
print(f"Model timestep: {timestep:.4f} seconds")

# 启动 GUI 可视化
with mujoco.viewer.launch_passive(model, data) as viewer:
    q_desired = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    delta = 0.0001 # v = 0.0001 / 0.001(mujuc推进时间步长) = 0.1 rad/s
    while viewer.is_running():
        start = time.time()
        q_desired += np.array([delta, delta, delta, delta, delta, delta])
        print(f"Desired Joint Positions: {q_desired}")
        data.ctrl[:] = q_desired
        mujoco.mj_step(model, data)  # 推进仿真, 每次推进的是 一个物理时间步（默认 0.002s）
        viewer.sync()
        # print the fk
        # ee_pos = data.site_xpos[data.site("ee_site").id]
        # print(f"End Effector Position: {ee_pos}")


        # time.sleep(1) # 如果加了这个sleep, 每秒只mj_step一次, 即推进仿真0.002s, 很慢看不出来
        # time.sleep(max(0, timestep - (time.time() - start))) # 按仿真时间步来sleep, 保持每次mj_step的间隔一致, 这样仿真才真实
        # 如果用于rl训练, 可以不sleep, 让mj_step尽可能快地推进仿真
