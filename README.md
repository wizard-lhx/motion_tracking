# motion_tracking

Active-adaptation project. Add tasks under `cfg/task/`, experiments under `cfg/exp/`.

# 数据集可视化
将处理好的 dataset 放在根目录，然后在 mujoco 中可视化参考动作
运行前先把`motion_tracking/__init__.py`内容注释掉
`uv run python motion_tracking/src/motion_tracking/visulizer.py -d lafan_all`

# 处理数据集
`uv run python motion_tracking/src/motion_tracking/build_kinematics.py --overwrite
`