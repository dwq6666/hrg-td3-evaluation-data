# HRG-TD3

Implementation of *HRG-TD3: Traversability-Aware Mapless Navigation via Joint Distance–Height Reweighting and Rule-Authorized Traversal Gating*. The `full` preset is the HRG-TD3 configuration.

The project includes the Python training and evaluation code and the ROS/Gazebo overlay for the navigation environments. The evaluation code generates episode manifests from the requested seed.

## Setup

Use a ROS Noetic and Gazebo environment with PyTorch, NumPy, OpenCV, Pillow, and squaternion. The ROS packages and robot meshes build on the MIT-licensed [DRL-robot-navigation project](https://github.com/reiniscimurs/DRL-robot-navigation). The upstream license is included in `LICENSE`.

From the root of this project, apply the bundled overlay to the upstream ROS workspace.

```sh
git clone https://github.com/reiniscimurs/DRL-robot-navigation.git ../hrg-td3-ros
cp -a catkin_ws/src/multi_robot_scenario/. ../hrg-td3-ros/catkin_ws/src/multi_robot_scenario/
(cd ../hrg-td3-ros/catkin_ws && catkin_make_isolated)
source ../hrg-td3-ros/catkin_ws/devel_isolated/setup.bash
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_PORT_SIM=11311
```

## Trained checkpoints

The study's three HRG-TD3 policies were trained for 500,000 environment steps each. The three original PyTorch checkpoint files used in the reported experiments are included at the paths below. Use `checkpoints/SHA256SUMS.txt` to verify each file.

| Training seed | Checkpoint path |
| --- | --- |
| 0 | `checkpoints/full_seed0/checkpoint.pt` |
| 1 | `checkpoints/full_seed1/checkpoint.pt` |
| 2 | `checkpoints/full_seed2/checkpoint.pt` |

## Evaluate a trained policy

Run from the root of this project with the ROS workspace sourced and the selected checkpoint present.

```sh
python3 TD3/test_velodyne_td3.py \
  --preset full --seed 0 --checkpoint checkpoints/full_seed0/checkpoint.pt \
  --launch-sim --launchfile paper_traversability.launch \
  --run-root runs/evaluation --eval-seed 100000 --episodes 10
```

The evaluation script loads the checkpoint and writes new episode results under `runs/evaluation/`. PyTorch automatically selects CUDA when available and CPU otherwise.

## Train

```sh
python3 TD3/train_velodyne_td3.py \
  --preset full --seed 0 --budget final --max-timesteps 500000 \
  --run-root runs/training --launch-sim --launchfile paper_traversability.launch
```
