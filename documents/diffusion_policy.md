# Diffusion Policy

## Directory Structure

```
.
├── data # original data dir, and used for saving outputs
├── tests # env or functions testing
├── <font color='red'>diffusion_policy</font> # key dir
├── eval_real_robot.py # real env
├── multirun_metrics.py # compute metrics using logs
└── train.py # training file
```

**diffusion_policy/**

```

├── codecs # images processing
├── common # utils
├── config # configuration dir
│   ├── defaults.yaml # configuration for workspace (defaults/overrides/policy/dataloader/opt/training/logging/checkpoint/hydra)
│   ├── task # configuration for task (env_runner/dataset/shape_meta[obs&acts description])
├── dataset # dataset class for per task
├── env # env class for per task (reset/step/render/_get_obs...)
├── env_runner # execute policy
├── gym_util # render issues
├── model # Neural Network Architecture
├── policy # predict actions/compute loss
├── real_world # real robot
├── scripts # buffer saving 
├── shared_memory # real robot
└── workspace # execute env_runner -> train/validation 
```

<font color='red' size=4>ReplayBuffer -> storing a demonstration dataset</font>



### PushT Task

![image-20230810142139952](assets/image-20230810142139952.png)

![image-20230810143802441](assets/image-20230810143802441.png)

**obs_dim (20)** -> 9\*2 keypoints + 2 state

**action_dim (2)**

### Policy (Diffusion Unit Low Dim Policy)

```python
def conditional_sample(**kwargs) # used for inference
def predict_action(**kwargs) # predict actions with `conditaional_sample`
```

![image-20230810144640975](assets/image-20230810144640975.png)

### Network Achitecture

![image-20230810144622188](assets/image-20230810144622188.png)

```python
"""
x: (B,T,input_dim)
timestep: (B,) or int, diffusion step
local_cond: (B,T,local_cond_dim)
global_cond: (B,global_cond_dim)
output: (B,T,input_dim)
"""
# diffusion_step_encoder
# if local_cond -> local_cond_encoder else none
# down_modules
# up_modules
# final_conv
```

### How to Train (workspace.run)



### Dataset

**diffusion_policy/dataset/**

```
├── base_dataset.py
├── blockpush_lowdim_dataset.py
├── kitchen_lowdim_dataset.py
├── kitchen_mjl_lowdim_dataset.py
├── pusht_dataset.py
├── pusht_image_dataset.py
├── real_pusht_image_dataset.py
├── robomimic_replay_image_dataset.py
└── robomimic_replay_lowdim_dataset.py
```

**base_dataset: **includes class `BaseLowdimDataset` and class `BaseImageDataset`, there are closely the same. 

| pusht_image_dataset                          | pusht_dataset                                |
| -------------------------------------------- | -------------------------------------------- |
| ReplayBuffer -> img, state, action           | ReplayBuffer -> keypoint, state, action      |
| obs -> image(T, 3, 96, 96) + agent_pos(T, 2) | obs -> cat(kpoint(T, 9, 2), agent_pos(T, 2)) |

**Original Data (`__name__.zarr`)** <font color='red'>Load by ReplayBuffer</font>

```tex
$ lowdim push T/2D without image input $
 ├── data
 │   ├── action (25650, 2) float32
 │   ├── img (25650, 96, 96, 3) float32
 │   ├── keypoint (25650, 9, 2) float32
 │   ├── n_contacts (25650, 1) float32
 │   └── state (25650, 5) float32
 └── meta
     └── episode_ends (206,) int64 -> start & end
```

**Training Data**

```tex
$ lowdim push T/2D without image input $
- obs		torch.Size([bs, horizon, 20])
- action	torch.Size([bs, horizon, 2])

$ image push T $
- obs		torch.Size([bs, horizon, channels, h, w])
- agent_pos	torch.Size([bs, horizon, 2])
- action	torch.Size([bs, horizon, 2])

$ block push $
- obs		torch.Size([bs, horizon, 16])
- action	torch.Size([bs, horizon, 2])
```

:point_down:**Data Sampling Algorithm**

```tex
# pad_before + episode_length + pad_after
# min_start -> -pad_before
# max_start -> episode_length - sequence_lenth + pad_after
$ indices -> [ts - horizon * len(episode_ends), 4] $
```

Sample sequences like this is more reasonable.

- [x] Get `push block` task data 

- [ ] obs/actions data :question:

- [ ] algorithm, data -> policy :dart:<font color='blue'>sample algorithm</font>

- [ ] diffusion policy / ACT 