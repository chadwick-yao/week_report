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

### Action Space and Observation Space

**Tasks:**

![image-20230813104532124](assets/image-20230813104532124.png)

> - **Lift:**
>   - Observation Space (10 dim), includes cube position(3), cube quaternion(7), and cube position relative to the robot end effector(3).
>   - Action Space (7 dim), includes desired translation of EEF(3), desired delta rotation from current EEF(3), and opening and closing of the gripper fingers.
> - **Can:**
>   - Observation Space (14 dim), includes absolute Can position and quaternion(7), Can position and quaternion relative to the EEF(7)
>   - Action Space (7 dim), the same
> - **Square:**
>   - Observation Space (14 dim), the same as Can, but here is square nut
>   - Action Space (7 dim), the same
> - **Transport:**
>   - Observation Space (41 dim), includes absolution position and quaternion of hammer(7), absolute pos and quat of the trash cube(7), absolute pos and quat of lid handle(7), target bin position(3), trash bin position(3), the relative pos of the hammer and trash cube with respect to the 1st and 2nd EEF(12), a binary indicator for the hammer reaching the target bin(1), a binary indicator for the trash reaching the trash bin(1)
>   - Action Space (14 dim), double
> - **Tool Hang:**
>   - Observation Space (44 dim), includes the absolute position and quaternion and relative pose and quaternion with respect to the end effector of the base frame (14-dim), the insertion hook (14-dim), and the ratcheting wrench (14-dim), as well as binary indicators for whether the stand was assembled (1-dim) and whether the tool was successfully placed on the stand (1-dim)
>   - Action Space (7 dim), the same
> - **Push-T:**
>   - Observation Space (3, 96, 96), i.e. the camera image
>   - Action Space (2), agent 2D position
> - **BlockPush:**
>   - Observation Space (16), includes
>   - Action Space (2), the same as Push-T
> - **Kitchen:**
>   - Observation Space (30 dim), includes 7 objects positions
>   - Action Space (8 dim), includes 6 DoF for the arm and 2 for the gripper.
> - **Real Pour:**
>   - Observation Space (2, 3, 320, 340)
>   - Action Space (6 dim), 6 DoF


## Algorithm

### :point_down:Data Sampling Algorithm

```tex
# pad_before + episode_length + pad_after
# min_start -> -pad_before
# max_start -> episode_length - sequence_lenth + pad_after
$ indices -> [ts - horizon * len(episode_ends), 4] $
```

Sample sequences like this is more reasonable.

### Model
- Conditional Unet 1D

- For UNET, this is how it obtain local_cond or global_cond

```python
if self.obs_as_local_cond:
    # condition through local feature
    # all zero except first To timesteps
    local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
    local_cond[:,:To] = nobs[:,:To]
    shape = (B, T, Da)
    cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
elif self.obs_as_global_cond:
    # condition throught global feature
    global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
    shape = (B, T, Da)
    if self.pred_action_steps_only:
        shape = (B, self.n_action_steps, Da)
    cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
else:
    # condition through impainting
    shape = (B, T, Da+Do)
    cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    cond_data[:,:To,Da:] = nobs[:,:To]
    cond_mask[:,:To,Da:] = True
```

**Module Details**
```python
# Diffusion Step Encoder
diffusion_step_encoder = nn.Sequential(
    SinusoidalPosEmb(dsed),
    nn.Linear(dsed, dsed * 4),
    nn.Mish(),
    nn.Linear(dsed * 4, dsed),
)

# local_cond_encoder
local_cond_encoder = nn.ModuleList([
    # down encoder
    ConditionalResidualBlock1D
    # up encoder
    ConditionalResidualBlock1D
])

# Mid Modules
mid_modules = nn.ModuleList([
    ConditionalResidualBlock1D,
    ConditionalResidualBlock1D
])

# Down Modules
down_modules = nn.ModuleList([])
for ind, (dim_in, dim_out) in enumerate(in_out):
    down_modules.append(nn.ModuleList([
        ConditionalResidualBlock1D,
        ConditionalResidualBlock1D,
        Downsample1d(dim_out) if not is_last else nn.Identity()
    ]))

# Up Modules
up_modules = nn.ModuleList([])
for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
    up_modules.append(nn.ModuleList([
        ConditionalResidualBlock1D,
        ConditionalResidualBlock1D,
        Upsample1d(dim_in) if not is_last else nn.Identity()
    ]))
```

![Alt text](assets/image.png)

**Transformer Module**
![Alt text](assets/image_1.png)
- [x] Get `push block` task data 

- [x] obs/actions data :question:

- [x] algorithm, data -> policy :dart: <font color='blue'>sample algorithm</font>

- [ ] diffusion policy / ACT 



![image-20230813100337416](assets/image-20230813100337416.png)