# DDPM

## Data

### Square

:arrow_forward: **observation**

在square任务中，它的obs包括一个全局上帝视角的图片，一个wrist视角的图片，以及终端执行器的pos和quat，最后还有用两个维度描诉gripper的关节。

```tex
agentview_image:
  shape: [3, 84, 84]
  type: rgb
robot0_eye_in_hand_image:
  shape: [3, 84, 84]
  type: rgb
robot0_eef_pos:
  shape: [3]
  # type default: low_dim
robot0_eef_quat:
  shape: [4]
robot0_gripper_qpos:
  shape: [2]
```

:arrow_forward: **action**

action前三维度指的是EEF位置的变化，接着三维是角度的变化，最后一维是gripper的开合状态。

```
desired translation of EEF(3), desired delta rotation from current EEF(3), and opening and closing of the gripper fingers:
	shape: [7]
```

### Can

:arrow_forward: **observation**

和square任务是一样的

```
agentview_image:
  shape: [3, 84, 84]
  type: rgb
robot0_eye_in_hand_image:
  shape: [3, 84, 84]
  type: rgb
robot0_eef_pos:
  shape: [3]
  # type default: low_dim
robot0_eef_quat:
  shape: [4]
robot0_gripper_qpos:
  shape: [2]
```

:arrow_forward: **action**

和square任务是一样的

```
desired translation of EEF(3), desired delta rotation from current EEF(3), and opening and closing of the gripper fingers:
	shape: [7]
```

## Training

### Transformer

<img src="assets/image_1.png" alt="transformer architecture" style="zoom: 33%;" />

**Inputs**

Here inputs are quite crucial, `X/sample` is a sequence of noised action (its dimension can be [bs, horizon, action_dim], [B, n_action_step, action_dim] or [bs, horizon, action_dim + obs_feature_dim]). `timesteps` is the number of diffusion steps, for instance, if `timesteps=10` then it has 10 steps from the original sample. 

The default value of `cond` is `None`, but when `obs_as_cond` is set `True`, which means the model would take observation as a condition, and the detailed procedures are below. 

**TESTING:** Because we are using To steps of observations to do prediction, so first it obtain `this_nobs` from the firt To of `nobs`. Then, `this_nobs` will be passed through `obs_encoder` to get its features, named `cond`. Conversely, if `obs_as_cond` is `False`, it will do condition through impainting. 

**TRAINING: **the same.

```python
""" TESTING """
if self.obs_as_cond:
    this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
    nobs_features = self.obs_encoder(this_nobs)
    # reshape back to B, To, Do
    cond = nobs_features.reshape(B, To, -1)
    shape = (B, T, Da)
    if self.pred_action_steps_only:
        shape = (B, self.n_action_steps, Da)
    cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
else:
    # condition through impainting
    this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
    nobs_features = self.obs_encoder(this_nobs)
    # reshape back to B, T, Do
    nobs_features = nobs_features.reshape(B, T, -1)
    shape = (B, T, Da+Do)
    cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    cond_data[:,:To,Da:] = nobs_features
    cond_mask[:,:To,Da:] = True
""" TRAINING """
if self.obs_as_cond:
    # reshape B, T, ... to B*T
    this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
    nobs_features = self.obs_encoder(this_nobs)
    # reshape back to B, T, Do
    cond = nobs_features.reshape(batch_size, To, -1)
    if self.pred_action_steps_only:
        start = To - 1
        end = start + self.n_action_steps
        trajectory = nactions[:,start:end]
else:
    # reshape B, T, ... to B*T
    this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
    nobs_features = self.obs_encoder(this_nobs)
    # reshape back to B, T, Do
    nobs_features = nobs_features.reshape(batch_size, horizon, -1)
    trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

# generate impainting mask
if self.pred_action_steps_only:
    condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
else:
    condition_mask = self.mask_generator(trajectory.shape)
```

Encoder & Decoder

Encoder is designed to encode `cond` and `timesteps`. `n_cond_layers` can be set in configuration files, and if it’s > 0, transformer encoder will replace MLP encoder. Both transformer encoder and decoder are using torch.nn module.

```python
self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
if n_cond_layers > 0:
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=n_emb,
        nhead=n_head,
        dim_feedforward=4*n_emb,
        dropout=p_drop_attn,
        activation='gelu',
        batch_first=True,
        norm_first=True
    )
    self.encoder = nn.TransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=n_cond_layers
    )
else:
    self.encoder = nn.Sequential(
        nn.Linear(n_emb, 4 * n_emb),
        nn.Mish(),
        nn.Linear(4 * n_emb, n_emb)
    )
# decoder
decoder_layer = nn.TransformerDecoderLayer(
    d_model=n_emb,
    nhead=n_head,
    dim_feedforward=4*n_emb,
    dropout=p_drop_attn,
    activation='gelu',
    batch_first=True,
    norm_first=True # important for stability
)
self.decoder = nn.TransformerDecoder(
    decoder_layer=decoder_layer,
    num_layers=n_layer
)
```

**Loss Function**

It uses a DDPM to approximate the conditional distribution $p(A_t|O_t)$ for planning. This formulation allows the model to predict actions conditioned on observations without the cost of inferring future states, speeding up the diffusion process and improving the accurary of generated antions. To capture the conditional distribution,it has:

$A_t^{k-1}=\alpha(A_t^k-\gamma \epsilon_{\theta}(O_t, A_t^k), k) + N(0, \sigma^2I)$

The traning loss is below:

$Loss = MSE(\epsilon^k, \epsilon_{\theta}(O_t, A_t^0+\epsilon^k,k))$

> `compute_loss` is a method of policy, it is only used in training. However, when testing or doing rollout, it uses `predict_action`, another method of policy. With respect to compute_loss, now we have a original trajectory, first, we produce noise by torch.randn, whose shape is the same as the original trajectory. As talked in transformer model above, except for X/sample (here is trajectory) and `cond`, we also need `timesteps`, here it uses torch.randint. 
>
> When we have noise, and all the required data, then it uses `noise_scheduler` (DDPM algorithm) to add noise in the original trajectory. And this process can be regarded as the forward, therefore, next step is the backward to predict noise with Diffusion Model (DM is actually a noise predictor.). Finally, we use MSE to calculate the loss. 

```python
# this is how the model to calculate loss, and the loss function it uses is MSE_LOSS.
def compute_loss(self, batch):
    # normalize input
    assert 'valid_mask' not in batch
    nobs = self.normalizer.normalize(batch['obs'])
    nactions = self.normalizer['action'].normalize(batch['action'])
    batch_size = nactions.shape[0]
    horizon = nactions.shape[1]
    To = self.n_obs_steps

    # handle different ways of passing observation
    cond = None
    trajectory = nactions
    if self.obs_as_cond:
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, T, Do
        cond = nobs_features.reshape(batch_size, To, -1)
        if self.pred_action_steps_only:
            start = To - 1
            end = start + self.n_action_steps
            trajectory = nactions[:,start:end]
    else:
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, T, Do
        nobs_features = nobs_features.reshape(batch_size, horizon, -1)
        trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

    # generate impainting mask
    if self.pred_action_steps_only:
        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
    else:
        condition_mask = self.mask_generator(trajectory.shape)
""" NOTE: the codes above has been introduced """
    # Sample noise that we'll add to the images
    noise = torch.randn(trajectory.shape, device=trajectory.device)
    bsz = trajectory.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(
        0, self.noise_scheduler.config.num_train_timesteps, 
        (bsz,), device=trajectory.device
    ).long()
    # Add noise to the clean images according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_trajectory = self.noise_scheduler.add_noise(
        trajectory, noise, timesteps)

    # compute loss mask
    loss_mask = ~condition_mask

    # apply conditioning
    noisy_trajectory[condition_mask] = trajectory[condition_mask]

    # Predict the noise residual
    pred = self.model(noisy_trajectory, timesteps, cond)

    pred_type = self.noise_scheduler.config.prediction_type 
    if pred_type == 'epsilon':
        target = noise
    elif pred_type == 'sample':
        target = trajectory
    else:
        raise ValueError(f"Unsupported prediction type {pred_type}")

    loss = F.mse_loss(pred, target, reduction='none')
    loss = loss * loss_mask.type(loss.dtype)
    loss = reduce(loss, 'b ... -> b (...)', 'mean')
    loss = loss.mean()
    return loss
```





**HyperParameters**

下面是policy使用的DDPM算法的超参数：

| 参数名称      | 含义                      | 数值              |
| ------------- | ------------------------- | ----------------- |
| beta_start    | inference过程中beta起始值 | 0.0001            |
| beta_end      | inference过程中beta最终值 | 0.02              |
| beta_schedule | 映射策略                  | squaredcos_cap_v2 |

policy的task配置

| 参数名称       | 含义                | 数值 |
| -------------- | ------------------- | ---- |
| horizon        | 预测action的step数  | 10   |
| n_action_steps | 执行action的step数  | 8    |
| n_obs_steps    | 预测依赖obs的step数 | 2    |

policy的image超参数

| 参数名称   | 含义                 | 数值 |
| ---------- | -------------------- | ---- |
| crop_shape | 图片经过裁剪后的维度 | 10   |

policy使用的model超参数：

| 参数名称    | 含义                              | 数值 |
| ----------- | --------------------------------- | ---- |
| crop_shape  | decoder/encoder的层数             | 8    |
| n_head      | 多头注意力的头数                  | 4    |
| n_emb       | 嵌入层纬度                        | 256  |
| p_drop_emb  | 在encoder/decoder前的drop概率     | 0.0  |
| p_drop_attn | transformer layer的内部drop的概率 | 0.3  |

如果使用ema的超参数：

| 参数名称  | 含义                   | 数值   |
| --------- | ---------------------- | ------ |
| inv_gamma | EMA warmup的逆乘法因子 | 1.0    |
| power     | EMA warmup的指数因子   | 0.75   |
| min_value | 最小 EMA 衰减率        | 0.0    |
| max_value | 最大 EMA 衰减率        | 0.9999 |

dataloader的超参数：

| 参数名称    | 含义               | 数值 |
| ----------- | ------------------ | ---- |
| batch_size  | 批次大小           | 64   |
| num_workers | 读取数据时的进程数 | 8    |

optimizer的超参数：

| 参数名称                 | 含义                                   | 数值        |
| ------------------------ | -------------------------------------- | ----------- |
| transformer_weight_decay | trans权重衰减率                        | 1.0e-3      |
| obs_encoder_weight_decay | obs encoder权重衰减率                  | 1.0e-6      |
| learning_rate            | 学习率                                 | 1.0e-4      |
| betas                    | 用于计算梯度及其平方的运行平均值的系数 | [0.9, 0.95] |

```yaml
policy: # policy configuration
	_target_: DiffusionTransformerHybridImagePolicy # policy type
	
	shape_meta: # observations and actions specification
        obs:
            agentview_image:
                shape: [3, 84, 84]
                type: rgb
            robot0_eye_in_hand_image:
                shape: [3, 84, 84]
                type: rgb
            robot0_eef_pos:
                shape: [3]
                # type default: low_dim
            robot0_eef_quat:
                shape: [4]
            robot0_gripper_qpos:
                shape: [2]
        action: 
			shape: [7]
    
    noise_scheduler: # DDPM algorithm's hyperparameters
    	_target: DDPMScheduler	# algorithm type
    	num_train_timesteps: 100
    	beta_start: 0.0001
    	beta_end: 0.02
    	beta_schedule: squaredcos_cap_v2
        variance_type: fixed_small # Yilun's paper uses fixed_small_log instead, but easy to cause Nan
        clip_sample: True # required when predict_epsilon=False
        prediction_type: epsilon # or sample
    # task cfg
    horizon: 10 # dataset sequence length
    n_action_steps: 8	# number of steps of action will be executed
    n_obs_steps: 2 # the latest steps of observations data as input
    num_inference_steps: 100
    # image cfg
    crop_shape: [76, 76]	# images will be cropped into [76, 76]
    obs_encoder_group_norm: False,
    # arch
    n_layer: 8	# transformer decoder/encoder layer number
    n_cond_layers: 0  # >0: use transformer encoder for cond, otherwise use MLP
    n_head: 4	# head number
    n_emb: 256	# embedding dim (input dim --(emb)--> n_emb)
    p_drop_emb: 0.0	# dropout prob (before encoder&decoder)
    p_drop_attn: 0.3	# encoder_layer dropout prob
    causal_attn: True	# mask or not
    time_as_cond: True # if false, use BERT like encoder only arch, time as input
    obs_as_cond: True

# if ema is true
ema:
    _target_: diffusion_policy.model.diffusion.ema_model.EMAModel
    update_after_step: 0
    inv_gamma: 1.0
    power: 0.75
    min_value: 0.0
    max_value: 0.9999
dataloader:
    batch_size: 64
    num_workers: 8
    shuffle: True
    pin_memory: True
    persistent_workers: False

val_dataloader:
    batch_size: 64
    num_workers: 8
    shuffle: False
    pin_memory: True
    persistent_workers: False

optimizer:
    transformer_weight_decay: 1.0e-3
    obs_encoder_weight_decay: 1.0e-6
    learning_rate: 1.0e-4
    betas: [0.9, 0.95]

training:
    device: "cuda:0"
    seed: 42
    debug: False
    resume: True
    # optimization
    lr_scheduler: cosine
    # Transformer needs LR warmup
    lr_warmup_steps: 10
    num_epochs: 100
    gradient_accumulate_every: 1
    # EMA destroys performance when used with BatchNorm
    # replace BatchNorm with GroupNorm.
    use_ema: True
    # training loop control
    # in epochs
    rollout_every: 10
    checkpoint_every: 10
    val_every: 1
    sample_every: 5
    # steps per epoch
    max_train_steps: null
    max_val_steps: null
    # misc
    tqdm_interval_sec: 1.0
```
## Inference

When doing inference/testing/rollout, it will use predict_action function of policy. The difference between inference and training is that, it would not do backward to update parameters, secondly instead of just getting noise to compute loss, it uses `noise_scheduler.step` to acquire the original trajectory.

```python
# ========= inference  ============
def conditional_sample(self, 
        condition_data, condition_mask,
        cond=None, generator=None,
        # keyword arguments to scheduler.step
        **kwargs
        ):
    model = self.model
    scheduler = self.noise_scheduler

    trajectory = torch.randn(
        size=condition_data.shape, 
        dtype=condition_data.dtype,
        device=condition_data.device,
        generator=generator)

    # set step values
    scheduler.set_timesteps(self.num_inference_steps)

    for t in scheduler.timesteps:
        # 1. apply conditioning
        trajectory[condition_mask] = condition_data[condition_mask]

        # 2. predict model output
        model_output = model(trajectory, t, cond)

        # 3. compute previous image: x_t -> x_t-1
        trajectory = scheduler.step(
            model_output, t, trajectory, 
            generator=generator,
            **kwargs
            ).prev_sample

    # finally make sure conditioning is enforced
    trajectory[condition_mask] = condition_data[condition_mask]        

    return trajectory

def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    obs_dict: must include "obs" key
    result: must include "action" key
    """
	...
	...
    # run sampling
    nsample = self.conditional_sample(
        cond_data, 
        cond_mask,
        cond=cond,
        **self.kwargs)

    # unnormalize prediction
    naction_pred = nsample[...,:Da]
    action_pred = self.normalizer['action'].unnormalize(naction_pred)

    # get action
    if self.pred_action_steps_only:
        action = action_pred
    else:
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

    result = {
        'action': action,
        'action_pred': action_pred
    }
    return result
```

Here is the algorithm of `noise_scheduler.step`. (x_t -> x_t-1)

```python
# 1. compute alphas, betas
# 2. compute predicted original sample from predicted noise also called
# 3. Clip "predicted x_0"
# 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
# 5. Compute predicted previous sample µ_t
pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
# 6. Add noise
pred_prev_sample = pred_prev_sample + variance
```

The whole inference process is implemented in `env_runner.run`, which takes in a pre-trained policy.

```python
while not done:
    # create obs dict
    np_obs_dict = dict(obs)
    if self.past_action and (past_action is not None):
        # TODO: not tested
        np_obs_dict['past_action'] = past_action[
            :,-(self.n_obs_steps-1):].astype(np.float32)

    # device transfer
    obs_dict = dict_apply(np_obs_dict, 
        lambda x: torch.from_numpy(x).to(
            device=device))

    # run policy
    with torch.no_grad():
        action_dict = policy.predict_action(obs_dict)

    # device_transfer
    np_action_dict = dict_apply(action_dict,
        lambda x: x.detach().to('cpu').numpy())

    action = np_action_dict['action']
    if not np.all(np.isfinite(action)):
        print(action)
        raise RuntimeError("Nan or Inf action")

    # step env
    env_action = action
    if self.abs_action:
        env_action = self.undo_transform_action(action)

    obs, reward, done, info = env.step(env_action)
    done = np.all(done)
    past_action = action
```

