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

**Input**

```python
# X/sampler: low_dim input, like eef position, eef quaternion, etc.
# timestep: sample from the discrete timesteps used for the diffusion chain
# cond: 
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
```

Encoder & Decoder

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

HyperParameters

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

## Inference

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
```

