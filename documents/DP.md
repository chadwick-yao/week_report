# DDPM

## Data

### Square

:arrow_forward: **observation**

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

```
desired translation of EEF(3), desired delta rotation from current EEF(3), and opening and closing of the gripper fingers:
	shape: [7]
```

### Can

:arrow_forward: **observation**

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

```
desired translation of EEF(3), desired delta rotation from current EEF(3), and opening and closing of the gripper fingers:
	shape: [7]
```

## Training

### Unet

![unet architecture](./assets/image.png)

The training main backbone can be seen above, the `low_dim` type data will pass through Downsample Module, then Mid Module, and finally Upsample Module. With respect to `image`, it's processed by `obs_encoder` from robomimic package. 

**Input**

```python
# X/sampler: low_dim input, like eef position, eef quaternion, etc.
# timestep: sample from the discrete timesteps used for the diffusion chain
# global cond & local cond: 
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

Encoder & Decoder

```python
""" Encoder """
# Diffusion Step Encoder for Timestep
diffusion_step_encoder = nn.Sequential(
    SinusoidalPosEmb(dsed),
    nn.Linear(dsed, dsed * 4),
    nn.Mish(),
    nn.Linear(dsed * 4, dsed),
)
# local_cond_encoder for local cond
local_cond_encoder = nn.ModuleList([
    # down encoder
    ConditionalResidualBlock1D
    # up encoder
    ConditionalResidualBlock1D
])

# x/sample encoder
down_modules = nn.ModuleList([])
for ind, (dim_in, dim_out) in enumerate(in_out):
    down_modules.append(nn.ModuleList([
        ConditionalResidualBlock1D,
        ConditionalResidualBlock1D,
        Downsample1d(dim_out) if not is_last else nn.Identity()
    ]))
""" Image Encoder """
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
policy: PolicyAlgo = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=action_dim,
        device='cpu',
    )
obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
print(policy)
""" Decoder """
# x/sample decoder
up_modules = nn.ModuleList([])
for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
    up_modules.append(nn.ModuleList([
        ConditionalResidualBlock1D,
        ConditionalResidualBlock1D,
        Upsample1d(dim_in) if not is_last else nn.Identity()
    ]))
```

**Loss Function**

It uses a DDPM to approximate the conditional distribution $p(A_t|O_t)$ for planning. This formulation allows the model to predict actions conditioned on observations without the cost of inferring future states, speeding up the diffusion process and improving the accurary of generated antions. To capture the conditional distribution,it has:
$$
A_t^{k-1}=\alpha(A_t^k-\gamma \epsilon_{\theta}(O_t, A_t^k), k) + N(0, \sigma^2I)
$$
The traning loss is below:
$$
Loss = MSE(\epsilon^k, \epsilon_{\theta}(O_t, A_t^0+\epsilon^k,k))
$$


```python
    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
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
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

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

