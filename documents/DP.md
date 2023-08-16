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

