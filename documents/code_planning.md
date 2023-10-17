# ACT

## CVAE Encoder

It will concatenate embedded CLS (bs, 1, dim), embedded joint positions (bs, 1 , dim), and action sequence (bs, horizon, dim) which has time history information.

Here, in each transformer encoder layer, it will do one time position embedding, which is a big difference from pytorch package. (Here position embedding is not learnable)

## Visual Encoder (Resnet 18)

It selects the output of {'layer4': "0"} (shape: 3, h, w -> shape: dim, h’, w’) to represent images features, whose shape is (bs. dim, cam_num * h’ * w’). And its position information is based on the relative position of (h’, w’). (not learnable)

Secondly, it changes a part of resnet18, which uses FrozenBatchNorm2d to replace all NormLayers.

## CVAE Decoder transformer

**Encoder**

In transformer encoder, it concatenates images features (bs, cam_num * h * 2, dim), joints positions (bs, 1, dim) and latent code (bs, 1, dim). And do position embedding in each encoder layer. (Here one part of position embedding is from Visual Encoder, another is nn.Embedding.weights).

**Decoder**

The input (namely queries) are learnable parameters. In every decoder layer, it would not just embed during self-attention, but also embed in multi-head attention.

```python
q = k = self.with_pos_embed(tgt, query_pos)
tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                      key_padding_mask=tgt_key_padding_mask)[0]
tgt = tgt + self.dropout1(tgt2)
tgt = self.norm1(tgt)
tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                           key=self.with_pos_embed(memory, pos),
                           value=memory, attn_mask=memory_mask,
                           key_padding_mask=memory_key_padding_mask)[0]
```

# DP

## Visual Encoder

The visual encoder here is from Robomimic, which is well designed for Robomimic tasks. Due to receding horizon policy, the observation input here has history information (bs, To, c, h, w). And finally it gets observation features (bs, To, 64 * cam_num + low_dim). 

## Transformer Encoder

Added `timestep`, before feeding to transformer encoder, it will do one time position embedding, which is learnable parameters. 

## Transformer Decoder

The input is noised action sequence, not learnable. 
