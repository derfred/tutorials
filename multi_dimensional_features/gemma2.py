from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.gemma2 import ModelArgs, TransformerBlock, RMSNorm

def build_hooks(hooks):
  if hooks is None:
    return lambda x, l=None: x
  return hooks

@staticmethod
def load(model_path: str = "mlx-community/gemma-2-9b-8bit"):
  import glob
  from mlx_lm.utils import get_model_path, load_config, load_tokenizer

  model_path = get_model_path(model_path)
  config     = load_config(model_path)

  weight_files = glob.glob(str(model_path / "model*.safetensors"))
  weights = {}
  for wf in weight_files:
    weights.update(mx.load(wf))

  model   = Model(ModelArgs.from_dict(config))
  weights = model.sanitize(weights)

  if (quantization := config.get("quantization", None)) is not None:
    nn.quantize(model, **quantization)

  model.load_weights(list(weights.items()))
  mx.eval(model.parameters())
  model.eval()

  return model, load_tokenizer(model_path)

class Model(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args              = args
    self.vocab_size        = args.vocab_size
    self.num_hidden_layers = args.num_hidden_layers
    assert self.vocab_size > 0
    self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
    self.layers       = [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
    self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

  @property
  def head_dim(self):
    return self.args.head_dim

  @property
  def n_kv_heads(self):
    return self.args.num_key_value_heads

  def sanitize(self, weights):
    result = {}
    for k, v in weights.items():
      if k.startswith("model."):
        result[k[6:]] = v
      else:
        result[k] = v
    return result

  def __call__(self, inputs: mx.array, cache=None, hooks=None):
    hooks = build_hooks(hooks)
    if cache is None:
      cache = [None] * len(self.layers)

    h    = self.embed_tokens(inputs)
    h    = h * (self.args.hidden_size**0.5)
    mask = create_attention_mask(h, cache)
    h    = hooks(h, "hook_embed")

    for i, (layer, c) in enumerate(zip(self.layers, cache)):
      h = hooks(h, f"blocks.{i}.hook_resid_pre")

      r = layer.self_attn(layer.input_layernorm(h.astype(mx.float32)), mask, c)
      r = layer.post_attention_layernorm(r)
      h = h + hooks(r, f"blocks.{i}.hook_attn_out")

      h = hooks(h, f"blocks.{i}.hook_resid_mid")

      r = layer.mlp(layer.pre_feedforward_layernorm(h).astype(mx.float16)).astype(
          mx.float32
      )
      r = layer.post_feedforward_layernorm(r)
      h = h + hooks(r, f"blocks.{i}.hook_mlp_out")

      h = hooks(h, f"blocks.{i}.hook_resid_post")

    out = hooks(self.norm(h), "ln_final.hook_normalized")
    out = self.embed_tokens.as_linear(out)
    out = mx.tanh(out / self.args.final_logit_softcapping)
    out = out * self.args.final_logit_softcapping

    return hooks(out)
