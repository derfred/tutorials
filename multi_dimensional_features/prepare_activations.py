import glob, math, os.path, sys
import numpy as np
import mlx.core as mx
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.getcwd(), "..", "support"))
import gemma2

class SAEHook:
  @staticmethod
  def apply(model, layer, batch):
    hook = SAEHook(layer)
    try:
      model(mx.array(batch), hook=hook)
    except StopIteration:
      pass
    return np.array(hook.acts)

  def __init__(self, layer):
    self.layer = layer
    self.acts  = None

  def __call__(self, array, label=None):
    if label == f"blocks.{self.layer}.hook_resid_post":
      self.acts = array
      raise StopIteration()
    return array

def llm_acts_file_name(args, suffix):
  return os.path.join(args.dir, f"llm_acts_layer_{args.layer}_split_{suffix}.npz")

def sae_acts_file_name(args, suffix):
  return os.path.join(args.dir, f"sae_acts_layer_{args.layer}_width_{args.sae_width}_l0_{args.sae_sparsity}_split_{suffix}.npz")

def load_sae(args):
  return gemma2.JumpReLUSAE.load(layer=args.layer, width=args.sae_width, l0=args.sae_sparsity)

def find_start_point(args, layer):
  result = 0
  for fname in glob.glob(sae_acts_file_name(args, "*")):
    acts    = np.load(fname)["sae_acts"]
    result += len(acts)
  return result

def residuals_from_dataset(model, tokenizer, layer, to_skip, batch_size=16, context_length=256, dataset_name="monology/pile-uncopyrighted", split="train", streaming=True):
  from datasets import load_dataset
  dataset   = load_dataset(dataset_name, split=split, streaming=streaming)
  batch     = []
  generated = 0
  for x in dataset:
    if generated < to_skip:
      generated += len(x["text"])
    else:
      tokens = tokenizer.encode(x["text"])
      if len(tokens) > context_length:
        batch.append(tokens[:context_length])
        if len(batch) == batch_size:
          yield SAEHook.apply(model, layer, np.array(batch)), np.array(batch)
          batch = []
  if len(batch) > 0:
    yield SAEHook.apply(model, layer, np.array(batch)), np.array(batch)

def extract_llm_activations(args):
  model, tokenizer = gemma2.load(args.model)
  sae              = load_sae(args)

  llm_acts_generated = 0
  llm_acts_split     = 0
  llm_acts           = np.zeros((args.split_size, args.context_length, model.hidden_size), dtype=np.float16)
  llm_tokens         = np.zeros((args.split_size, args.context_length), dtype=int)

  existing = find_start_point(args, args.layer)
  pbar     = tqdm(total=args.prompts_to_save, position=existing)
  for batch, tokens in residuals_from_dataset(model, tokenizer, args.layer, existing, batch_size=args.batch_size, context_length=args.context_length, dataset_name=args.dataset_name, split=args.split, streaming=args.streaming):
    offset_in_epoch     = llm_acts_generated % args.split_size
    end_offset_in_epoch = min(offset_in_epoch + len(batch), args.split_size)

    llm_acts[offset_in_epoch:end_offset_in_epoch]   = batch
    llm_tokens[offset_in_epoch:end_offset_in_epoch] = tokens

    if end_offset_in_epoch == args.split_size:
      if args.store_residuals:
        np.savez(llm_acts_file_name(args, "llm_acts", args.layer, str(llm_acts_split).zfill(4)), llm_acts=llm_acts, llm_tokens=llm_tokens)

      sae_acts = np.zeros((llm_acts.shape[0], args.context_length, args.features_to_save), dtype=np.float16)
      sae_idxs = np.zeros((llm_acts.shape[0], args.context_length, args.features_to_save), dtype=np.int16)
      for i, acts_batch in enumerate(llm_acts):
        sae_act     = sae.encode(mx.array(acts_batch))
        sae_idxs[i] = np.argsort(sae_act, axis=1)[:, -args.features_to_save:]
        sae_acts[i] = np.sort(sae_act, axis=1)[:, -args.features_to_save:]
      np.savez(sae_acts_file_name(args, str(llm_acts_split).zfill(4)), sae_acts=sae_acts, sae_idxs=sae_idxs, llm_tokens=llm_tokens)

      llm_acts_split += 1

      if llm_acts_generated > args.prompts_to_save:
        break

    llm_acts_generated += len(batch)
    pbar.set_description(f"Current split = {llm_acts_split}({(100*offset_in_epoch/float(args.split_size)):.1f}%)/{math.ceil(args.prompts_to_save/args.split_size)}")
    pbar.update(len(batch))

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("dir",                type=str,  help="Directory to save activations in")
  parser.add_argument("--model",            type=str,  default="mlx-community/gemma-2-9b-8bit")
  parser.add_argument("--layer",            type=int,  default=20)
  parser.add_argument("--prompts_to_save",  type=int,  default=10**6)
  parser.add_argument("--features_to_save", type=int,  default=64, help="For each token only keep the top_k features")
  parser.add_argument("--store_residuals",  type=bool, default=False)
  parser.add_argument("--split_size",       type=int,  default=2**12)
  parser.add_argument("--batch_size",       type=int,  default=16)
  parser.add_argument("--context_length",   type=int,  default=256)
  parser.add_argument("--sae_width",        type=str,  default="16k")
  parser.add_argument("--sae_sparsity",     type=int,  default=58)
  parser.add_argument("--dataset_name",     type=str,  default="monology/pile-uncopyrighted")
  parser.add_argument("--split",            type=str,  default="train")
  parser.add_argument("--streaming",        type=bool, default=True)
  args = parser.parse_args()

  extract_llm_activations(args)
