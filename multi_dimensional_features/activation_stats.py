import concurrent.futures, pickle, glob
import numpy as np
from tqdm import tqdm

def save_feature_token_activations(num_features, vocab_size, min_activations, inpaths, outfile):
  def for_split(token_sae_acts, token_sae_idxs, llm_tokens, feature_idx):
    idxs, ranks = np.where((token_sae_acts > 0) & (token_sae_idxs == feature_idx))
    activations = token_sae_acts[idxs, ranks].astype(np.float32)

    tokens, inv_idxs = np.unique(llm_tokens[idxs], return_inverse=True)
    token_acts       = [activations[inv_idxs == i] for i in range(len(tokens))]
    token_act_counts = np.array([len(acts) for acts in token_acts])
    token_act_means  = np.array([np.mean(acts, axis=0) for acts in token_acts])
    token_act_m2s    = np.array([np.sum((acts - mean)**2, axis=0) for acts, mean in zip(token_acts, token_act_means)])

    return feature_idx, tokens, token_act_counts, token_act_means, token_act_m2s

  feat_counts = np.zeros((num_features, vocab_size), dtype=np.int32)
  feat_means  = np.zeros((num_features, vocab_size), dtype=np.float32)
  feat_m2s    = np.zeros((num_features, vocab_size), dtype=np.float32)

  for fname in tqdm(list(glob.glob(inpaths))):
    split          = np.load(fname)
    token_sae_acts = split["sae_acts"].reshape(-1, split["sae_idxs"].shape[-1])
    token_sae_idxs = split["sae_idxs"].reshape(-1, split["sae_idxs"].shape[-1])
    llm_tokens     = split["llm_tokens"].reshape(-1)

    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [executor.submit(for_split, token_sae_acts, token_sae_idxs, llm_tokens, i) for i in range(num_features)]
      for future in concurrent.futures.as_completed(futures):
        feat_idx, tokens, act_counts, act_means, act_m2s = future.result()

        new_count = feat_counts[feat_idx, tokens] + act_counts
        delta     = act_means - feat_means[feat_idx, tokens]

        feat_means[feat_idx, tokens] += delta * (act_counts / new_count)
        feat_m2s[feat_idx, tokens]   += act_m2s + delta**2 * (feat_counts[feat_idx, tokens] * act_counts / new_count)
        feat_counts[feat_idx, tokens] = new_count

  tok_idxs = [np.where(feat_counts[i] > min_activations)[0] for i in range(feat_counts.shape[0])]
  tok_cnts = [feat_counts[i, idxs] for i, idxs in enumerate(tok_idxs)]
  tok_mns  = [feat_means[i, idxs] for i, idxs in enumerate(tok_idxs)]
  tok_m2s  = [feat_m2s[i, idxs] for i, idxs in enumerate(tok_idxs)]

  with open(outfile, "wb") as f:
    pickle.dump((tok_idxs, tok_cnts, tok_mns, tok_m2s), f)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("infiles",           type=str, help="Glob pattern for the SAE activations")
  parser.add_argument("outfile",           type=str, help="Output file")
  parser.add_argument("--num_features",    type=int, default=2**14,  help="Number of features, this is equal to the width of the SAE")
  parser.add_argument("--vocab_size",      type=int, default=256128, help="Vocabulary size")
  parser.add_argument("--min_activations", type=int, default=10, help="Minimum number of activations to consider a token")
  args = parser.parse_args()

  save_feature_token_activations(args.num_features, args.vocab_size, args.min_activations, args.infiles, args.outfile)
