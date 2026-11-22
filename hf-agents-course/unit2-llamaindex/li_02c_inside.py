"""UNIT 2.2 · SIDE-QUEST — WHERE DO THE 384 NUMBERS COME FROM?

Not in the course. Opens up the embedding model and shows every stage between
"a string" and "384 floats". Everything printed here is read off the model on
your disk - nothing is asserted from memory.
"""
import torch
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("BAAI/bge-small-en-v1.5")
bert = m[0].auto_model
cfg = bert.config

print("=" * 78)
print("THE PIPELINE  (what sentence-transformers actually runs, in order)")
print("=" * 78)
print(m)

print("\n" + "=" * 78)
print("THE NETWORK")
print("=" * 78)
print(f"  architecture        {cfg.architectures}")
print(f"  vocabulary          {cfg.vocab_size:,} subword tokens")
print(f"  hidden size         {cfg.hidden_size}      <- this is where 384 comes from")
print(f"  transformer layers  {cfg.num_hidden_layers}")
print(f"  attention heads     {cfg.num_attention_heads}")
print(f"  max input length    {cfg.max_position_embeddings} tokens")
print(f"  parameters          {sum(p.numel() for p in m.parameters()):,}")

TEXT = "Main: wild mushroom wellington (vegetarian), served with braised leeks"
print("\n" + "=" * 78)
print("STAGE 1 — TOKENISE:  string  ->  integers")
print("=" * 78)
toks = m.tokenizer.tokenize(TEXT)
ids  = m.tokenizer.convert_tokens_to_ids(toks)
print(f"  input: {TEXT!r}\n")
print(f"  {len(toks)} subword tokens (note how rare words get split):")
for t, i in list(zip(toks, ids)):
    print(f"      {t:<18} -> id {i}")
print("\n  Plus [CLS] at the front and [SEP] at the end, added automatically.")

print("\n" + "=" * 78)
print("STAGE 2+3 — LOOKUP, then 12 LAYERS OF SELF-ATTENTION")
print("=" * 78)
# sentence-transformers moved the model to the Apple GPU ("mps"). Tensors must
# live on the same device as the weights, or torch raises
# "Passed CPU tensor to MPS op". This is the fix.
device = next(bert.parameters()).device
print(f"  (model is on device: {device})\n")
enc = m.tokenizer(TEXT, return_tensors="pt")
enc = {k: v.to(device) for k, v in enc.items()}
with torch.no_grad():
    out = bert(**enc, output_hidden_states=True)
hs = out.hidden_states
print(f"  hidden_states: {len(hs)} tensors (1 embedding lookup + {cfg.num_hidden_layers} layers)")
print(f"  each of shape {tuple(hs[0].shape)}  =  (batch, tokens, {cfg.hidden_size})\n")
print("  EVERY TOKEN keeps its own 384-vector all the way through. The layers")
print("  rewrite each token's vector using every other token in the sentence -")
print("  which is how 'wellington' next to 'mushroom' ends up somewhere different")
print("  from 'wellington' next to 'boot'. Watch one token move:\n")
pos = min(3, hs[0].shape[1] - 1)
tok = m.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])[pos]
print(f"      token {pos} = {tok!r}, first 4 of its 384 numbers at each depth")
for layer in (0, 1, 6, cfg.num_hidden_layers):
    v = hs[layer][0, pos, :4].tolist()
    label = "lookup table" if layer == 0 else f"after layer {layer:2}"
    print(f"      {label:>16}: {[round(x, 3) for x in v]}")

print("\n" + "=" * 78)
print("STAGE 4 — POOL:  many token-vectors  ->  ONE sentence-vector")
print("=" * 78)
try:
    print(f"  pooling module config: {m[1].get_config_dict()}")
except Exception:
    print(f"  pooling module: {m[1]}")
print("\n  BGE uses CLS pooling: it throws away every token vector except the")
print("  first one ([CLS]), which the 12 layers have been quietly filling with a")
print("  summary of the whole sentence. That single 384-vector IS the embedding.")
cls_vec = hs[-1][0, 0, :]
print(f"\n  [CLS] after the last layer, first 4: {[round(x,4) for x in cls_vec[:4].tolist()]}")

print("\n" + "=" * 78)
print("STAGE 5 — NORMALISE to length 1")
print("=" * 78)
final = m.encode(TEXT)
print(f"  final embedding, first 4: {[round(float(x), 4) for x in final[:4]]}")
print(f"  length (L2 norm)        : {float((final ** 2).sum() ** 0.5):.6f}")
print("\n  Because every vector has length 1, cosine similarity is just the dot")
print("  product, and squared euclidean distance is 2 - 2*cosine. Same geometry,")
print("  two ways of writing it. This matters in li_03b.")

print("\n" + "=" * 78)
print("AND THE PART THAT ACTUALLY MAKES IT WORK: HOW IT WAS TRAINED")
print("=" * 78)
print("""  Nothing above explains why 'what did we eat' lands near a menu. The
  architecture only guarantees SOME 384 numbers come out. The arrangement
  comes from the training objective, which was CONTRASTIVE:

      take a (question, passage-that-answers-it) pair   -> pull vectors TOGETHER
      take that question and an unrelated passage       -> push vectors APART

  Repeat over hundreds of millions of pairs. The loss is defined directly on
  the distances, so "similar meaning is nearby" is not emergent or mysterious -
  it is literally the thing that was optimised. That is the whole idea.

  Consequence you can test: BGE was trained with an instruction prefix on the
  QUERY side only. LlamaIndex adds it for you in get_query_embedding():
      "Represent this sentence for searching relevant passages: "
  Queries and documents are therefore embedded slightly differently on purpose.""")
