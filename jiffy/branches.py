"""Causal prefix reuse with independent, unpadded question branches.

Pinned to Transformers 5.17's out-of-place DynamicCache updates. No model
weights or attention topology are changed. Never share mutable layer objects.
"""
import copy
import time
from collections import defaultdict
from types import SimpleNamespace

import torch
from transformers import DynamicCache

from .diffusion_decisions import answer_scores


def fork_cache(cache, batch_size):
    if not isinstance(cache, DynamicCache) or cache.is_compileable:
        raise ValueError("prefix branching requires an ordinary DynamicCache")
    fork = copy.copy(cache)
    fork.layers = [copy.copy(layer) for layer in cache.layers]
    for layer in fork.layers:
        # DynamicLayer.update concatenates into new tensors, leaving these views intact.
        layer.keys = layer.keys.expand(batch_size, -1, -1, -1)
        layer.values = layer.values.expand(batch_size, -1, -1, -1)
    return fork


@torch.inference_mode()
def shared_document(backend, state, contracts, *, seed=20260921, batch_size=4):
    contracts = list(contracts)
    if not contracts or type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError("provide contracts and a batch size between 1 and 8")
    if any(len(c.questions) != 1 for c in contracts):
        raise ValueError("each isolated branch must contain exactly one question")
    model, device = backend.model, backend.device
    if model.config.text_config.use_bidirectional_attention not in ("vision", False):
        raise ValueError("shared document cache requires causal text attention")
    if model.config._attn_implementation != "sdpa":
        raise ValueError("shared document path is validated only with SDPA")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    inputs, ending = backend.tokenize(state, SimpleNamespace(prompt=""), split_document=True)
    inputs = inputs.to(device)
    prefix_length = inputs["input_ids"].shape[1]
    suffixes = [backend.processor.tokenizer.encode(c.prompt + ending, add_special_tokens=False) for c in contracts]
    canvas_length = model.config.canvas_length
    if any(prefix_length + len(s) + canvas_length > backend.max_length for s in suffixes):
        raise ValueError("document plus question plus canvas exceeds context limit; no truncation")
    config = copy.deepcopy(model.generation_config)
    config.cache_implementation = "dynamic"
    cache = model._prepare_cache_for_generation(config, 1, backend.max_length)
    attention = inputs["attention_mask"].bool()
    positions = torch.arange(prefix_length, dtype=torch.int32, device=device)[None]
    extras = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    ids, masks = model._prepare_encoder_inputs(inputs["input_ids"], attention, positions, cache,
                                               True, canvas_length, 1, **extras)
    cache = model.model.encoder(input_ids=ids, attention_mask=masks, position_ids=positions,
                                past_key_values=cache, **extras).past_key_values
    torch.cuda.synchronize(device)
    document_ms = (time.perf_counter() - started) * 1000
    # Exact-length buckets avoid padding changing sliding-cache eviction or RoPE positions.
    buckets = defaultdict(list)
    for index, suffix in enumerate(suffixes):
        buckets[len(suffix)].append(index)
    results = [None] * len(contracts)
    batches = 0
    sampler = model._prepare_sampler(config)
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(seed)
        random_canvas = sampler.initialize_canvas(1, device)
    for length, indices in buckets.items():
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset:offset + batch_size]
            size = len(selected)
            branch = fork_cache(cache, size)
            ids = torch.tensor([suffixes[i] for i in selected], device=device)
            positions = torch.arange(prefix_length, prefix_length + length, dtype=torch.int32, device=device)[None].expand(size, -1)
            attention = torch.ones(size, prefix_length + length, dtype=torch.bool, device=device)
            ids, masks = model._prepare_encoder_inputs(ids, attention, positions, branch, True, canvas_length, size)
            branch = model.model.encoder(input_ids=ids, attention_mask=masks, position_ids=positions,
                                         past_key_values=branch).past_key_values
            canvas = torch.stack([contracts[i].scaffold for i in selected]).to(device)
            for row, index in enumerate(selected):
                canvas[row, list(contracts[index].positions)] = random_canvas[0, list(contracts[index].positions)]
            positions = torch.arange(prefix_length + length, prefix_length + length + canvas_length,
                                     dtype=torch.int32, device=device)[None].expand(size, -1)
            hidden = model.model.decoder(decoder_input_ids=canvas, past_key_values=branch,
                decoder_attention_mask={"full_attention": None, "sliding_attention": None},
                decoder_position_ids=positions).last_hidden_state
            readouts = torch.stack([hidden[row, contracts[index].positions[0]] for row, index in enumerate(selected)])
            logits = model.lm_head(readouts).float()
            cap = model.final_logit_softcapping
            logits = (logits / cap).tanh() * cap
            for row, index in enumerate(selected):
                contract = contracts[index]
                entry = answer_scores(logits[row:row + 1, None, :], contract)[0]
                results[index] = {"answers": {contract.names[0]: contract.questions[0].answer(entry["probabilities"])},
                                  "diagnostics": {"prefix_tokens": prefix_length + length,
                                                  "allowed_mass": entry["allowed_mass"]}}
            batches += 1
            del branch, hidden, logits
    torch.cuda.synchronize(device)
    return {"results": results, "diagnostics": {"document_tokens": prefix_length,
            "input_tokens": prefix_length + sum(map(len, suffixes)), "document_prefills": 1,
            "branch_batches": batches, "branches": len(contracts), "document_ms": document_ms,
            "total_ms": (time.perf_counter() - started) * 1000}}
