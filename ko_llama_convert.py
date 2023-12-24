import argparse
import copy
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from llama_chatbot import Llama, ModelArgs, sanitize_config
from mlx.utils import tree_flatten, tree_map, tree_unflatten

def ko_llama(model_path):
    try:
        import transformers
    except ImportError as e:
        print("The transformers package must be installed for this model conversion:")
        print("pip install transformers")
        exit(0)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path)
    ).state_dict()
    config = transformers.AutoConfig.from_pretrained(model_path)

    # things to change
    # 1. there's no "model." in the weight names
    model = {k.replace("model.", ""): v for k, v in model.items()}

    # 2. mlp is called feed_forward
    model = {k.replace("mlp", "feed_forward"): v for k, v in model.items()}

    # 3. up_proj, down_proj, gate_proj
    model = {k.replace("down_proj", "w2"): v for k, v in model.items()}
    model = {k.replace("up_proj", "w3"): v for k, v in model.items()}
    model = {k.replace("gate_proj", "w1"): v for k, v in model.items()}

    # 4. layernorms
    model = {
        k.replace("input_layernorm", "attention_norm"): v for k, v in model.items()
    }
    model = {
        k.replace("post_attention_layernorm", "ffn_norm"): v for k, v in model.items()
    }

    # 5. lm head
    model = {k.replace("lm_head", "output"): v for k, v in model.items()}

    # 6. token emb
    model = {k.replace("embed_tokens", "tok_embeddings"): v for k, v in model.items()}

    # 7. attention
    model = {k.replace("self_attn", "attention"): v for k, v in model.items()}
    model = {k.replace("q_proj", "wq"): v for k, v in model.items()}
    model = {k.replace("k_proj", "wk"): v for k, v in model.items()}
    model = {k.replace("v_proj", "wv"): v for k, v in model.items()}
    model = {k.replace("o_proj", "wo"): v for k, v in model.items()}

    params = {}
    params["dim"] = config.hidden_size
    params["hidden_dim"] = config.intermediate_size
    params["n_heads"] = config.num_attention_heads
    if hasattr(config, "num_key_value_heads"):
        params["n_kv_heads"] = config.num_key_value_heads
    params["n_layers"] = config.num_hidden_layers
    params["vocab_size"] = config.vocab_size
    params["norm_eps"] = config.rms_norm_eps
    params["rope_traditional"] = False
    weights = {k: v.to(torch.float16).numpy() for k, v in model.items()}

    return weights, params


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    config = sanitize_config(config, weights)
    model = Llama(ModelArgs(**config))
    weights = tree_map(mx.array, weights)
    model.update(tree_unflatten(list(weights.items())))

    # Quantize the model:
    nn.QuantizedLinear.quantize_module(model, args.q_group_size, args.q_bits)

    # Update the config:
    quantized_config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits,
    }
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama-ko weights to MLX")
    parser.add_argument(
        "--torch-path",
        type=str,
        help="Path to the PyTorch model.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="Path to save the MLX model.",
    )
    parser.add_argument(
        "--model-name",
        help=(
            "Name of the model to convert. Use 'llama' for models in the "
            "Llama family distributed by Meta including Llama 1, Llama 2, "
            "Code Llama, and Llama chat."
        ),
        choices=["tiny_llama", "ko_llama"],
        default="ko_llama",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        help="Generate a quantized model.",
        action="store_true",
    )
    parser.add_argument(
        "--q_group_size",
        help="Group size for quantization.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--q_bits",
        help="Bits per weight for quantization.",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    torch_path = Path(args.torch_path)
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading")
    weights, params = globals()[args.model_name](torch_path)
    params["model_type"] = "llama"
    if args.quantize:
        print("[INFO] Quantizing")
        weights, params = quantize(weights, params, args)

    print("[INFO] Saving")
    mlx_path_str = str(mlx_path)
    if mlx_path_str + "/tokenizer.model" not in glob.glob(mlx_path_str+"/*"):
        shutil.copyfile(
            str(torch_path / "tokenizer.model"),
            str(mlx_path / "tokenizer.model"),
        )
    np.savez(str(mlx_path / "weights.npz"), **weights)
    with open(mlx_path / "config.json", "w") as fid:
        json.dump(params, fid, indent=4)
