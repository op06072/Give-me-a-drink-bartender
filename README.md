# Give me a drink bartender

# Introducing
## What is `Give me a drink bartender`

A Python-based chatbot which use llama to recommend a whiskey.

# Usage
**Note:** Tool usage is listed by `python3 llama_whiskey.py -h`
```shell
usage: llama_whiskey.py [-h] [--num-tokens NUM_TOKENS] [--write-every WRITE_EVERY] [--temp TEMP] [--seed SEED] [--log-limit LOG_LIMIT] [--ko] model

Llama inference script

positional arguments:
  model                 Path to the model directory containing the MLX weights

options:
  -h, --help            show this help message and exit
  --num-tokens NUM_TOKENS, -n NUM_TOKENS
                        How many tokens to generate
  --write-every WRITE_EVERY
                        After how many tokens to detokenize
  --temp TEMP           The sampling temperature
  --seed SEED           The PRNG seed
  --log-limit LOG_LIMIT
                        The number of saved chat logs to use in LLM.
  --ko, -k              Option for custom llama model to use korean.
```
