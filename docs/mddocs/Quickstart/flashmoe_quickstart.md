# FlashMoE
The `FlashMoe` support in `ipex-llm` allows you to run ***DeepSeek V3/R1 671B*** and ***Qwen3Moe 235B*** models with just 1 or 2 Intel Arc GPU.

## Install
### Prerequisites
Check your GPU driver version, and update it if needed; we recommend following [Intel client GPU driver installation guide](https://dgpu-docs.intel.com/driver/client/overview.html) to install your GPU driver.

### Download and Extract
1. Download IPEX-LLM llama.cpp portable tgz for Linux from the [link](https://github.com/ipex-llm/ipex-llm/releases/tag/v2.3.0-nightly).

2. Extract the tgz file to a folder.

3. Open a "Terminal", and enter the extracted folder through `cd /PATH/TO/EXTRACTED/FOLDER`

> [!NOTE]
> Hardware Requirements: 
> - 380GB CPU memory for ***DeepSeek V3/R1 671B*** INT4 model 
> - 1-8 ARC A770 or B580
> - 500GB Disk space

## Run
Before running, you should download or copy community GGUF model to your local directory. For instance,  `DeepSeek-R1-Q4_K_M.gguf` of [DeepSeek-R1-Q4_K_M.gguf](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q4_K_M).

Run `DeepSeek-R1-Q4_K_M.gguf`as shown below (change `/PATH/TO/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf` to your model path)

#### CLI
The CLI version of `flashmoe` is built on top of `llama.cpp llama-cli`:

```bash
./flash-moe -m /PATH/TO/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf --prompt "What's AI?" -no-cnv
```

Part of outputs

```bash
llama_kv_cache_init:      SYCL0 KV buffer size =  1280.00 MiB
llama_kv_cache_init:      SYCL1 KV buffer size =  1280.00 MiB
llama_kv_cache_init:      SYCL2 KV buffer size =  1280.00 MiB
llama_kv_cache_init:      SYCL3 KV buffer size =  1280.00 MiB
llama_kv_cache_init:      SYCL4 KV buffer size =  1120.00 MiB
llama_kv_cache_init:      SYCL5 KV buffer size =  1280.00 MiB
llama_kv_cache_init:      SYCL6 KV buffer size =  1280.00 MiB
llama_kv_cache_init:      SYCL7 KV buffer size =   960.00 MiB
llama_new_context_with_model: KV self size  = 9760.00 MiB, K (i8): 5856.00 MiB, V (i8): 3904.00 MiB
llama_new_context_with_model:  SYCL_Host  output buffer size =     0.49 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      SYCL0 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL1 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL2 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL3 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL4 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL5 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL6 compute buffer size =  2076.02 MiB
llama_new_context_with_model:      SYCL7 compute buffer size =  3264.00 MiB
llama_new_context_with_model:  SYCL_Host compute buffer size =  1332.05 MiB
llama_new_context_with_model: graph nodes  = 5184 (with bs=4096), 4720 (with bs=1)
llama_new_context_with_model: graph splits = 125
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
main: llama threadpool init, n_threads = 48

system_info: n_threads = 48 (n_threads_batch = 48) / 192 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 |

sampler seed: 2052631435
sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = -1
        top_k = 40, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> dry -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist
generate: n_ctx = 4096, n_batch = 4096, n_predict = -1, n_keep = 1

<think>
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
</think>

<answer>XXXX</answer> [end of text]
```

#### Serving
The serving version of `flashmoe` is built on top of `llama.cpp server`:

```bash
./flash-moe -m /PATH/TO/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf --serve -n 512 -np 2 -c 4096
```
> `-n` means number of tokens to predict, `-np` means number of parallel sequences to decode, `-c` means the size of whole context, you can adjust these values based on your requirements.

Part of outputs

```bash
...
llama_init_from_model: graph nodes  = 3560
llama_init_from_model: graph splits = 121
common_init_from_params: setting dry_penalty_last_n to ctx_size = 4096
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
srv          init: initializing slots, n_slots = 2
slot         init: id  0 | task -1 | new slot n_ctx_slot = 2048
slot         init: id  1 | task -1 | new slot n_ctx_slot = 2048
main: model loaded
main: chat template, chat_template: {% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true) %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- if ns.is_first_sp %}{% set ns.system_prompt = ns.system_prompt + message['content'] %}{% set ns.is_first_sp = false %}{%- else %}{% set ns.system_prompt = ns.system_prompt + '\n\n' + message['content'] %}{%- endif %}{%- endif %}{%- endfor %}{{ bos_token }}{{ ns.system_prompt }}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and 'tool_calls' in message %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls'] %}{%- if not ns.is_first %}{%- if message['content'] is none %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- else %}{{'<｜Assistant｜>' + message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- endfor %}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- if message['role'] == 'assistant' and 'tool_calls' not in message %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}, example_format: 'You are a helpful assistant

<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>'
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle
```

## Notes 
- Larger models and higher precisions may require more resources.
- For 1 ARC A770 platform, please reduce context length (e.g., 1024) to avoid OOM. Add this option `-c 1024` at the CLI command.
- For dual-sockets Xeon system, consider enabling `SNC (Sub-NUMA Clustering)` in BIOS and add `numactl --interleave=all` before launch command for *better decoding performance*.
