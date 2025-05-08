# vLLM Serving with IPEX-LLM on Intel GPUs via Docker

This guide provides step-by-step instructions for running `vLLM` serving with `IPEX-LLM` on Intel GPUs using Docker.

---

## 1. Install Docker

Follow the instructions in [this guide](./docker_windows_gpu.md#linux) to install Docker on Linux.

---

## 2. Prepare the Docker Image

You can either **pull the prebuilt image** from DockerHub:

```bash
docker pull intelanalytics/ipex-llm-serving-xpu:0.8.3-b19
```

Or **build the image locally** from source:

```bash
cd docker/llm/serving/xpu/docker
docker build \
  --build-arg http_proxy=... \
  --build-arg https_proxy=... \
  --build-arg no_proxy=... \
  --rm --no-cache -t vllm-serving:test .
```

---

## 3. Start the Docker Container

To enable GPU access, map the device by adding `--device=/dev/dri`. Replace `/path/to/models` with the path to your local model directory.

```bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:latest
export CONTAINER_NAME=multi-arc-container

sudo docker run -itd \
  --net=host \
  --privileged \
  --device=/dev/dri \
  -v /path/to/models:/llm/models \
  -e no_proxy=localhost,127.0.0.1 \
  -e http_proxy=$HTTP_PROXY \
  -e https_proxy=$HTTPS_PROXY \
  --name=$CONTAINER_NAME \
  --shm-size="16g" \
  --entrypoint /bin/bash \
  $DOCKER_IMAGE
```

To enter the running container:

```bash
docker exec -it multi-arc-container /bin/bash
```

### Verify GPU Access

Run `sycl-ls` inside the container to confirm GPU devices are visible. A successful output on a machine with Intel Arc A770 GPUs may look like:

```bash
root@ws-arc-001:/llm# sycl-ls
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Arc(TM) A770 Graphics 12.55.8 [1.6.32224.500000]
[level_zero:gpu][level_zero:1] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Arc(TM) A770 Graphics 12.55.8 [1.6.32224.500000]
[level_zero:gpu][level_zero:2] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Arc(TM) A770 Graphics 12.55.8 [1.6.32224.500000]
[level_zero:gpu][level_zero:3] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Arc(TM) A770 Graphics 12.55.8 [1.6.32224.500000]
[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Xeon(R) w5-3435X OpenCL 3.0 (Build 0) [2024.18.12.0.05_160000]
[opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [24.52.32224.5]
[opencl:gpu][opencl:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [24.52.32224.5]
[opencl:gpu][opencl:3] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [24.52.32224.5]
[opencl:gpu][opencl:4] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [24.52.32224.5]
```

---

## 4. Run vLLM Serving with IPEX-LLM

> [!TIP]  
> Before running benchmarks, it's recommended to lock CPU and GPU frequencies to ensure more stable, reliable, and better performance data.
>  
> **Lock CPU Frequency:**  
> Use the following command to set the minimum CPU frequency (adjust based on your CPU model):  
>  
> ```bash  
> sudo cpupower frequency-set -d 3.8GHz  
> ```  
>  
> **Lock GPU Frequencies:**  
> Use these commands to lock GPU frequencies to 2400MHz:  
>  
> ```bash  
> sudo xpu-smi config -d 0 -t 0 --frequencyrange 2400,2400  
> sudo xpu-smi config -d 1 -t 0 --frequencyrange 2400,2400  
> sudo xpu-smi config -d 2 -t 0 --frequencyrange 2400,2400  
> sudo xpu-smi config -d 3 -t 0 --frequencyrange 2400,2400  
> ```  


---

### Start the vLLM Service

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0
export SYCL_CACHE_PERSISTENT=1

export FI_PROVIDER=shm
export CCL_WORKER_COUNT=2
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1
export CCL_SAME_STREAM=1
export CCL_BLOCKING_WAIT=0

export VLLM_USE_V1=0
export IPEX_LLM_LOWBIT="fp8"

source /opt/intel/1ccl-wks/setvars.sh

numactl -C 0-11 python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --port 8000 \
  --model "/llm/models/Qwen2.5-7B-Instruct/" \
  --served-model-name "Qwen2.5-7B-Instruct" \
  --trust-remote-code \
  --gpu-memory-utilization "0.95" \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit "fp8" \
  --max-model-len "2000" \
  --max-num-batched-tokens "3000" \
  --max-num-seqs "256" \
  --tensor-parallel-size "2" \
  --pipeline-parallel-size "1" \
  --block-size 8 \
  --distributed-executor-backend ray \
  --disable-async-output-proc
```

#### Parameter Descriptions

|parameters|explanation|
|:---|:---|
|`--model`| the model path in docker, for example `"/llm/models/Llama-2-7b-chat-hf"`|
|`--served-model-name`| the model name, for example `"Llama-2-7b-chat-hf"`|
|`--load-in-low-bit`| model quantization accuracy, acceptable ``'sym_int4'``, ``'asym_int4'``,  ``'fp6'``, ``'fp8'``, ``'fp8_e4m3'``, ``'fp8_e5m2'``,  ``'fp16'``; ``'sym_int4'`` means symmetric int 4, ``'asym_int4'`` means asymmetric int 4, etc. Relevant low bit optimizations will be applied to the model. default is ``'fp8'``, which is the same as ``'fp8_e5m2'``|
|`--tensor-parallel-size`| number of tensor parallel replicas, default is `1`|
|`--pipeline-parallel-size`| number of pipeline stages, default is `1`|
|`--gpu-memory-utilization`| The fraction of GPU memory to be used for the model executor, which can range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory utilization. If unspecified, will use the default value of 0.9.|
|`--max-model-len`| Model context length. If unspecified, will be automatically derived from the model config.|
|`--max-num-batched-token`| Maximum number of batched tokens per iteration.|
|`--max-num-seq`| Maximum number of sequences per iteration. Default: 256|
|`--block-size`| vLLM block size. Set to 8 to achieve a performance boost.|
|`--quantization`| If you want to enable low-bit quantization (e.g., AWQ or GPTQ), you can add the `--quantization` argument. The available values include `awq` and `gptq`. Note that `load_in_low_bit` should be set to `asym_int4` instead of `int4`. |


The script mentioned above has been included in the start-vllm-service.sh within the container. You can also modify the parameters in `/llm/start-vllm-service.sh` and start the service using the built-in script:

```bash
bash /llm/start-vllm-service.sh
```

A successful startup will show logs similar to the screenshot:

  <a href="https://llm-assets.readthedocs.io/en/latest/_images/start-vllm-service.png" target="_blank">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/start-vllm-service.png" width=100%; />

  </a>

---

## 5. Test the vLLM Service

Send a test completion request using `curl`:

```bash
curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "llama2-7b-chat",
           "prompt": "San Francisco is a",
           "max_tokens": 128
         }'
```

---

## 6. Benchmarking

If you want to run benchmarks, we recommend using the official vLLM benchmarking script to collect performance data. You can execute the following command inside the container.

- --num_prompt specifies the number of concurrent requests
- --random-input-len sets the number of input tokens
- --random-output-len sets the number of output tokens


```bash
python /llm/vllm/benchmarks/benchmark_serving.py \
  --model /llm/models/Qwen2.5-7B-Instruct \
  --dataset-name random \
  --trust_remote_code \
  --num_prompt $batch_size \
  --random-input-len=$input_length \
  --random-output-len=$output_length
```

### Sample Benchmark Output

```
============ Serving Benchmark Result ============
Successful requests:                     1
Benchmark duration (s):                  17.06
Total input tokens:                      1024
Total generated tokens:                  512
Request throughput (req/s):              0.06
Output token throughput (tok/s):         30.01
Total Token throughput (tok/s):          90.02
---------------Time to First Token----------------
Mean TTFT (ms):                          520.10
Median TTFT (ms):                        520.10
P99 TTFT (ms):                           520.10
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          32.37
Median TPOT (ms):                        32.37
P99 TPOT (ms):                           32.37
---------------Inter-token Latency----------------
Mean ITL (ms):                           64.87
Median ITL (ms):                         64.84
P99 ITL (ms):                            66.34
==================================================
```

## 7. Miscellaneous Tools



### 7.1 Offline Inference and Model Quantization (Optional) 
<details>
If real-time services are not required, you can choose to use offline inference for testing or evaluation. Additionally, IPEX-LLM supports various model quantization schemes (such as FP8, INT4, AWQ, GPTQ) to reduce memory usage and improve performance.

Edit the parameters of the `LLM` class in `/llm/vllm_offline_inference.py`, such as:

**Example 1:** Standard Model + IPEX-LLM Low-bit Format

```python
llm = LLM(model="/llm/models/Llama-2-7b-chat-hf",
          device="xpu",
          dtype="float16",
          enforce_eager=True,
          load_in_low_bit="sym_int4",  # Optional values: sym_int4, asym_int4, fp6, fp8, fp16
          tensor_parallel_size=1,
          trust_remote_code=True)
```

**Example 2:** AWQ Model (e.g., Llama-2-7B-Chat-AWQ)

```python
llm = LLM(model="/llm/models/Llama-2-7B-Chat-AWQ/",
          quantization="AWQ",
          load_in_low_bit="asym_int4",  # Note: use asym_int4 here
          device="xpu",
          dtype="float16",
          enforce_eager=True,
          tensor_parallel_size=1)
```

**Example 3:** GPTQ Model (e.g., llama2-7b-chat-GPTQ)

```python
llm = LLM(model="/llm/models/llama2-7b-chat-GPTQ/",
          quantization="GPTQ",
          load_in_low_bit="sym_int4",  # GPTQ recommends using sym_int4
          device="xpu",
          dtype="float16",
          enforce_eager=True,
          tensor_parallel_size=1)
```

Run the command:

```bash
python vllm_offline_inference.py
```

You should see output similar to the following:

```plaintext
Prompt: 'The capital of France is', Generated text: ' Paris.'
Prompt: 'The future of AI is', Generated text: ' promising and transformative...'
```
</details>

### 7.2. Benchmarking
<details>

#### 7.2.1 Online Benchmark through API Server

To benchmark the API server and estimate TPS (transactions per second), follow these steps:

1. Start the service as per the instructions in this [section](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/DockerGuides/vllm_docker_quickstart.md#Serving).
2. Run the benchmark using `vllm_online_benchmark.py`:

```bash
python vllm_online_benchmark.py $model_name $max_seqs $input_length $output_length
```

If `input_length` and `output_length` are not provided, the script defaults to values of 1024 and 512 tokens, respectively. The output will look something like:

```bash
model_name: Qwen1.5-14B-Chat
max_seq: 12
Warm Up: 100%|█████████████████████████████████████████████████████| 24/24 [01:36<00:00,  4.03s/req]
Benchmarking: 100%|████████████████████████████████████████████████| 60/60 [04:03<00:00,  4.05s/req]
Total time for 60 requests with 12 concurrent requests: xxx seconds.
Average response time: xxx
Token throughput: xxx

Average first token latency: xxx milliseconds.
P90 first token latency: xxx milliseconds.
P95 first token latency: xxx milliseconds.

Average next token latency: xxx milliseconds.
P90 next token latency: xxx milliseconds.
P95 next token latency: xxx milliseconds.
```

#### 7.2.2 Online Benchmark with Multimodal Input

After starting the vLLM service, you can benchmark multimodal inputs using `vllm_online_benchmark_multimodal.py`:

```bash
wget -O /llm/models/test.webp https://gd-hbimg.huaban.com/b7764d5f9c19b3e433d54ba66cce6a112050783e8182-Cjds3e_fw1200webp
export image_url="/llm/models/test.webp"
python vllm_online_benchmark_multimodal.py --model-name $model_name --image-url $image_url --port 8000 --max-seq 1 --input-length 512 --output-length 100
```

The `image_url` can be a local path (e.g., `/llm/xxx.jpg`) or an external URL (e.g., `"http://xxx.jpg`).

The output will be similar to the example in the API benchmarking section.

#### 7.2.3 Online Benchmark through wrk

In the container, modify the `payload-1024.lua` to ensure the "model" attribute is correct. By default, it uses a prompt of about 1024 tokens.

Then, start the benchmark using `wrk`:

```bash
cd /llm
wrk -t12 -c12 -d15m -s payload-1024.lua http://localhost:8000/v1/completions --timeout 1h
```

#### 7.2.4 Offline Benchmark through `benchmark_vllm_throughput.py`

To use the `benchmark_vllm_throughput.py` script, first download the test dataset:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Then, run the benchmark:

```bash
cd /llm/

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

export MODEL="YOUR_MODEL"

python3 /llm/benchmark_vllm_throughput.py \
    --backend vllm \
    --dataset /llm/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model $MODEL \
    --num-prompts 1000 \
    --seed 42 \
    --trust-remote-code \
    --enforce-eager \
    --dtype float16 \
    --device xpu \
    --load-in-low-bit sym_int4 \
    --gpu-memory-utilization 0.85
```
</details>


### 7.3 Automatically Launching the Service via Container
<details>
You can configure the container to automatically start the service with the desired model and parallel settings by using environment variables:

```bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:latest

sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        --privileged \
        --memory="32G" \
        --name=CONTAINER_NAME \
        --shm-size="16g" \
        -e MODEL_PATH="/llm/models" \
        -e SERVED_MODEL_NAME="my_model" \
        -e TENSOR_PARALLEL_SIZE=4 \
        -v /home/intel/LLM/:/llm/models/ \
        $DOCKER_IMAGE
```

* `MODEL_PATH`, `SERVED_MODEL_NAME`, and `TENSOR_PARALLEL_SIZE` control the model used and degree of parallelism.
* Mount the model directory using `-v`.

To view logs and confirm service status:

```bash
docker logs CONTAINER_NAME
```
</details>



## 8. Advanced Features

#### Multi-modal Model
<details>
vLLM serving with IPEX-LLM supports multi-modal models, such as [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6), which can accept image and text input at the same time and respond.

1. Start MiniCPM service: change the `model` and `served_model_name` value in `/llm/start-vllm-service.sh`

2. Send request with image url and prompt text. (For successfully download image from url, you may need set `http_proxy` and `https_proxy` in docker before the vllm service started)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-2_6",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "图片里有什么?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 128
  }'
```

3. Expect result should be like:

```bash
{"id":"chat-0c8ea64a2f8e42d9a8f352c160972455","object":"chat.completion","created":1728373105,"model":"MiniCPM-V-2_6","choices":[{"index":0,"message":{"role":"assistant","content":"这幅图片展示了一个小孩，可能是女孩，根据服装和发型来判断。她穿着一件有红色和白色条纹的连衣裙，一个可见的白色蝴蝶结，以及一个白色的 头饰，上面有红色的点缀。孩子右手拿着一个白色泰迪熊，泰迪熊穿着一个粉色的裙子，带有褶边，它的左脸颊上有一个红色的心形图案。背景模糊，但显示出一个自然户外的环境，可能是一个花园或庭院，有红花和石头墙。阳光照亮了整个场景，暗示这可能是正午或下午。整体氛围是欢乐和天真。","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":225,"total_tokens":353,"completion_tokens":128}}
```
</details>

#### Preifx Caching
<details>
Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part.

1. Set `enable_prefix_caching=True` in vLLM engine to enable APC. Here is an example python script to show the time reduce of APC:

```python
import time
from vllm import SamplingParams
from ipex_llm.vllm.xpu.engine import IPEXLLMClass as LLM


# A prompt containing a large markdown table. The table is randomly generated by GPT-4.
LONG_PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as follows.\n# Table\n" + """
| ID  | Name          | Age | Occupation    | Country       | Email                  | Phone Number   | Address                       |
|-----|---------------|-----|---------------|---------------|------------------------|----------------|------------------------------|
| 1   | John Doe      | 29  | Engineer      | USA           | john.doe@example.com   | 555-1234       | 123 Elm St, Springfield, IL  |
| 2   | Jane Smith    | 34  | Doctor        | Canada        | jane.smith@example.com | 555-5678       | 456 Oak St, Toronto, ON      |
| 3   | Alice Johnson | 27  | Teacher       | UK            | alice.j@example.com    | 555-8765       | 789 Pine St, London, UK      |
| 4   | Bob Brown     | 45  | Artist        | Australia     | bob.b@example.com      | 555-4321       | 321 Maple St, Sydney, NSW    |
| 5   | Carol White   | 31  | Scientist     | New Zealand   | carol.w@example.com    | 555-6789       | 654 Birch St, Wellington, NZ |
| 6   | Dave Green    | 28  | Lawyer        | Ireland       | dave.g@example.com     | 555-3456       | 987 Cedar St, Dublin, IE     |
| 7   | Emma Black    | 40  | Musician      | USA           | emma.b@example.com     | 555-1111       | 246 Ash St, New York, NY     |
| 8   | Frank Blue    | 37  | Chef          | Canada        | frank.b@example.com    | 555-2222       | 135 Spruce St, Vancouver, BC |
| 9   | Grace Yellow  | 50  | Engineer      | UK            | grace.y@example.com    | 555-3333       | 864 Fir St, Manchester, UK   |
| 10  | Henry Violet  | 32  | Artist        | Australia     | henry.v@example.com    | 555-4444       | 753 Willow St, Melbourne, VIC|
| 11  | Irene Orange  | 26  | Scientist     | New Zealand   | irene.o@example.com    | 555-5555       | 912 Poplar St, Auckland, NZ  |
| 12  | Jack Indigo   | 38  | Teacher       | Ireland       | jack.i@example.com     | 555-6666       | 159 Elm St, Cork, IE         |
| 13  | Karen Red     | 41  | Lawyer        | USA           | karen.r@example.com    | 555-7777       | 357 Cedar St, Boston, MA     |
| 14  | Leo Brown     | 30  | Chef          | Canada        | leo.b@example.com      | 555-8888       | 246 Oak St, Calgary, AB      |
| 15  | Mia Green     | 33  | Musician      | UK            | mia.g@example.com      | 555-9999       | 975 Pine St, Edinburgh, UK   |
| 16  | Noah Yellow   | 29  | Doctor        | Australia     | noah.y@example.com     | 555-0000       | 864 Birch St, Brisbane, QLD  |
| 17  | Olivia Blue   | 35  | Engineer      | New Zealand   | olivia.b@example.com   | 555-1212       | 753 Maple St, Hamilton, NZ   |
| 18  | Peter Black   | 42  | Artist        | Ireland       | peter.b@example.com    | 555-3434       | 912 Fir St, Limerick, IE     |
| 19  | Quinn White   | 28  | Scientist     | USA           | quinn.w@example.com    | 555-5656       | 159 Willow St, Seattle, WA   |
| 20  | Rachel Red    | 31  | Teacher       | Canada        | rachel.r@example.com   | 555-7878       | 357 Poplar St, Ottawa, ON    |
| 21  | Steve Green   | 44  | Lawyer        | UK            | steve.g@example.com    | 555-9090       | 753 Elm St, Birmingham, UK   |
| 22  | Tina Blue     | 36  | Musician      | Australia     | tina.b@example.com     | 555-1213       | 864 Cedar St, Perth, WA      |
| 23  | Umar Black    | 39  | Chef          | New Zealand   | umar.b@example.com     | 555-3435       | 975 Spruce St, Christchurch, NZ|
| 24  | Victor Yellow | 43  | Engineer      | Ireland       | victor.y@example.com   | 555-5657       | 246 Willow St, Galway, IE    |
| 25  | Wendy Orange  | 27  | Artist        | USA           | wendy.o@example.com    | 555-7879       | 135 Elm St, Denver, CO       |
| 26  | Xavier Green  | 34  | Scientist     | Canada        | xavier.g@example.com   | 555-9091       | 357 Oak St, Montreal, QC     |
| 27  | Yara Red      | 41  | Teacher       | UK            | yara.r@example.com     | 555-1214       | 975 Pine St, Leeds, UK       |
| 28  | Zack Blue     | 30  | Lawyer        | Australia     | zack.b@example.com     | 555-3436       | 135 Birch St, Adelaide, SA   |
| 29  | Amy White     | 33  | Musician      | New Zealand   | amy.w@example.com      | 555-5658       | 159 Maple St, Wellington, NZ |
| 30  | Ben Black     | 38  | Chef          | Ireland       | ben.b@example.com      | 555-7870       | 246 Fir St, Waterford, IE    |
"""


def get_generation_time(llm, sampling_params, prompts):
    # time the generation
    start_time = time.time()
    output = llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()
    # print the output and generation time
    print(f"Output: {output[0].outputs[0].text}")
    print(f"Generation time: {end_time - start_time} seconds.")


# set enable_prefix_caching=True to enable APC
llm = LLM(model='/llm/models/Llama-2-7b-chat-hf',
          device="xpu",
          dtype="float16",
          enforce_eager=True,
          load_in_low_bit="fp8",
          tensor_parallel_size=1,
          max_model_len=2000,
          max_num_batched_tokens=2000,
          enable_prefix_caching=True)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Querying the age of John Doe
get_generation_time(
        llm,
        sampling_params,
        LONG_PROMPT + "Question: what is the age of John Doe? Your answer: The age of John Doe is ",
        )

# Querying the age of Zack Blue
# This query will be faster since vllm avoids computing the KV cache of LONG_PROMPT again.
get_generation_time(
        llm,
        sampling_params,
        LONG_PROMPT + "Question: what is the age of Zack Blue? Your answer: The age of Zack Blue is ",
        )

```

2. Expected output is shown as below: APC greatly reduces the generation time of the question related to the same table.

```bash
INFO 10-09 15:43:21 block_manager_v1.py:247] Automatic prefix caching is enabled.
Processed prompts: 100%|█████████████████████████████████████████████████| 1/1 [00:21<00:00, 21.97s/it, est. speed input: 84.57 toks/s, output: 0.73 toks/s]
Output: 29.
Question: What is the occupation of Jane Smith? Your answer
Generation time: 21.972806453704834 seconds.
Processed prompts: 100%|██████████████████████████████████████████████| 1/1 [00:00<00:00,  1.04it/s, est. speed input: 1929.67 toks/s, output: 16.63 toks/s]
Output: 30.
Generation time: 0.9657604694366455 seconds.
```
</details>

#### LoRA Adapter
<details>
This chapter shows how to use LoRA adapters with vLLM on top of a base model. Adapters can be efficiently served on a per request basis with minimal overhead.

1. Download the adapter(s) and save them locally first, for example, for `llama-2-7b`:

```bash
git clone https://huggingface.co/yard1/llama-2-7b-sql-lora-test
```

2. Start vllm server with LoRA adapter, setting `--enable-lora` and `--lora-modules` is necessary

```bash
export SQL_LOARA=your_sql_lora_model_path
python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --served-model-name Llama-2-7b-hf \
  --port 8000 \
  --model meta-llama/Llama-2-7b-hf \
  --trust-remote-code \
  --gpu-memory-utilization 0.75 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit fp8 \
  --max-model-len 4096 \
  --max-num-batched-tokens 10240 \
  --tensor-parallel-size 1 \
  --distributed-executor-backend ray \
  --enable-lora \
  --lora-modules sql-lora=$SQL_LOARA
```

3. Send a request to sql-lora

```bash
curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
     "model": "sql-lora",
     "prompt": "San Francisco is a",
     "max_tokens": 128,
     "temperature": 0
     }'
```

4. Result expected show below:

```json
{
    "id": "cmpl-d6fa55b2bc404628bd9c9cf817326b7e",
    "object": "text_completion",
    "created": 1727367966,
    "model": "Llama-2-7b-hf",
    "choices": [
        {
            "index": 0,
            "text": " city in Northern California that is known for its vibrant cultural scene, beautiful architecture, and iconic landmarks like the Golden Gate Bridge and Alcatraz Island. Here are some of the best things to do in San Francisco:\n\n1. Explore Golden Gate Park: This sprawling urban park is home to several museums, gardens, and the famous Japanese Tea Garden. It's a great place to escape the hustle and bustle of the city and enjoy some fresh air and greenery.\n2. Visit Alcatraz Island: Take a ferry to the former prison and",
            "logprobs": null,
            "finish_reason": "length",
            "stop_reason": null
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 133,
        "completion_tokens": 128
    }
}
```

5. For multi lora adapters, modify the sever start script's `--lora-modules` like this:

```bash
export SQL_LOARA_1=your_sql_lora_model_path_1
export SQL_LOARA_2=your_sql_lora_model_path_2
python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  #other codes...
  --enable-lora \
  --lora-modules sql-lora-1=$SQL_LOARA_1 sql-lora-2=$SQL_LOARA_2

```
</details>

#### OpenAI API Backend
<details>
vLLM Serving can be deployed as a server that implements the OpenAI API protocol. This allows vLLM to be used as backend for web applications such as [open-webui](https://github.com/open-webui/open-webui/) using OpenAI API.

1. Start vLLM Serving with `api-key`, just setting any string to `api-key` in `start-vllm-service.sh`, and run it.

```bash
#!/bin/bash
model="/llm/models/Qwen1.5-14B-Chat"
served_model_name="Qwen1.5-14B-Chat"

#export SYCL_CACHE_PERSISTENT=1
export CCL_WORKER_COUNT=4
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0

export VLLM_USE_V1=0
export IPEX_LLM_LOWBIT=fp8

source /opt/intel/1ccl-wks/setvars.sh

python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 8000 \
  --model $model \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit $IPEX_LLM_LOWBIT \
  --max-model-len 2048 \
  --max-num-batched-tokens 4000 \
  --api-key <your-api-key> \
  --tensor-parallel-size 4 \
  --distributed-executor-backend ray
```

2. Send http request with `api-key` header to verify the model has deployed successfully.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <your-api-key>" \
    -d '{
    "model": "Qwen1.5-14B-Chat",
    "prompt": "San Francisco is a",
    "max_tokens": 128
    }'
```

3. Start open-webui serving with following scripts. Note that the `OPENAI_API_KEY` must be consistent with the backend value. The `<host-ip>` in `OPENAI_API_BASE_URL` is the ipv4 address of the host that starts docker. For relevant details, please refer to official document [link](https://docs.openwebui.com/#installation-for-openai-api-usage-only) of open-webui.

```bash
#!/bin/bash
export DOCKER_IMAGE=ghcr.io/open-webui/open-webui:main
export CONTAINER_NAME=<your-docker-container-name>

docker rm -f $CONTAINER_NAME

docker run -itd \
           -p 3000:8080 \
           -e OPENAI_API_KEY=<your-api-key> \
           -e OPENAI_API_BASE_URL=http://<host-ip>:8000/v1 \
           -v open-webui:/app/backend/data \
           --name $CONTAINER_NAME \
           --restart always $DOCKER_IMAGE  
```

Then you should start the docker on host that make sure you can visit vLLM backend serving.

4. After installation, you can access Open WebUI at <http://localhost:3000>. Enjoy! 😄

#### Serving with FastChat

We can set up model serving using `IPEX-LLM` as backend using FastChat, the following steps gives an example of how to deploy a demo using FastChat.

1. **Start the Docker Container**

    Run the following command to launch a Docker container with device access:

    ```bash
    #/bin/bash
    export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:latest

    sudo docker run -itd \
            --net=host \
            --device=/dev/dri \
            --name=demo-container \
            # Example: map host model directory to container
            -v /LLM_MODELS/:/llm/models/ \  
            --shm-size="16g" \
            # Optional: set proxy if needed
            -e http_proxy=... \ 
            -e https_proxy=... \
            -e no_proxy="127.0.0.1,localhost" \
            --entrypoint /bin/bash \
            $DOCKER_IMAGE
    ```

2. **Start the FastChat Service**

    Enter the container and start the FastChat service:

    ```bash
    #/bin/bash

    # This command assumes that you have mapped the host model directory to the container
    # and the model directory is /llm/models/
    # we take Yi-1.5-34B as an example, and you can replace it with your own model

    ps -ef | grep "fastchat" | awk '{print $2}' | xargs kill -9
    pip install -U gradio==4.43.0
    
    # start controller
    python -m fastchat.serve.controller &

    export USE_XETLA=OFF
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
    
    export TORCH_LLM_ALLREDUCE=0
    export CCL_DG2_ALLREDUCE=1
    # CCL needed environment variables
    export CCL_WORKER_COUNT=4
    # pin ccl worker to cores
    # export CCL_WORKER_AFFINITY=32,33,34,35
    export FI_PROVIDER=shm
    export CCL_ATL_TRANSPORT=ofi
    export CCL_ZE_IPC_EXCHANGE=sockets
    export CCL_ATL_SHM=1
    
    source /opt/intel/1ccl-wks/setvars.sh
    
    python -m ipex_llm.serving.fastchat.vllm_worker \
    --model-path /llm/models/Yi-1.5-34B \
    --device xpu \
    --enforce-eager \
    --disable-async-output-proc \
    --distributed-executor-backend ray \
    --dtype float16 \
    --load-in-low-bit fp8 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --max-num-batched-tokens 8000 &
    
    sleep 120
    
    python -m fastchat.serve.gradio_web_server &
    ```

This quick setup allows you to deploy FastChat with IPEX-LLM efficiently.
</details>

### Validated Models List

| models (fp8)     | gpus  |
| ---------------- | :---: |
| llama-3-8b       |   1   |
| Llama-2-7B       |   1   |
| Qwen2-7B         |   1   |
| Qwen1.5-7B       |   1   |
| GLM4-9B          |   1   |
| chatglm3-6b      |   1   |
| Baichuan2-7B     |   1   |
| Codegeex4-all-9b |   1   |
| Llama-2-13B      |   2   |
| Qwen1.5-14b      |   2   |
| TeleChat-13B     |   2   |
| Qwen1.5-32b      |   4   |
| Yi-1.5-34B       |   4   |
| CodeLlama-34B    |   4   |
