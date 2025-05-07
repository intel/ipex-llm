import requests
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent
import numpy as np
from tqdm import tqdm
import json
import random
import sys
import argparse


PROMPT_128 = "In a distant future, humanity has expanded across the galaxy, establishing colonies on numerous planets. The interstellar community thrives under the guidance of the United Galactic Federation, which ensures peace and prosperity. However, a new threat emerges from the unknown regions of space, challenging the stability and security of the galaxy. Brave explorers and seasoned warriors must unite to uncover the secrets of this mysterious force and protect the future of all sentient beings.  Please continue the above story as long as possible, preferably more than 1000 tokens."

PROMPT_1024 = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. However, her parents were always telling her to stay close to home, to be careful, and to avoid any danger. But the little girl was stubborn, and she wanted to see what was on the other side of the mountain. So she sneaked out of the house one night, leaving a note for her parents, and set off on her journey. As she climbed the mountain, the little girl felt a sense of excitement and wonder. She had never been this far away from home before, and she couldnt wait to see what she would find on the other side. She climbed higher and higher, her lungs burning from the thin air, until she finally reached the top of the mountain. And there, she found a beautiful meadow filled with wildflowers and a sparkling stream. The little girl danced and played in the meadow, feeling free and alive. She knew she had to return home eventually, but for now, she was content to enjoy her adventure. As the sun began to set, the little girl reluctantly made her way back down the mountain, but she knew that she would never forget her adventure and the joy of discovering something new and exciting. And whenever she felt scared or unsure, she would remember the thrill of climbing the mountain and the beauty of the meadow on the other side, and she would know that she could face any challenge that came her way, with courage and determination. She carried the memories of her journey in her heart, a constant reminder of the strength she possessed. The little girl returned home to her worried parents, who had discovered her note and anxiously awaited her arrival. They scolded her for disobeying their instructions and venturing into the unknown. But as they looked into her sparkling eyes and saw the glow on her face, their anger softened. They realized that their little girl had grown, that she had experienced something extraordinary. The little girl shared her tales of the mountain and the meadow with her parents, painting vivid pictures with her words. She spoke of the breathtaking view from the mountaintop, where the world seemed to stretch endlessly before her. She described the delicate petals of the wildflowers, vibrant hues that danced in the gentle breeze. And she recounted the soothing melody of the sparkling stream, its waters reflecting the golden rays of the setting sun. Her parents listened intently, captivated by her story. They realized that their daughter had discovered a part of herself on that journey—a spirit of curiosity and a thirst for exploration. They saw that she had learned valuable lessons about independence, resilience, and the beauty that lies beyond ones comfort zone. From that day forward, the little girls parents encouraged her to pursue her dreams and embrace new experiences. They understood that while there were risks in the world, there were also rewards waiting to be discovered. They supported her as she continued to embark on adventures, always reminding her to stay safe but never stifling her spirit. As the years passed, the little girl grew into a remarkable woman, fearlessly exploring the world and making a difference wherever she went. The lessons she had learned on that fateful journey stayed with her, guiding her through challenges and inspiring her to live life to the fullest. And so, the once timid little girl became a symbol of courage and resilience, a reminder to all who knew her that the greatest joys in life often lie just beyond the mountains we fear to climb. Her story spread far and wide, inspiring others to embrace their own journeys and discover the wonders that awaited them. In the end, the little girls adventure became a timeless tale, passed down through generations, reminding us all that sometimes, the greatest rewards come to those who dare to step into the unknown and follow their hearts. With each passing day, the little girls story continued to inspire countless individuals, igniting a spark within their souls and encouraging them to embark on their own extraordinary adventures. The tale of her bravery and determination resonated deeply with people from all walks of life, reminding them of the limitless possibilities that awaited them beyond the boundaries of their comfort zones. People marveled at the little girls unwavering spirit and her unwavering belief in the power of dreams. They saw themselves reflected in her journey, finding solace in the knowledge that they too could overcome their fears and pursue their passions. The little girl\'s story became a beacon of hope, a testament to the human spirit"

model_name = ""


from typing import AsyncGenerator, List, Tuple
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer
def sample_requests(
    dataset_path: str,
    num_requests: int,
    model_path: str,
    seed=42
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    random.seed(seed)
    np.random.seed(seed)
    tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests

# 定义一个函数来执行单个请求
def perform_request(session, url, payload, headers):
    start_time = time.perf_counter()
    with session.post(url, json=payload, headers=headers, stream=True) as response:
        # 确保响应成功
        response.raise_for_status()

        # 开始接收streaming响应
        first_token_time = None
        last_token_time = 0
        first_token_inference_time = None
        next_token_inference_time = None
        next_token_time = []
        i = 0
        for line in response.iter_lines():

            token_time = time.perf_counter() - start_time
            if line:  # 忽略心跳
                data = line.decode('utf-8').strip()
                if data.startswith('data: '):
                    data = data[len('data: '):]
                    i = i + 1
                    # print(i, " ", data)
                    try:
                        json_data = json.loads(data)
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            choice = json_data['choices'][0]
                            if 'text' in choice or 'delta' in choice:
                                if first_token_time is None:
                                    first_token_time = token_time
                                else:
                                    # 记录后续token的时间
                                    next_token_time.append(token_time - last_token_time)
                                last_token_time = token_time
                    except json.JSONDecodeError:
                        pass  # 如果不是JSON数据，忽略错误

        # 返回第一个token和后续token的latency
        end_time = time.perf_counter()
        # print("length: ", len(next_token_time))

        return first_token_time, np.mean(next_token_time), end_time - start_time, first_token_inference_time, next_token_inference_time

def extend_list_to_length(lst, target_length):
    if target_length <= len(lst):
        return lst[:]

    # 计算每个元素需要复制的次数
    times = target_length // len(lst)
    # 计算不能整除的剩余部分
    remainder = target_length % len(lst)

    # 生成新列表：重复整个列表times次，再加上前remainder个元素
    extended_list = lst * times + lst[:remainder]

    return extended_list

# 定义一个函数来执行benchmark
def benchmark(llm_urls, model, prompt, image_url, num_requests, max_concurrent_requests, max_tokens, is_warmup=False, dataset=None):
    # 定义请求的payload和headers

    headers = {"Content-Type": "application/json"}

    first_token_latencies = []
    next_token_latencies = []
    total_responce_times = []
    first_token_inference_times = []
    next_token_inference_times = []
    cur_url_index = 0
    sampled_requests = []
    prompt_token_lens = []
    output_tokens_lens = []


    if not dataset is None:
        sampled_requests = sample_requests(dataset, num_requests, model)

    # 使用Session对象以便复用连接
    with requests.Session() as session:
        # 创建一个线程池
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            # 开始计时
            # time.sleep(1)
            llm_url = llm_urls[cur_url_index]
            cur_url_index = (cur_url_index + 1) % len(llm_urls)

            cur_llm_urls = extend_list_to_length(llm_urls, max_concurrent_requests)
            cur_len = len(cur_llm_urls)
            if image_url is not None:
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                }
                            ]
                        }
                    ],
                    "n": 1,
                    "best_of": 1,
                    "use_beam_search": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": max_tokens,
                    "ignore_eos": True,
                    "stream": True  # 开启streaming模式
                }
                futures = [executor.submit(perform_request, session, cur_llm_urls[index % cur_len], payload, headers) for index in range(num_requests)]

            start_time = time.perf_counter()

            if is_warmup:
                phase = "Warm Up"
            else:
                phase = "Benchmarking"
            with tqdm(total=num_requests, desc=phase, unit="req", ncols=100) as pbar:
                # 等待所有请求完成
                for future in concurrent.futures.as_completed(futures):
                    try:
                        first_token_latency, next_token_latency, total_responce_time, first_token_inference_time, next_token_inference_time = future.result()
                        first_token_latencies.append(first_token_latency)
                        next_token_latencies.append(next_token_latency)
                        total_responce_times.append(total_responce_time)
                        if first_token_inference_time:
                            first_token_inference_times.append(first_token_inference_time)
                        if next_token_inference_time:
                            next_token_inference_times.append(next_token_inference_time)
                    except Exception as e:
                        print(f"Request failed: {e}")
                    pbar.update(1)

            # 计算总用时
            if is_warmup:
                return
            total_time = time.perf_counter() - start_time
            print(f"Total time for {num_requests} requests with {max_concurrent_requests} concurrent requests: {total_time} seconds.")
            print(f"Average responce time: {np.mean(total_responce_times)}")
            if dataset is None:
                print(f"Token throughput: {num_requests * max_tokens / total_time}")
            else:
                print(f"Output token throughput: {sum(output_tokens_lens) / total_time}")
                print(f"Total token throughput: {(sum(prompt_token_lens) + sum(output_tokens_lens)) / total_time}")
            print()
            if first_token_latencies:
                print(f"first_token_latencies {first_token_latencies}")
                average_first_token_latency = sum(first_token_latencies) / len(first_token_latencies)
                p90_first_token_latency = np.percentile(first_token_latencies, 90)
                p95_first_token_latency = np.percentile(first_token_latencies, 95)
                average_first_token_inference_latency = np.mean(first_token_inference_times)
                print(f"Average first token latency: {average_first_token_latency * 1000} milliseconds.")
                print(f"P90 first token latency: {p90_first_token_latency * 1000} milliseconds.")
                print(f"P95 first token latency: {p95_first_token_latency * 1000} milliseconds.")
                #print(f"Average first token inference latency: {average_first_token_inference_latency * 1000} milliseconds.")
                print()
            if next_token_latencies:
                average_next_token_latency = sum(next_token_latencies) / len(next_token_latencies)
                p90_next_token_latency = np.percentile(next_token_latencies, 90)
                p95_next_token_latency = np.percentile(next_token_latencies, 95)
                average_next_token_inference_latency = np.mean(next_token_inference_times)
                print(f"Average next token latency: {average_next_token_latency * 1000} milliseconds.")
                print(f"P90 next token latency: {p90_next_token_latency * 1000} milliseconds.")
                print(f"P95 next token latency: {p95_next_token_latency * 1000} milliseconds.")
                #print(f"Average next token inference latency: {average_next_token_inference_latency * 1000} milliseconds.")
                print()

def main(args):
    global model_name
    model_name = args.model_name
    max_seq = args.max_seq
    image_url = args.image_url

    output_length = args.output_length if args.output_length else 100
    input_length = args.input_length if args.input_length else 512


    port = args.port
    # 设置benchmark参数
    LLM_URLS = [f"http://localhost:{PORT}/v1/chat/completions" for PORT in [port]]

    MODEL = "/llm/models/" + model_name
    MAX_TOKENS = output_length  # 修改 MAX_TOKENS 为 output_length

    PROMPT = PROMPT_128

    in_len = input_length
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    input_ids = tokenizer.encode(PROMPT_1024, return_tensors="pt")
    print("old input_ids.shape:"+ str(input_ids.shape))
    input_ids = input_ids[:, :in_len]
    print("latest input_ids.shape:"+ str(input_ids.shape))
    true_str = tokenizer.batch_decode(input_ids)[0]
    prompt = true_str


    max_batch=int(max_seq)

    for MAX_CONCURRENT_REQUESTS in [max_batch]:
        NUM_WARMUP = 1 * MAX_CONCURRENT_REQUESTS
        NUM_REQUESTS = 3 * MAX_CONCURRENT_REQUESTS  # 总请求次数

        # to avoid warm_up time out
        benchmark(LLM_URLS, MODEL, PROMPT, image_url, 2, 1, 32, is_warmup = True)
        benchmark(LLM_URLS, MODEL, prompt, image_url, NUM_WARMUP, MAX_CONCURRENT_REQUESTS, MAX_TOKENS, is_warmup = True)

        # 运行benchmark
        benchmark(LLM_URLS, MODEL, prompt, image_url, NUM_REQUESTS, MAX_CONCURRENT_REQUESTS, MAX_TOKENS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking script for LLM")
    parser.add_argument("--model-name", type=str, help="Model name")
    parser.add_argument("--max-seq", type=int, default=1, help="Maximum sequence length")
    parser.add_argument("--image-url", type=str, help="image_url for model to generate")
    parser.add_argument("--input-length", type=int, default=512, help="Input length")
    parser.add_argument("--output-length", type=int, default=100, help="Output length")
    parser.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()
    main(args)