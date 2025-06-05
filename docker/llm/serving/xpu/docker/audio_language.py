import os
from dataclasses import asdict
from typing import NamedTuple, Optional

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from ipex_llm.vllm.xpu.engine import IPEXLLMClass as LLM
from vllm.assets.audio import AudioAsset
from vllm.utils import FlexibleArgumentParser

audio_assets = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]
question_per_audio_count = {
    0: "What is 1+1?",
    1: "What is recited in the audio?",
    2: "What sport and what nursery rhyme are referenced?"
}

model_path = "/llm/models/whisper-large-v3-turbo"
#model_path = "/llm/models/whisper-medium"
#model_path = "/llm/models/Phi-4-multimodal-instruct"

# Phi-4-multimodal-instruct
def run_phi4mm(question: str, audio_count: int):
    placeholders = "".join([f"<|audio_{i+1}|>" for i in range(audio_count)])

    prompt = f"<|user|>{placeholders}{question}<|end|><|assistant|>"

    return prompt


# Whisper
def run_whisper(question: str, audio_count: int):
    assert audio_count == 1, (
        "Whisper only support single audio input per prompt")

    prompt = "<|startoftranscript|>"

    return prompt


model_example_map = {
    "phi4mm": run_phi4mm,
    "whisper": run_whisper,
}


if "whisper" in model_path:
    model_len=448
    low_bit="fp16"
else:
    model_len = 5500
    low_bit="sym_int4"

def main(args):
    audio_count = args.num_audios

    llm = LLM(
            model=model_path,
            device="xpu",
            dtype="float16",
            limit_mm_per_prompt={"audio": audio_count},
            enforce_eager=True,
            mm_processor_kwargs=None,
            load_in_low_bit=low_bit,
            tensor_parallel_size=1,
            max_num_seqs=8,
            gpu_memory_utilization=0.95,
            disable_async_output_proc=True,
            distributed_executor_backend="ray",
            max_model_len=model_len,
            trust_remote_code=True,
            block_size=8,
            max_num_batched_tokens=model_len)

    model = llm.llm_engine.model_config.hf_config.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    prompt = model_example_map[model](question_per_audio_count[audio_count], audio_count)

    sampling_params = SamplingParams(temperature=0.1,
                                     top_p=0.001,
                                     repetition_penalty=1.05,
                                     max_tokens=128,
                                     skip_special_tokens=False
                                     )

    mm_data = {}
    if audio_count > 0:
        mm_data = {
            "audio": [
                asset.audio_and_sample_rate
                for asset in audio_assets[:audio_count]
            ]
        }

    assert args.num_prompts > 0
    inputs = {"prompt": prompt, "multi_modal_data": mm_data}
    if args.num_prompts > 1:
        # Batch inference
        inputs = [inputs] * args.num_prompts

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'audio language models')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')
    parser.add_argument("--num-audios",
                        type=int,
                        default=1,
                        choices=[0, 1, 2],
                        help="Number of audio items per prompt.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Set the seed when initializing `vllm.LLM`.")

    args = parser.parse_args()
    main(args)
