# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# First stage: build oneccl
FROM intel/oneapi-basekit:2025.0.1-0-devel-ubuntu24.04 AS build

ARG http_proxy
ARG https_proxy
ARG PIP_NO_CACHE_DIR=false

# Set environment variables
ENV TZ=Asia/Shanghai PYTHONUNBUFFERED=1

# Copy patch file and benchmark scripts
COPY ./ccl_torch.patch /tmp/
COPY ./vllm_online_benchmark.py ./vllm_offline_inference.py ./vllm_offline_inference_vision_language.py \
     ./payload-1024.lua ./start-vllm-service.sh ./benchmark_vllm_throughput.py ./benchmark_vllm_latency.py \
     ./vllm_online_benchmark_multimodal.py ./start-pp_serving-service.sh /llm/
COPY ./1ccl_for_multi_arc.patch /build/
     
RUN set -eux && \
    #
    # Update and install basic dependencies
    apt-get update && \
    apt-get install -y --no-install-recommends \
      curl wget git libunwind8-dev vim less gnupg gpg-agent software-properties-common \
      libfabric-dev wrk libaio-dev numactl && \
    # 
    # Set timezone
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    # 
    # Install Python 3.11
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-distutils python3-wheel && \
    rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    # 
    # Install pip and essential Python packages
    wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3 get-pip.py && rm get-pip.py && \
    #
    # Install Intel GPU OpenCL Driver and Compute Runtime
    mkdir -p /tmp/neo && \
    cd /tmp/neo && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu-dbgsym_1.6.32224.5_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd-dbgsym_24.52.32224.5_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb && \
    dpkg -i *.deb && \
    #
    # Install Intel PyTorch extension for LLM inference
    pip install --pre --upgrade 'ipex-llm[xpu_2.6]>=2.3.0b0,<2.3.0rc1' --extra-index-url https://download.pytorch.org/whl/xpu && \
    pip install intel-extension-for-pytorch==2.6.10+xpu --extra-index-url=https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/ && \
    #
    # Build torch-ccl
    mkdir -p /build && \
    cd /build && \
    git clone https://github.com/intel/torch-ccl.git && \
    cd torch-ccl && \
    git checkout ccl_torch2.6.0+xpu && \
    git submodule sync && \
    git submodule update --init --recursive && \
    # This patch will enable build torch-ccl with pytorch 2.6 environment
    # git apply /tmp/ccl_torch.patch && \
    USE_SYSTEM_ONECCL=ON COMPUTE_BACKEND=dpcpp python setup.py bdist_wheel && \
    # File path: /build/torch-ccl/dist/oneccl_bind_pt-2.6.0+xpu-cp311-cp311-linux_x86_64.whl
    # Build oneCCL
    pip install ninja && \
    cd /build/ && \
    git clone https://github.com/analytics-zoo/oneCCL.git && \
    cd oneCCL && \
    git checkout 3afa1bb7936f57683a2503c34b29c0daca6a9ccb && \
    git apply /build/1ccl_for_multi_arc.patch && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS="-fsycl" -DCOMPUTE_BACKEND=dpcpp  -DCMAKE_BUILD_TYPE=MinSizeRel && \
    # File path: /build/oneCCL/build/src/libccl.so.1.0
    ninja


# Second stage: Final runtime image
FROM intel/oneapi-basekit:2025.0.1-0-devel-ubuntu24.04

# Copy the built torch-ccl package from the build stage
COPY --from=build /build/torch-ccl/dist/oneccl_bind_pt-2.6.0+xpu-cp311-cp311-linux_x86_64.whl /opt/
COPY --from=build /llm/ /llm/
COPY --from=build /build/oneCCL/build/src/libccl.so.1.0 /opt/intel/1ccl-wks/lib/
COPY --from=build /build/oneCCL/build/src/libccl.so.1 /opt/intel/1ccl-wks/lib/
COPY --from=build /build/oneCCL/build/src/libccl.so /opt/intel/1ccl-wks/lib/
COPY ./vllm_for_multi_arc.patch /llm/
COPY ./setvars.sh /opt/intel/1ccl-wks/

ARG http_proxy
ARG https_proxy
ARG PIP_NO_CACHE_DIR=false

# Set environment variables
ENV TZ=Asia/Shanghai PYTHONUNBUFFERED=1 VLLM_RPC_TIMEOUT=100000

RUN set -eux && \
    #
    # Update and install basic dependencies, upgrade linux-libc-dev to fix CT7 CVEs
    apt-get update && \
    apt-get install -y --no-install-recommends \
      linux-libc-dev \
      curl wget git libunwind8-dev vim less gnupg gpg-agent software-properties-common \
      libfabric-dev wrk libaio-dev numactl && \
    # 
    # Set timezone
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    # 
    # Install Python 3.11
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-distutils python3-wheel && \
    rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    # 
    # Install pip and essential Python packages
    wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3 get-pip.py && rm get-pip.py && \
    pip install --upgrade requests argparse urllib3 && \
    pip install --pre --upgrade 'ipex-llm[xpu_2.6]>=2.3.0b0,<2.3.0rc1' --extra-index-url https://download.pytorch.org/whl/xpu && \
    pip install transformers_stream_generator einops tiktoken librosa && \
    pip install --upgrade colorama && \
    # 
    git clone https://github.com/intel/ipex-llm.git && \
    cp -r ./ipex-llm/python/llm/dev/benchmark/ ./benchmark && \
    cp -r ./ipex-llm/python/llm/example/GPU/HuggingFace/LLM ./examples && \
    cp -r ./ipex-llm/python/llm/example/GPU/vLLM-Serving/ ./vLLM-Serving && \
    #
    # Download pp_serving
    mkdir -p /llm/pp_serving && \
    cp ./ipex-llm/python/llm/example/GPU/Pipeline-Parallel-Serving/*.py /llm/pp_serving/ && \
    #
    # Download lightweight_serving
    mkdir -p /llm/lightweight_serving && \
    cp ./ipex-llm/python/llm/example/GPU/Lightweight-Serving/*.py /llm/lightweight_serving/ && \
    rm -rf ./ipex-llm && \
    #
    # Install vllm dependencies
    pip install --upgrade fastapi && \
    pip install --upgrade "uvicorn[standard]" && \
    pip install intel-extension-for-pytorch==2.6.10+xpu --extra-index-url=https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/ && \
    #
    # Install torch-ccl
    pip install /opt/oneccl_bind_pt-2.6.0+xpu-cp311-cp311-linux_x86_64.whl && \
    #
    apt-get update && \
    apt-get install -y --no-install-recommends libfabric-dev wrk libaio-dev numactl && \
    # 
    # Remove breaks install packages
    apt-get remove -y libze-dev libze-intel-gpu1 && \
    #
    # Install compute runtime
    mkdir -p /tmp/neo && \
    cd /tmp/neo && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.8.3/intel-igc-core-2_2.8.3+18762_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.8.3/intel-igc-opencl-2_2.8.3+18762_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.8.3/intel-igc-opencl-devel_2.8.3+18762_amd64.deb && \
    dpkg -i *.deb && \
    rm -rf *.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.09.32961.7/libigdgmm12_22.6.0_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.09.32961.7/intel-level-zero-gpu-dbgsym_1.6.32961.7_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.09.32961.7/intel-level-zero-gpu_1.6.32961.7_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.09.32961.7/intel-opencl-icd-dbgsym_25.09.32961.7_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.09.32961.7/intel-opencl-icd_25.09.32961.7_amd64.deb && \
    dpkg -i *.deb && \
    dpkg -i *.ddeb && \
    mkdir -p /llm && \
    cd /llm && \
    rm -rf /tmp/neo && \
    #
    # Install vllm
    git clone -b v0.8.3 https://github.com/vllm-project/vllm /llm/vllm && \
    cd /llm/vllm && \
    git apply /llm/vllm_for_multi_arc.patch && \
    pip install setuptools-scm==8.2.0 setuptools==78.1.0 && \
    pip install --upgrade cmake && \
    pip install -v -r requirements/xpu.txt && \
    VLLM_TARGET_DEVICE=xpu python setup.py install && \
    pip install intel-extension-for-pytorch==2.6.10+xpu --extra-index-url=https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/ && \
    pip uninstall -y oneccl oneccl-devel && \
    pip install intel-opencl-rt==2025.0.2 intel-openmp==2025.0.2 && \
    rm -rf /llm/vllm_for_multi_arc.patch && \
    pip install mpi4py fastapi uvicorn openai && \
    pip install ray numba && \
    # Link libze_loader.so
    cd /usr/lib/x86_64-linux-gnu && \
    ln -s libze_loader.so.1 libze_loader.so && \
    sed -i 's/qtype = ggml_tensor_qtype\["woq_int4"\]/qtype = ggml_tensor_qtype["sym_int4"]/' /usr/local/lib/python3.11/dist-packages/ipex_llm/transformers/convert.py

# Set environment variables to force use batch_forward_new()
ENV IPEX_LLM_FORCE_BATCH_FORWARD=1

WORKDIR /llm/
ENTRYPOINT ["bash", "/llm/start-vllm-service.sh"]
