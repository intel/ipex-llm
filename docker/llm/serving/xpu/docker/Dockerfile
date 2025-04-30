# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# First stage: build oneccl
FROM intel/oneapi-basekit:2025.0.1-0-devel-ubuntu22.04 AS build

ARG http_proxy
ARG https_proxy
ARG PIP_NO_CACHE_DIR=false

# Set environment variables
ENV TZ=Asia/Shanghai PYTHONUNBUFFERED=1

# Copy patch file and benchmark scripts relative to project root
COPY docker/llm/serving/xpu/docker/ccl_torch.patch /tmp/
COPY docker/llm/serving/xpu/docker/vllm_online_benchmark.py \
     docker/llm/serving/xpu/docker/vllm_offline_inference.py \
     docker/llm/serving/xpu/docker/vllm_offline_inference_vision_language.py \
     docker/llm/serving/xpu/docker/payload-1024.lua \
     docker/llm/serving/xpu/docker/start-vllm-service.sh \
     docker/llm/serving/xpu/docker/benchmark_vllm_throughput.py \
     docker/llm/serving/xpu/docker/benchmark_vllm_latency.py \
     docker/llm/serving/xpu/docker/start-pp_serving-service.sh \
     /llm/
COPY docker/llm/serving/xpu/docker/1ccl_for_multi_arc.patch /build/

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
    apt-get install -y --no-install-recommends python3.11 python3-pip python3.11-dev python3.11-distutils python3-wheel && \
    rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    #
    # Install pip and essential Python packages
    wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3 get-pip.py && rm get-pip.py

# Install Intel GPU OpenCL Driver and Compute Runtime
RUN set -eux && \
    mkdir -p /tmp/neo && \
    cd /tmp/neo && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu-dbgsym_1.6.32224.5_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd-dbgsym_24.52.32224.5_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb && \
    dpkg -i *.deb
# Install Intel PyTorch extension for LLM inference
RUN set -eux && \
    pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/xpu && \
    pip install intel-extension-for-pytorch==2.6.10+xpu --extra-index-url=https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/

# Build torch-ccl
RUN set -eux && \
    mkdir -p /build && \
    cd /build && \
    git clone --depth 1 --branch ccl_torch2.6.0+xpu https://github.com/intel/torch-ccl.git torch-ccl && \
    cd torch-ccl && \
    git submodule sync && \
    git submodule update --init --recursive --depth 1 && \
    # This patch will enable build torch-ccl with pytorch 2.6 environment
    # git apply /tmp/ccl_torch.patch && \
    USE_SYSTEM_ONECCL=ON COMPUTE_BACKEND=dpcpp python setup.py bdist_wheel
    # File path: /build/torch-ccl/dist/oneccl_bind_pt-2.6.0+xpu-cp311-cp311-linux_x86_64.whl

RUN set -eux && \
    # Build oneCCL
    pip install ninja && \
    cd /build/ && \
    git clone https://github.com/analytics-zoo/oneCCL.git oneCCL && \
    cd oneCCL && \
    git checkout 3afa1bb7936f57683a2503c34b29c0daca6a9ccb && \
    git apply /build/1ccl_for_multi_arc.patch && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS="-fsycl" -DCOMPUTE_BACKEND=dpcpp  -DCMAKE_BUILD_TYPE=MinSizeRel && \
    # File path: /build/oneCCL/build/src/libccl.so.1.0
    ninja


# Second stage: Final runtime image
FROM intel/oneapi-basekit:2025.0.1-0-devel-ubuntu22.04

# Copy the built torch-ccl package from the build stage
COPY --from=build /build/torch-ccl/dist/oneccl_bind_pt-2.6.0+xpu-cp311-cp311-linux_x86_64.whl /opt/
COPY --from=build /llm/ /llm/
COPY --from=build /build/oneCCL/build/src/libccl.so.1.0 /opt/intel/1ccl-wks/lib/
COPY --from=build /build/oneCCL/build/src/libccl.so.1 /opt/intel/1ccl-wks/lib/
COPY --from=build /build/oneCCL/build/src/libccl.so /opt/intel/1ccl-wks/lib/

# Copy patch files and scripts relative to project root
COPY docker/llm/serving/xpu/docker/vllm_for_multi_arc.patch /llm/
COPY docker/llm/serving/xpu/docker/setvars.sh /opt/intel/1ccl-wks/

# Copy required code from the build context (project root) instead of cloning
# Copy the main ipex-llm source code
COPY python/llm /llm/ipex-llm-src
COPY python/llm/dev/benchmark/ /llm/benchmark/
COPY python/llm/example/GPU/HuggingFace/LLM /llm/examples/
COPY python/llm/example/GPU/vLLM-Serving/ /llm/vLLM-Serving/
# Create serving directories and copy scripts early
RUN mkdir -p /llm/pp_serving && \
    mkdir -p /llm/lightweight_serving
COPY python/llm/example/GPU/Pipeline-Parallel-Serving/*.py /llm/pp_serving/
COPY python/llm/example/GPU/Lightweight-Serving/*.py /llm/lightweight_serving/

ARG http_proxy
ARG https_proxy
ARG PIP_NO_CACHE_DIR=false

# Set environment variables
ENV TZ=Asia/Shanghai PYTHONUNBUFFERED=1 VLLM_RPC_TIMEOUT=100000

# Update and install basic dependencies, upgrade linux-libc-dev to fix CT7 CVEs
RUN set -eux && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      linux-libc-dev \
      curl wget git libunwind8-dev vim less gnupg gpg-agent software-properties-common \
      libfabric-dev wrk libaio-dev numactl

# Set timezone
RUN set -eux && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# Install Python 3.11
RUN set -eux && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3-pip python3.11-dev python3.11-distutils python3-wheel && \
    rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python

# Install pip
RUN set -eux && \
    wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3 get-pip.py && rm get-pip.py

# Install essential Python packages
RUN set -eux && \
    pip install --upgrade requests argparse urllib3

# Install ipex-llm from the copied local source directory, using the extra index for dependencies
RUN set -eux && \
    pip install /llm/ipex-llm-src[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/xpu

# Install other Python packages
RUN set -eux && \
    pip install transformers_stream_generator einops tiktoken && \
    pip install --upgrade colorama

# Install vllm dependencies
RUN set -eux && \
    pip install --upgrade fastapi && \
    pip install --upgrade "uvicorn[standard]"

# Install Intel Extension for PyTorch
RUN set -eux && \
    pip install intel-extension-for-pytorch==2.6.10+xpu --extra-index-url=https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/

# Install torch-ccl
RUN set -eux && \
    pip install /opt/oneccl_bind_pt-2.6.0+xpu-cp311-cp311-linux_x86_64.whl

# Install system dependencies for vLLM/CCL
RUN set -eux && \
    apt-get update && \
    apt-get install -y --no-install-recommends libfabric-dev wrk libaio-dev numactl

# Remove potentially conflicting packages
RUN set -eux && \
    apt-get remove -y libze-dev libze-intel-gpu1 || true # Allow failure if packages are not installed

# Install compute runtime
RUN set -eux && \
    mkdir -p /tmp/neo && \
    cd /tmp/neo && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu-dbgsym_1.6.32224.5_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd-dbgsym_24.52.32224.5_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb && \
    dpkg -i *.deb && rm -rf /tmp/neo

# Clone, patch, and install vllm
RUN set -eux && \
    mkdir -p /llm && \
    cd /llm && \
    git clone --depth 1 -b v0.8.3 https://github.com/vllm-project/vllm /llm/vllm && \
    cd /llm/vllm && \
    git apply /llm/vllm_for_multi_arc.patch && \
    pip install setuptools-scm==8.2.0 setuptools==78.1.0 && \
    pip install --upgrade cmake && \
    pip install -v -r requirements/xpu.txt && \
    VLLM_TARGET_DEVICE=xpu python setup.py install && \
    # Reinstall IPEX to ensure compatibility after vLLM install
    pip install intel-extension-for-pytorch==2.6.10+xpu --extra-index-url=https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/ && \
    pip uninstall -y oneccl oneccl-devel && \
    rm -rf /llm/vllm_for_multi_arc.patch

# Install final serving/MPI dependencies
RUN set -eux && \
    pip install mpi4py fastapi uvicorn openai && \
    pip install ray numba


WORKDIR /llm/
ENTRYPOINT ["bash", "/llm/start-vllm-service.sh"]
