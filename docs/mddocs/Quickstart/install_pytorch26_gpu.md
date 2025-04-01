# Install IPEX-LLM on Intel GPU with PyTorch 2.6

This guide demonstrates how to install IPEX-LLM on Intel GPUs with PyTorch 2.6 support.

IPEX-LLM with PyTorch 2.6 provides a simpler prerequisites setup process, without requiring manual installation of oneAPI. Besides, it offers broader platform support with AOT (Ahead of Time) Compilation.

> [!TIP]
> For details on which device IPEX-LLM PyTorch 2.6 supports with AOT compilation, you could refer to here ([Windows](https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.6.10%2Bxpu&os=windows&package=pip#:~:text=following%20system%20requirements%3A-,1.1.%20Hardware,-Supported%20by%20prebuilt) or [Linux](https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.6.10%2Bxpu&os=linux%2Fwsl2&package=pip#:~:text=following%20system%20requirements%3A-,1.1.%20Hardware,-Supported%20by%20prebuilt)) for more information.

## Table of Contents
- [Windows Quickstart](#windows-quickstart)
  - [Install Prerequisites](#install-prerequisites)
  - [Install `ipex-llm`](#install-ipex-llm)
  - [Runtime Configurations](#runtime-configurations)
  - [Verify Installation](#verify-installation)
- [Linux Quickstart](#linux-quickstart)
  - [Install Prerequisites](#install-prerequisites-1)
  - [Install `ipex-llm`](#install-ipex-llm-1)
  - [Runtime Configurations](#runtime-configurations-1)
  - [Verify Installation](#verify-installation-1)

## Windows Quickstart

### Install Prerequisites

#### Update GPU Driver

We recommend updating your GPU driver to the [latest](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html). A system reboot is necessary to apply the changes after the installation is complete.

#### Setup Python Environment

Visit [Miniforge installation page](https://conda-forge.org/download/), download the **Miniforge installer for Windows**, and follow the instructions to complete the installation.

<div align="center">
<img src="https://llm-assets.readthedocs.io/en/latest/_images/quickstart_windows_gpu_miniforge_download.png"  width=80%/>
</div>

After installation, open the **Miniforge Prompt**, create a new python environment `llm-pt26`:
```cmd
conda create -n llm-pt26 python=3.11
```
Activate the newly created environment `llm-pt26`:
```cmd
conda activate llm-pt26
```

### Install `ipex-llm`

With the `llm-pt26` environment active, use `pip` to install `ipex-llm` for GPU:

- For **Intel Core™ Ultra Processors (Series 2) with processor number 2xxH (code name Arrow Lake)**:

  Choose either US or CN website for `extra-index-url`:

  - For **US**:

    ```cmd
    pip install --pre --upgrade ipex-llm[xpu_2.6_arl] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/arl/us/
    ```

  - For **CN**:

    ```cmd
    pip install --pre --upgrade ipex-llm[xpu_2.6_arl] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/arl/cn/
    ```

> [!TIP]
> For other Intel Core™ Ultra Processors, such as 2xxHX, please refer to the installation instruction below (i.e. for **other Intel iGPU and dGPU**).

- For **other Intel iGPU and dGPU**:

   ```cmd
   pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/xpu
   ```

### Runtime Configurations

For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.

With the `llm-pt26` environment active:

- For **Intel Arc™ A-Series GPU (code name Alchemist)**

  ```cmd
  set SYCL_CACHE_PERSISTENT=1
  set UR_L0_USE_IMMEDIATE_COMMANDLISTS=0
  ```

> [!TIP]
> It is recommanded to experiment with `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` or `1` for best performance on Intel Arc™ A-Series GPU.

- For **other Intel iGPU and dGPU**:

  ```cmd
  set SYCL_CACHE_PERSISTENT=1
  :: [optional] The following environment variable may improve performance, but in some cases, it may also lead to performance degradation
  set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  ```

> [!NOTE]
> The environment variable `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` determines the usage of immediate command lists for task submission to the GPU. It is highly recommanded to experiment with `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` or `0` on your device for best performance.
>
> You could refer to [here](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html) regarding more information about Level Zero Immediate Command Lists.

### Verify Installation

You can verify if `ipex-llm` is successfully installed following below steps:

- Open the **Miniforge Prompt** and activate the Python environment `llm-pt26` you previously created:

  ```cmd
  conda activate llm-pt26
  ```

- Set environment variables according to the [Runtime Configurations section](#runtime-configurations).

- Launch the Python interactive shell by typing `python` in the Miniforge Prompt window and then press Enter.

- Copy following code to Miniforge Prompt **line by line** and press Enter **after copying each line**.

  ```python
  import torch
  from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
  tensor_1 = torch.randn(1, 1, 40, 128).to('xpu')
  tensor_2 = torch.randn(1, 1, 128, 40).to('xpu')
  print(torch.matmul(tensor_1, tensor_2).size())
  ```

  It should output following content at the end:

  ```
  torch.Size([1, 1, 40, 40])
  ```

- To exit the Python interactive shell, simply press Ctrl+Z then press Enter (or input `exit()` then press Enter).


## Linux Quickstart

### Install Prerequisites

#### Install GPU Driver

We recommend following [Intel client GPU driver installation guide](https://dgpu-docs.intel.com/driver/client/overview.html) to install your GPU driver.

#### Setup Python Environment
 
Download and install the Miniforge as follows if you don't have conda installed on your machine:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
source ~/.bashrc
```

You can use `conda --version` to verify you conda installation.

After installation, create a new python environment `llm-pt26`:
```bash
conda create -n llm-pt26 python=3.11
```
Activate the newly created environment `llm-pt26`:
```bash
conda activate llm-pt26
```

### Install `ipex-llm`

With the `llm-pt26` environment active, use `pip` to install `ipex-llm` for GPU:

```bash
pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/xpu
```

### Runtime Configurations

For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.

With the `llm-pt26` environment active:

```bash
unset OCL_ICD_VENDORS
export SYCL_CACHE_PERSISTENT=1
# [optional] The following environment variable may improve performance, but in some cases, it may also lead to performance degradation
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

> [!NOTE]
> The environment variable `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` determines the usage of immediate command lists for task submission to the GPU. It is highly recommanded to experiment with `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` or `0` on your device for best performance.
>
> You could refer to [here](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html) regarding more information about Level Zero Immediate Command Lists.

### Verify Installation

You can verify if `ipex-llm` is successfully installed following below steps:

- Activate the Python environment `llm-pt26` you previously created:

  ```cmd
  conda activate llm-pt26
  ```

- Set environment variables according to the [Runtime Configurations section](#runtime-configurations-1).

- Launch the Python interactive shell by typing `python` in the terminal and then press Enter.

- Copy following code to Miniforge Prompt **line by line** and press Enter **after copying each line**.

  ```python
  import torch
  from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
  tensor_1 = torch.randn(1, 1, 40, 128).to('xpu')
  tensor_2 = torch.randn(1, 1, 128, 40).to('xpu')
  print(torch.matmul(tensor_1, tensor_2).size())
  ```

  It should output following content at the end:

  ```
  torch.Size([1, 1, 40, 40])
  ```

- To exit the Python interactive shell, simply press Ctrl+C then press Enter (or input `exit()` then press Enter).
