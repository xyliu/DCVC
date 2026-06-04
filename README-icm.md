# DCVC运行环境配置

## 总结

在 `$USER` 用户下创建了 `py312_dcvcrt` conda 环境，安装了 PyTorch 2.12.0+cu130、项目依赖，并编译了两个 C++/CUDA 扩展（`MLCodec_extensions_cpp` 和 `inference_extensions_cuda`）。使用清华 TUNA 镜像加速 conda/pip。

## 详细复现步骤

```bash
# 1. 安装 Miniconda（如无）
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3 && rm /tmp/miniconda.sh

# 2. 配置 conda 清华镜像
cat > ~/.condarc << 'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# 3. 创建环境
~/miniconda3/bin/conda clean -i -y
~/miniconda3/bin/conda create -n py312_dcvcrt python=3.12 -y

# 4. 配置 pip 清华镜像
~/miniconda3/envs/py312_dcvcrt/bin/pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 5. 安装 PyTorch + CUDA 13.0（从官方源，耗时依赖网络）
export PATH=/usr/local/cuda/bin:$PATH
~/miniconda3/envs/py312_dcvcrt/bin/pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 6. 安装项目依赖
~/miniconda3/envs/py312_dcvcrt/bin/pip install -r requirements.txt

# 7. 安装系统编译工具（如无）
sudo apt-get install -y g++ ninja-build cmake

# 8. 编译 MLCodec_extensions_cpp
~/miniconda3/envs/py312_dcvcrt/bin/pip install --no-build-isolation ./src/cpp/
~/miniconda3/envs/py312_dcvcrt/bin/python -c "import MLCodec_extensions_cpp; print('OK')"

# 9. 编译 inference_extensions_cuda
cd src/layers/extensions/inference/
export PATH=/usr/local/cuda/bin:/usr/bin:$PATH
~/miniconda3/envs/py312_dcvcrt/bin/python setup.py build_ext
~/miniconda3/envs/py312_dcvcrt/bin/python setup.py install
~/miniconda3/envs/py312_dcvcrt/bin/python -c "import torch; import inference_extensions_cuda; print('OK')"

# 10. 验证
~/miniconda3/envs/py312_dcvcrt/bin/python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.device_count()}')
"
```

### 激活 & 运行

```bash
conda activate py312_dcvcrt
export PATH=/usr/local/cuda/bin:$PATH
bash test_video.sh
```

### 关键环境信息

- **系统 CUDA**: 13.1, **PyTorch**: 2.12.0+cu130 (cu130 wheel)
- `common.h` 中 `operator>` 的模板特化 patch 已应用（见 `install_c_extern.sh`）
- 清华镜像无 PyTorch cu130 pip wheel，故 PyTorch 从 `download.pytorch.org` 安装；其余包从清华镜像加速

---


