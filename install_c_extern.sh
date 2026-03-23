#/bin/bash

# Create python env
#conda create -n py312_dcvcrt python=3.12
#conda activate py312_dcvcrt
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
#pip install -r requirements.txt

conda activate py312_dcvcrt

# 确保 nvcc 在 PATH 中
export PATH=/usr/local/cuda/bin:$PATH

DCVC_ROOT=~/github/DCVC
# ─── Step 1: 安装 MLCodec_extensions_cpp ───────────────────────────────────
pip install pybind11
cd "$DCVC_ROOT/src/cpp/"
pip install --no-build-isolation .
python -c "import MLCodec_extensions_cpp; print('MLCodec_extensions_cpp OK')"

# ─── Step 2: 修复 common.h 并安装 inference_extensions_cuda ────────────────
# 注意：此修复已应用到 common.h，如果是全新 clone 的仓库，
#       需要先应用下面的 patch：
#
#cd "$DCVC_ROOT/src/layers/extensions/inference/"
#patch -p0 << 'EOF'
#--- a/common.h
#+++ b/common.h
#@@ -271,5 +271,7 @@
# template <typename T1, typename T2>
#-__forceinline__ __device__ bool4 operator>(const T1& a, const T2& b)
#+__forceinline__ __device__
#+typename std::enable_if<std::is_same<T1, float4>::value || std::is_same<T1, Half4>::value, bool4>::type
#+operator>(const T1& a, const T2& b)
# {
#     return make_vec4(a.x > b, a.y > b, a.z > b, a.w > b);
# }
#EOF

cd "$DCVC_ROOT/src/layers/extensions/inference/"
python setup.py build_ext   # 编译（耗时约 1~2 分钟）
python setup.py install     # 安装到 site-packages
python -c "import torch; import inference_extensions_cuda; print('inference_extensions_cuda OK')"

echo "All extensions installed successfully!"
