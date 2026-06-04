# DCVC-RT 测试程序的代码分析文档

## Goal

从 `test_video.sh` 入口出发，深入分析 DCVC-RT 的完整数据流：YUV 输入→编码→解码→质量评估的完整链路。

## 文件一览

| 文件 | 角色 |
|---|---|
| `test_video.sh` | 运行入口脚本 |
| `test_video.py` | 主控：加载配置、多进程分发、编码+解码+评估全流程 |
| `src/models/image_model.py` | 帧内编码模型 `DMCI`（I帧） |
| `src/models/video_model.py` | 帧间编码模型 `DMC`（P帧 + DPB + 特征适配） |
| `src/models/common_model.py` | 基类 `CompressionModel`：上下文熵编码（2x/4x 掩码分块编码） |
| `src/models/entropy_models.py` | 熵编码：`EntropyCoder`（rANS）、`BitEstimator`（z通道）、`GaussianEncoder`（y通道） |
| `src/layers/layers.py` | 网络基本单元：`DepthConvBlock`、`SubpelConv2x`、`ResidualBlockWithStride2`、`ResidualBlockUpsample` |
| `src/layers/cuda_inference.py` | CUDA 自定义算子封装 + PyTorch fallback |
| `src/utils/video_reader.py` | `YUV420Reader` / `PNGReader`：读取原始视频帧 |
| `src/utils/video_writer.py` | `YUV420Writer` / `PNGWriter`：写入重建视频 |
| `src/utils/transforms.py` | YCbCr ↔ RGB、YUV420 ↔ 444 颜色空间转换 |
| `src/utils/metrics.py` | PSNR、MS-SSIM 计算 |
| `src/utils/stream_helper.py` | 码流打包：SPS、I/P帧 NAL 单元读写 |
| `src/utils/common.py` | 辅助：状态加载、JSON日志输出 |

## 完整数据流

### 阶段一：视频加载与预处理

```
YUV文件 → YUV420Reader.read_one_frame()
  → y: [1,H,W] uint8, uv: [2,H/2,W/2] uint8
  → ycbcr420_to_444_np(y, uv)  # uv 最近邻上采样 2x
  → yuv_444: [3,H,W] float32
  → np_image_to_tensor(): /255 → [1,3,H,W] float32 → .half() → float16
```

### 阶段二：帧类型决策

```
frame_idx == 0 → I帧 (DMCI)
intra_period > 0 && frame_idx % intra_period == 0 → I帧
其余 → P帧 (DMC)

特殊逻辑：
  reset_interval: 每64帧的第1个P帧 → adaptor_i = 1（重置参考特征）
  fa_idx = index_map[frame_idx % 8]: 8帧循环 [0,1,0,2,0,2,0,2]
    → qp_shift[fa_idx] 实现QP细调（0/8/4）
```

### 阶段三：I帧编码（DMCI · `image_model.py`）

```
x_padded: [1,3,H_pad,W_pad]  → replicate_pad 对齐到16的倍数

① IntraEncoder:
   pixel_unshuffle(x, 8): [1,192,H/8,W/8]  (3×8×8=192)
   DepthConvBlock ×7 → Conv2d stride=2 → [1,256,H/16,W/16]

② q_scale_enc[qp]: 乘性量化步长嵌入编码器输出

③ HyperEncoder (z通道):
   DepthConvBlock → 2× ResidualBlockWithStride2 → [1,128,H/64,W/64]
   round_and_to_int8: 取整 + int8量化 → z_hat

④ HyperDecoder → y_prior_fusion → PriorParams

⑤ 4× 掩码分块上下文熵编码 (compress_prior_4x):
   将y在通道维分成4块，每块用棋盘格掩码(mask_0~3)逐步编解码
   每个子块依赖已重构的前序子块 → 空间上下文建模

⑥ IntraDecoder:
   13× DepthConvBlock → pixel_shuffle(8) → [1,3,H,W]
   clamp(0,1) → x_hat

⑦ 熵编码写码流:
   bit_estimator_z.encode_z(z_hat_write, qp)
   gaussian_encoder.encode_y(y_q_w_0~3, s_w_0~3)
   rANS → bit_stream
```

### 阶段四：P帧编码（DMC · `video_model.py`）

```
① 参考特征提取 (apply_feature_adaptor):
   第1个P帧(DPB帧为空): feature_adaptor_i 从重建帧提取
   后续P帧: feature_adaptor_p 从运动特征提取

② FeatureExtractor:
   DepthConvBlock×6 → ctx(时空上下文) + ctx_t(时间先验)

③ Encoder:
   pixel_unshuffle(x,8): [1,192,H/8,W/8]
   Conv2d(1×1) → cat(ctx) → DepthConvBlock×3 → ×q_encoder → Conv2d stride=2
   → y: [1,128,H/16,W/16]

④ HyperEncoder → round_to_int8 → z_hat

⑤ PriorFusion (res_prior_param_decoder):
   hyper_decoder(z_hat) + temporal_prior_encoder(ctx_t) → cat → PriorFusion
   → 含q_dec, scales, means

⑥ 2× 掩码分块熵编码 (compress_prior_2x):
   mask_0 → 编解码一半通道 → y_spatial_prior 条件预测
   mask_1 → 编解码另一半通道

⑦ Decoder:
   SubpelConv2x ↑2 → cat(ctx) → DepthConvBlock×3 → Conv2d(1×1)
   → feature: [1,256,H/8,W/8]

⑧ ReconGeneration:
   DepthConvBlock×4 → Conv2d → pixel_shuffle(8) → clamp(0,1)
   → x_hat: [1,3,H,W]

⑨ DPB更新: add_ref_frame(feature, None) 存入参考帧列表
```

### 阶段五：码流打包（`stream_helper.py`）

```
二进制格式:
  [NAL_SPS|NAL_I|NAL_P] × N帧
  SPS: nal_type(4b) + sps_id(4b) + height + width + ec_part + use_ada_i
  I/P帧: nal_type(4b) + sps_id(4b) + qp(8b) + stream_length(variable) + bit_stream

写入流程 (write_ip):
  flag = (NAL_I/NAL_P << 4) + sps_id
  write_uchars(flag) + write_uchars(qp) + write_uint_adaptive(length) + write_bytes(stream)
```

只复用了H.264/H.265概念层面的命名，语法完全是自定义的：

| 概念 | 本代码 | H.264/H.265 标准 |
|---|---|---|
| **NAL header** | 1字节：`nal_type(4b) + sps_id(4b)` | H.264: `forbidden(1) + ref_idc(2) + type(5)`；H.265: 2字节含 `layer_id(6) + tid(3)` |
| **SPS** | 仅存 resolution + 2个flag，读写即 `write_uint_adaptive` | 完备的 `profile/level/pic_parameter_set_id/chroma_format/...` 等上百个语法元素 |
| **熵编码** | 直接 raw bytes（rANS 已在模型内完成） | `Exp-Golomb` + `CABAC`/`CAVLC` 语法元素级编码 |
| **变长整数** | `write_uint_adaptive`：第1字节最高位=0 时1字节，否则2或4字节 | 标准用 `ue(v)/se(v)`（无符号/有符号指数哥伦布编码） |
| **RBSP/EBSP** | 无 | 标准有起始码 `0x000001` + 防竞争字节机制 |
| **PPS/VPS/SEI** | 无 | 标准有完备的参数集和辅助信息 |

核心区别：标准中 NAL 单元承载的是**宏块/CTU 级的预测残差、运动矢量等语法元素**；这里 NAL 单元只做 **rANS 码流的简单容器**——真正的压缩信息（概率分布、量化系数）都在 rANS 码流里，不由这个层解析。


### 阶段六：解码（反向流程）

```
① 读 SPS header → 解析 resolution / ec_part / use_ada_i

② I帧解码 (DMCI.decompress):
   bit_stream → set_stream
   decode_z → hyper_dec → prior_fusion → decompress_prior_4x → IntraDecoder → x_hat

③ P帧解码 (DMC.decompress):
   apply_feature_adaptor → FeatureExtractor.part1
   decode_z → res_prior_param_decoder → decompress_prior_2x_part1
   FeatureExtractor.part2 → decompress_prior_2x_part2
   Decoder → ReconGeneration → x_hat
   add_ref_frame(feature, x_hat)
```

### 阶段七：质量评估

```
x_hat (= recon) : [1,3,H,W] float16 (0~1)
→ 裁剪到原始分辨率(去掉padding)
→ yuv_444_to_420(x_hat):
    y取第1通道, uv用 avg_pool2d(k=2) 下采样
→ clamp(*255, 0, 255).squeeze(0).cpu().numpy()
→ y: [H,W], u/v: [H/2,W/2]

PSNR计算 (calc_psnr):
  MSE = mean((original - reconstructed)^2)
  PSNR = 10 * log10(255^2 / MSE)

YUV420加权PSNR:
  PSNR = (6*PSNR_Y + PSNR_U + PSNR_V) / 8

MS-SSIM: 多尺度结构相似性, 5级金字塔
  (Y平面, U/V分别计算, 同样加权平均)

每帧bit数: stream_bytes * 8 + sps_bytes * 8
输出JSON: 含ave_all_frame_bpp / psnr / msssim, 分I帧H.264/H.265/P帧统计
```

### 阶段八：速度测量

```
encoding_time / decoding_time = 每帧CUDA同步后计时
bypass前10帧(warmup)后计算平均值
输出: avg_frame_encoding_time / avg_frame_decoding_time (秒)
```

## 核心压缩原理

1. **超先验 (Hyperprior)**: z通道作为超先验，指导y的分布估计
2. **上下文熵编码 (Checkboard context model)**: 2x/4x 棋盘格掩码实现并行因果上下文建模
3. **rANS 熵编码**: 基于 CDF 表的非对称数值系统编解码
4. **特征互参考 (Temporal Feature Memory)**: P帧编码器输出特征存入DPB，供后续帧参考（而非传统运动估计+补偿）
5. **QP控制**: 64级量化参数 + 8/4的帧类型偏移微调，实现码率连续可调
6. **深度可分离卷积**: DepthConv + Pointwise + WSiLU 实现高效特征变换
---

## 量化精度问题

核心策略是网络推理用 float16（GPU），精度敏感操作回退 float32 或 CPU：
1. rANS 熵编码全程走 CPU
   - entropy_models.py:48,51,64,67 — 编解码时符号/CDF表全部 .cpu().numpy() 传给 C++ 扩展，不在 GPU 上处理
   - set_cdf_info() — CDF 表存为 CPU numpy，float16 的精度不足以表示 CDF 累积概率
2. CDF 表计算时显式升 float32
   - entropy_models.py:195 — self.forward(maxima.to(torch.float32), index)，BitEstimator 前向强制 float32 防精度丢失
3. force_zero_thres 机制
   - --force_zero_thres 0.12：当高斯 scale ≤ 阈值时，量化后的 symbol 直接强制归零，跳过熵编码。避免极小 scale 下 CDF 查表精度不足导致的码率浪费和画质损失
4. 模型整体 .half()
   - test_video.py:404,414 — i_frame_net.half()，全部网络层转 float16 推理，换来速度提升
5. CUDA 算子 vs PyTorch fallback
   - cuda_inference.py — 提供 round_and_to_int8、process_with_mask、build_index 等 CUDA 自定义算子；若编译失败则 fallback 到纯 PyTorch 实现

对比 `DCVC-FM` 还有更明显的例子：flow_warp 中 grid_sample 先升 float32 再降回 float16，因为 warp 在 float16 下会产生 NaN。`DCVC-RT` 没有光流模块，所以这一条不适用。