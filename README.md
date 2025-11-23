# CUDA WSL Installer

Automated CUDA installation scripts for Windows 11 developers using WSL2. The
repository detects the GPU’s compute capability from inside WSL and installs
the matching CUDA toolkit, samples, and developer prerequisites.

## Why this repo?

Upgrading between CUDA major versions on WSL often requires uninstalling and
reinstalling multiple packages, repos, and samples. These scripts encapsulate
known-good flows for two scenarios:

| Hardware profile                          | Compute capability | CUDA track |
|------------------------------------------|--------------------|------------|
| Pascal / early Turing (e.g., GTX 1080 Ti) | < 7.0              | 12.5       |
| Recent Turing, Ampere, Ada, Blackwell     | ≥ 7.5              | 13.0       |

The script auto-detects the compute capability via `nvidia-smi`. You can also
force a track via CLI flags when testing.

## What is CUDA & why run it inside WSL2?

[CUDA](https://developer.nvidia.com/cuda-zone) is NVIDIA’s parallel computing
platform that exposes the GPU for general-purpose workloads. When combined with
WSL2 you can:

1. **Prototype in Linux without leaving Windows** – run the exact CLI tooling,
   package managers, and build chains that your Linux servers use.
2. **Accelerate AI/ML frameworks** – libraries like [PyTorch](https://pytorch.org/),
   [TensorFlow](https://www.tensorflow.org/install/pip#windows_setup), and
   [JAX](https://jax.readthedocs.io/en/latest/) automatically leverage CUDA for
   GPU-backed training/inference.
3. **Speed up data engineering & analytics** – [RAPIDS](https://rapids.ai/)
   provides GPU-accelerated pandas/cuDF, cuML, and cuGraph pipelines.
4. **Enable simulation, rendering, and HPC codes** – e.g., [Blender Cycles](https://www.blender.org/),
   [LAMMPS](https://www.lammps.org/), or in-house CUDA kernels.
5. **Run modern GenAI tooling locally** – Stable Diffusion pipelines, LLM
   fine-tuning, and other CUDA-dependent projects work seamlessly when WSL has
   GPU access.

Installing CUDA in WSL means your Windows laptops/desktops act like Linux CUDA
workstations without dual-booting, while still sharing the same NVIDIA driver
stack maintained on Windows.

## Requirements

* Windows 11 with WSL2 (Ubuntu 22.04/24.04) already configured
* Latest NVIDIA Windows driver with WSL GPU support
* WSL distro must have `sudo` privileges and network access

## Quick start

```bash
git clone https://github.com/<your-org>/cuda-wsl-installer.git
cd cuda-wsl-installer
bash scripts/install_cuda.sh
```

The installer will:

1. Verify it is running under WSL and that `nvidia-smi` works
2. Detect the GPU’s compute capability (major.minor)
3. Choose CUDA 12.5 or CUDA 13.0 based on that capability (unless overridden)
4. Install/refresh the NVIDIA apt repository (adds `cuda-keyring` if missing)
5. Remove conflicting CUDA packages, then install the target toolkit meta-package
6. Set `/usr/local/cuda` via `update-alternatives`
7. Ensure your `~/.bashrc` exports `/usr/local/cuda/bin` and `lib64`
8. Clone the matching `cuda-samples` tag, build them, and run `deviceQuery`

### Command-line options

* `--force-track {12.5|13.0}` – bypass hardware detection (useful for testing)
* `--skip-samples` – install the toolkit only (skips cloning/building samples)
* `--dry-run` – print the plan without mutating the system

## Verification

After the script finishes, you can re-run the sample check at any time:

```bash
/usr/local/cuda/samples/bin/x86_64/linux/release/deviceQuery
```

A final line of `Result = PASS` confirms that CUDA sees your GPU from WSL.

### Sample output (GTX 1080 Ti on CUDA 12.5 track)

Below is the expected `deviceQuery` output on a Pascal card that routes to the
12.5 toolchain. Use it as a reference to confirm your installation matches:

```
Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce GTX 1080 Ti"
  CUDA Driver Version / Runtime Version          13.0 / 12.5
  CUDA Capability Major/Minor version number:    6.1
  Total amount of global memory:                 11264 MBytes (11811028992 bytes)
  (028) Multiprocessors, (128) CUDA Cores/MP:    3584 CUDA Cores
  GPU Max Clock rate:                            1582 MHz (1.58 GHz)
  Memory Clock rate:                             5505 Mhz
  Memory Bus Width:                              352-bit
  L2 Cache Size:                                 2883584 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        98304 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 13.0, CUDA Runtime Version = 12.5, NumDevs = 1
Result = PASS
```

## Benchmark ideas (before vs. after)

Give teammates a way to quantify the benefit of enabling CUDA in WSL. Run the
GPU version first, then repeat with `CUDA_VISIBLE_DEVICES=` (empty) or
`CUDA_VISIBLE_DEVICES=-1` to force CPU-only execution.

1. **CUDA Samples benchmarks** – great for quick sanity/perf checks using NVIDIA’s reference kernels (GPU-only)
   ```bash
   # N-body simulation
   /usr/local/cuda/samples/bin/x86_64/linux/release/nbody -benchmark -numbodies=15360

   # Matrix multiply throughput
   /usr/local/cuda/samples/bin/x86_64/linux/release/matrixMul
   ```

2. **PyTorch micro-benchmark** – stresses dense linear algebra (train/infer core) and highlights GPU matmul gains (install once: `pip install torch torchvision`)
   ```python
   import torch, time

   x = torch.randn(4096, 4096, device="cuda")
   torch.cuda.synchronize()
   t0 = time.time(); _ = x @ x; torch.cuda.synchronize()
   print("GPU matmul:", time.time() - t0)

   y = x.cpu()
   t0 = time.time(); _ = y @ y
   print("CPU matmul:", time.time() - t0)
   ```

3. **TensorFlow CNN snippet** – trains a tiny MNIST CNN to see end-to-end training speed differences (install: `pip install tensorflow-cpu tensorflow`)
   ```python
   import tensorflow as tf

   (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
   x_train = x_train[..., None]/255.0
   model = tf.keras.Sequential([
       tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10)
   ])
   model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
   model.fit(x_train, y_train, epochs=1, batch_size=256)
   ```
   Run once normally, then re-run with `CUDA_VISIBLE_DEVICES=-1` to show the CPU
   slowdown.

4. **RAPIDS cuDF vs pandas** – GPU-accelerated analytics workflow to compare ETL/groupby latency against CPU pandas (install: `pip install cudf-cu12 dask-cudf --extra-index-url=https://pypi.nvidia.com`)
   ```python
   import cudf, numpy as np
   df = cudf.DataFrame({"a": np.random.randint(0, 1000, 5_000_000), "b": np.random.rand(5_000_000)})
   %time df.groupby("a").b.mean()
   ```
   Switch to pandas for the CPU baseline.

5. **Monitor utilization while benchmarking** – watching `nvidia-smi` confirms workloads are actually engaging the GPU instead of silently falling back to CPU
   ```bash
   watch -n 1 nvidia-smi
   ```
   GPU utilization/memory rising during the test demonstrates the CUDA path is
   active.

## Troubleshooting

* **`nvidia-smi` missing:** Install/repair the NVIDIA Windows driver, then
  restart WSL (`wsl --shutdown`).
* **APT failures:** Ensure `sudo apt-get update` works independently and that
  your distro has outbound HTTPS access.
* **Custom GPU thresholds:** Edit `scripts/install_cuda.sh` to adjust the
  capability cutoff or add new tracks (e.g., future CUDA versions).

## Next steps

* Extend the matrix for additional GPU tiers (Hopper, etc.)
* Add CI smoke tests using WSL containers once available
* Package the script as a deb/rpm for centralized IT deployments
