# Image model guide

This app tests **image classification inference**: a model receives RGB image
tensors and returns one score per class. It measures execution and scheduling.
The procedural images have no correct labels, so no accuracy score is produced.

## Choose a model

The supported families are torchvision's `mobilenet_v3_small`, `resnet18`,
`efficientnet_b0`, and `squeezenet1_1`. A family is the network's structure;
weights are the learned numbers loaded into that structure. Selecting ResNet18
does not make a MobileNet checkpoint compatible with it.

| Model family | Official weights selected by this app | Approximate download |
| --- | --- | --- |
| MobileNetV3-Small | `MobileNet_V3_Small_Weights.IMAGENET1K_V1` | 10 MB |
| ResNet18 | `ResNet18_Weights.IMAGENET1K_V1` | 45 MB |
| EfficientNet-B0 | `EfficientNet_B0_Weights.IMAGENET1K_V1` | 21 MB |
| SqueezeNet1.1 | `SqueezeNet1_1_Weights.IMAGENET1K_V1` | 5 MB |

Official weights use 1,000 classes. Start at 224 pixels square and batch 2 on
CPU. Choose a family and **Fetch model** to download its official weights
explicitly. Later runs reuse the cache without downloading. Model files are
excluded from source and application archives.

The app accepts square image sizes from 64 through 512 pixels, class counts
from 1 through 10,000, and batches up to 16. Official weights require 1,000
classes; another class count requires a matching custom checkpoint. Keep a
small batch/image size until you know the model fits your RAM/VRAM. Checkpoint
imports are capped at 256 MiB; these bounds limit this lab's scope rather than
claiming that all combinations fit every computer.

The architecture, weight enum and preprocessing references come from the
[versioned torchvision model documentation](https://docs.pytorch.org/vision/0.27/models.html).
These are four specific supported families; support for every image model is
not implied. The original MobileNet experiment remains available.

## Import your own weights

1. Know which of the four supported families your model uses and how many
   output classes it has. Ask the checkpoint's author if that information is
   missing. A filename alone does not establish the architecture.
2. Select the matching image model architecture, choose **Imported weights**
   as the weights source, and set the output class count and square image size.
3. Choose **Import weights...** and select a trusted `.pth` or `.pt` checkpoint.
   This is a local import: the file is copied into this app's cache, not sent
   to a website or another person.
4. Run **Device test** first. It loads and validates the model and measures
   batch execution without simulated arrivals. Then use **Run** to compare
   FIFO and deadline batching under traffic.
5. Save the report. Record the family, image size, class count, file SHA256 and
   actual CPU/CUDA device when comparing results.

The accepted file contains a PyTorch **state dictionary**: named tensors saved
from a compatible model. A model owner can export their already trained model:

```python
import torch

# trained_model must already be one of the supported torchvision families.
torch.save(trained_model.state_dict(), "my_image_classifier.pth")
```

A single `state_dict` or `model_state_dict` wrapper is also accepted. Full
training checkpoint dictionaries containing optimizer/epoch metadata are outside
this contract; export just the model state. Tensor names must match exactly,
including any prefixes. Quantized/sparse tensors and non-finite parameter values
are rejected. Parameter and buffer dtypes must match the selected architecture:
float32 learned weights and the expected integer buffers. Half-precision,
float64 and complex checkpoints are rejected rather than silently converted.
The owner of a compatible floating-point trained model can convert it to float32
and export with `torch.save(trained_model.float().state_dict(), path)`. This does not
convert a quantized model into a supported architecture. The current execution
path uses float32 inference.

The app creates the selected architecture locally, then requires matching
learned parameters. It never uses the architecture's temporary initial values
as a substitute for missing weights. A bad file or mismatch produces an error.
Changing the class count does not retrain or adapt a checkpoint.

Some older official BatchNorm checkpoints omit the integer training-step
counter `num_batches_tracked`. PyTorch's versioned state-dictionary compatibility
rule may supply that counter as zero; it does not fill missing learned weights
and does not affect eval-mode inference. Other missing, extra, wrong-shaped or
wrong-dtype parameters are errors. Checkpoint validation also rejects non-finite
values before and after loading.

Selecting **Official pretrained** uses 1,000 classes. Changing architecture
resets the previous import and restores 224 pixels/1,000 classes, so weights for
one structure are not silently reused for another. A selected imported path and
hash identify the active cached file. **System check** describes the installed
runtime; it does not run a model performance test. Use **Device test** for that.

Supported custom models keep the same torchvision architecture, three RGB input
channels, and a single class-score output. Modifying the network's internal
layers, width or input channels needs a new adapter in the source. Default
ImageNet-style normalization and procedural inputs are an execution fixture;
they cannot establish accuracy for a custom model's training distribution.

ONNX, TensorFlow/Keras, TensorRT engines, TorchScript programs, arbitrary Python
model code, text/audio models, image generators, object detectors and segmentation
models are outside this version's import contract. Renaming a file to `.pth`
does not convert it. See [SECURITY.md](../SECURITY.md) before importing weights.

## Read a device test

A device test is a small repeatable measurement for the **selected model and
device**. It runs warmup first, then repeats batch 1 and the selected maximum
batch. Results include measured forward/service timing, variability, images per
second, parameter count, checkpoint identity and available memory measurements.

Stop is cooperative: it is checked between preparation/warmup forwards and
measured samples. An already running CPU call or CUDA kernel returns safely
before cancellation completes. A cancelled profile preserves its settings,
cancellation state and any measured sample counts, including zero samples.

CUDA device timing uses synchronized CUDA events. Host service timing includes
the surrounding preparation/transfers/prediction work; the report labels its
scope. CPU forward time is measured by a host clock. A larger batch can process
more images per second while taking longer to return one batch.

Parameter count describes the model's size. It is not a measure of accuracy.
Model throughput is workload-specific; it is not a universal CPU/GPU power
score, a FLOPS rating, a temperature/power measurement, or a hardware stress
certification. Compare identical model weights, image size, batch, backend and
timing scope, and retain repeat variability. A short test is not a sustained
capacity guarantee.

The separate traffic experiment adds arrival timing, queueing, deadlines and
discarded work. Its end-to-end latency can be higher than a device-test forward
time because requests also wait before model execution.

## Command-line examples

Use the project's interpreter from its folder; no global activation is needed:

```powershell
.\.venv\Scripts\python.exe -m deadline_engine fetch-weights --model resnet18
.\.venv\Scripts\python.exe -m deadline_engine models
.\.venv\Scripts\python.exe -m deadline_engine profile --device cpu --model resnet18 --batch 2 --samples 10 --output results/resnet18-device
.\.venv\Scripts\python.exe -m deadline_engine demo --device cpu --model resnet18 --duration 1 --rate 20 --deadline 1000 --batch 2 --output results/resnet18-traffic
.\.venv\Scripts\python.exe -m deadline_engine import-weights 'C:\My Models\classifier.pth' --model resnet18 --classes 10 --image-size 224
.\.venv\Scripts\python.exe -m deadline_engine profile --device cpu --model resnet18 --weights 'C:\My Models\classifier.pth' --classes 10 --image-size 224 --batch 2 --samples 10 --output results/custom-device
```

`profile` writes `profile.json` and `profile.csv` inside the chosen output
folder. Its default is 10 measured samples per batch, bounded to 3–100. The
command-line `--weights` path is validated directly; `import-weights` makes a
managed cached copy and prints its path/identity. Pass that returned path for a
run independent of the original file's location.

On Linux/macOS, use `./.venv/bin/python` instead. Choose `--device cuda` only
with working NVIDIA/CUDA dependencies. The portable Windows CPU executable
continues to use CPU even on a computer that has an NVIDIA GPU.

## Provenance and permissions

Official checkpoints come from URLs in the pinned torchvision release, with
their identities recorded in the exported report. Download checks catch damage
or unexpected content; a SHA256 does not establish ownership or permission to
use someone else's custom weights. Custom uploads are not automatically called
pretrained, accurate or licensed. Keep the author's documentation and license.

torchvision cautions that pretrained models may carry terms derived from their
training datasets. Review the upstream terms for your use. This project's MIT
license covers its own code, not all imported weights. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
