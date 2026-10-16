# Third-party software and model notice

The project's own scheduling engine, desktop interface, scripts, and documentation
are under the MIT license in `LICENSE`. That license does not replace the terms
for Python, PyTorch, pretrained weights, NVIDIA runtime components, or other
dependencies. No affiliation with or endorsement by these upstream projects is
claimed.

## Libraries and runtime

| Component | Used for | Upstream terms |
| --- | --- | --- |
| CPython | Python runtime and standard library | [Python license](https://docs.python.org/3.14/license.html) |
| Tcl/Tk | Native desktop widgets through `tkinter` | [Tcl/Tk license](https://www.tcl-lang.org/software/tcltk/license.html) |
| OpenSSL | TLS functionality included with the bundled Python runtime | [OpenSSL 3.5.7 license](https://github.com/openssl/openssl/blob/openssl-3.5.7/LICENSE.txt) |
| PyTorch | Tensor operations, inference mode, CPU/CUDA execution and CUDA events | [PyTorch license and bundled notices](https://github.com/pytorch/pytorch/blob/v2.12.0/LICENSE) |
| torchvision | Supported image-classification architectures, weight metadata and checkpoint retrieval | [torchvision BSD 3-Clause license](https://github.com/pytorch/vision/blob/v0.27.0/LICENSE) |
| NumPy, Pillow, and other pinned dependencies | Dependencies of the inference libraries | Their installed distribution license files and metadata |
| PyInstaller, when used to build a Windows app | Freezing Python and dependencies into an application directory | [PyInstaller license and distribution exception](https://pyinstaller.org/en/stable/license.html) |

Source installations obtain upstream packages separately through pip. A binary
release must include the applicable dependency license texts and copyright
notices, including transitive/native components. Keep installed license files
when freezing or repackaging the application; a table of links alone does not
replace those notices. The locally built CPU application includes a
`THIRD_PARTY_LICENSES` directory containing installed package notices, Python's
license, the Tcl/Tk 9.0.4 and OpenSSL 3.5.7 notices from `licenses/`, and a
dependency version manifest. Keep that directory with the distributed app.
`BUILD_INFO.json` records the bundle's actual interpreter/dependency versions.
Audit these files whenever dependencies or the runtime change. CUDA builds may
include NVIDIA runtime libraries with their own redistributable terms; the
supplied CPU build avoids that NVIDIA runtime dependency.

## Pretrained weights

The configured official choices are `mobilenet_v3_small`, `resnet18`,
`efficientnet_b0`, and `squeezenet1_1`, each with its family's explicit
`IMAGENET1K_V1` enum. Weights are downloaded separately from the URL supplied by
the pinned torchvision version. The app records the URL, enum and file SHA256
in result metadata. [MODELS.md](docs/MODELS.md) lists exact enum names. Cached
weights are excluded from source and application archives.

A custom checkpoint retains its owner's terms. Local import does not grant a
license, establish its training history or make the checkpoint part of this
project's MIT code. The app requires a compatible tensor state dictionary and
does not execute downloaded architecture code. Keep the original model card and
license, and obtain permission for your use before sharing weights.

torchvision's [versioned model documentation](https://docs.pytorch.org/vision/0.27/models.html)
explains that model terms can depend on the training dataset. The project MIT
license therefore does not grant a new license to ImageNet-derived weights.
Users must review the upstream model and dataset terms for their use. The app
does not redistribute ImageNet images, and the procedural images used here do
not measure model accuracy.

## Implementation boundary

PyTorch and torchvision supply the neural network and execution kernels. This
project implements request representation, bounded queues, deadline selection,
batch collection limits, load replay, cancellation, accounting, measurement,
export, and the desktop experiment interface. It does not implement a neural
network architecture or custom CUDA kernels. No upstream model accuracy is
claimed as a result produced by this project.
