# CUDA-based implementation of the BIMP keypoint extractor

## Authors
* **Kasim Terzic** - *Original Implementation* (kt54@st-andrews.ac.uk)
* **Daniel Koudouna** - *GPU Implementation* (dzk@st-andrews.ac.uk)

## Relevant literature
```
@article{terzic14realtime,
    author={K. Terzi{\' c} and J.M.F. Rodrigues and J.M.H. {du Buf}},
    title={BIMP: A Real-Time Biological Model of Multi-Scale Keypoint Detection in V1},
    journal={Neurocomputing},
    volume={150},
    year={2015},
    pages={227--237},
    url={https://kt54.host.cs.st-andrews.ac.uk/Papers/neurocomputing2014.pdf},
}
```

## Requirements

* CMake 3.9+
* OpenCV 3+
* OpenCV Non-Free Modules
* NVIDIA CUDA + Appropriate device drivers

## Bulding (Linux)

Building instructions for OpenCV are included in the scripts/ folder

``` shell
mkdir build
cd build
cmake ..
make
```

## Execution

An executable is created for testing the BIMP library. It accepts a variety of options, accessed by a help interface:

``` shell
./build/bimp_example/bimp_example --help
```

## Submodules

* bimp           -- The GPU BIMP implementation
* utils          -- Helper library for utility, debugging and logging functions
* bimp_example      -- Example usage of the BIMP library


## License
Released under the GNU Lesser General Public License v2.1 or later
