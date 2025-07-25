# Neural Networks in C++

> **A low-level, from-scratch, ongoing implementation of neural network operations and learning algorithms in modern C++**

---

## 🚀 About This Project

Welcome! This repository is a personal journey and open playground to **implement neural network building blocks from scratch in C++**—without relying on high-level frameworks.

- **Goal:** To deeply understand, document, and optimize the core operations behind modern deep learning systems, all in C++.
- **Approach:** Start with basics (activations, linear layers, MLPs) and incrementally add more advanced ops (convolution, pooling, softmax, optimizers, etc.).
- **Why C++?** For high performance, portability, and to explore what it takes to build ML infrastructure closer to the metal.

> **I will be adding new features and operations almost every day.**  
> Follow the repo to watch the progress and learn step by step!

---

## ✨ Features (Planned and Completed)

- [x] Matrix multiplication, dot product
- [x] Sigmoid activation and its derivative
- [x] Tanh and ReLU activations
- [x] Mean Squared Error loss
- [x] Manual forward/backward for 1-hidden-layer MLP
- [x] SGD optimizer
- [x] Softmax and Cross Entropy
- [ ] Convolution (1D/2D)
- [ ] Pooling layers
- [ ] Batch Normalization
- [ ] RNN/LSTM ops
- [ ] Serialization (saving/loading models)
- [ ] Fixed-point support for embedded use
- [ ] Unit tests and benchmarks
- [ ] More...

*See the [Project Board](https://github.com/yourusername/Neural-Networks-in-Cpp/projects) for progress and ideas!*

---

## 🛠️ How to Build & Run

### Requirements

- C++17 or later
- CMake (recommended)
- Linux, macOS, or Windows

### Build Instructions

```bash
git clone https://github.com/yourusername/Neural-Networks-in-Cpp.git
cd Neural-Networks-in-Cpp
mkdir build && cd build
cmake ..
make
./NN_cpp      # or whatever the executable is called
```

## 📁 Project Structure
```bash
/src           # Source code
/include       # Header files
/tests         # Unit tests
/examples      # Demo apps & experiments
CMakeLists.txt # Build configuration
README.md      # This file
```

## 👨‍💻 Contributing
- Ideas, issues, and pull requests are welcome!
- See CONTRIBUTING.md (to be written) for how to get involved.
- This project is intended as a personal learning journey but is open for collaboration and feedback.

## ⭐️ Follow the Journey
- I’m building this repo incrementally—committing new ops, layers, and demos almost daily.
- Star and watch the repo to learn together!