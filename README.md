# noahgrad

not perfect, but it works so far. ship fast, fix later. this is a learning exercise, after all.

inspirations / references: [micrograd](https://github.com/karpathy/micrograd), [tinygrad](https://github.com/tinygrad/tinygrad), [pytorch](https://github.com/pytorch/pytorch), [paperswithcode](https://paperswithcode.com/)

**features**

- tensors w/ autodiff
- feed-forward and backpropagation
- loss functions (mse, crossentropy, etc.)
- optimizers (sgd, adam, etc.)
- extendable optimizer class
- modules (linear, sequential, etc.)
- extendable module class
- activations (relu, tanh, etc.)
- functional lib (like torch.nn.functional)
- persistent and temporary buffers
- saving state dicts

**todo** (not guaranteed)

- gpu support (via jax?)
- loading state dicts
