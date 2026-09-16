# Glossary

One line per term, in the order the project meets them. Each links to the chapter that introduces it.

## Phase 0: Skeleton and scenario

- **Attribute**: one categorical property of a scene (location, weather, time, companion, item) with a fixed vocabulary, so every image has integer labels. [Chapter 0](00-skeleton/README.md)
- **Supersampling**: rendering at a larger size and shrinking, so edges in a tiny image are smooth instead of jagged. [Chapter 0](00-skeleton/README.md)
- **Seed**: the starting state of a random number generator; the same seed gives the same "random" numbers, which makes experiments repeatable. [Chapter 0](00-skeleton/README.md)

## Phase 1: Autograd from scratch

- **Computational graph**: the record of every operation that produced a value, as nodes and edges; the thing backpropagation walks. [Chapter 1](01-autograd/README.md)
- **Gradient**: the vector of partial derivatives of a scalar (the loss) with respect to each parameter; it points in the direction that increases the loss fastest. [Chapter 1](01-autograd/README.md)
- **Chain rule**: the derivative of a composition is the product of the derivatives of its parts; backpropagation is the chain rule applied along a graph. [Chapter 1](01-autograd/README.md)
- **Backpropagation**: visiting the computational graph from the output backward and accumulating each node's gradient from the nodes that used it. [Chapter 1](01-autograd/README.md)
- **Autograd (automatic differentiation)**: software that builds the computational graph as you compute and runs backpropagation for you. [Chapter 1](01-autograd/README.md)
- **Topological order**: an ordering of graph nodes where every node comes after all of its inputs; reversed, it is the order backpropagation must visit nodes. [Chapter 1](01-autograd/README.md)
- **Gradient descent**: repeatedly nudging parameters a small step against the gradient to lower the loss. [Chapter 1](01-autograd/README.md)
- **Learning rate**: the size of that step. [Chapter 1](01-autograd/README.md)
- **Loss**: a single number measuring how wrong the model's outputs are on a batch; training minimizes it. [Chapter 1](01-autograd/README.md)
- **Finite differences**: approximating a derivative by evaluating the function at two nearby points; slow but independent of any backward code, so the ground truth for gradient checking. [Chapter 1](01-autograd/README.md)
- **Gradient check**: comparing autograd's gradients with finite differences to catch bugs in backward functions. [Chapter 1](01-autograd/README.md)
- **Neuron**: a weighted sum of inputs plus a bias, passed through a nonlinearity. [Chapter 1](01-autograd/README.md)
- **MLP (multilayer perceptron)**: layers of neurons stacked so each layer's outputs are the next layer's inputs. [Chapter 1](01-autograd/README.md)
- **Nonlinearity (activation function)**: a fixed function like tanh or ReLU applied after each layer; without one, stacked layers collapse into a single linear map. [Chapter 1](01-autograd/README.md)
- **Decision boundary**: the set of inputs where a classifier is undecided; visualizing it shows what the model has learned. [Chapter 1](01-autograd/README.md)
- **Numerically stable**: a formula rearranged so intermediate values cannot overflow, underflow, or hit log(0); softplus in place of log(sigmoid) is the Phase 1 example. [Chapter 1](01-autograd/README.md)
- **Softplus**: log(1 + e^x), a smooth version of max(x, 0) whose derivative is the sigmoid; the safe way to write cross-entropy from logits. [Chapter 1](01-autograd/README.md)
- **Standardization**: shifting and scaling each input feature to zero mean and unit variance so no feature dominates because of its units. [Chapter 1](01-autograd/README.md)
- **Logit**: the raw, unbounded output of a classifier before a sigmoid or softmax turns it into a probability. [Chapter 1](01-autograd/README.md)

## Phase 2: Tensors and PyTorch

- **Tensor**: an n-dimensional array with a shape, a dtype, and a device; the unit PyTorch computes on, replacing Phase 1's scalars. [Chapter 2](02-pytorch-mlp/README.md)
- **Shape**: the size of each dimension of a tensor, e.g. (128, 1, 28, 28) for a batch of 128 grayscale 28x28 images. [Chapter 2](02-pytorch-mlp/README.md)
- **dtype**: the number format of a tensor's elements; float32 is the default and float64 must be asked for. [Chapter 2](02-pytorch-mlp/README.md)
- **Device**: where a tensor lives (cpu, mps, cuda); tensors in one operation must share a device. [Chapter 2](02-pytorch-mlp/README.md)
- **Broadcasting**: the rule that lets tensors of different shapes combine by stretching size-1 dimensions, aligned from the right. [Chapter 2](02-pytorch-mlp/README.md)
- **Batch**: a group of examples processed together as one tensor, so one matrix multiply does the work of many neurons. [Chapter 2](02-pytorch-mlp/README.md)
- **Minibatch gradient descent**: estimating the gradient from a batch instead of the whole dataset; noisier but far cheaper per step. [Chapter 2](02-pytorch-mlp/README.md)
- **Epoch**: one pass over the whole training set. [Chapter 2](02-pytorch-mlp/README.md)
- **Softmax**: turns a vector of logits into probabilities that sum to one by exponentiating and normalizing. [Chapter 2](02-pytorch-mlp/README.md)
- **Cross-entropy**: the negative log-probability the model assigns to the correct class; the standard classification loss. [Chapter 2](02-pytorch-mlp/README.md)
- **Log-sum-exp trick**: subtracting the maximum before exponentiating so softmax never overflows. [Chapter 2](02-pytorch-mlp/README.md)
- **Module (`nn.Module`)**: PyTorch's container for parameters and sub-modules; gives `parameters()`, `to(device)`, `state_dict()` for free. [Chapter 2](02-pytorch-mlp/README.md)
- **Parameter**: a tensor registered with a module so the optimizer can find it and autograd tracks it. [Chapter 2](02-pytorch-mlp/README.md)
- **Momentum**: an optimizer's running average of past gradients, which smooths the path and speeds up consistent directions. [Chapter 2](02-pytorch-mlp/README.md)
- **Adam**: an optimizer that scales each parameter's step by the running size of its own gradient, so one learning rate works across layers. [Chapter 2](02-pytorch-mlp/README.md)
- **Train / validation / test split**: the data the model learns from, the data used to make decisions about training, and the data touched once at the end. [Chapter 2](02-pytorch-mlp/README.md)
- **Overfitting**: training loss keeps falling while validation loss rises; the model is memorizing examples instead of learning the rule. [Chapter 2](02-pytorch-mlp/README.md)
- **Generalization**: how well a model does on data it never trained on; the thing we actually care about. [Chapter 2](02-pytorch-mlp/README.md)
- **Checkpoint**: a saved `state_dict` so a trained model can be reloaded without retraining. [Chapter 2](02-pytorch-mlp/README.md)
- **Confusion matrix**: a table of true class against predicted class; shows which classes get mixed up. [Chapter 2](02-pytorch-mlp/README.md)
