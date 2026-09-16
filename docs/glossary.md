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
