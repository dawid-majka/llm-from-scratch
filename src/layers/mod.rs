// All layer impls for GPT-2

// Each layer provices forward and backward passes

// activation: GELU activation function (forward and backward)
// linear: Fully connected layer
// layer_norm: Layer normalization
// dropout: Dropout regularization
// mlp: Multi-layer perceptron (feedforward network)
// attention: Self-attention mechanism
// block: Complete transformer block

pub mod activation;
pub mod attention;
pub mod block;
pub mod dropout;
pub mod layer_norm;
pub mod linear;
pub mod mlp;
