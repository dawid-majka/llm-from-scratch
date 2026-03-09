use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::tensor::Tensor;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub block_size: usize,
    pub dropout_rate: f32,
}

impl Config {
    // ~50k params - 2-5 min training
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            n_embd: 64,        // Very small embedding
            n_heads: 1,        // Single-head attention
            n_layers: 2,       // Shallow
            block_size: 64,    // Short context
            dropout_rate: 0.1, // Dropout probability
        }
    }

    // ~200k params
    pub fn small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            n_embd: 128,       // Small embedding
            n_heads: 1,        // Single-head attention
            n_layers: 3,       // Medium depth
            block_size: 128,   // Medium context
            dropout_rate: 0.1, // Dropout probability
        }
    }

    // ~4M params
    pub fn medium(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            n_embd: 256,       // Medium embedding
            n_heads: 4,        // Multi-head attention
            n_layers: 4,       // Medium depth
            block_size: 256,   // Medium context
            dropout_rate: 0.1, // Dropout probability
        }
    }

    // ~163M parameters (with GPT-2 vocab of 50257 tokens)
    // ~86M parameters (with smaller demo vocab of 512 tokens)
    //
    // This matches OpenAI's GPT-2 Small architecture:
    // - 768 dimensional embeddings
    // - 12 transformer layers
    // - 12 attention heads
    // - 1024 token context window
    pub fn gpt2_small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            n_embd: 768,       // GPT-2 Small
            n_heads: 12,       // GPT-2 Small
            n_layers: 12,      // GPT-2 Small
            block_size: 1024,  // GPT-2 standard context
            dropout_rate: 0.1, // Dropout probability
        }
    }
}

// ACTIVATION FN

// GELU > ReLU - Smoother gradients
pub fn gelu(x: &Tensor) -> Tensor {
    // Constants for the approximation
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    let coeff = 0.044715_f32;

    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let result: Vec<f32> = x
        .data
        .iter()
        .map(|&val| {
            let x_cubed = val * val * val;
            let inner = sqrt_2_over_pi * (val + coeff * x_cubed);
            0.5 * val * (1.0 + inner.tanh())
        })
        .collect();

    Tensor::new(result, x.shape.clone())
}

// EMBEDDING

// Converts token Ids to fixed-size embedding vector counterparts
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    // Initializes with random values
    // Mean 0
    // Variance - 0.02
    //
    // vocab_size - num of tokens in vocab
    // n_embd - embedding dimension
    pub fn new(vocab_size: usize, n_embd: usize) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap();
        let weight_data: Vec<f32> = (0..vocab_size * n_embd)
            .map(|_| normal.sample(&mut rng))
            .collect();

        Self {
            weight: Tensor::new(weight_data, vec![vocab_size, n_embd]),
        }
    }

    // Look up embeddings for token Ids
    // token_ids - [batch, seq_len] with token indices
    pub fn forward(&self, token_ids: &[Vec<usize>]) -> Tensor {
        let batch_size = token_ids.len();
        let seq_len = token_ids[0].len();
        let n_embd = self.weight.shape[1];

        let mut output = Vec::with_capacity(batch_size * seq_len * n_embd);

        for batch in token_ids {
            for &token_id in batch {
                assert!(
                    token_id < self.weight.shape[0],
                    "Token ID {} out of vocab range (vocab_size = {})",
                    token_id,
                    self.weight.shape[0]
                );
                // Copy the embedding vector for this token
                let start = token_id * n_embd;
                let end = start + n_embd;
                output.extend_from_slice(&self.weight.data[start..end]);
            }
        }

        Tensor::new(output, vec![batch_size, seq_len, n_embd])
    }
}

// Layer Normalization

// Normalizes activations to have zero mean and unit variance
// Applied before each sublayer (attention and MLP)

pub struct LayerNorm {
    // Scale parameter (learnable): [n_embd]
    pub gamma: Tensor,
    // Shift parameter (learnable): [n_embd]
    pub beta: Tensor,
    // Small constant for numerical stability
    pub eps: f32,
}

impl LayerNorm {
    // n_embd - Feature dimension to normalize over
    // eps - Small constant to prevent division by zero (default: 1e-5)
    pub fn new(n_embd: usize, eps: f32) -> Self {
        // Initialize gamma to 1 (no scaling initially)
        let gamma = Tensor::new(vec![1.0; n_embd], vec![n_embd]);
        // Initialize beta to 0 (no shift initially)
        let beta = Tensor::new(vec![0.0; n_embd], vec![n_embd]);

        Self { gamma, beta, eps }
    }

    // Normalize along last dimension
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Compute mean and variance along last dimension
        let mean = x.mean(-1, true);
        let variance = x.var(-1, true);

        // Normalize: (x - mean) / sqrt(var + eps)
        let normalized = x.sub(&mean).div(&variance.add_scalar(self.eps).sqrt());

        // Scale and shift: normalized * gamma + beta
        normalized.mul(&self.gamma).add(&self.beta)
    }
}

// Linear Layer

// Applies an affine transformation: `y = x @ W + b`
//
// Input:  [*, in_features]
// Output: [*, out_features]
//
// where `*` represents any number of leading dimensions (batch, sequence, etc.)
pub struct Linear {
    // Weight matrix: [in_features, out_features]
    pub weight: Tensor,
    // Bias vector: [out_features]
    pub bias: Tensor,
}

impl Linear {
    // Random init vals
    // Weights are initialized from N(0, 0.02) following GPT-2.
    //
    // Mean - 0, Standard Deviation - 0.02
    //
    // Bias is initialized to zeros.
    //
    // in_features - Input dimension
    // out_features - Output dimension
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| normal.sample(&mut rng))
            .collect();

        let weight = Tensor::new(weight_data, vec![in_features, out_features]);
        let bias = Tensor::zeros(vec![out_features]);

        Self { weight, bias }
    }

    // Forward pass: y = x @ W + b
    //
    // x - Input tensor [..., in_features]
    //
    // Output tensor [..., out_features]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // For simplicity, we handle 3D input [batch, seq, in_features]
        // Reshape to 2D, matmul, then reshape back
        let batch_size = x.shape[0];
        let seq_len = x.shape[1];
        let in_features = x.shape[2];

        // Reshape to [batch * seq, in_features]
        let x_2d = x.reshape(&[batch_size * seq_len, in_features]);

        // Matrix multiply
        let y_2d = x_2d.matmul(&self.weight);

        // Reshape back to [batch, seq, out_features]
        let out_features = self.weight.shape[1];
        let y_3d = y_2d.reshape(&[batch_size, seq_len, out_features]);

        // Add bias (broadcasts automatically)
        y_3d.add(&self.bias)
    }
}

// Attention Mechanism

pub struct Attention {
    // Combined Q, K, V projection: [n_embd, 3 * n_embd]
    pub c_attn: Linear,
    // Output projection: [n_embd, n_embd]
    pub c_proj: Linear,
    // Number of attention heads
    pub n_heads: usize,
    // Dimension per head (n_embd / n_heads)
    pub head_dim: usize,
}

impl Attention {
    pub fn new(n_embd: usize, n_heads: usize) -> Self {
        assert_eq!(n_embd % n_heads, 0, "n_embd must be divisible by n_heads");

        let head_dim = n_embd / n_heads;

        // Single linear layer computes Q, K, V in one shot
        let c_attn = Linear::new(n_embd, 3 * n_embd);
        // Output projection after concatenating heads
        let c_proj = Linear::new(n_embd, n_embd);

        Self {
            c_attn,
            c_proj,
            n_heads,
            head_dim,
        }
    }

    // Compute multi-head self-attention
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let batch_size = x.shape[0];
        let seq_len = x.shape[1];
        let n_embd = x.shape[2];

        // === 1. Compute Q, K, V ===
        // c_attn projects to 3*n_embd (stacked Q, K, V)
        let qkv = self.c_attn.forward(x); // [batch, seq, 3*n_embd]

        // Split into Q, K, V
        let mut q_data = Vec::with_capacity(batch_size * seq_len * n_embd);
        let mut k_data = Vec::with_capacity(batch_size * seq_len * n_embd);
        let mut v_data = Vec::with_capacity(batch_size * seq_len * n_embd);

        for i in 0..batch_size * seq_len {
            let start = i * 3 * n_embd;
            q_data.extend_from_slice(&qkv.data[start..start + n_embd]);
            k_data.extend_from_slice(&qkv.data[start + n_embd..start + 2 * n_embd]);
            v_data.extend_from_slice(&qkv.data[start + 2 * n_embd..start + 3 * n_embd]);
        }

        let q = Tensor::new(q_data, vec![batch_size, seq_len, n_embd]);
        let k = Tensor::new(k_data, vec![batch_size, seq_len, n_embd]);
        let v = Tensor::new(v_data, vec![batch_size, seq_len, n_embd]);

        // === 2. Reshape for multi-head attention ===
        // [batch, seq, n_embd] -> [batch, n_heads, seq, head_dim]
        let q = self.split_heads(&q, batch_size, seq_len);
        let k = self.split_heads(&k, batch_size, seq_len);
        let v = self.split_heads(&v, batch_size, seq_len);

        // === 3. Transpose K for attention scores ===
        // K: [batch, n_heads, seq, head_dim] -> [batch, n_heads, head_dim, seq]
        let k_t = k.transpose(2, 3);

        // === 4. Compute attention scores ===
        // Q @ K^T: [batch, n_heads, seq, head_dim] @ [batch, n_heads, head_dim, seq]
        //       -> [batch, n_heads, seq, seq]
        let scores = q.matmul(&k_t);

        // === 5. Scale by sqrt(head_dim) ===
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = scores.mul_scalar(scale);

        // === 6. Apply causal mask ===
        let mask = self.create_causal_mask(seq_len);
        let scores = scores.masked_fill(&mask, f32::NEG_INFINITY);

        // === 7. Softmax to get attention weights ===
        let attn = scores.softmax(-1); // [batch, n_heads, seq, seq]

        // === 8. Apply attention to values ===
        // attn @ V: [batch, n_heads, seq, seq] @ [batch, n_heads, seq, head_dim]
        //        -> [batch, n_heads, seq, head_dim]
        let out = attn.matmul(&v);

        // === 9. Concatenate heads ===
        let out = self.merge_heads(&out, batch_size, seq_len);

        // === 10. Output projection ===
        self.c_proj.forward(&out)
    }

    // Split into multiple attention heads
    //
    // [batch, seq, n_embd] -> [batch, n_heads, seq, head_dim]
    fn split_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        // Reshape and transpose
        let mut result = vec![0.0; batch_size * self.n_heads * seq_len * self.head_dim];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.n_heads {
                    for d in 0..self.head_dim {
                        let src_idx = (b * seq_len + s) * (self.n_heads * self.head_dim)
                            + h * self.head_dim
                            + d;
                        let dst_idx = ((b * self.n_heads + h) * seq_len + s) * self.head_dim + d;
                        result[dst_idx] = x.data[src_idx];
                    }
                }
            }
        }

        Tensor::new(
            result,
            vec![batch_size, self.n_heads, seq_len, self.head_dim],
        )
    }

    // Merge attention heads back
    //
    // [batch, n_heads, seq, head_dim] -> [batch, seq, n_embd]
    fn merge_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        let n_embd = self.n_heads * self.head_dim;
        let mut result = vec![0.0; batch_size * seq_len * n_embd];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.n_heads {
                    for d in 0..self.head_dim {
                        let src_idx = ((b * self.n_heads + h) * seq_len + s) * self.head_dim + d;
                        let dst_idx = (b * seq_len + s) * n_embd + h * self.head_dim + d;
                        result[dst_idx] = x.data[src_idx];
                    }
                }
            }
        }

        Tensor::new(result, vec![batch_size, seq_len, n_embd])
    }

    // Create causal attention mask
    //
    // For seq_len=4, mask looks like:
    // [0 1 1 1]  position 0 can only see itself
    // [0 0 1 1]  position 1 can see 0,1
    // [0 0 0 1]  position 2 can see 0,1,2
    // [0 0 0 0]  position 3 can see all
    fn create_causal_mask(&self, seq_len: usize) -> Tensor {
        let mut mask_data = vec![0.0; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    // j is in the future relative to i
                    mask_data[i * seq_len + j] = 1.0;
                }
            }
        }

        Tensor::new(mask_data, vec![seq_len, seq_len])
    }
}

// MLP Layer

// Applied in transformer after each attention block
//
// 1. Expand 4x
// 2. GELU activation
// 3. Project back to previous size ( /4 )

pub struct MLP {
    // First linear layer: [n_embd, 4*n_embd]
    pub c_fc: Linear,
    // Second linear layer: [4*n_embd, n_embd]
    pub c_proj: Linear,
}

impl MLP {
    pub fn new(n_embd: usize) -> Self {
        let hidden_dim = 4 * n_embd;
        let c_fc = Linear::new(n_embd, hidden_dim);
        let c_proj = Linear::new(hidden_dim, n_embd);

        Self { c_fc, c_proj }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Expand to hidden dimension
        let h = self.c_fc.forward(x);
        // Apply GELU activation
        let h = gelu(&h);
        // Project back to n_embd
        self.c_proj.forward(&h)
    }
}

// Transformer Block

pub struct Block {
    // Layer norm before attention
    pub ln_1: LayerNorm,
    // Multi-head attention
    pub attn: Attention,
    // Layer norm before MLP
    pub ln_2: LayerNorm,
    // Feedforward network
    pub mlp: MLP,
}

impl Block {
    pub fn new(n_embd: usize, n_heads: usize) -> Self {
        Self {
            ln_1: LayerNorm::new(n_embd, 1e-5),
            attn: Attention::new(n_embd, n_heads),
            ln_2: LayerNorm::new(n_embd, 1e-5),
            mlp: MLP::new(n_embd),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Attention block with residual connection
        let x = x.add(&self.attn.forward(&self.ln_1.forward(x)));

        // MLP block with residual connection
        x.add(&self.mlp.forward(&self.ln_2.forward(&x)))
    }
}

// GPT-2 Model

pub struct GPT2 {
    // Model configuration
    pub config: Config,
    // Token embedding layer
    pub token_embedding: Embedding,
    // Position embedding layer
    pub position_embedding: Embedding,
    // Stack of transformer blocks
    pub blocks: Vec<Block>,
    // Final layer normalization
    pub ln_f: LayerNorm,
    // Output projection to vocabulary (unembedding)
    pub lm_head: Linear,
}

impl GPT2 {
    pub fn new(config: &Config) -> Self {
        // Create embeddings
        let token_embedding = Embedding::new(config.vocab_size, config.n_embd);
        let position_embedding = Embedding::new(config.block_size, config.n_embd);

        // Create transformer blocks
        let blocks = (0..config.n_layers)
            .map(|_| Block::new(config.n_embd, config.n_heads))
            .collect();

        // Final layer norm
        let ln_f = LayerNorm::new(config.n_embd, 1e-5);

        // Output projection to vocabulary
        let lm_head = Linear::new(config.n_embd, config.vocab_size);

        Self {
            config: config.clone(),
            token_embedding,
            position_embedding,
            blocks,
            ln_f,
            lm_head,
        }
    }

    pub fn forward(&self, token_ids: &[Vec<usize>]) -> Tensor {
        let batch_size = token_ids.len();
        let seq_len = token_ids[0].len();

        assert!(
            seq_len <= self.config.block_size,
            "Sequence length {} exceeds block_size {}",
            seq_len,
            self.config.block_size
        );

        // === 1. Token embeddings ===
        let mut x = self.token_embedding.forward(token_ids);

        // === 2. Position embeddings ===
        // Create position indices [0, 1, 2, ..., seq_len-1]
        let positions: Vec<Vec<usize>> = vec![(0..seq_len).collect()];
        let pos_emb = self.position_embedding.forward(&positions);

        // Broadcast position embeddings to batch size and add
        // pos_emb: [1, seq_len, n_embd] -> broadcast to [batch, seq_len, n_embd]
        for b in 0..batch_size {
            for s in 0..seq_len {
                for e in 0..self.config.n_embd {
                    let idx = (b * seq_len + s) * self.config.n_embd + e;
                    let pos_idx = s * self.config.n_embd + e;
                    x.data[idx] += pos_emb.data[pos_idx];
                }
            }
        }

        // === 3. Pass through transformer blocks ===
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // === 4. Final layer norm ===
        x = self.ln_f.forward(&x);

        // === 5. Project to vocabulary ===
        self.lm_head.forward(&x)
    }

    pub fn count_parameters(&self) -> usize {
        let mut total = 0;

        // Token and position embeddings
        total += self.token_embedding.weight.data.len();
        total += self.position_embedding.weight.data.len();

        // Transformer blocks
        for block in &self.blocks {
            // Attention
            total += block.attn.c_attn.weight.data.len();
            total += block.attn.c_attn.bias.data.len();
            total += block.attn.c_proj.weight.data.len();
            total += block.attn.c_proj.bias.data.len();

            // MLP
            total += block.mlp.c_fc.weight.data.len();
            total += block.mlp.c_fc.bias.data.len();
            total += block.mlp.c_proj.weight.data.len();
            total += block.mlp.c_proj.bias.data.len();

            // Layer norms
            total += block.ln_1.gamma.data.len();
            total += block.ln_1.beta.data.len();
            total += block.ln_2.gamma.data.len();
            total += block.ln_2.beta.data.len();
        }

        // Final layer norm
        total += self.ln_f.gamma.data.len();
        total += self.ln_f.beta.data.len();

        // LM head
        total += self.lm_head.weight.data.len();
        total += self.lm_head.bias.data.len();

        total
    }
}
