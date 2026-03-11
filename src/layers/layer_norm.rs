use crate::tensor::Tensor;

/// Layer normalization layer
///
/// Normalizes activations across the feature dimension and applies learnable
/// scale and shift.
pub struct TrainableLayerNorm {
    pub gamma: Tensor, // Scale parameter [n_embd]
    pub beta: Tensor,  // Shift parameter [n_embd]
    pub eps: f32,      // Small constant for numerical stability
}

impl TrainableLayerNorm {
    /// Create a new layer normalization layer
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - Size of the feature dimension to normalize
    ///
    /// # Initialization
    ///
    /// - gamma initialized to 1.0 (no scaling initially)
    /// - beta initialized to 0.0 (no shift initially)
    /// - eps = 1e-5 (standard value)
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            gamma: Tensor::new(vec![1.0; normalized_shape], vec![normalized_shape]),
            beta: Tensor::new(vec![0.0; normalized_shape], vec![normalized_shape]),
            eps: 1e-5,
        }
    }

    /// Forward pass
    ///
    /// Normalizes input to zero mean and unit variance, then applies scale/shift
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor [seq_len, n_embd]
    ///
    /// # Returns
    ///
    /// Tuple of (output, cache) where:
    /// - output: Normalized tensor [seq_len, n_embd]
    /// - cache: Stores values needed for backward pass
    pub fn forward(&self, x: &Tensor) -> (Tensor, LayerNormCache) {
        // Compute statistics along last dimension
        let mean = x.mean(-1, true);
        let variance = x.var(-1, true);
        let std = variance.add_scalar(self.eps).sqrt();

        // Normalize
        let x_centered = x.sub(&mean);
        let x_norm = x_centered.div(&std);

        // Apply learnable scale and shift
        let y = x_norm.mul(&self.gamma).add(&self.beta);

        let cache = LayerNormCache {
            x: x.clone(),
            x_norm,
            #[allow(dead_code)]
            mean,
            std,
        };

        (y, cache)
    }

    /// Backward pass
    ///
    /// Computes gradients for gamma, beta, and input. The input gradient is
    /// complex because normalization creates dependencies between all elements.
    ///
    /// # Arguments
    ///
    /// * `grad_out` - Gradient from next layer [seq_len, n_embd]
    /// * `cache` - Cached values from forward pass
    ///
    /// # Returns
    ///
    /// Gradients for gamma, beta, and input
    pub fn backward(&self, grad_out: &Tensor, cache: &LayerNormCache) -> LayerNormGradients {
        let n_embd = self.gamma.data.len();
        let seq_len = grad_out.shape[0];

        // Compute grad_gamma and grad_beta by accumulating over sequence
        let mut grad_gamma = vec![0.0; n_embd];
        let mut grad_beta = vec![0.0; n_embd];
        for i in 0..seq_len {
            for j in 0..n_embd {
                let idx = i * n_embd + j;
                grad_gamma[j] += grad_out.data[idx] * cache.x_norm.data[idx];
                grad_beta[j] += grad_out.data[idx];
            }
        }

        // Backprop through scale: grad_x_norm = grad_out * gamma
        let grad_x_norm = grad_out.mul(&self.gamma);

        // Backprop through normalization (the complex part!)
        // This accounts for mean and variance dependencies
        let mut grad_x_data = vec![0.0; seq_len * n_embd];

        for i in 0..seq_len {
            let row_start = i * n_embd;
            let row_end = row_start + n_embd;

            let grad_x_norm_row = &grad_x_norm.data[row_start..row_end];
            let x_norm_row = &cache.x_norm.data[row_start..row_end];
            let std_val = cache.std.data[i];

            // Compute mean of gradients (accounts for mean dependency)
            let mean_grad: f32 = grad_x_norm_row.iter().sum::<f32>() / n_embd as f32;

            // Compute mean of (grad * x_norm) (accounts for variance dependency)
            let mean_grad_x: f32 = grad_x_norm_row
                .iter()
                .zip(x_norm_row.iter())
                .map(|(g, x)| g * x)
                .sum::<f32>()
                / n_embd as f32;

            // Final gradient formula
            for j in 0..n_embd {
                let idx = row_start + j;
                grad_x_data[idx] =
                    (grad_x_norm_row[j] - mean_grad - x_norm_row[j] * mean_grad_x) / std_val;
            }
        }

        LayerNormGradients {
            gamma: Tensor::new(grad_gamma, vec![n_embd]),
            beta: Tensor::new(grad_beta, vec![n_embd]),
            x: Tensor::new(grad_x_data, cache.x.shape.clone()),
        }
    }
}

/// Cache for layer norm backward pass
pub struct LayerNormCache {
    pub x: Tensor,
    pub x_norm: Tensor,
    #[allow(dead_code)]
    pub mean: Tensor,
    pub std: Tensor,
}

/// Gradients for layer norm
pub struct LayerNormGradients {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub x: Tensor,
}
