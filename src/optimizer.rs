// AdamW optimizer

use crate::{
    gpt2_trainable::{GPT2Gradients, TrainableGPT2},
    tensor::Tensor,
};

use rayon::prelude::*;

pub struct AdamWOptimizer {
    // First moment (momentum) - matches GPT2Gradients structure
    pub m_token_embedding: Tensor,
    pub m_position_embedding: Tensor,
    pub m_block_states: Vec<BlockAdamState>,
    pub m_ln_final_gamma: Tensor,
    pub m_ln_final_beta: Tensor,
    pub m_output_weight: Tensor,

    // Second moment (variance) - matches GPT2Gradients structure
    pub v_token_embedding: Tensor,
    pub v_position_embedding: Tensor,
    pub v_block_states: Vec<BlockAdamState>,
    pub v_ln_final_gamma: Tensor,
    pub v_ln_final_beta: Tensor,
    pub v_output_weight: Tensor,

    // Hyperparameters
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub step: usize,
}

// Optimizer state for a single transformer block
pub struct BlockAdamState {
    pub ln1_gamma: Tensor,
    pub ln1_beta: Tensor,
    pub attn: AttentionAdamState,
    pub ln2_gamma: Tensor,
    pub ln2_beta: Tensor,
    pub mlp: MLPAdamState,
}

// Optimizer state for attention mechanism
pub struct AttentionAdamState {
    pub q_weight: Tensor,
    pub q_bias: Tensor,
    pub k_weight: Tensor,
    pub k_bias: Tensor,
    pub v_weight: Tensor,
    pub v_bias: Tensor,
    pub out_weight: Tensor,
    pub out_bias: Tensor,
}

// Optimizer state for MLP (feedforward network
pub struct MLPAdamState {
    pub fc1_weight: Tensor,
    pub fc1_bias: Tensor,
    pub fc2_weight: Tensor,
    pub fc2_bias: Tensor,
}

impl AdamWOptimizer {
    // Create a new AdamW optimizer for the given model
    //
    // Initializes all moment estimates to zero. The optimizer state mirrors
    // the model structure exactly, ensuring every parameter has optimizer state.
    pub fn new(model: &TrainableGPT2) -> Self {
        // Initialize all momentum and variance tensors to zero
        let m_token_embedding = Tensor::zeros(model.token_embedding.shape.clone());
        let m_position_embedding = Tensor::zeros(model.position_embedding.shape.clone());
        let m_ln_final_gamma = Tensor::zeros(model.ln_final.gamma.shape.clone());
        let m_ln_final_beta = Tensor::zeros(model.ln_final.beta.shape.clone());
        let m_output_weight = Tensor::zeros(model.output_weight.shape.clone());

        let v_token_embedding = Tensor::zeros(model.token_embedding.shape.clone());
        let v_position_embedding = Tensor::zeros(model.position_embedding.shape.clone());
        let v_ln_final_gamma = Tensor::zeros(model.ln_final.gamma.shape.clone());
        let v_ln_final_beta = Tensor::zeros(model.ln_final.beta.shape.clone());
        let v_output_weight = Tensor::zeros(model.output_weight.shape.clone());

        let mut m_block_states = Vec::new();
        let mut v_block_states = Vec::new();

        for block in &model.blocks {
            let m_block = BlockAdamState {
                ln1_gamma: Tensor::zeros(block.ln1.gamma.shape.clone()),
                ln1_beta: Tensor::zeros(block.ln1.beta.shape.clone()),
                attn: AttentionAdamState {
                    q_weight: Tensor::zeros(block.attn.q_proj.weight.shape.clone()),
                    q_bias: Tensor::zeros(block.attn.q_proj.bias.shape.clone()),
                    k_weight: Tensor::zeros(block.attn.k_proj.weight.shape.clone()),
                    k_bias: Tensor::zeros(block.attn.k_proj.bias.shape.clone()),
                    v_weight: Tensor::zeros(block.attn.v_proj.weight.shape.clone()),
                    v_bias: Tensor::zeros(block.attn.v_proj.bias.shape.clone()),
                    out_weight: Tensor::zeros(block.attn.out_proj.weight.shape.clone()),
                    out_bias: Tensor::zeros(block.attn.out_proj.bias.shape.clone()),
                },
                ln2_gamma: Tensor::zeros(block.ln2.gamma.shape.clone()),
                ln2_beta: Tensor::zeros(block.ln2.beta.shape.clone()),
                mlp: MLPAdamState {
                    fc1_weight: Tensor::zeros(block.mlp.fc1.weight.shape.clone()),
                    fc1_bias: Tensor::zeros(block.mlp.fc1.bias.shape.clone()),
                    fc2_weight: Tensor::zeros(block.mlp.fc2.weight.shape.clone()),
                    fc2_bias: Tensor::zeros(block.mlp.fc2.bias.shape.clone()),
                },
            };

            let v_block = BlockAdamState {
                ln1_gamma: Tensor::zeros(block.ln1.gamma.shape.clone()),
                ln1_beta: Tensor::zeros(block.ln1.beta.shape.clone()),
                attn: AttentionAdamState {
                    q_weight: Tensor::zeros(block.attn.q_proj.weight.shape.clone()),
                    q_bias: Tensor::zeros(block.attn.q_proj.bias.shape.clone()),
                    k_weight: Tensor::zeros(block.attn.k_proj.weight.shape.clone()),
                    k_bias: Tensor::zeros(block.attn.k_proj.bias.shape.clone()),
                    v_weight: Tensor::zeros(block.attn.v_proj.weight.shape.clone()),
                    v_bias: Tensor::zeros(block.attn.v_proj.bias.shape.clone()),
                    out_weight: Tensor::zeros(block.attn.out_proj.weight.shape.clone()),
                    out_bias: Tensor::zeros(block.attn.out_proj.bias.shape.clone()),
                },
                ln2_gamma: Tensor::zeros(block.ln2.gamma.shape.clone()),
                ln2_beta: Tensor::zeros(block.ln2.beta.shape.clone()),
                mlp: MLPAdamState {
                    fc1_weight: Tensor::zeros(block.mlp.fc1.weight.shape.clone()),
                    fc1_bias: Tensor::zeros(block.mlp.fc1.bias.shape.clone()),
                    fc2_weight: Tensor::zeros(block.mlp.fc2.weight.shape.clone()),
                    fc2_bias: Tensor::zeros(block.mlp.fc2.bias.shape.clone()),
                },
            };

            m_block_states.push(m_block);
            v_block_states.push(v_block);
        }

        Self {
            m_token_embedding,
            m_position_embedding,
            m_block_states,
            m_ln_final_gamma,
            m_ln_final_beta,
            m_output_weight,
            v_token_embedding,
            v_position_embedding,
            v_block_states,
            v_ln_final_gamma,
            v_ln_final_beta,
            v_output_weight,
            beta1: 0.9,
            beta2: 0.95, // Lower than Adam's 0.999, standard for transformers
            epsilon: 1e-8,
            step: 0,
        }
    }

    // Create a shallow copy for checkpointing
    //
    // Clones all tensors to create an independent copy of the optimizer state.
    // Used when saving checkpoints to disk.
    pub fn clone_shallow(&self) -> Self {
        Self {
            m_token_embedding: self.m_token_embedding.clone(),
            m_position_embedding: self.m_position_embedding.clone(),
            m_block_states: self
                .m_block_states
                .iter()
                .map(|b| BlockAdamState {
                    ln1_gamma: b.ln1_gamma.clone(),
                    ln1_beta: b.ln1_beta.clone(),
                    attn: AttentionAdamState {
                        q_weight: b.attn.q_weight.clone(),
                        q_bias: b.attn.q_bias.clone(),
                        k_weight: b.attn.k_weight.clone(),
                        k_bias: b.attn.k_bias.clone(),
                        v_weight: b.attn.v_weight.clone(),
                        v_bias: b.attn.v_bias.clone(),
                        out_weight: b.attn.out_weight.clone(),
                        out_bias: b.attn.out_bias.clone(),
                    },
                    ln2_gamma: b.ln2_gamma.clone(),
                    ln2_beta: b.ln2_beta.clone(),
                    mlp: MLPAdamState {
                        fc1_weight: b.mlp.fc1_weight.clone(),
                        fc1_bias: b.mlp.fc1_bias.clone(),
                        fc2_weight: b.mlp.fc2_weight.clone(),
                        fc2_bias: b.mlp.fc2_bias.clone(),
                    },
                })
                .collect(),
            m_ln_final_gamma: self.m_ln_final_gamma.clone(),
            m_ln_final_beta: self.m_ln_final_beta.clone(),
            m_output_weight: self.m_output_weight.clone(),
            v_token_embedding: self.v_token_embedding.clone(),
            v_position_embedding: self.v_position_embedding.clone(),
            v_block_states: self
                .v_block_states
                .iter()
                .map(|b| BlockAdamState {
                    ln1_gamma: b.ln1_gamma.clone(),
                    ln1_beta: b.ln1_beta.clone(),
                    attn: AttentionAdamState {
                        q_weight: b.attn.q_weight.clone(),
                        q_bias: b.attn.q_bias.clone(),
                        k_weight: b.attn.k_weight.clone(),
                        k_bias: b.attn.k_bias.clone(),
                        v_weight: b.attn.v_weight.clone(),
                        v_bias: b.attn.v_bias.clone(),
                        out_weight: b.attn.out_weight.clone(),
                        out_bias: b.attn.out_bias.clone(),
                    },
                    ln2_gamma: b.ln2_gamma.clone(),
                    ln2_beta: b.ln2_beta.clone(),
                    mlp: MLPAdamState {
                        fc1_weight: b.mlp.fc1_weight.clone(),
                        fc1_bias: b.mlp.fc1_bias.clone(),
                        fc2_weight: b.mlp.fc2_weight.clone(),
                        fc2_bias: b.mlp.fc2_bias.clone(),
                    },
                })
                .collect(),
            v_ln_final_gamma: self.v_ln_final_gamma.clone(),
            v_ln_final_beta: self.v_ln_final_beta.clone(),
            v_output_weight: self.v_output_weight.clone(),
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            step: self.step,
        }
    }
}

// AdamW optimizer parameter update
pub fn adamw_update(
    model: &mut TrainableGPT2,
    grads: &GPT2Gradients,
    optimizer: &mut AdamWOptimizer,
    lr: f32,
    weight_decay: f32,
) {
    optimizer.step += 1;
    let step = optimizer.step as f32;

    // Bias correction factors
    // These correct for initialization bias (m and v start at 0)
    let bias_correction1 = 1.0 - optimizer.beta1.powf(step);
    let bias_correction2 = 1.0 - optimizer.beta2.powf(step);

    let beta1 = optimizer.beta1;
    let beta2 = optimizer.beta2;
    let epsilon = optimizer.epsilon;

    // Helper macro to update a parameter with AdamW
    // Parallelizes for large tensors, sequential for small ones
    // apply_decay: whether to apply weight decay (only for 2D weight matrices)
    macro_rules! adamw_update_param {
        ($param:expr, $grad:expr, $m:expr, $v:expr, $apply_decay:expr) => {
            // Parallelize for large tensors (>1000 elements)
            if $param.data.len() > 1000 {
                $param
                    .data
                    .par_iter_mut()
                    .zip($grad.data.par_iter())
                    .zip($m.data.par_iter_mut().zip($v.data.par_iter_mut()))
                    .for_each(|((param_val, &grad_val), (m_val, v_val))| {
                        // WEIGHT DECAY: Apply before Adam update (decoupled)
                        if $apply_decay {
                            *param_val *= 1.0 - lr * weight_decay;
                        }

                        // Update biased first moment estimate (momentum)
                        *m_val = beta1 * *m_val + (1.0 - beta1) * grad_val;

                        // Update biased second moment estimate (variance)
                        *v_val = beta2 * *v_val + (1.0 - beta2) * grad_val * grad_val;

                        // Compute bias-corrected first moment
                        let m_hat = *m_val / bias_correction1;

                        // Compute bias-corrected second moment
                        let v_hat = *v_val / bias_correction2;

                        // Update parameter (Adam step)
                        *param_val -= lr * m_hat / (v_hat.sqrt() + epsilon);
                    });
            } else {
                // Sequential for small tensors to avoid parallelization overhead
                for i in 0..$param.data.len() {
                    // WEIGHT DECAY: Apply before Adam update (decoupled)
                    if $apply_decay {
                        $param.data[i] *= 1.0 - lr * weight_decay;
                    }

                    let g = $grad.data[i];

                    // Update biased first moment estimate (momentum)
                    $m.data[i] = beta1 * $m.data[i] + (1.0 - beta1) * g;

                    // Update biased second moment estimate (variance)
                    $v.data[i] = beta2 * $v.data[i] + (1.0 - beta2) * g * g;

                    // Compute bias-corrected first moment
                    let m_hat = $m.data[i] / bias_correction1;

                    // Compute bias-corrected second moment
                    let v_hat = $v.data[i] / bias_correction2;

                    // Update parameter (Adam step)
                    $param.data[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
                }
            }
        };
    }

    // Update embeddings (no weight decay - common practice)
    adamw_update_param!(
        model.token_embedding,
        grads.token_embedding,
        optimizer.m_token_embedding,
        optimizer.v_token_embedding,
        false // No decay on embeddings
    );
    adamw_update_param!(
        model.position_embedding,
        grads.position_embedding,
        optimizer.m_position_embedding,
        optimizer.v_position_embedding,
        false // No decay on embeddings
    );

    // Update all transformer blocks
    for ((block, block_grads), (m_block, v_block)) in
        model.blocks.iter_mut().zip(&grads.block_grads).zip(
            optimizer
                .m_block_states
                .iter_mut()
                .zip(optimizer.v_block_states.iter_mut()),
        )
    {
        // LayerNorm 1 (no decay on 1D scale/shift parameters)
        adamw_update_param!(
            block.ln1.gamma,
            block_grads.ln1_gamma,
            m_block.ln1_gamma,
            v_block.ln1_gamma,
            false // No decay on LayerNorm
        );
        adamw_update_param!(
            block.ln1.beta,
            block_grads.ln1_beta,
            m_block.ln1_beta,
            v_block.ln1_beta,
            false // No decay on LayerNorm
        );

        // Self-attention (decay on 2D weight matrices, not on 1D biases)
        adamw_update_param!(
            block.attn.q_proj.weight,
            block_grads.attn.q_weight,
            m_block.attn.q_weight,
            v_block.attn.q_weight,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.attn.q_proj.bias,
            block_grads.attn.q_bias,
            m_block.attn.q_bias,
            v_block.attn.q_bias,
            false // No decay on bias
        );
        adamw_update_param!(
            block.attn.k_proj.weight,
            block_grads.attn.k_weight,
            m_block.attn.k_weight,
            v_block.attn.k_weight,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.attn.k_proj.bias,
            block_grads.attn.k_bias,
            m_block.attn.k_bias,
            v_block.attn.k_bias,
            false // No decay on bias
        );
        adamw_update_param!(
            block.attn.v_proj.weight,
            block_grads.attn.v_weight,
            m_block.attn.v_weight,
            v_block.attn.v_weight,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.attn.v_proj.bias,
            block_grads.attn.v_bias,
            m_block.attn.v_bias,
            v_block.attn.v_bias,
            false // No decay on bias
        );
        adamw_update_param!(
            block.attn.out_proj.weight,
            block_grads.attn.out_weight,
            m_block.attn.out_weight,
            v_block.attn.out_weight,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.attn.out_proj.bias,
            block_grads.attn.out_bias,
            m_block.attn.out_bias,
            v_block.attn.out_bias,
            false // No decay on bias
        );

        // LayerNorm 2 (no decay on 1D scale/shift parameters)
        adamw_update_param!(
            block.ln2.gamma,
            block_grads.ln2_gamma,
            m_block.ln2_gamma,
            v_block.ln2_gamma,
            false // No decay on LayerNorm
        );
        adamw_update_param!(
            block.ln2.beta,
            block_grads.ln2_beta,
            m_block.ln2_beta,
            v_block.ln2_beta,
            false // No decay on LayerNorm
        );

        // MLP (decay on 2D weight matrices, not on 1D biases)
        adamw_update_param!(
            block.mlp.fc1.weight,
            block_grads.mlp.fc1_weight,
            m_block.mlp.fc1_weight,
            v_block.mlp.fc1_weight,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.mlp.fc1.bias,
            block_grads.mlp.fc1_bias,
            m_block.mlp.fc1_bias,
            v_block.mlp.fc1_bias,
            false // No decay on bias
        );
        adamw_update_param!(
            block.mlp.fc2.weight,
            block_grads.mlp.fc2_weight,
            m_block.mlp.fc2_weight,
            v_block.mlp.fc2_weight,
            true // Decay on weight matrix
        );
        adamw_update_param!(
            block.mlp.fc2.bias,
            block_grads.mlp.fc2_bias,
            m_block.mlp.fc2_bias,
            v_block.mlp.fc2_bias,
            false // No decay on bias
        );
    }

    // Final layer norm (no decay on 1D scale/shift parameters)
    adamw_update_param!(
        model.ln_final.gamma,
        grads.ln_final_gamma,
        optimizer.m_ln_final_gamma,
        optimizer.v_ln_final_gamma,
        false // No decay on LayerNorm
    );
    adamw_update_param!(
        model.ln_final.beta,
        grads.ln_final_beta,
        optimizer.m_ln_final_beta,
        optimizer.v_ln_final_beta,
        false // No decay on LayerNorm
    );

    // Output projection weight (decay on 2D weight matrix)
    adamw_update_param!(
        model.output_weight,
        grads.output_weight,
        optimizer.m_output_weight,
        optimizer.v_output_weight,
        true // Decay on weight matrix
    );
}
