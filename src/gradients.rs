// Utils for working with gradients

use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::gpt2_trainable::GPT2Gradients;

// Compute the L2 norm of all gradients
//
// The gradient norm is the square root of the sum of all squared gradient values
// across all parameters in the model. This gives a single number representing
// the overall magnitude of the gradient update.
pub fn compute_grad_norm(grads: &GPT2Gradients) -> f32 {
    // Helper to compute sum of squares in parallel
    let sum_sq_parallel = |data: &Vec<f32>| -> f32 { data.par_iter().map(|&val| val * val).sum() };

    let mut sum_sq = 0.0;

    // Token and position embeddings
    sum_sq += sum_sq_parallel(&grads.token_embedding.data);
    sum_sq += sum_sq_parallel(&grads.position_embedding.data);

    // All transformer blocks
    for block_grad in &grads.block_grads {
        // LayerNorm 1
        sum_sq += sum_sq_parallel(&block_grad.ln1_gamma.data);
        sum_sq += sum_sq_parallel(&block_grad.ln1_beta.data);

        // Attention
        sum_sq += sum_sq_parallel(&block_grad.attn.q_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.q_bias.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.k_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.k_bias.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.v_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.v_bias.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.out_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.attn.out_bias.data);

        // LayerNorm 2
        sum_sq += sum_sq_parallel(&block_grad.ln2_gamma.data);
        sum_sq += sum_sq_parallel(&block_grad.ln2_beta.data);

        // MLP (feedforward network)
        sum_sq += sum_sq_parallel(&block_grad.mlp.fc1_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.mlp.fc1_bias.data);
        sum_sq += sum_sq_parallel(&block_grad.mlp.fc2_weight.data);
        sum_sq += sum_sq_parallel(&block_grad.mlp.fc2_bias.data);
    }

    // Final layer norm
    sum_sq += sum_sq_parallel(&grads.ln_final_gamma.data);
    sum_sq += sum_sq_parallel(&grads.ln_final_beta.data);

    // Output projection weight
    sum_sq += sum_sq_parallel(&grads.output_weight.data);

    sum_sq.sqrt()
}

// Clip gradients to a maximum norm
//
// When the gradient norm exceeds `max_norm`, all gradients are scaled
// proportionally to bring the norm down to exactly `max_norm`. This prevents
// gradient explosion while preserving the direction of the gradient update.
pub fn clip_gradients(grads: &mut GPT2Gradients, max_norm: f32) {
    let norm = compute_grad_norm(grads);

    // Only clip if norm exceeds threshold
    if norm > max_norm {
        let scale = max_norm / norm;

        // Helper to scale tensor data in parallel
        let scale_parallel = |data: &mut Vec<f32>| {
            data.par_iter_mut().for_each(|val| *val *= scale);
        };

        // Scale all gradients by the same factor

        // Token and position embeddings
        scale_parallel(&mut grads.token_embedding.data);
        scale_parallel(&mut grads.position_embedding.data);

        // All transformer blocks
        for block_grad in &mut grads.block_grads {
            // LayerNorm 1
            scale_parallel(&mut block_grad.ln1_gamma.data);
            scale_parallel(&mut block_grad.ln1_beta.data);

            // Attention
            scale_parallel(&mut block_grad.attn.q_weight.data);
            scale_parallel(&mut block_grad.attn.q_bias.data);
            scale_parallel(&mut block_grad.attn.k_weight.data);
            scale_parallel(&mut block_grad.attn.k_bias.data);
            scale_parallel(&mut block_grad.attn.v_weight.data);
            scale_parallel(&mut block_grad.attn.v_bias.data);
            scale_parallel(&mut block_grad.attn.out_weight.data);
            scale_parallel(&mut block_grad.attn.out_bias.data);

            // LayerNorm 2
            scale_parallel(&mut block_grad.ln2_gamma.data);
            scale_parallel(&mut block_grad.ln2_beta.data);

            // MLP (feedforward network)
            scale_parallel(&mut block_grad.mlp.fc1_weight.data);
            scale_parallel(&mut block_grad.mlp.fc1_bias.data);
            scale_parallel(&mut block_grad.mlp.fc2_weight.data);
            scale_parallel(&mut block_grad.mlp.fc2_bias.data);
        }

        // Final layer norm
        scale_parallel(&mut grads.ln_final_gamma.data);
        scale_parallel(&mut grads.ln_final_beta.data);

        // Output projection weight
        scale_parallel(&mut grads.output_weight.data);
    }
}
