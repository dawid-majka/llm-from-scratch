use rayon::prelude::*;

use crate::tensor::Tensor;

pub fn gelu_forward(x: &Tensor) -> Tensor {
    let result = x
        .data
        .par_iter()
        .map(|&val| {
            0.5 * val
                * (1.0
                    + ((2.0 / std::f32::consts::PI).sqrt() * (val + 0.044715 * val.powi(3))).tanh())
        })
        .collect();
    Tensor::new(result, x.shape.clone())
}

pub fn gelu_backward(grad_out: &Tensor, x: &Tensor) -> Tensor {
    let grad_data: Vec<f32> = x
        .data
        .par_iter()
        .zip(&grad_out.data)
        .map(|(&x_val, &grad_val)| {
            let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
            let inner = sqrt_2_pi * (x_val + 0.044715 * x_val.powi(3));
            let tanh_inner = inner.tanh();
            let sech_sq = 1.0 - tanh_inner * tanh_inner;

            let grad_gelu = 0.5 * (1.0 + tanh_inner)
                + 0.5 * x_val * sech_sq * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x_val.powi(2));

            grad_val * grad_gelu
        })
        .collect();

    Tensor::new(grad_data, x.shape.clone())
}
