use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};

// Multidimensional array for neural network computations
// Stores data with contiguous Vec<f32> and information about shape of data
//
// For shape [2,3] data is stored as [r0c0, r0c1, r0c2, r1c0, r1c1, r1c2]
// and strides are [3,1] meaning
// moving one step in zero dimension advances position by 3 in data
// moving one step in first dimension advances position by 1 in data
#[derive(Clone, Debug)]
pub struct Tensor {
    // Flat array of all tensor elements
    pub data: Vec<f32>,
    // Shape of tensor (tensor dimensions)
    pub shape: Vec<usize>,
    // Data computed from shape indicating how to move across dimensions
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_size: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_size,
            "Data length ({}) doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_size
        );

        let strides = Self::compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let data = vec![0.0; size];

        Self::new(data, shape)
    }

    // for shape [d0, d1, d2] strides are [d1*d2, d2, 1]
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];

        // 1,0
        for i in (0..shape.len().saturating_sub(1)).rev() {
            // println!("{}", i);
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        strides
    }

    // MATH

    // Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // 2D x 2D

        if self.shape.len() == 2 && other.shape.len() == 2 {
            assert_eq!(
                self.shape[1], other.shape[0],
                "Matrix dimensions incompatible: [{}, {}] @ [{}, {}]",
                self.shape[0], self.shape[1], other.shape[0], other.shape[1]
            );

            let m = self.shape[0];
            let n = other.shape[1];
            let k = self.shape[1];

            // Use parallel version for larger matrices (work threshold: 1000 operations)
            // This threshold balances parallel overhead against performance gains
            if m * n * k >= 1_000 {
                return self.matmul_parallel_blocked(other, m, n, k);
            }

            // Sequential version for small matrices (avoids parallel overhead)
            let mut result = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += self.data[i * k + l] * other.data[l * n + j];
                    }
                    result[i * n + j] = sum;
                }
            }

            return Tensor::new(result, vec![m, n]);
        }

        // 4D x 4D

        if self.shape.len() == 4 && other.shape.len() == 4 {
            let batch = self.shape[0];
            let n_heads = self.shape[1];
            let seq1 = self.shape[2];
            let inner_dim = self.shape[3];
            let seq2 = other.shape[3];

            assert_eq!(
                other.shape[2], inner_dim,
                "Inner dimensions must match for batched matmul"
            );

            let total_size = batch * n_heads * seq1 * seq2;
            let mut result = vec![0.0; total_size];

            // Parallelize over (batch, head) combinations
            // Each (batch, head) pair computes an independent seq1×seq2 matrix multiplication
            result
                .par_chunks_mut(seq1 * seq2)
                .enumerate()
                .for_each(|(bh_idx, chunk)| {
                    let b = bh_idx / n_heads;
                    let h = bh_idx % n_heads;

                    // Compute 2D matmul for this batch/head
                    for i in 0..seq1 {
                        for j in 0..seq2 {
                            let mut sum = 0.0;
                            for l in 0..inner_dim {
                                let self_idx = ((b * n_heads + h) * seq1 + i) * inner_dim + l;
                                let other_idx = ((b * n_heads + h) * inner_dim + l) * seq2 + j;
                                sum += self.data[self_idx] * other.data[other_idx];
                            }
                            chunk[i * seq2 + j] = sum;
                        }
                    }
                });

            return Tensor::new(result, vec![batch, n_heads, seq1, seq2]);
        }

        panic!(
            "Unsupported matmul shapes: {:?} @ {:?}",
            self.shape, other.shape
        );
    }

    // Parallel multiplication of 2D matrices
    // Uses cache blocking
    // Inner loops access memory sequentially
    fn matmul_parallel_blocked(&self, other: &Tensor, m: usize, n: usize, k: usize) -> Tensor {
        // Block size for cache optimization
        // 8×8 blocks = 256 bytes (fits well in L1 cache: typically 32-64KB)
        const BLOCK_SIZE: usize = 8;

        let mut result = vec![0.0; m * n];

        // Parallelize over output row blocks
        // Each thread processes BLOCK_SIZE rows independently
        result
            .par_chunks_mut(BLOCK_SIZE * n)
            .enumerate()
            .for_each(|(block_i, result_block)| {
                let i_start = block_i * BLOCK_SIZE;
                let i_end = (i_start + BLOCK_SIZE).min(m);

                // Iterate over column blocks
                for j_start in (0..n).step_by(BLOCK_SIZE) {
                    let j_end = (j_start + BLOCK_SIZE).min(n);

                    // Iterate over inner dimension blocks
                    for k_start in (0..k).step_by(BLOCK_SIZE) {
                        let k_end = (k_start + BLOCK_SIZE).min(k);

                        // Compute this block (cache-friendly inner loops)
                        for i in i_start..i_end {
                            let row_offset = (i - i_start) * n;
                            for k_idx in k_start..k_end {
                                let a_val = self.data[i * k + k_idx];

                                // SIMD-optimized innermost loop
                                Self::matmul_inner_simd(
                                    a_val,
                                    &other.data[k_idx * n + j_start..k_idx * n + j_end],
                                    &mut result_block[row_offset + j_start..row_offset + j_end],
                                );
                            }
                        }
                    }
                }
            });

        Tensor::new(result, vec![m, n])
    }

    // SIMD
    #[inline(always)]
    fn matmul_inner_simd(a_val: f32, b: &[f32], result: &mut [f32]) {
        // Simple loop that LLVM can auto-vectorize
        for (r, &b_val) in result.iter_mut().zip(b.iter()) {
            *r += a_val * b_val;
        }
    }

    // Computes softmax along the specified axis
    // Softmax converts prediction scores into probabilities (percentage value)
    pub fn softmax(&self, axis: isize) -> Tensor {
        // Convert negative axis to positive
        let axis_pos = if axis < 0 {
            (self.shape.len() as isize + axis) as usize
        } else {
            axis as usize
        };

        // === 2D SOFTMAX PER ROW (common case for attention) ===
        if self.shape.len() == 2 && axis_pos == 1 {
            let rows = self.shape[0];
            let cols = self.shape[1];

            // Parallel softmax computation per row
            let result: Vec<f32> = (0..rows)
                .into_par_iter()
                .flat_map_iter(|i| {
                    let start = i * cols;
                    let end = start + cols;
                    let row = &self.data[start..end];

                    // Find max for numerical stability
                    let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                    // Compute exp(x - max)
                    let exp_values: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();

                    // Normalize
                    let sum: f32 = exp_values.iter().sum();
                    exp_values.into_iter().map(move |val| val / sum)
                })
                .collect();

            return Tensor::new(result, self.shape.clone());
        }

        // === FALLBACK: GLOBAL SOFTMAX ===
        // Less common, but included for completeness
        let max = self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = self.data.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exp_values.iter().sum();
        let result = exp_values.iter().map(|&x| x / sum).collect();

        Tensor::new(result, self.shape.clone())
    }

    // Element-wise addition with broadcasting support
    pub fn add(&self, other: &Tensor) -> Tensor {
        // === EXACT MATCH: Same shape ===
        if self.shape == other.shape {
            let result = self
                .data
                .par_iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect();
            return Tensor::new(result, self.shape.clone());
        }

        // === BROADCAST BATCH: [batch, seq, dim] + [seq, dim] ===
        if self.shape.len() == 3 && other.shape.len() == 2 {
            let batch_size = self.shape[0];
            let seq_len = self.shape[1];
            let dim = self.shape[2];

            assert_eq!(
                other.shape[0], seq_len,
                "Sequence length must match for broadcasting"
            );
            assert_eq!(other.shape[1], dim, "Dimension must match for broadcasting");

            let result: Vec<f32> = (0..batch_size * seq_len * dim)
                .into_par_iter()
                .map(|i| {
                    let s = (i / dim) % seq_len;
                    let d = i % dim;
                    let other_idx = s * dim + d;
                    self.data[i] + other.data[other_idx]
                })
                .collect();
            return Tensor::new(result, self.shape.clone());
        }

        // === BROADCAST LAST DIM: [*, n] + [n] (e.g., bias addition) ===
        if self.shape.len() > other.shape.len() {
            let last_dim = *self.shape.last().unwrap();
            if other.data.len() == last_dim {
                let result: Vec<f32> = (0..self.data.len())
                    .into_par_iter()
                    .map(|i| {
                        let other_idx = i % last_dim;
                        self.data[i] + other.data[other_idx]
                    })
                    .collect();
                return Tensor::new(result, self.shape.clone());
            }
        }

        panic!(
            "Unsupported broadcast for add: {:?} + {:?}",
            self.shape, other.shape
        );
    }

    // Element-wise multiplication with broadcasting
    pub fn mul(&self, other: &Tensor) -> Tensor {
        // Exact match
        if self.shape == other.shape {
            let result = self
                .data
                .par_iter()
                .zip(&other.data)
                .map(|(a, b)| a * b)
                .collect();
            return Tensor::new(result, self.shape.clone());
        }

        // Broadcast last dimension
        if self.shape.len() > other.shape.len() {
            let last_dim = *self.shape.last().unwrap();
            if other.data.len() == last_dim {
                let result: Vec<f32> = (0..self.data.len())
                    .into_par_iter()
                    .map(|i| {
                        let other_idx = i % last_dim;
                        self.data[i] * other.data[other_idx]
                    })
                    .collect();
                return Tensor::new(result, self.shape.clone());
            }
        }

        panic!(
            "Unsupported broadcast for mul: {:?} * {:?}",
            self.shape, other.shape
        );
    }

    // Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shapes must match for subtraction");
        let result = self
            .data
            .par_iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Tensor::new(result, self.shape.clone())
    }

    // Element-wise division
    pub fn div(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shapes must match for division");
        let result = self
            .data
            .par_iter()
            .zip(&other.data)
            .map(|(a, b)| a / b)
            .collect();
        Tensor::new(result, self.shape.clone())
    }

    // Add scalar to all elements
    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let result = self.data.par_iter().map(|&x| x + scalar).collect();
        Tensor::new(result, self.shape.clone())
    }

    // Multiply all elements by scalar
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let result = self.data.par_iter().map(|&x| x * scalar).collect();
        Tensor::new(result, self.shape.clone())
    }

    // Divide all elements by scalar
    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        let result = self.data.par_iter().map(|&x| x / scalar).collect();
        Tensor::new(result, self.shape.clone())
    }

    // Element-wise square root
    pub fn sqrt(&self) -> Tensor {
        let result = self.data.par_iter().map(|&x| x.sqrt()).collect();
        Tensor::new(result, self.shape.clone())
    }

    // Reshape tensor to new shape
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            self.data.len(),
            new_size,
            "Cannot reshape: element count mismatch"
        );
        Tensor::new(self.data.clone(), new_shape.to_vec())
    }

    // Transpose two dimensions
    pub fn transpose(&self, dim1: isize, dim2: isize) -> Tensor {
        let ndim = self.shape.len() as isize;

        // Convert negative indices
        let d1 = if dim1 < 0 { ndim + dim1 } else { dim1 } as usize;
        let d2 = if dim2 < 0 { ndim + dim2 } else { dim2 } as usize;

        // Create new shape with swapped dimensions
        let mut new_shape = self.shape.clone();
        new_shape.swap(d1, d2);

        // For 2D matrices, we can use a simple transpose
        if self.shape.len() == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let mut result = vec![0.0; rows * cols];

            for i in 0..rows {
                for j in 0..cols {
                    result[j * rows + i] = self.data[i * cols + j];
                }
            }

            return Tensor::new(result, new_shape);
        }

        // For higher dimensions, do full transpose with stride remapping
        let old_strides = &self.strides;
        let mut new_strides = old_strides.clone();
        new_strides.swap(d1, d2);

        let total_size = self.data.len();
        let mut result = vec![0.0; total_size];

        for (i, item) in result.iter_mut().enumerate().take(total_size) {
            // Compute old multi-index from flat index
            let mut old_idx = 0;
            let mut remaining = i;

            for (dim_idx, &stride) in new_strides.iter().enumerate() {
                let coord = remaining / stride;
                remaining %= stride;
                old_idx += coord * old_strides[dim_idx];
            }

            *item = self.data[old_idx];
        }

        Tensor::new(result, new_shape)
    }

    // Replace values where mask is true with given value
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Tensor {
        assert_eq!(self.shape, mask.shape, "Mask shape must match tensor shape");
        let result = self
            .data
            .par_iter()
            .zip(&mask.data)
            .map(|(&x, &m)| if m != 0.0 { value } else { x })
            .collect();
        Tensor::new(result, self.shape.clone())
    }

    // Compute mean (average) along an axis
    pub fn mean(&self, axis: isize, keepdim: bool) -> Tensor {
        let axis_pos = if axis < 0 {
            (self.shape.len() as isize + axis) as usize
        } else {
            axis as usize
        };

        // For 2D tensor, compute mean along specified axis
        if self.shape.len() == 2 && axis_pos == 1 {
            // Mean along columns (result has shape [rows, 1] or [rows])
            let rows = self.shape[0];
            let cols = self.shape[1];

            let result: Vec<f32> = (0..rows)
                .into_par_iter()
                .map(|i| {
                    let start = i * cols;
                    let end = start + cols;
                    let sum: f32 = self.data[start..end].iter().sum();
                    sum / cols as f32
                })
                .collect();

            let new_shape = if keepdim { vec![rows, 1] } else { vec![rows] };
            return Tensor::new(result, new_shape);
        }

        // For 3D tensor [batch, seq, dim], compute mean along last axis
        if self.shape.len() == 3 && axis_pos == 2 {
            let batch = self.shape[0];
            let seq = self.shape[1];
            let dim = self.shape[2];

            let result: Vec<f32> = (0..batch * seq)
                .into_par_iter()
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    let sum: f32 = self.data[start..end].iter().sum();
                    sum / dim as f32
                })
                .collect();

            let new_shape = if keepdim {
                vec![batch, seq, 1]
            } else {
                vec![batch, seq]
            };
            return Tensor::new(result, new_shape);
        }

        panic!("Unsupported mean operation for shape {:?}", self.shape);
    }

    // Compute variance (how spread out are values from mean/average) along an axis
    pub fn var(&self, axis: isize, keepdim: bool) -> Tensor {
        let axis_pos = if axis < 0 {
            (self.shape.len() as isize + axis) as usize
        } else {
            axis as usize
        };

        // For 3D tensor [batch, seq, dim], compute variance along last axis
        if self.shape.len() == 3 && axis_pos == 2 {
            let batch = self.shape[0];
            let seq = self.shape[1];
            let dim = self.shape[2];

            let result: Vec<f32> = (0..batch * seq)
                .into_par_iter()
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    let slice = &self.data[start..end];

                    // Compute mean
                    let mean: f32 = slice.iter().sum::<f32>() / dim as f32;

                    // Compute variance
                    let variance: f32 = slice
                        .iter()
                        .map(|&x| {
                            let diff = x - mean;
                            diff * diff
                        })
                        .sum::<f32>()
                        / dim as f32;

                    variance
                })
                .collect();

            let new_shape = if keepdim {
                vec![batch, seq, 1]
            } else {
                vec![batch, seq]
            };
            return Tensor::new(result, new_shape);
        }

        panic!("Unsupported var operation for shape {:?}", self.shape);
    }

    // Create a tensor with sequential integers
    pub fn arange(start: usize, end: usize) -> Tensor {
        let data: Vec<f32> = (start..end).map(|i| i as f32).collect();
        let len = data.len();
        Tensor::new(data, vec![len])
    }
}
