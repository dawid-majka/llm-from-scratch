use std::io::Write;
use std::{fs::File, time::Instant};

// Logs training metrics to console and CSV
pub struct TrainingLogger {
    log_file: File,
    start_time: Instant,
    last_log_time: Instant,
}

impl TrainingLogger {
    pub fn new(log_path: &str) -> std::io::Result<Self> {
        let mut log_file = File::create(log_path)?;

        // Write CSV header
        writeln!(
            log_file,
            "step,elapsed_seconds,learning_rate,train_loss,val_loss,train_perplexity,val_perplexity,sample"
        )?;

        let now = Instant::now();
        Ok(Self {
            log_file,
            start_time: now,
            last_log_time: now,
        })
    }

    // Writes metrics to CSV and prints to console with timing information.
    pub fn log(
        &mut self,
        step: usize,
        learning_rate: f32,
        train_loss: f32,
        val_loss: f32,
        sample: Option<&str>,
    ) -> std::io::Result<()> {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        // Perplexity = exp(loss)
        // This is a more interpretable metric than raw loss
        let train_perplexity = train_loss.exp();
        let val_perplexity = val_loss.exp();

        // Escape quotes in sample text for CSV format
        let sample_escaped = sample.map(|s| s.replace('"', "\"\"")).unwrap_or_default();

        // Write to CSV file
        writeln!(
            self.log_file,
            "{},{:.2},{:.6},{:.4},{:.4},{:.2},{:.2},\"{}\"",
            step,
            elapsed,
            learning_rate,
            train_loss,
            val_loss,
            train_perplexity,
            val_perplexity,
            sample_escaped
        )?;

        // Flush to ensure data is written immediately
        // This is important if training crashes - we don't lose data
        self.log_file.flush()?;

        // Print to console with timing info
        let step_time = self.last_log_time.elapsed().as_secs_f32();
        println!(
            "Step {:4} | Time: {:7.1}s (+{:.1}s) | LR: {:.6} | Train: {:.4} | Val: {:.4} | Perplexity: {:.2}",
            step, elapsed, step_time, learning_rate, train_loss, val_loss, val_perplexity
        );

        if let Some(text) = sample {
            println!("  Sample: \"{}\"", text);
        }

        self.last_log_time = Instant::now();
        Ok(())
    }
}

// Split tokenized data into training and validation sets
pub fn train_val_split(tokens: &[usize], val_fraction: f32) -> (&[usize], &[usize]) {
    let split_idx = ((tokens.len() as f32) * (1.0 - val_fraction)) as usize;
    (&tokens[..split_idx], &tokens[split_idx..])
}

// Compute average loss over a dataset
// tokens - Tokenized dataset
// seq_len - Sequence length per example
// num_batches - Number of batches to evaluate (limited by dataset size)
// compute_loss_fn - Function that computes loss for a single batch
pub fn compute_dataset_loss<F>(
    tokens: &[usize],
    seq_len: usize,
    num_batches: usize,
    mut compute_loss_fn: F,
) -> f32
where
    F: FnMut(&[usize], &[usize]) -> f32,
{
    // Need at least seq_len + 1 tokens (input + target)
    if tokens.len() < seq_len + 1 {
        return 0.0;
    }

    let mut total_loss = 0.0;

    // Limit num_batches to what's actually available in the dataset
    let max_batches = (tokens.len() - seq_len - 1) / seq_len;
    let num_batches = num_batches.min(max_batches);

    for batch_idx in 0..num_batches {
        // Extract input and target sequences
        // Target is shifted by 1 position (next token prediction)
        let start = (batch_idx * seq_len) % (tokens.len() - seq_len - 1);
        let input_seq = &tokens[start..start + seq_len];
        let target_seq = &tokens[start + 1..start + seq_len + 1];

        let loss = compute_loss_fn(input_seq, target_seq);
        total_loss += loss;
    }

    total_loss / num_batches as f32
}
