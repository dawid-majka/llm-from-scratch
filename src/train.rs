use crate::tokenizer::Tokenizer;

// Each element is a Vec<Vec<usize>> with shape [batch_size][seq_len]
pub type Batch = (Vec<Vec<usize>>, Vec<Vec<usize>>);

// Loads text, tokenizes it, and provides batches of (input, target) sequence pairs
// for training language models.
pub struct TextDataLoader {
    tokens: Vec<usize>,
    seq_len: usize,
    batch_size: usize,
    position: usize,
}

impl TextDataLoader {
    // From text
    pub fn new(text: &str, tokenizer: &Tokenizer, seq_len: usize, batch_size: usize) -> Self {
        let tokens = tokenizer.encode(text);
        println!("Loaded {} tokens from text", tokens.len());

        Self {
            tokens,
            seq_len,
            batch_size,
            position: 0,
        }
    }

    // From file
    pub fn from_file(
        path: &str,
        tokenizer: &Tokenizer,
        seq_len: usize,
        batch_size: usize,
    ) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(Self::new(&text, tokenizer, seq_len, batch_size))
    }

    // Get the next batch of training data
    //
    // Returns a batch of (input, target) sequence pairs. The target is always
    // the input shifted by one position (next token prediction).
    pub fn next_batch(&mut self) -> Option<Batch> {
        // Check if we have enough tokens left for a full batch
        if self.position + self.batch_size * (self.seq_len + 1) >= self.tokens.len() {
            // Reset to beginning (epoch complete)
            self.position = 0;
            return None;
        }

        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        // Build batch by extracting sequences
        for _ in 0..self.batch_size {
            // Ensure we have enough tokens for this sequence
            if self.position + self.seq_len + 1 >= self.tokens.len() {
                break;
            }

            // Extract input sequence: tokens[pos..pos+seq_len]
            let input_seq = self.tokens[self.position..self.position + self.seq_len].to_vec();

            // Extract target sequence: tokens[pos+1..pos+seq_len+1]
            // This is the input shifted by 1 (next token prediction)
            let target_seq =
                self.tokens[self.position + 1..self.position + self.seq_len + 1].to_vec();

            inputs.push(input_seq);
            targets.push(target_seq);

            // Move forward by seq_len (non-overlapping sequences)
            self.position += self.seq_len;
        }

        if inputs.is_empty() {
            None
        } else {
            Some((inputs, targets))
        }
    }

    // Reset loader to begining
    pub fn reset(&mut self) {
        self.position = 0;
    }

    // Total num of batches per epoch
    pub fn num_batches(&self) -> usize {
        self.tokens.len() / (self.batch_size * self.seq_len)
    }
}

pub struct TrainingConfig {
    // Learning rate for optimizer
    pub learning_rate: f32,
    // Number of passes through the dataset
    pub num_epochs: usize,
    // Number of sequences per batch
    pub batch_size: usize,
    // Length of each training sequence
    pub seq_len: usize,
    // Print metrics every N steps
    pub print_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            num_epochs: 1,
            batch_size: 4,
            seq_len: 64,
            print_every: 100,
        }
    }
}

impl TrainingConfig {
    // TrainingConfig with small batch size and short sequences
    pub fn tiny() -> Self {
        Self {
            learning_rate: 3e-4,
            num_epochs: 3,
            batch_size: 8,
            seq_len: 64,
            print_every: 50,
        }
    }

    // Create a small configuration for medium experiments (overnight run)
    pub fn small() -> Self {
        Self {
            learning_rate: 3e-4,
            num_epochs: 5,
            batch_size: 16,
            seq_len: 128,
            print_every: 100,
        }
    }
}
