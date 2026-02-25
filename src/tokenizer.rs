use std::{collections::HashMap, fs, path::Path};

use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use serde::{Deserialize, Serialize};

// BPE tokenizer
#[derive(Serialize, Deserialize)]
pub struct Tokenizer {
    // Maps token strings to IDs
    vocabulary: HashMap<String, usize>,
    // Merge rules in order
    merges: Vec<(String, String)>,
    //
    unk_token: String,
}

impl Tokenizer {
    // Each possible byte value (0-255) is represented as tokens (encoded as hex strings ("<00>", "<01>", ...))
    pub fn new(_vocab_size: usize) -> Self {
        let mut vocabulary = HashMap::new();

        for byte in 0..=255 {
            vocabulary.insert(format!("<{:02x}>", byte), vocabulary.len());
        }

        Self {
            vocabulary,
            merges: Vec::new(),
            unk_token: "unk".to_string(),
        }
    }

    // We store tokens as hex strings
    // This can be improwed using integers
    // 3 numbers: token_a, token_b, new_token
    pub fn train(&mut self, text: &str, vocab_size: usize) {
        // we already have vocabulary with 256 tokens created in constructor
        if vocab_size <= 256 {
            return;
        }

        println!("Training tokenizer...");
        println!("Starting vocabulary size: {}", self.vocabulary.len());
        println!("Target vocabulary size: {}", vocab_size);
        println!("Training text size: {}", text.len());

        let num_merges = vocab_size - 256;

        // If num_of_merges/desired vocabulary size is large we will train on smaller chunk of text - 200kb
        // Sufficent to learn but much faster

        let training_text = if num_merges > 2000 && text.len() > 200_000 {
            let sample_size = 200_000;
            println!("Training on 200kb sample size for performance");
            &text[..sample_size]
        } else {
            text
        };

        // Convert text to tokens
        let mut tokens: Vec<String> = training_text
            .bytes()
            .map(|b| format!("<{:02x}>", b))
            .collect();

        let mut new_tokens = Vec::with_capacity(tokens.len());

        for merge_idx in 0..num_merges {
            // Parallelization
            let chunk_size = 50_000.max(tokens.len() / rayon::current_num_threads().max(1));

            let pair_counts = tokens
                // parallel iterator over chunks of chunk_size size
                .par_chunks(chunk_size)
                // enumerated (idx, chunk)
                .enumerate()
                // Each thread gets local_counts initialized to HashMap::new (identity),
                .fold(HashMap::new, |mut local_counts, (chunk_idx, chunk)| {
                    // for each chunk we advance window of 2 items (our pair) and are adding it to map or inc its count
                    for window in chunk.windows(2) {
                        let pair = (window[0].clone(), window[1].clone());
                        *local_counts.entry(pair).or_insert(0) += 1;
                    }

                    // chunk boundaries - if pair spanning to next chunk

                    // If not last chunk
                    if chunk_idx * chunk_size + chunk.len() < tokens.len() {
                        // Get last token from chunk
                        if let Some(last) = chunk.last() {
                            // Get first token from next chunk
                            if let Some(next) = tokens.get(chunk_idx * chunk_size + chunk.len()) {
                                let pair = (last.clone(), next.clone());
                                *local_counts.entry(pair).or_insert(0) += 1;
                            }
                        }
                    }
                    local_counts
                })
                // combine all pairs from all chunks
                .reduce(HashMap::new, |mut a, b| {
                    for (pair, count) in b {
                        *a.entry(pair).or_insert(0) += count;
                    }
                    a
                });

            if pair_counts.is_empty() {
                break;
            }

            // HashMap iteration is random
            // Sort pairs to always have same merges
            // 1. By count desc
            // 2. lexicographically asc

            let mut pairs: Vec<((String, String), usize)> = pair_counts.into_iter().collect();
            pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            // We only care about one pair that has biggest count (is lex.. earlier if counts are same)
            // We then replace it by new token
            // And run merge all over again as next "strongest" pair count can be different then second best from this run
            // It can be pair containg our current best pair
            let (best_pair, count) = pairs[0].clone();

            let new_token = format!("{}{}", best_pair.0, best_pair.1);

            // Add pair to vocabulary
            self.vocabulary
                .insert(new_token.clone(), self.vocabulary.len());
            // Add pair to merges list
            self.merges.push(best_pair.clone());

            // We updated vocabulary and merges list
            // Last thing is to update out text tokens list

            new_tokens.clear();

            let mut i = 0;
            while i < tokens.len() {
                // We found pair, replace it with merged token
                if i < tokens.len() - 1 && tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1
                {
                    new_tokens.push(new_token.clone());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }

            std::mem::swap(&mut tokens, &mut new_tokens);

            // log progress every 50 merges
            if merge_idx % 50 == 0 {
                println!(
                    "  Merge {}/{}: {:?} (count: {}) -> vocabulary size: {}",
                    merge_idx + 1,
                    num_merges,
                    best_pair,
                    count,
                    self.vocabulary.len()
                );
            }
        }

        println!(
            "Training complete! Final vocabulary size: {}",
            self.vocabulary.len()
        );
        println!("Learned {} merges\n", self.merges.len());
    }

    // Text -> Bytes -> Token Ids
    // Uses parallelization if text > 200kb

    pub fn encode(&self, text: &str) -> Vec<usize> {
        const CHUNK_SIZE: usize = 100_000;

        // If more then 2 chunks we use parallelization
        if text.len() > CHUNK_SIZE * 2 {
            // Split text to chunks
            // Do not apply merges across chunk boundaries as they are rare (chunk size is large == small number of boundaries)
            // This has small impact on compression
            // So if it is on boundary it will be just 2 smaller tokens instead of one large

            let mut chunks = Vec::new();
            let mut start = 0;

            while start < text.len() {
                let mut end = (start + CHUNK_SIZE).min(text.len());

                // Advancing end of chunk to not split utf-8 chars
                while end < text.len() && !text.is_char_boundary(end) {
                    end += 1;
                }

                chunks.push(&text[start..end]);

                start = end;
            }

            let encoded_chunks: Vec<Vec<usize>> = chunks
                .par_iter()
                .map(|chunk| self.encode_sequential(chunk))
                .collect();

            let mut result = Vec::new();

            for chunk in encoded_chunks {
                result.extend_from_slice(&chunk);
            }

            result
        } else {
            self.encode_sequential(text)
        }
    }

    // Encodes text to token ids
    // Can be improved using trie-based lookup or cached encoding
    fn encode_sequential(&self, text: &str) -> Vec<usize> {
        // Convert bytes to tokens
        let mut tokens = self.byte_encode(text);

        let mut new_tokens = Vec::with_capacity(tokens.len());

        // Apply merges
        for (pair_a, pair_b) in &self.merges {
            let merged = format!("{}{}", pair_a, pair_b);

            new_tokens.clear();

            let mut i = 0;

            while i < tokens.len() {
                if i < tokens.len() - 1 && tokens[i] == *pair_a && tokens[i + 1] == *pair_b {
                    new_tokens.push(merged.clone());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }

            std::mem::swap(&mut tokens, &mut new_tokens);
        }

        // Convert text with merged tokens to token ids
        tokens
            .iter()
            .map(|token| *self.vocabulary.get(token).unwrap_or(&0))
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let id_to_token: HashMap<usize, String> = self
            .vocabulary
            .iter()
            .map(|(token, id)| (*id, token.clone()))
            .collect();

        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|id| id_to_token.get(id).cloned())
            .collect();

        let merged = tokens.join("");
        self.decode_token(&merged)
    }

    // hex encoded token string: "<68><65><6c><6c><6f>"
    // to utf-8 text
    // Reminder:
    // token string even with merged tokens looks still the same like above
    // merges only are visible at ids level so
    // above token string can be represented as 5, 4 or ... tokens based on merges.
    // What i want to say is: You were confused why we decode each <xx> separately but we apply merges.
    // but merges are only at token id  level, when we go from token ids to tokens we get string of separate token
    // like the one above
    fn decode_token(&self, token: &str) -> String {
        let mut bytes = Vec::new();

        let mut chars = token.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '<' {
                let mut hex_str = String::new();
                while let Some(&next_ch) = chars.peek() {
                    if next_ch == '>' {
                        chars.next();
                        break;
                    }
                    hex_str.push(chars.next().unwrap());
                }

                if let Ok(byte) = u8::from_str_radix(&hex_str, 16) {
                    bytes.push(byte);
                }
            }
        }

        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let tokenizer = serde_json::from_str(&json)?;
        Ok(tokenizer)
    }

    pub fn analyze_vocabulary(&self, sample_text: &str) {
        println!("\n=== Vocabulary Analysis ===\n");

        // Find human-readable tokens (merged tokens, not just base bytes)
        let mut readable_tokens: Vec<(String, usize)> = self
            .vocabulary
            .iter()
            .filter(|(token, _)| !token.starts_with('<') || token.len() > 4)
            .map(|(token, id)| (token.clone(), *id))
            .collect();

        // Sort by token ID (roughly reflects merge order during training)
        readable_tokens.sort_by_key(|(_, id)| *id);

        // Display token type breakdown
        let base_tokens = 256;
        let merged_tokens = self.vocabulary.len() - base_tokens;
        println!("Token Composition:");
        println!("  Base tokens (bytes): {}", base_tokens);
        println!("  Learned merges: {}", merged_tokens);
        println!("  Total vocabulary: {}\n", self.vocabulary.len());

        // Show sample of learned tokens
        println!("Sample of Learned Tokens (first 30):");
        let display_count = 30.min(readable_tokens.len());
        for (token, id) in readable_tokens.iter().take(display_count) {
            // Try to decode token for display
            let decoded = self.decode_token(token);
            if decoded.len() <= 20 && !decoded.is_empty() {
                println!("  [{}] \"{}\"", id, decoded);
            }
        }

        // Analyze compression on sample text
        if !sample_text.is_empty() {
            println!("\nCompression Analysis (on sample):");
            let sample_chars: String = sample_text.chars().take(10000).collect();
            let tokens = self.encode(&sample_chars);
            let char_count = sample_chars.len();
            let token_count = tokens.len();
            let compression_ratio = char_count as f32 / token_count as f32;

            println!("  Sample size: {} characters", char_count);
            println!("  Token count: {} tokens", token_count);
            println!("  Compression ratio: {:.2}x", compression_ratio);
            println!("  Avg chars per token: {:.1}", compression_ratio);
        }

        // Show example tokenizations
        println!("\nExample Tokenizations:");
        let examples = vec![
            "To be, or not to be",
            "Romeo and Juliet",
            "Wherefore art thou",
            "The quality of mercy",
        ];

        for example in examples {
            let tokens = self.encode(example);
            let token_strs: Vec<String> = tokens
                .iter()
                .map(|&id| {
                    // Find token string for this ID
                    self.vocabulary
                        .iter()
                        .find(|(_, v)| **v == id)
                        .map(|(k, _)| self.decode_token(k))
                        .unwrap_or_else(|| "?".to_string())
                })
                .collect();
            println!(
                "  \"{}\" -> {} tokens: [{}]",
                example,
                tokens.len(),
                token_strs.join("|")
            );
        }

        println!("\n{}\n", "=".repeat(60));
    }

    fn byte_encode(&self, text: &str) -> Vec<String> {
        text.bytes().map(|b| format!("<{:02x}>", b)).collect()
    }

    pub fn stats(&self) -> TokenizerStats {
        TokenizerStats {
            vocab_size: self.vocab_size(),
            num_merges: self.merges.len(),
            base_tokens: 256,
        }
    }
}

// Note about stats:
// Higher size of vocab gives better compression
// But slows down encoding time.
// Need to find sweet spot
#[derive(Debug)]
pub struct TokenizerStats {
    // base tokens + learned merges
    pub vocab_size: usize,
    // num of merge rules learned
    pub num_merges: usize,
    // always 256 for byte-level BPE
    pub base_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_token_single_byte() {
        let tokenizer = Tokenizer::new(256);

        // Single byte token: 'h' = 0x68
        let result = tokenizer.decode_token("<68>");
        assert_eq!(result, "h");
    }

    #[test]
    fn test_decode_token_multiple_bytes() {
        let tokenizer = Tokenizer::new(256);

        // Multiple bytes: "hello"
        let result = tokenizer.decode_token("<68><65><6c><6c><6f>");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_decode_token_with_space() {
        let tokenizer = Tokenizer::new(256);

        // "hi " (with space, 0x20)
        let result = tokenizer.decode_token("<68><69><20>");
        assert_eq!(result, "hi ");
    }

    #[test]
    fn test_decode_token_utf8_multibyte() {
        let tokenizer = Tokenizer::new(256);

        // "é" in UTF-8 is [0xc3, 0xa9]
        let result = tokenizer.decode_token("<c3><a9>");
        assert_eq!(result, "é");
    }

    #[test]
    fn test_decode_token_empty() {
        let tokenizer = Tokenizer::new(256);

        let result = tokenizer.decode_token("");
        assert_eq!(result, "");
    }

    #[test]
    fn test_decode_basic() {
        let tokenizer = Tokenizer::new(256);

        // Create a simple test: encode "hello" as individual bytes
        let text = "hello";
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tokenizer = Tokenizer::new(256);

        let test_cases = vec![
            "hello",
            "Hello, world!",
            "To be, or not to be",
            "123 456 789",
            "special chars: !@#$%^&*()",
            "newline\nand\ttab",
            "UTF-8: café, naïve, 日本語",
        ];

        for text in test_cases {
            let encoded = tokenizer.encode(text);
            let decoded = tokenizer.decode(&encoded);
            assert_eq!(decoded, text, "Failed roundtrip for: {}", text);
        }
    }

    #[test]
    fn test_encode_decode_with_merges() {
        // Create tokenizer and train it
        let mut tokenizer = Tokenizer::new(300);
        let training_text = "hello hello world world hello";
        tokenizer.train(training_text, 300);

        // Test that encode/decode still works after training
        let test_text = "hello world";
        let encoded = tokenizer.encode(test_text);
        let decoded = tokenizer.decode(&encoded);

        assert_eq!(decoded, test_text);
    }

    #[test]
    fn test_decode_token_consistency_with_decode() {
        let tokenizer = Tokenizer::new(256);

        // Test that decode_token produces same result as decode for single token
        let token_str = "<68><65><6c><6c><6f>"; // "hello"

        // Direct decode_token
        let direct_result = tokenizer.decode_token(token_str);

        // Simulate what decode does: parse the token string as if it came from vocab
        let simulated_result = tokenizer.decode_token(token_str);

        assert_eq!(direct_result, simulated_result);
        assert_eq!(direct_result, "hello");
    }

    #[test]
    fn test_decode_token_concatenated() {
        let tokenizer = Tokenizer::new(256);

        // Multiple tokens concatenated (what decode does before calling decode_token)
        let concatenated = "<68><65><6c><6c><6f><20><77><6f><72><6c><64>";
        let result = tokenizer.decode_token(concatenated);

        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_vocab_size() {
        let tokenizer = Tokenizer::new(256);
        assert_eq!(tokenizer.vocab_size(), 256);

        let mut tokenizer2 = Tokenizer::new(512);
        tokenizer2.train("hello hello world", 512);
        // Note: actual vocab size depends on how many unique pairs exist in the corpus
        // Small corpus won't reach target vocab size, so just verify it increased
        assert!(tokenizer2.vocab_size() > 256);
        assert!(tokenizer2.vocab_size() <= 512);
    }

    #[test]
    fn test_base_vocab_coverage() {
        let tokenizer = Tokenizer::new(256);

        // All byte values should be encodable
        for byte in 0u8..=255u8 {
            let text = String::from_utf8(vec![byte]).unwrap_or_else(|_| {
                // For invalid UTF-8, create string from bytes using from_utf8_lossy
                String::from_utf8_lossy(&[byte]).to_string()
            });

            let encoded = tokenizer.encode(&text);
            let decoded = tokenizer.decode(&encoded);

            // Should roundtrip correctly
            assert_eq!(decoded, text);
        }
    }
}
