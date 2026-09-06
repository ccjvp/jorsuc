use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::hash::Hash;
use std::io::BufRead;
use std::ops::Add;

pub const N_CONTEXT: usize = 64;
const N_TOKENS: usize = 255;
const N_CHARS: usize = 6;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Token {
    pub bytes: [Option<u8>; N_CHARS],
}

impl Token {
    fn new(b: u8) -> Self {
        let mut bytes: [Option<u8>; N_CHARS] = [None; N_CHARS];
        bytes[0] = Some(b);

        Self { bytes }
    }
}

impl Add for Token {
    type Output = Token;

    fn add(self, other: Self) -> Token {
        let mut bytes = self.bytes;

        let mut j = 0;
        for i in 0..N_CHARS {
            if self.bytes[i].is_some() {
                continue;
            }

            if other.bytes[j].is_none() {
                break;
            }

            bytes[i] = other.bytes[j];
            j += 1;
        }

        Token { bytes }
    }
}

pub struct Tokenizer {
    pub rules: HashMap<(Token, Token), usize>,
    pub encoder: HashMap<Token, usize>,
    pub vocab: Vec<Token>,
    pub sequence: Vec<Token>,
    pub prev: Vec<Option<usize>>,
    pub next: Vec<Option<usize>>,
    pub dead: Vec<bool>,
    pub pairs: HashMap<(Token, Token), HashSet<usize>>,
    pub heap: BinaryHeap<(usize, (Token, Token))>,
    pub bos: usize,
}

impl Tokenizer {
    fn add_input(&mut self, input: &str) {
        let bytes = input.as_bytes();
        for (i, pair) in bytes.windows(2).enumerate() {
            let left = Token::new(pair[0]);
            let right = Token::new(pair[1]);
            let pos = self.sequence.len();

            self.sequence.push(left);
            self.pairs.entry((left, right)).or_default().insert(pos);
            self.prev.push(if i == 0 { None } else { Some(pos - 1) });
            self.next.push(Some(pos + 1));
        }

        if let Some(&b) = bytes.last() {
            let token = Token::new(b);
            self.prev.push(Some(self.sequence.len() - 1));
            self.sequence.push(token);
            self.next.push(None);
        }

        self.dead.resize(self.sequence.len(), false);
    }

    // Move position from one key (pair) to another
    fn rekey_position(
        &mut self,
        from_key: (Token, Token),
        to_key: (Token, Token),
        old_pos: usize,
        new_pos: usize,
        train: bool,
    ) {
        if let Some(positions) = self.pairs.get_mut(&from_key) {
            positions.remove(&old_pos);
        }

        self.pairs.entry(to_key).or_default().insert(new_pos);

        if train {
            let freq = self.pairs[&to_key].len();
            self.heap.push((freq, to_key));
        } else {
            if let Some(&rank) = self.rules.get(&to_key) {
                self.heap.push((self.rules.len() - rank, to_key));
            }
        }
    }

    fn merge_pair(&mut self, pair: (Token, Token), train: bool) {
        let mut rule_added = train == false;
        if let Some(positions) = self.pairs.remove(&pair) {
            for start in positions {
                if self.dead[start] {
                    continue;
                }

                if let Some(end) = self.next[start] {
                    // Tokens might have been mutated invalidating the pair
                    if (self.sequence[start], self.sequence[end]) != pair {
                        continue;
                    }

                    let merged = self.sequence[start] + self.sequence[end];
                    if rule_added == false {
                        let rank = self.rules.len();
                        self.rules.insert(pair, rank);
                        rule_added = true
                    };

                    if let Some(left) = self.prev[start] {
                        let from_key = (self.sequence[left], self.sequence[start]);
                        let to_key = (self.sequence[left], merged);
                        self.rekey_position(from_key, to_key, left, left, train);
                    }

                    // Unlink end from the chain start -> end -> right
                    let right = self.next[end];
                    self.next[start] = right;
                    self.dead[end] = true;
                    if let Some(right) = right {
                        self.prev[right] = Some(start);

                        let from_key = (self.sequence[end], self.sequence[right]);
                        let to_key = (merged, self.sequence[right]);
                        self.rekey_position(from_key, to_key, end, start, train);
                    }

                    self.sequence[start] = merged;
                }
            }
        }
    }

    fn get_vocab(&mut self) -> Vec<Token> {
        let mut vocab = HashSet::new();

        while let Some(rule) = &self.heap.pop() {
            let freq = rule.0;
            let pair = rule.1;

            if let Some(positions) = self.pairs.get(&pair) {
                let current_freq = positions.len();
                if freq != current_freq {
                    self.heap.push((current_freq, rule.1));
                    continue;
                }

                self.merge_pair(pair, true);
            }

            vocab.insert(pair.0);
            vocab.insert(pair.1);
            vocab.insert(pair.0 + pair.1);
            if vocab.len() == N_TOKENS {
                break;
            }
        }

        vocab.into_iter().collect()
    }

    fn reset_scratch(&mut self) {
        self.sequence.clear();
        self.prev.clear();
        self.next.clear();
        self.pairs.clear();
        self.heap.clear();
    }

    pub fn encode(&mut self, input: &str, out: &mut Vec<usize>) {
        self.reset_scratch();
        self.add_input(input);
        self.dead.fill(false);

        for key in self.pairs.keys() {
            if let Some(&rank) = self.rules.get(key) {
                self.heap.push((self.rules.len() - rank, *key));
            }
        }

        while let Some(rule) = self.heap.pop() {
            let token = rule.1;
            self.merge_pair(token, false);
        }

        out.clear();
        out.push(self.bos);

        for (i, &t) in self.sequence.iter().enumerate() {
            if self.dead[i] {
                continue;
            }

            if let Some(&j) = self.encoder.get(&t) {
                out.push(j)
            };
        }

        out.push(self.bos)
    }

    fn get_encoder(&self) -> HashMap<Token, usize> {
        self.vocab
            .iter()
            .copied()
            .enumerate()
            .map(|(i, t)| (t, i))
            .collect()
    }

    pub fn train(&mut self, path: &str) {
        let file = fs::File::open(path).expect("Input file not found");
        let reader = std::io::BufReader::new(file);

        for line in reader.lines() {
            let input = line.expect("Failed to read line");
            self.add_input(&input);
        }

        self.dead.resize(self.sequence.len(), false);
        self.dead.fill(false);

        for p in &self.pairs {
            let freq = p.1.len();
            self.heap.push((freq, *p.0));
        }

        self.vocab = self.get_vocab();
        self.encoder = self.get_encoder();
        self.bos = self.vocab.len();
    }

    pub fn new() -> Self {
        let sequence = vec![];
        let prev = vec![];
        let next = vec![];
        let dead = vec![];
        let vocab = vec![];
        let encoder = HashMap::new();
        let pairs = HashMap::new();
        let rules = HashMap::new();
        let heap = BinaryHeap::new();

        Self {
            sequence,
            prev,
            next,
            dead,
            vocab,
            encoder,
            pairs,
            rules,
            heap,
            bos: 0,
        }
    }
}
