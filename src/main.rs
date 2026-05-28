use rand::seq::SliceRandom;
use rand::thread_rng;
use std::rc::Rc;
use std::{
    fs,
    io::{self, BufRead},
    vec,
};

/*
let file = fs::File::open("input.txt")?;
    let reader = io::BufReader::new(file);

    let mut docs = vec![];
    for doc in reader.lines() {
        docs.push(doc.unwrap());
    }
    let mut rng = thread_rng();
    docs.shuffle(&mut rng);

    // Each unique character becomes a token assigned a unique integer
    let mut uchars: Vec<char> = docs.join("").chars().collect();
    uchars.sort();
    uchars.dedup();

    // ID for the special beginning of sequence character
    let BOS = uchars.len();
    let vocab_size = BOS + 1;

*/

fn main() -> std::io::Result<()> {
    

    let mut tape = Tape::new();
    let a = tape.leaf(2.0);
    let b = tape.leaf(3.0);
    let c = tape.mul(a, b);
    let _ = tape.add(c, a);

    tape.backward();
    println!("{}", tape.values[a].grad);
    println!("{}", tape.values[b].grad);

    Ok(())
}

#[derive(Clone)]
struct Value {
    data: f64,
    grad: f64,
    children: Vec<usize>,
    local_grads: Vec<f64>,
}


impl Value {
    fn new(data: f64, children: Vec<usize>, local_grads: Vec<f64>) -> Self {
        Self {
            data,
            grad: 0.0,
            children,
            local_grads,
        }
    }
}

struct Tape {
    values: Vec<Value>,
}

impl Tape {
    fn new() -> Self {
        Self {values: vec![]}
    }

    fn backward(&mut self) {
        let n = self.values.len();
        assert!(n > 0);
        
        self.values[n - 1].grad = 1.0;

        for i in (0..n).rev() {
            let v = self.values[i].clone();
            for j in 0..v.children.len() {
                let child_i = v.children[j];
                let local_grad = v.local_grads[j];

                self.values[child_i].grad += local_grad * v.grad;
            }
        }
    }

    fn push(&mut self, v: Value) -> usize {
        self.values.push(v);
        self.values.len() - 1
    }

    fn leaf(&mut self, data: f64) -> usize {
        self.push(Value::new(data, vec![], vec![]))
    }

    fn add(&mut self, a: usize, b: usize) -> usize {
        let data = self.values[a].data + self.values[b].data;
        self.push(Value::new(data, vec![a, b], vec![1.0, 1.0]))
    }

    fn mul(&mut self, a: usize, b: usize) -> usize {
        let local_grads = vec![self.values[b].data, self.values[a].data];
        let data = self.values[a].data * self.values[b].data;
        self.push(Value::new(data, vec![a,b], local_grads))
    }

    fn neg(&mut self, i: usize) -> usize {
        self.push(Value::new(-1.0, vec![], vec![]));
        self.mul(i, self.values.len() - 1)
    }

    fn pow(&mut self, i: usize, n: f64) -> usize {
        let local_grads = vec![n * self.values[i].data.powf(n - 1.0)] ;
        self.push(Value::new(self.values[i].data.powf(n), vec![i], local_grads))
    }

    fn log(&mut self, i: usize) -> usize {
        let local_grads = vec![1.0 / self.values[i].data] ;
        self.push(Value::new(self.values[i].data.ln(), vec![i], local_grads))
    }

    fn exp(&mut self, i: usize) -> usize {
        let data = self.values[i].data.exp();
        self.push(Value::new(data, vec![i], vec![data]))
    }

    fn relu(&mut self, i: usize) -> usize {
        let local_grads = vec![if self.values[i].data > 0.0 { 1.0 } else { 0.0 }];
        self.push(Value::new(self.values[i].data.max(0.0), vec![i], local_grads))
    }
}
