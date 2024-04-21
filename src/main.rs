mod activations;
pub use activations::{SIGMOID, IDENTITY, TANH, RELU};
mod error;
mod network;
pub use error::{Error, Result};
pub use network::Network;
// 0, 0 -> 0
// 0, 1 -> 1
// 1, 0 -> 1
// 1, 1 -> 0
fn main() -> Result<()>{
    let inputs = vec![vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0]];
    let targets = vec![vec![0.0],
       vec![1.0],
       vec![1.0],
       vec![0.0]];
    let mut network = Network::new(vec![2, 3, 1], SIGMOID, 0.8);
    println!("BEFORE:");
    println!("0 and 0: {:?}", network.feed_forward(vec![0.0, 0.0])?);
    println!("0 and 1: {:?}", network.feed_forward(vec![0.0, 1.0])?);
    println!("1 and 0: {:?}", network.feed_forward(vec![1.0, 0.0])?);
    println!("1 and 1: {:?}", network.feed_forward(vec![1.0, 1.0])?);
    network.train(inputs, targets, 10000)?;
    println!("AFTER:");
    println!("0 and 0: {:?}", network.feed_forward(vec![0.0, 0.0])?);
    println!("0 and 1: {:?}", network.feed_forward(vec![0.0, 1.0])?);
    println!("1 and 0: {:?}", network.feed_forward(vec![1.0, 0.0])?);
    println!("1 and 1: {:?}", network.feed_forward(vec![1.0, 1.0])?);
    Ok(())
}
