use std::io::{Read, Write};
use super::{Error, Result};
use matrix::{Matrix, Operation};
use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};
use crate::activations::Activation;

pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation<'a>,
    learning_rate: f64,
}

impl Network<'_> {
    pub fn new(layers: Vec<usize>, activation: Activation, learning_rate: f64) -> Network {
        let (mut weights, mut biases, data) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1))
        }
        Network {
            layers,
            weights,
            biases,
            data,
            activation,
            learning_rate
        }
    }
    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Result<Vec<f64>> {
        if inputs.len() != self.layers[0] {
            return Err(Error::InvalidNumberOfInputs);
        }
        let mut current = Matrix::from(vec![inputs]).transpose();
        self.data = vec![current.clone()];
        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .multiply(&current)?
                .dot_operation(&self.biases[i], Operation::Add)?.map(self.activation.function);
            self.data.push(current.clone())
        }
        Ok(current.transpose().data[0].to_owned())
    }
    pub fn back_propagate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) -> Result<()> {
        if targets.len() != self.layers[self.layers.len()-1] {
           return Err(Error::InvalidTargetLength);
        }
        let parsed = Matrix::from(vec![outputs]).transpose();
        let mut errors = Matrix::from(vec![targets]).transpose().dot_operation(&parsed, Operation::Subtract)?;
        let mut gradients = parsed.map(self.activation.derivative);
        for i in (0..self.layers.len()-1).rev() {
            gradients = gradients.dot_operation(&errors, Operation::Multiply)?.map(&|x| x* self.learning_rate);
            self.weights[i] = self.weights[i].dot_operation(&gradients.multiply(&self.data[i].transpose())?, Operation::Add)?;
            self.biases[i] = self.biases[i].dot_operation(&gradients, Operation::Add)?;
            errors = self.weights[i].transpose().multiply(&errors)?;
            gradients = self.data[i].map(self.activation.derivative);
        }
        Ok(())
    }
    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) -> Result<()> {
        for i in 1..=epochs {
            if epochs < 100 || i % (epochs/100) == 0 {
                println!("Epoch {i} of {epochs}")
            }
            for j in 0 ..inputs.len() {
                let outputs = self.feed_forward(inputs[j].clone())?;
                self.back_propagate(outputs, targets[j].clone())?
            }
        }
        Ok(())
    }
    pub fn save(&self, file: &str) -> Result<()> {
        let mut file = std::fs::File::create(file)?;
        file.write_all(
            json!({
                "weights": self.weights.clone().into_iter().map(|m|m.data).collect::<Vec<Vec<Vec<f64>>>>(),
                "biases": self.biases.clone().into_iter().map(|m|m.data).collect::<Vec<Vec<Vec<f64>>>>(),
            }).to_string().as_bytes(),
        )?;
        Ok(())
    }
    pub fn load(&mut self, file: &str) -> Result<()> {
        let mut file = std::fs::File::open(file)?;
        let mut buffer = String::new();
        file.read_to_string(&mut buffer)?;
        let save_data: SaveData = from_str(&buffer)?;
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for i in 0..self.layers.len()-1 {
            weights.push(Matrix::from(save_data.weights[i].clone()));
            biases.push(Matrix::from(save_data.biases[i].clone()))
        }
        self.weights = weights;
        self.biases = biases;
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct SaveData {
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<Vec<f64>>>,
}