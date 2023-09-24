use std::cell::RefCell;

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub activation: fn(f64) -> f64,
    pub output: RefCell<f64>,
}

impl Neuron {
    pub fn new(weights: Vec<f64>, bias: f64, activation: fn(f64) -> f64) -> Result<Neuron, String> {
        if weights.len() == 0 {
            return Err("The number of inputs must be greater than zero.".to_string());
        }
        Ok(Neuron {
            weights,
            bias,
            activation,
            output: RefCell::new(0.0),
        })
    }

    pub fn feed_forward(&self, inputs: &Vec<f64>) -> Result<f64, String> {
        if self.weights.len() != inputs.len() {
            return Err(format!(
                "The number of inputs {} must be equal to the number of weights {}.",
                inputs.len(),
                self.weights.len()
            )
            .to_string());
        }
        let mut total = 0.0;
        for (weight, &input) in self.weights.iter().zip(inputs.iter()) {
            total += input * weight;
        }
        *self.output.borrow_mut() = (self.activation)(total + self.bias);
        return Ok(*self.output.borrow());
    }

    // Calculate the derivative of the activation function at a given value 'x'
    pub fn derivative(&self, x: f64) -> f64 {
        // Implement the derivative of the sigmoid activation function
        // You can adjust this to handle different activation functions if needed
        let sigmoid_x = (self.activation)(x);
        sigmoid_x * (1.0 - sigmoid_x)
    }

    // Update weights based on error, learning rate, and input
    pub fn update_weights(&mut self, learning_rate: f64, error: f64, input: &Vec<f64>) {
        for (weight, &input_value) in self.weights.iter_mut().zip(input.iter()) {
            *weight -= learning_rate * error * input_value;
        }
    }

    // Update bias based on error and learning rate
    pub fn update_bias(&mut self, learning_rate: f64, error: f64) {
        self.bias -= learning_rate * error;
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert_eq!(sigmoid(0.0), 0.5);
    }
}
