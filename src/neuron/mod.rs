use crate::FunctionInterface;

pub mod activation;

struct Neuron {}
impl FunctionInterface for Neuron{
    pub fn forward(inputs: &Vec<f64>, weights: &Vec<f64>, bias: &f64, activation:fn(&f64) ->f64) -> Result<f64, String> {
        if weights.len() != inputs.len() {
            return Err(format!(
                "The number of inputs {} must be equal to the number of weights {}.",
                inputs.len(),
                weights.len()
            )
                .to_string());
        }
        let mut total = 0.0;
        for (weight, &input) in weights.iter().zip(inputs.iter()) {
            total += input * weight;
        }
        return Ok((activation)(&(total + bias)));
    }

    pub fn backward(output: &f64, target: &f64, inputs: &Vec<f64>, weights: &mut Vec<f64>, bias: &mut f64, activation: fn(&f64) -> f64, learning_rate: f64) {
        let error = output - target;
        let delta = error * derivation(output, activation);
        for (weight, &input) in weights.iter_mut().zip(inputs.iter()) {
            *weight -= delta * input * learning_rate;
        }
        *bias -= delta * learning_rate;
    }

    fn derivation(x: &f64, activation:fn(&f64) ->f64) -> f64 {
        let sigmoid_x = (activation)(x);
        return sigmoid_x * (1.0 - sigmoid_x);
    }
}

#[cfg(test)]
mod neuron_tests {
    use super::*;

    #[test]
    fn new_test() {
        // let layer = Layer::new(2, 3, |x| );
        // assert_eq!(sigmoid(0.0), 0.5);
    }
}
