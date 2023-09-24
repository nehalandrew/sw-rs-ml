use crate::structured::neuron::{sigmoid, Neuron};
use rand::{random, seq::index};

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn generate(
        number_inputs: usize,
        number_neurons: usize,
        activation: fn(f64) -> f64,
    ) -> Layer {
        println!("inputs number {number_inputs} number neurons {number_neurons}");
        let neurons: Vec<Neuron> = (0..number_neurons)
            .map(|_| {
                let weights: Vec<f64> = (0..number_inputs).map(|_| random::<f64>()).collect();
                Neuron::new(weights, 1.0, activation).unwrap()
            })
            .collect();

        Layer { neurons }
    }

    pub fn restore(weights: Vec<Vec<f64>>, biases: Vec<f64>, activation: fn(f64) -> f64) -> Layer {
        let neurons: Vec<Neuron> = weights
            .iter()
            .zip(&biases)
            .map(|(w, b)| Neuron::new(w.clone(), *b, activation).unwrap())
            .collect();

        Layer { neurons }
    }

    pub fn feed_forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = Vec::new();
        for neuron in &self.neurons {
            match neuron.feed_forward(inputs) {
                Ok(result) => outputs.push(result),
                Err(e) => println!("{}", e),
            }
        }
        outputs
    }

    pub fn train(
        &mut self,
        inputs: &Vec<Vec<f64>>,
        targets: &Vec<Vec<f64>>,
        iterations: usize,
        learning_rate: f64,
    ) -> Result<(), String> {
        if inputs.len() != targets.len() {
            return Err("Input and target datasets have different lengths".to_string());
        }

        for _ in 0..iterations {
            let mut index = 0;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                println!("in layer {} {} {}", index, input.len(), target.len());
                index += 1;
                let prediction = self.feed_forward(input);
                let errors: Vec<f64> = prediction
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| p - t)
                    .collect();
                println!("errors {:?}", errors);
                for (neuron, error) in self.neurons.iter_mut().zip(errors.iter()) {
                    let delta = error * neuron.derivative(*neuron.output.borrow());
                    neuron.update_weights(learning_rate, delta, input);
                    neuron.update_bias(learning_rate, delta);
                }
            }
        }

        Ok(())
    }
}
