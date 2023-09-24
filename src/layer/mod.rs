use rand::seq::index::sample_weighted;

pub struct Layer {
    pub lweights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: fn(&f64) -> f64,
}
impl Layer {
    pub fn new(number_inputs: u64, number_neurons: u64, activation: fn(&f64) -> f64) -> Layer {
        let mut lweights = Vec::new();
        let mut biases = Vec::new();
        for _ in 0..number_neurons {
            let mut neuron_weights = Vec::new();
            for _ in 0..number_inputs {
                neuron_weights.push(rand::random::<f64>());
            }
            lweights.push(neuron_weights);
            biases.push(rand::random::<f64>());
        }
        return Layer {
            lweights,
            biases,
            activation,
        };
    }

    pub fn load(
        lweights: Vec<Vec<f64>>,
        biases: Vec<f64>,
        activation: fn(&f64) -> f64,
    ) -> Result<Layer, String> {
        if lweights.len() != biases.len() {
            return Err("number of weights is not same as number of biases".to_string());
        }
        return Ok(Layer {
            lweights,
            biases,
            activation,
        });
    }

    pub fn predict(&self, inputs: &Vec<f64>) -> Result<Vec<f64>, String> {
        let mut outputs = Vec::new();
        for (nweights, bias) in self.lweights.iter().zip(self.biases.iter()) {
            if inputs.len() != nweights.len() {
                return Err("Inputs number is not same as nweights!".to_string());
            }
            match super::neuron::forward(inputs, nweights, bias, self.activation) {
                Ok(result) => outputs.push(result),
                Err(e) => println!("{}", e),
            }
        }
        return Ok(outputs);
    }

    pub fn train(
        &mut self,
        inputs: &Vec<f64>,
        targets: &Vec<f64>,
        learning_rate: f64,
    ) -> Result<(), String> {
        // if inputs.len() != targets.len() {
        //     return Err("Input and target datasets have different lengths".to_string());
        // }
            for input in inputs{
                let output = self.predict(inputs).unwrap();
                for i in 0..self.lweights.len() {
                    super::neuron::backward(
                        &output[i],
                        &targets[i],
                        &inputs,
                        &mut self.lweights[i], // Pass weights as a mutable reference
                        &mut self.biases[i],
                        self.activation,
                        learning_rate,
                    );
                }


            // let errors: Vec<f64> = prediction
            //     .iter()
            //     .zip(target.iter())
            //     .map(|(p, t)| p - t)
            //     .collect();
            // println!("errors {:?}", errors);
            // for (neuron, error) in self.neurons.iter_mut().zip(errors.iter()) {
            //     let delta = error * neuron.derivative(*neuron.output.borrow());
            //     neuron.update_weights(learning_rate, delta, input);
            //     neuron.update_bias(learning_rate, delta);
            // }
        }

        Ok(())
    }
}

#[cfg(test)]
mod layer_tests {
    use super::*;

    #[test]
    fn new_test() {
        let layer = Layer::new(2, 3, |x| 1.0 / (1.0 + (-x).exp()) );
        assert_eq!(layer.lweights.len(), 3);
    }
}
