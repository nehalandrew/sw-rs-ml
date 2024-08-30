// use super::layer::Layer;
//
// pub struct Model {
//     pub layers: Vec<Layer>,
// }
// impl Model {
//     pub fn new(layers: Vec<Layer>) -> Model {
//         Model {
//             layers,
//         }
//     }
//     pub fn feed_forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
//         let mut outputs = inputs.clone();
//         for layer in &self.layers {
//             outputs = layer.feed_forward(&outputs);
//         }
//         outputs
//     }
//     pub fn back_propagate(&mut self, inputs: &Vec<f64>, targets: &Vec<f64>, learning_rate: f64) {
//         // Feed forward
//         let mut outputs = inputs.clone();
//         let mut outputs_stack = Vec::new();
//         let mut inputs_stack = Vec::new();
//         for layer in &self.layers {
//             outputs_stack.push(outputs.clone());
//             inputs_stack.push(inputs.clone());
//             outputs = layer.feed_forward(&outputs);
//         }
//         // Back propagate
//         let mut next_layer_deltas = Vec::new();
//         for i in 0..outputs.len() {
//             next_layer_deltas.push(outputs[i] - targets[i]);
//         }
//         for i in (0..self.layers.len()).rev() {
//             let mut deltas = Vec::new();
//             for j in 0..self.layers[i].neurons.len() {
//                 let mut delta = 0.0;
//                 for k in 0..next_layer_deltas.len() {
//                     delta += next_layer_deltas[k] * self.layers[i].neurons[j].weights[k];
//                 }
//                 delta *= (self.layers[i].neurons[j].derivative)(outputs_stack[i][j]);
//                 deltas.push(delta);
//                 for k in 0..self.layers[i].neurons[j].weights.len() {
//                     self.layers[i].neurons[j].weights[k] -= learning_rate * next_layer_deltas[k] * outputs_stack[i][j];
//                 }
//                 self.layers[i].neurons[j].bias -= learning_rate * next_layer_deltas[j];
//             }
//             next_layer_deltas = deltas;
//         }
//     }
// }


