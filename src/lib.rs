pub mod datasets;
pub mod structured;
pub mod neuron;
pub mod layer;
pub mod model;

// # Create level documentation

pub trait FunctionInterface{
    fn new() -> Self;
    fn forward(&self, x: &f64) -> f64;
    fn backward(&self, x: &f64) -> f64;
}