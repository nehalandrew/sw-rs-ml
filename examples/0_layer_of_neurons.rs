use rsml::datasets::iris::{get_data, get_results};
use rsml::structured::neuron::sigmoid;
use rsml::structured::layer::Layer;

fn main() {
    let data: Vec<Vec<f64>> = get_data();
    let result: Vec<Vec<f64>> = get_results();

    println!("datasets {} result {}", data[0].len(), result[0].len());
    println!("datasets {} result {}", data.len(), result.len());

    // layer vector with 3 neurons
    let mut layer = Layer::generate(data[0].len(), result[0].len(), sigmoid);

    // train the layer
    layer.train(&data, &result, 1000, 0.1).unwrap();

    let output = layer.feed_forward(&vec![5.4, 3.7, 1.5, 0.2]);
    // layer.feed_forward(&vec![-1.0, 1.0], &output).unwrap();

    println!("{:?}", output);

    for neuron in layer.neurons.iter() {
        println!("weights {:?}", neuron.weights);
    }
}
