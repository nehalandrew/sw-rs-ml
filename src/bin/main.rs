use rand::random;
use rsml::datasets::iris::{get_data, get_results};
use rsml::datasets::iris_csv;
use rsml::neuron::activation::sigmoid;
use rand::seq::SliceRandom; // Import the SliceRandom trait

use rsml::layer::Layer;

fn main() {
    let (dataset,result) = iris_csv::load("data/iris.csv").unwrap();

    //    let (dataset,result) = iris::load_data();
    println!("datasets {} result {}", dataset.len(), result.len());
    println!("datasets {} result {}", dataset.len(), result.len());

    // layer vector with 3 neurons
    let mut layer = Layer::new(dataset[0].len() as u64, result[0].len() as u64, sigmoid);

    let mut rng = rand::thread_rng(); // Initialize the random number generator

    // Collect the zipped iterator into a Vec
    let dataset = dataset.iter().zip(result.iter()).collect::<Vec<_>>();

    // train the layer
    for i in 0..150 {
        let random_pair = dataset.choose(&mut rng);

        // Check if there's a pair (i.e., the vector is not empty)
        if let Some((data, result)) = random_pair {
            // Call your `layer.train` function with the selected data and result
            layer.train(data, result, 0.1).unwrap();
        }
    }


    let output = layer.predict(&vec![4.3,3.0,1.1,0.1]).unwrap();
    // layer.feed_forward(&vec![-1.0, 1.0], &output).unwrap();


    println!("{:?}", iris_csv::get_name(&output));
    println!("{:?}", output);

    for neuron in layer.lweights.iter() {
        println!("weights {:?}", neuron.iter());
    }
}
