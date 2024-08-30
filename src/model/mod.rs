#[cfg(test)]
mod layer_tests {
    use super::*;

    #[test]
    fn new_test() {

        // assert_eq!(sigmoid(0.0), 0.5);
    }
}


struct Model {
    layers: Vec<Layer>,
}
impl Model {
    def forward(){

    }
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

}