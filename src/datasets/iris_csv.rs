extern crate serde;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;

#[derive(Debug, Deserialize)]
struct IrisRecord {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    target: String,
}

pub fn load(path: &str) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), Box<dyn Error>> {
    // Open the CSV file
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    // Define vectors to hold the input and target values
    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<Vec<f64>> = Vec::new();

    // Iterate over the CSV records
    for result in rdr.deserialize() {
        let record: IrisRecord = result?;

        // Convert the target column to a one-hot encoded vector
        let target_vec: Vec<f64> = match record.target.as_str() {
            "setosa" => vec![1.0, 0.0, 0.0],
            "versicolor" => vec![0.0, 1.0, 0.0],
            "virginica" => vec![0.0, 0.0, 1.0],
            _ => vec![0.0, 0.0, 0.0], // Handle unknown values as needed
        };

        // Push the input and target values into their respective vectors
        inputs.push(vec![
            record.sepal_length,
            record.sepal_width,
            record.petal_length,
            record.petal_width,
        ]);
        targets.push(target_vec);
    }

    Ok((inputs, targets))
}
pub fn get_name(target: &Vec<f64>) -> String {
    // Define the variant names as an array
    let variant_names = ["setosa", "versicolor", "virginica", "unknown"];

    // Check if the target vector is empty
    if target.is_empty() {
        return "No data provided".to_string();
    }

    // Find the index of the maximum value in the target vector
    if let Some(max_index) = target.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) {
        let max_variant_index = max_index.0;

        // Return the name of the biggest variant
        if max_variant_index < variant_names.len() {
            return variant_names[max_variant_index].to_string();
        }
    }

    // If no maximum value is found or the index is out of bounds, return "unknown"
    "unknown".to_string()
}
#[cfg(test)]
mod iris_csv_tests {
    use super::*;

    #[test]
    fn load_test() {
        match load("data/iris.csv") {
            Ok((inputs, targets)) => {
                assert_eq!(inputs.len(), targets.len());
                // for (input, target) in inputs.iter().zip(targets.iter()) {
                //     println!("Input: {:?}, Target: {:?}", input, target);
                // }
            }
            Err(err) => eprintln!("Error: {}", err),
        }

    }

    #[test]
    fn data_test() {
        match load("data/iris.csv") {
            Ok((inputs, targets)) => {
                assert_eq!(inputs[0].len(), 4);
                // for (input, target) in inputs.iter().zip(targets.iter()) {
                //     println!("Input: {:?}, Target: {:?}", input, target);
                // }
            }
            Err(err) => eprintln!("Error: {}", err),
        }

    }
}

