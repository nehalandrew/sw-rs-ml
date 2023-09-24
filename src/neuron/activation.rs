pub fn sigmoid(x: &f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod activation_tests {
    use super::*;

    #[test]
    fn new_test() {
        // assert_eq!();
    }
}
