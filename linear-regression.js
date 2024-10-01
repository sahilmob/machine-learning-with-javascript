const tf = require("@tensorflow/tfjs");

class LinearRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    const { mean, variance } = tf.moments(features, 0);

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
      },
      options
    );

    this.weights = tf.zeros([2, 1]);
  }

  gradientDescent() {
    const currentGuesses = this.features.matMul(this.weights);

    const differences = currentGuesses.sub(this.labels);

    const slopes = this.features
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);
    const ssr = testLabels.sub(predictions).pow(2).sum().arraySync();
    const sst = testLabels.sub(testLabels.mean()).pow(2).sum().arraySync();

    const r2 = 1 - ssr / sst;

    console.log(r2);
  }

  processFeatures(features) {
    features = tf.tensor(features);
    features = tf.ones([features.shape[0], 1]).concat(features, 1);
    return features;
  }
}

module.exports = LinearRegression;
