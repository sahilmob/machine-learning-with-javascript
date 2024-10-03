require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./logistic-regression");
const mnist = require("mnist-data");
const _ = require("lodash");

const mnistData = mnist.training(0, 60000);

const features = mnistData.images.values.map((image) => _.flatMap(image));

const encodedLabels = mnistData.labels.values.map((v) => {
  const arr = Array(10).fill(0);
  arr[v] = 1;
  return arr;
});

const regression = new LogisticRegression(features, encodedLabels, {
  learningRate: 1,
  iterations: 20,
  batchSize: 100,
});

regression.train();

const testMnistData = mnist.testing(0, 1000);
const testFeatures = testMnistData.images.values.map((v) => _.flatMap(v));
const testLabels = testMnistData.labels.values.map((v) => {
  const arr = Array(10).fill(0);
  arr[v] = 1;
  return arr;
});

console.log(regression.test(testFeatures, testLabels));
