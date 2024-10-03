require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./logistic-regression");
const mnist = require("mnist-data");
const _ = require("lodash");

const mnistData = mnist.training(0, 10);

const features = mnistData.images.values.map((image) => _.flatMap(image));

const encodedLabels = mnistData.labels.values.map((v) => {
  const arr = Array(10).fill(0);
  arr[v] = 1;
  return arr;
});

console.log(encodedLabels);
