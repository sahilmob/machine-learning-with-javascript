require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./logistic-regression");
const mnist = require("mnist-data");
const _ = require("lodash");

const mnistData = mnist.training(0, 1);

const features = mnistData.images.values.map((image) => _.flatMap(image));

console.log(features);
