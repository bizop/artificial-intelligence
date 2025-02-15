import * as math from 'mathjs';
import fs from 'fs';

class ActivationFunctions {
  static sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  static sigmoidDerivative(x) {
    return x * (1 - x);
  }

  static relu(x) {
    return Math.max(0, x);
  }

  static reluDerivative(x) {
    return x > 0 ? 1 : 0;
  }

  static tanh(x) {
    return Math.tanh(x);
  }

  static tanhDerivative(x) {
    return 1 - Math.tanh(x) ** 2;
  }
}

class NeuralNetwork {
  constructor(architecture) {
    this.architecture = architecture;
    this.weights = [];
    this.biases = [];
    this.initializeWeightsAndBiases();
  }

  initializeWeightsAndBiases() {
    const { hiddenLayers } = this.architecture;
    let inputSize = 2; // Input layer size for XOR problem

    for (let i = 0; i < hiddenLayers.length; i++) {
      this.weights.push(math.random([inputSize, hiddenLayers[i]], -1, 1));
      this.biases.push(math.zeros([1, hiddenLayers[i]]));
      inputSize = hiddenLayers[i];
    }

    // Output layer
    this.weights.push(math.random([inputSize, 1], -1, 1));
    this.biases.push(math.zeros([1, 1]));
  }

  forward(X) {
    const { hiddenActivations } = this.architecture;
    let activation = X;
    this.layerOutputs = [activation];

    for (let i = 0; i < this.weights.length - 1; i++) {
      const z = math.add(math.multiply(activation, this.weights[i]), this.biases[i]);
      activation = math.map(z, ActivationFunctions[hiddenActivations[i]]);
      this.layerOutputs.push(activation);
    }

    const outputZ = math.add(math.multiply(activation, this.weights[this.weights.length - 1]), this.biases[this.biases.length - 1]);
    const output = math.map(outputZ, ActivationFunctions.sigmoid);
    this.layerOutputs.push(output);

    return output;
  }

  backward(X, y, learningRate) {
    const { hiddenActivations } = this.architecture;
    const m = X.length;
    const gradients = { weights: [], biases: [] };

    let dZ = math.subtract(this.layerOutputs[this.layerOutputs.length - 1], y);
    for (let i = this.weights.length - 1; i >= 0; i--) {
      gradients.weights.unshift(math.divide(math.multiply(math.transpose(this.layerOutputs[i]), dZ), m));
      gradients.biases.unshift(math.mean(dZ, 0));

      if (i > 0) {
        const dA = math.multiply(dZ, math.transpose(this.weights[i]));
        const activationDerivative = hiddenActivations[i - 1] + 'Derivative';
        dZ = math.dotMultiply(dA, math.map(this.layerOutputs[i], ActivationFunctions[activationDerivative]));
      }
    }

    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = math.subtract(this.weights[i], math.multiply(gradients.weights[i], learningRate));
      this.biases[i] = math.subtract(this.biases[i], math.multiply(gradients.biases[i], learningRate));
    }
  }

  train(X, y, epochs, learningRate) {
    for (let i = 0; i < epochs; i++) {
      const output = this.forward(X);
      this.backward(X, y, learningRate);

      if (i % 1000 === 0) {
        const loss = math.mean(math.map(math.subtract(y, output), (x) => x * x));
        console.log(`Epoch ${i}, Loss: ${loss}`);
      }
    }

    const finalOutput = this.forward(X);
    return math.mean(math.map(math.subtract(y, finalOutput), (x) => x * x));
  }

  predict(X) {
    const output = this.forward(X);
    return math.map(output, (x) => (x > 0.5 ? 1 : 0));
  }

  serialize() {
    return JSON.stringify({
      architecture: this.architecture,
      weights: this.weights,
      biases: this.biases,
    });
  }

  static deserialize(json) {
    const data = JSON.parse(json);
    const nn = new NeuralNetwork(data.architecture);
    nn.weights = data.weights;
    nn.biases = data.biases;
    return nn;
  }

  saveModel(filename) {
    const serialized = this.serialize();
    fs.writeFileSync(filename, serialized);
    console.log(`Model saved to ${filename}`);
  }

  static loadModel(filename) {
    const json = fs.readFileSync(filename, 'utf8');
    return NeuralNetwork.deserialize(json);
  }
}

class GeneticAlgorithm {
  constructor(populationSize, generations) {
    this.populationSize = populationSize;
    this.generations = generations;
    this.population = this.initializePopulation();
    this.bestArchitecture = null;
    this.worstArchitecture = null;
    this.bestFitness = Infinity;
    this.worstFitness = -Infinity;
  }

  initializePopulation() {
    return Array(this.populationSize).fill().map(this.createRandomArchitecture);
  }

  createRandomArchitecture() {
    // const layerProbabilities = [0.2, 0.3, 0.5]; // 20% chance of 1 layer, 30% chance of 2, 50% chance of 3
    // const numHiddenLayers = layerProbabilities.findIndex((p) => Math.random() < p) + 1;
    const numHiddenLayers = Math.floor(Math.random() * Math.random() * 5) + 1;
    const hiddenLayers = Array(numHiddenLayers)
      .fill()
      .map(() => Math.floor(Math.random() * 16) + 1); // Limit to 1-16 neurons per layer
    const activations = ['sigmoid', 'relu', 'tanh'];
    const hiddenActivations = Array(numHiddenLayers)
      .fill()
      .map(() => activations[Math.floor(Math.random() * activations.length)]);
    return { hiddenLayers, hiddenActivations };
  }

  evolve(X, y) {
    for (let gen = 0; gen < this.generations; gen++) {
      console.log(`Generation ${gen + 1}`);

      const fitness = this.population.map((arch) => {
        try {
          const nn = new NeuralNetwork(arch);
          return nn.train(X, y, 10000, 0.1);
        } catch (error) {
          console.log('Error in architecture:', arch);
          console.error(error);
          return Infinity;
        }
      });

      const sortedPopulation = this.population.map((arch, index) => ({ arch, fitness: fitness[index] })).sort((a, b) => a.fitness - b.fitness);

      // Update best and worst architectures
      if (sortedPopulation[0].fitness < this.bestFitness) {
        this.bestArchitecture = sortedPopulation[0].arch;
        this.bestFitness = sortedPopulation[0].fitness;
      }

      if (sortedPopulation[sortedPopulation.length - 1].fitness > this.worstFitness) {
        this.worstArchitecture = sortedPopulation[sortedPopulation.length - 1].arch;
        this.worstFitness = sortedPopulation[sortedPopulation.length - 1].fitness;
      }

      console.log(`Best fitness: ${this.bestFitness}`);
      console.log(`Worst fitness: ${this.worstFitness}`);

      const parents = sortedPopulation.slice(0, 2).map((item) => item.arch);
      const children = this.crossoverAndMutate(parents);

      this.population = [...parents, ...children];
    }

    return {
      best: { architecture: this.bestArchitecture, fitness: this.bestFitness },
      worst: { architecture: this.worstArchitecture, fitness: this.worstFitness },
    };
  }

  crossoverAndMutate(parents) {
    const children = [];
    for (let i = 0; i < this.populationSize - 2; i++) {
      const parent1 = parents[0];
      const parent2 = parents[1];
      const childLayers = Math.min(parent1.hiddenLayers.length, parent2.hiddenLayers.length);
      const child = {
        hiddenLayers: Array(childLayers)
          .fill()
          .map((_, index) => (Math.random() < 0.5 ? parent1.hiddenLayers[index] : parent2.hiddenLayers[index])),
        hiddenActivations: Array(childLayers)
          .fill()
          .map((_, index) => (Math.random() < 0.5 ? parent1.hiddenActivations[index] : parent2.hiddenActivations[index])),
      };

      if (Math.random() < 0.1) {
        const layerToMutate = Math.floor(Math.random() * child.hiddenLayers.length);
        child.hiddenLayers[layerToMutate] = Math.floor(Math.random() * 8) + 1;
      }
      if (Math.random() < 0.1) {
        const actToMutate = Math.floor(Math.random() * child.hiddenActivations.length);
        child.hiddenActivations[actToMutate] = ['sigmoid', 'relu', 'tanh'][Math.floor(Math.random() * 3)];
      }

      children.push(child);
    }
    return children;
  }
}

// Main execution
const X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const y = [[0], [1], [1], [0]];

const ga = new GeneticAlgorithm(10, 3);
const { best, worst } = ga.evolve(X, y);

console.log('Best architecture:', best.architecture);
console.log('Best fitness:', best.fitness);
console.log('Worst architecture:', worst.architecture);
console.log('Worst fitness:', worst.fitness);

const finalNN = new NeuralNetwork(best.architecture);
const finalLoss = finalNN.train(X, y, 100000, 0.01);
console.log('Final Loss:', finalLoss);

// Save the best model
finalNN.saveModel('best_xor_model.json');

console.log('Testing the trained network:');
X.forEach((input, index) => {
  const output = finalNN.predict([input]);
  console.log(`Input: ${input}, Predicted Output: ${output[0][0]}, Expected Output: ${y[index][0]}`);
});

// Example of loading and using the saved model
const loadedNN = NeuralNetwork.loadModel('best_xor_model.json');
console.log('\nTesting the loaded model:');
X.forEach((input, index) => {
  const output = loadedNN.predict([input]);
  console.log(`Input: ${input}, Predicted Output: ${output[0][0]}, Expected Output: ${y[index][0]}`);
});
