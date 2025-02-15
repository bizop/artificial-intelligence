# Evolving Neural Networks with Genetic Algorithms for the XOR Problem

[@NathanWilbanks_](https://x.com/NathanWilbanks_)

This paper presents a comprehensive study of a research project aimed at training a neural network to solve the XOR problem using conventional gradient descent and further evolving its architecture with a genetic algorithm. The project is implemented in JavaScript using the mathjs library and Node.js file system module to handle model persistence.

In this paper, we detail the code structure, mathematics, and the methodology used. We include step‐by‐step explanations of each code section, from activation functions to evolutionary architecture search.

---

## 1. Introduction

The XOR problem is a classic benchmark for neural networks because it represents a non-linearly separable dataset. To solve this, we use a neural network with one or more hidden layers and apply a genetic algorithm (GA) to optimize its architecture. The GA iteratively refines potential architectures based on their fitness, evaluated via the training loss.

---

## 2. Methodology

### 2.1 Neural Network Training

The network is trained using a backpropagation algorithm with mean squared error (MSE) as the loss function. For a dataset with inputsXand outputsy, the loss is computed as:

L=1m∑i=1m(yi−yi^)2

where:

- mis the number of training samples,
- yi^is the predicted output.

The network uses several common activation functions such as sigmoid, ReLU, and tanh. Each activation function includes both the function and its derivative, critical for the backpropagation step.

### 2.2 Genetic Algorithm for Architecture Evolution

To discover an optimal network configuration, we use a genetic algorithm that:

- **Initializes** a population of random network architectures.
- **Evaluates** each architecture’s fitness (training loss).
- **Selects** the best-performing candidates.
- **Generates** new architectures through crossover (merging two parent architectures) and mutation (random perturbations of network parameters).

Fitness is evaluated by training the network on the XOR dataset for a fixed number of epochs, and the architecture with lower MSE is considered more "fit."

---

## 3. Detailed Code Analysis

Below, we provide and explain the code sections in detail.

### 3.1 Activation Functions

The `ActivationFunctions` class implements the following functions:

- **Sigmoid Function and Derivative:**
    
    sigmoid(x)=11+e−x,sigmoidDerivative(x)=x(1−x)
    
- **ReLU and Its Derivative:**
    
    relu(x)=max(0,x),reluDerivative(x)={1if x>0 0otherwise
    
- **Tanh and Its Derivative:**
    
    tanh(x)=tanh⁡(x),tanhDerivative(x)=1−tanh⁡(x)2
    

### 3.2 Neural Network Implementation

The `NeuralNetwork` class encapsulates the architecture, weight and bias initialization, feedforward, and backpropagation processes.

- **Weight and Bias Initialization:**  
    We initialize weights randomly within a range and set biases to zero. For a given network architecture, the first hidden layer takes a fixed input size (for XOR, it is 2).
    
- **Forward Propagation:**  
    The `forward()` method propagates inputs through the network. For each hidden layer, the network computes
    
    z=W⋅a+b,
    
    where:
    
- Wis the weight matrix,
    
- ais the input activation,
    
- bis the bias.
    
    The activation is then computed by applying the selected function element-wise.
    
- **Backpropagation:**  
    The `backward()` method computes gradients of the loss with respect to weights and biases using the chain rule. Gradients are calculated layer by layer in reverse order (starting from the output).
    
- **Training:**  
    Training consists of multiple epochs. Loss is periodically logged, and the final mean squared error is returned.
    
- **Prediction & Serialization:**  
    The network can predict outputs for new data and be saved/loaded via JSON serialization.
    

Below is the complete implementation of the neural network and supporting methods:

```kotlin
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
```

### 3.3 Genetic Algorithm Implementation

The `GeneticAlgorithm` class generates a population of network architectures and iteratively improves them:

- **Initialization:**  
    Random architectures are generated with a varying number of hidden layers and neurons per layer.
    
- **Fitness Evaluation:**  
    Each architecture is trained on the XOR problem, and its fitness is defined by the training loss.
    
- **Selection, Crossover, and Mutation:**  
    The top two architectures serve as parent candidates. They crossover to produce children networks. Occasional mutations alter the network parameters, ensuring diversity.
    

Below is the genetic algorithm code:

```kotlin
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
```

### 3.4 Main Execution and Experiment

The main execution script does the following:

1. **Defines the XOR dataset:**
    
    X=[00 01 10 11],y=[0 1 1 0]
    
2. **Runs the Genetic Algorithm:**  
    Evolve multiple architectures over 3 generations for a population size of 10. The GA prints the best and worst architectures along with their fitness values.
    
3. **Trains the Final Network:**  
    The best architecture is used to instantiate a new network, which is then trained further (100000 epochs using a smaller learning rate) to minimize the loss.
    
4. **Model Persistence:**  
    The trained neural network model is saved to a file named "best_xor_model.json" and also loaded back, demonstrating model persistence.
    
5. **Testing:**  
    Finally, the network predictions for the XOR inputs are printed for both the newly trained model and the loaded model.
    

Below is the main execution code:

```javascript
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
```

---

## 4. Reproducing the Results

To reproduce the research outcomes:

6. **Prerequisites:**

- Install Node.js and npm.
    
- Install the mathjs package:
    
    ```undefined
    npm install mathjs
    ```
    

7. **Run the Script:**

- Save the provided code into a file, e.g., `xor_ga.js`.
    
- Execute the script with:
    
    ```undefined
    node xor_ga.js
    ```
    

8. **Study the Output:**

- The console will display training loss at intervals.
- Best and worst architectures are printed along with fitness values.
- Predictions on the XOR dataset are logged both for the final trained network and the loaded model.

---

## 5. Conclusions

This work demonstrates the combined use of neural networks and genetic algorithms to solve a non-linear classification task—the XOR problem. The approach illustrates:

- The effectiveness of backpropagation with well-defined activation functions.
- The potential of evolutionary algorithms to optimize neural network architecture parameters.
- Practical implementation details, including model saving and loading.

Future work may extend this methodology to more complex datasets and deeper architectures, possibly integrating additional evolutionary strategies or advanced hyperparameter tuning techniques.

---

This concludes the detailed research paper and step-by-step guide on using the provided code to build, train, and evolve a neural network using both traditional training and genetic algorithms.
