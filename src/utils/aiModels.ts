/**
 * Expands each feature array with engineered features:
 * 1. Interaction term: voltage * current
 * 2. Power density: voltage / pulseOnTime
 * 3. Duty cycle: pulseOnTime / (pulseOnTime + pulseOffTime)
 * Assumes input order: [voltage, current, pulseOnTime, pulseOffTime, wireSpeed, dielectricFlow]
 */
export function engineerFeatures(inputs: number[][]): number[][] {
  return inputs.map(sample => {
    const [voltage, current, pulseOnTime, pulseOffTime] = sample;
    const interaction = voltage * current;
    const powerDensity = pulseOnTime !== 0 ? voltage / pulseOnTime : 0;
    const dutyCycle = (pulseOnTime + pulseOffTime) !== 0 ? pulseOnTime / (pulseOnTime + pulseOffTime) : 0;
    return [...sample, interaction, powerDensity, dutyCycle];
  });
}
// Real AI model implementations for Wire EDM simulation
import * as tf from '@tensorflow/tfjs';
import { loadEDMDataset, EDMTrainingData } from './datasetLoader';

export interface ModelResult {
  rSquared: number;
  trainingTime: number;
  samples: number;
  rmse: number;
  predict: (params: any) => {
    materialRemovalRate: number;
    surfaceRoughness: number;
    dimensionalAccuracy: number;
    processingTime: number;
  };
  weights?: number[];
  modelData?: any;
}

// ...existing code...

// Support Vector Machine implementation
export async function trainSVM(
  data: { trainData: EDMTrainingData[]; testData: EDMTrainingData[] }
): Promise<ModelResult> {
  const startTime = Date.now();
  const train = data.trainData;
  const test = data.testData;
  console.log(`Training SVM with ${train.length} training samples and ${test.length} test samples`);

  // Prepare features and targets for train and test
  const trainFeatures = train.map(d => [
    d.voltage / 300,
    d.current / 50,
    d.pulseOnTime / 100,
    d.pulseOffTime / 200,
    d.wireSpeed / 500,
    d.dielectricFlow / 20
  ]);
  const trainTargets = train.map(d => [
    d.materialRemovalRate,
    d.surfaceRoughness,
    d.dimensionalAccuracy,
    d.processingTime
  ]);
  const testFeatures = test.map(d => [
    d.voltage / 300,
    d.current / 50,
    d.pulseOnTime / 100,
    d.pulseOffTime / 200,
    d.wireSpeed / 500,
    d.dielectricFlow / 20
  ]);
  const testTargets = test.map(d => [
    d.materialRemovalRate,
    d.surfaceRoughness,
    d.dimensionalAccuracy,
    d.processingTime
  ]);

  // Initialize weights using least squares approximation
  const weights: number[][] = [];
  for (let output = 0; output < 4; output++) {
    const y = trainTargets.map(t => t[output]);
    const w = solveLeastSquares(trainFeatures, y);
    weights.push(w);
  }

  // Evaluate on test data
  let totalError = 0;
  let n = 0;
  let flatTargets = testTargets.flat();
  let mean = flatTargets.reduce((sum, v) => sum + v, 0) / flatTargets.length;
  let residualSum = 0;
  let totalSumSquares = 0;
  for (let i = 0; i < testFeatures.length; i++) {
    const predicted = predictSVM(testFeatures[i], weights);
    const actual = testTargets[i];
    for (let j = 0; j < 4; j++) {
      const diff = predicted[j] - actual[j];
      totalError += diff * diff;
      residualSum += diff * diff;
      totalSumSquares += (actual[j] - mean) * (actual[j] - mean);
      n++;
    }
  }
  const rmse = Math.sqrt(totalError / n);
  const rSquared = totalSumSquares === 0 ? 0 : 1 - (residualSum / totalSumSquares);

  const predict = (params: any) => {
    const input = [
      (params.laserPower || 3) / 6,
      (params.speed || 3000) / 6000,
      (params.thickness || 4) / 10,
      (params.linearEnergy || 60) / 250,
      (params.speed || 3000) / 6000,
      (params.surfaceRoughness || 1.3) / 5
    ];
    const result = predictSVM(input, weights);
    return {
      materialRemovalRate: Math.max(0.1, result[0] * 2),
      surfaceRoughness: Math.max(0.1, Math.min(5, result[1])),
      dimensionalAccuracy: Math.max(1, Math.min(100, result[2] * 10)),
      processingTime: Math.max(1, Math.min(300, result[3] * 5))
    };
  };

  return {
    rSquared,
    trainingTime: Date.now() - startTime,
    samples: train.length,
    rmse,
    weights: weights.flat(),
    predict
  };
}

function solveLeastSquares(X: number[][], y: number[]): number[] {
  const n = X.length;
  const m = X[0].length;
  
  // Add bias term
  const XWithBias = X.map(row => [1, ...row]);
  
  // Normal equation: w = (X^T * X)^(-1) * X^T * y
  const XT = transpose(XWithBias);
  const XTX = matrixMultiply(XT, XWithBias);
  const XTy = matrixVectorMultiply(XT, y);
  
  // Solve using Gaussian elimination
  return gaussianElimination(XTX, XTy);
}

function predictSVM(input: number[], weights: number[][]): number[] {
  const inputWithBias = [1, ...input];
  return weights.map(w => {
    let sum = 0;
    for (let i = 0; i < inputWithBias.length && i < w.length; i++) {
      sum += inputWithBias[i] * w[i];
    }
    return sum;
  });
}

// Artificial Neural Network implementation

export interface ANNConfig {
  learningRate: number;
  epochs: number;
  hiddenUnits: number;
}


export async function trainANN(
  data: { trainData: EDMTrainingData[]; testData: EDMTrainingData[] },
  config: ANNConfig = { learningRate: 0.01, epochs: 50, hiddenUnits: 12 },
  useFeatureEngineering: boolean = true
): Promise<ModelResult> {
  const startTime = Date.now();

  const train = data.trainData;
  const test = data.testData;
  console.log(`Training ANN (TF.js) with ${train.length} training samples and ${test.length} test samples`);


  // Prepare training features and targets
  const trainFeatures = train.map(d => [
    d.voltage / 300,
    d.current / 50,
    d.pulseOnTime / 100,
    d.pulseOffTime / 200,
    d.wireSpeed / 500,
    d.dielectricFlow / 20
  ]);
  const testFeatures = test.map(d => [
    d.voltage / 300,
    d.current / 50,
    d.pulseOnTime / 100,
    d.pulseOffTime / 200,
    d.wireSpeed / 500,
    d.dielectricFlow / 20
  ]);
  // Conditionally apply feature engineering
  const engineeredTrainFeatures = useFeatureEngineering ? engineerFeatures(trainFeatures) : trainFeatures;
  const engineeredTestFeatures = useFeatureEngineering ? engineerFeatures(testFeatures) : testFeatures;
  const trainTargets = train.map(d => [
    d.materialRemovalRate / 10,
    d.surfaceRoughness / 5,
    d.dimensionalAccuracy / 100,
    d.processingTime / 100
  ]);
  const testTargets = test.map(d => [
    d.materialRemovalRate / 10,
    d.surfaceRoughness / 5,
    d.dimensionalAccuracy / 100,
    d.processingTime / 100
  ]);

  // Convert all to tensors
  const xs = tf.tensor2d(engineeredTrainFeatures);
  const ys = tf.tensor2d(trainTargets);
  const xsTest = tf.tensor2d(engineeredTestFeatures);
  const ysTest = tf.tensor2d(testTargets);

  // Build model
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [useFeatureEngineering ? 9 : 6], units: config.hiddenUnits, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 4, activation: 'linear' }));

  model.compile({ optimizer: tf.train.adam(config.learningRate), loss: 'meanSquaredError' });

  // Train model
  const history = await model.fit(xs, ys, {
    epochs: config.epochs,
    batchSize: 8,
    verbose: 0,
    validationData: [xsTest, ysTest],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 10 === 0) {
          console.log(`Epoch ${epoch}, Loss: ${logs?.loss}, Val Loss: ${logs?.val_loss}`);
        }
      }
    }
  });

  // Evaluate model: calculate RMSE and R-squared on test data
  const predsTestTensor = model.predict(xsTest) as tf.Tensor;
  const predsTestArr = await predsTestTensor.array() as number[][];
  const testTargetsArr = await ysTest.array() as number[][];

  // Calculate RMSE
  let totalError = 0;
  let mean = 0;
  let n = 0;
  // Flatten testTargetsArr for mean calculation
  const flatTargets = testTargetsArr.flat();
  mean = flatTargets.reduce((sum, v) => sum + v, 0) / flatTargets.length;
  // Calculate residual sum of squares and total sum of squares
  let residualSum = 0;
  let totalSumSquares = 0;
  for (let i = 0; i < predsTestArr.length; i++) {
    for (let j = 0; j < 4; j++) {
      const pred = predsTestArr[i][j];
      const actual = testTargetsArr[i][j];
      const diff = pred - actual;
      totalError += diff * diff;
      residualSum += diff * diff;
      totalSumSquares += (actual - mean) * (actual - mean);
      n++;
    }
  }
  const rmse = Math.sqrt(totalError / n);
  const rSquared = totalSumSquares === 0 ? 0 : 1 - (residualSum / totalSumSquares);

  // Prediction function
  const predict = (params: any) => {
    // Normalize input params to match training normalization
    const input = [
      (params.laserPower || 3) / 6,
      (params.speed || 3000) / 6000,
      (params.thickness || 4) / 10,
      (params.linearEnergy || 60) / 250,
      (params.speed || 3000) / 6000,
      (params.surfaceRoughness || 1.3) / 5
    ];
    // Apply feature engineering to the input
    const engineeredInput = engineerFeatures([input])[0];
    // Create a 2D tensor of shape [1, 9]
    const inputTensor = tf.tensor2d([engineeredInput], [1, 9]);
    // Predict using the trained model
    const outputTensor = model.predict(inputTensor) as tf.Tensor;
    // Extract data from the output tensor
    const result = outputTensor.dataSync();
    return {
      materialRemovalRate: Math.max(0.1, result[0] * 10),
      surfaceRoughness: Math.max(0.1, Math.min(5, result[1] * 5)),
      dimensionalAccuracy: Math.max(1, Math.min(100, result[2] * 100)),
      processingTime: Math.max(1, Math.min(300, result[3] * 100))
    };
  };

  return {
    rSquared,
    trainingTime: Date.now() - startTime,
    samples: train.length,
    rmse,
    predict
  };
}

// ...removed old predictANN using Matrix and sigmoid...

// Extreme Learning Machine implementation
export async function trainELM(
  data: { trainData: EDMTrainingData[]; testData: EDMTrainingData[] }
): Promise<ModelResult> {
  const startTime = Date.now();
  const train = data.trainData;
  const test = data.testData;
  console.log(`Training ELM with ${train.length} training samples and ${test.length} test samples`);

  const inputSize = 6;
  const hiddenSize = 20;

  // Random input weights and biases (fixed during training)
  const inputWeights = Array(hiddenSize).fill(0).map(() => 
    Array(inputSize).fill(0).map(() => Math.random() * 2 - 1)
  );
  const biases = Array(hiddenSize).fill(0).map(() => Math.random() * 2 - 1);

  // Prepare training and test data
  const trainFeatures = train.map(d => [
    d.voltage / 300,
    d.current / 50,
    d.pulseOnTime / 100,
    d.pulseOffTime / 200,
    d.wireSpeed / 500,
    d.dielectricFlow / 20
  ]);
  const trainTargets = train.map(d => [
    d.materialRemovalRate / 10,
    d.surfaceRoughness / 5,
    d.dimensionalAccuracy / 100,
    d.processingTime / 100
  ]);
  const testFeatures = test.map(d => [
    d.voltage / 300,
    d.current / 50,
    d.pulseOnTime / 100,
    d.pulseOffTime / 200,
    d.wireSpeed / 500,
    d.dielectricFlow / 20
  ]);
  const testTargets = test.map(d => [
    d.materialRemovalRate / 10,
    d.surfaceRoughness / 5,
    d.dimensionalAccuracy / 100,
    d.processingTime / 100
  ]);

  // Calculate hidden layer output matrix H for training
  const H: number[][] = [];
  for (let i = 0; i < trainFeatures.length; i++) {
    const hiddenOutput: number[] = [];
    for (let j = 0; j < hiddenSize; j++) {
      let sum = biases[j];
      for (let k = 0; k < inputSize; k++) {
        sum += trainFeatures[i][k] * inputWeights[j][k];
      }
      hiddenOutput.push(1 / (1 + Math.exp(-sum)));
    }
    H.push(hiddenOutput);
  }

  // Calculate output weights using Moore-Penrose pseudoinverse
  const HT = transpose(H);
  const HTH = matrixMultiply(HT, H);
  for (let i = 0; i < HTH.length; i++) {
    HTH[i][i] += 0.001;
  }
  const HTHInv = matrixInverse(HTH);
  const HTHInvHT = matrixMultiply(HTHInv, HT);
  const outputWeights = matrixMultiply(HTHInvHT, trainTargets);

  // Evaluate on test data
  let totalError = 0;
  let n = 0;
  let flatTargets = testTargets.flat();
  let mean = flatTargets.reduce((sum, v) => sum + v, 0) / flatTargets.length;
  let residualSum = 0;
  let totalSumSquares = 0;
  for (let i = 0; i < testFeatures.length; i++) {
    const predicted = predictELM(testFeatures[i], inputWeights, biases, outputWeights);
    const actual = testTargets[i];
    for (let j = 0; j < 4; j++) {
      const diff = predicted[j] - actual[j];
      totalError += diff * diff;
      residualSum += diff * diff;
      totalSumSquares += (actual[j] - mean) * (actual[j] - mean);
      n++;
    }
  }
  const rmse = Math.sqrt(totalError / n);
  const rSquared = totalSumSquares === 0 ? 0 : 1 - (residualSum / totalSumSquares);

  const predict = (params: any) => {
    const input = [
      (params.laserPower || 3) / 6,
      (params.speed || 3000) / 6000,
      (params.thickness || 4) / 10,
      (params.linearEnergy || 60) / 250,
      (params.speed || 3000) / 6000,
      (params.surfaceRoughness || 1.3) / 5
    ];
    const result = predictELM(input, inputWeights, biases, outputWeights);
    return {
      materialRemovalRate: Math.max(0.1, result[0] * 10),
      surfaceRoughness: Math.max(0.1, Math.min(5, result[1] * 5)),
      dimensionalAccuracy: Math.max(1, Math.min(100, result[2] * 100)),
      processingTime: Math.max(1, Math.min(300, result[3] * 100))
    };
  };

  return {
    rSquared,
    trainingTime: Date.now() - startTime,
    samples: train.length,
    rmse,
    modelData: { inputWeights, biases, outputWeights },
    predict
  };
}

function predictELM(input: number[], inputWeights: number[][], biases: number[], outputWeights: number[][]): number[] {
  // Calculate hidden layer output
  const hiddenOutput: number[] = [];
  for (let i = 0; i < inputWeights.length; i++) {
    let sum = biases[i];
    for (let j = 0; j < input.length; j++) {
      sum += input[j] * inputWeights[i][j];
    }
  // TODO: Replace with tfjs or Math.sigmoid if needed
  hiddenOutput.push(1 / (1 + Math.exp(-sum)));
  }
  
  // Calculate final output
  const output: number[] = [];
  for (let i = 0; i < outputWeights.length; i++) {
    let sum = 0;
    for (let j = 0; j < hiddenOutput.length; j++) {
      sum += hiddenOutput[j] * outputWeights[i][j];
    }
    output.push(sum);
  }
  
  return output;
}

// Genetic Algorithm implementation
export async function trainGA(
  data: { trainData: EDMTrainingData[]; testData: EDMTrainingData[] }
): Promise<ModelResult> {
  const startTime = Date.now();
  const train = data.trainData;
  const test = data.testData;
  console.log(`Training GA with ${train.length} training samples and ${test.length} test samples`);

  const populationSize = 50;
  const generations = 100;
  const mutationRate = 0.1;
  const crossoverRate = 0.8;
  const chromosomeLength = 6 * 4 + 4;

  // Prepare training and test data
  const trainFeatures = train.map(d => [
    d.voltage / 300,
    d.current / 50,
    d.pulseOnTime / 100,
    d.pulseOffTime / 200,
    d.wireSpeed / 500,
    d.dielectricFlow / 20
  ]);
  const trainTargets = train.map(d => [
    d.materialRemovalRate / 10,
    d.surfaceRoughness / 5,
    d.dimensionalAccuracy / 100,
    d.processingTime / 100
  ]);
  const testFeatures = test.map(d => [
    d.voltage / 300,
    d.current / 50,
    d.pulseOnTime / 100,
    d.pulseOffTime / 200,
    d.wireSpeed / 500,
    d.dielectricFlow / 20
  ]);
  const testTargets = test.map(d => [
    d.materialRemovalRate / 10,
    d.surfaceRoughness / 5,
    d.dimensionalAccuracy / 100,
    d.processingTime / 100
  ]);

  // Initialize population
  let population: number[][] = [];
  for (let i = 0; i < populationSize; i++) {
    const chromosome: number[] = [];
    for (let j = 0; j < chromosomeLength; j++) {
      chromosome.push(Math.random() * 2 - 1);
    }
    population.push(chromosome);
  }

  // Evolution loop
  for (let gen = 0; gen < generations; gen++) {
    // Evaluate fitness on training data
    const fitness = population.map(chromosome => evaluateFitness(chromosome, trainFeatures, trainTargets));
    // Selection and reproduction
    const newPopulation: number[][] = [];
    // Elitism - keep best 10%
    const sortedIndices = fitness.map((f, i) => ({ fitness: f, index: i }))
      .sort((a, b) => b.fitness - a.fitness);
    const eliteCount = Math.floor(populationSize * 0.1);
    for (let i = 0; i < eliteCount; i++) {
      newPopulation.push([...population[sortedIndices[i].index]]);
    }
    // Generate rest through crossover and mutation
    while (newPopulation.length < populationSize) {
      const parent1 = tournamentSelection(population, fitness);
      const parent2 = tournamentSelection(population, fitness);
      let [child1, child2] = crossover(parent1, parent2, crossoverRate);
      child1 = mutate(child1, mutationRate);
      child2 = mutate(child2, mutationRate);
      newPopulation.push(child1);
      if (newPopulation.length < populationSize) {
        newPopulation.push(child2);
      }
    }
    population = newPopulation;
    if (gen % 20 === 0) {
      const bestFitness = Math.max(...fitness);
      console.log(`Generation ${gen}, Best Fitness: ${bestFitness}`);
    }
  }

  // Get best individual
  const finalFitness = population.map(chromosome => evaluateFitness(chromosome, trainFeatures, trainTargets));
  const bestIndex = finalFitness.indexOf(Math.max(...finalFitness));
  const bestChromosome = population[bestIndex];

  // Evaluate on test data
  let totalError = 0;
  let n = 0;
  let flatTargets = testTargets.flat();
  let mean = flatTargets.reduce((sum, v) => sum + v, 0) / flatTargets.length;
  let residualSum = 0;
  let totalSumSquares = 0;
  for (let i = 0; i < testFeatures.length; i++) {
    const predicted = predictGA(testFeatures[i], bestChromosome);
    const actual = testTargets[i];
    for (let j = 0; j < 4; j++) {
      const diff = predicted[j] - actual[j];
      totalError += diff * diff;
      residualSum += diff * diff;
      totalSumSquares += (actual[j] - mean) * (actual[j] - mean);
      n++;
    }
  }
  const rmse = Math.sqrt(totalError / n);
  const rSquared = totalSumSquares === 0 ? 0 : 1 - (residualSum / totalSumSquares);

  const predict = (params: any) => {
    const input = [
      (params.laserPower || 3) / 6,
      (params.speed || 3000) / 6000,
      (params.thickness || 4) / 10,
      (params.linearEnergy || 60) / 250,
      (params.speed || 3000) / 6000,
      (params.surfaceRoughness || 1.3) / 5
    ];
    const result = predictGA(input, bestChromosome);
    return {
      materialRemovalRate: Math.max(0.1, result[0] * 10),
      surfaceRoughness: Math.max(0.1, Math.min(5, result[1] * 5)),
      dimensionalAccuracy: Math.max(1, Math.min(100, result[2] * 100)),
      processingTime: Math.max(1, Math.min(300, result[3] * 100))
    };
  };

  return {
    rSquared,
    trainingTime: Date.now() - startTime,
    samples: train.length,
    rmse,
    weights: bestChromosome,
    predict
  };
}

function evaluateFitness(chromosome: number[], inputs: number[][], targets: number[][]): number {
  let totalError = 0;
  
  for (let i = 0; i < inputs.length; i++) {
    const predicted = predictGA(inputs[i], chromosome);
    const actual = targets[i];
    
    for (let j = 0; j < 4; j++) {
      totalError += Math.pow(predicted[j] - actual[j], 2);
    }
  }
  
  const mse = totalError / (inputs.length * 4);
  return 1 / (1 + mse); // Convert to fitness (higher is better)
}

function predictGA(input: number[], chromosome: number[]): number[] {
  const output: number[] = [];
  
  for (let i = 0; i < 4; i++) {
    let sum = chromosome[6 * 4 + i]; // bias
    for (let j = 0; j < 6; j++) {
      sum += input[j] * chromosome[i * 6 + j];
    }
  // TODO: Replace with tfjs or Math.tanh if needed
  output.push(Math.tanh(sum));
  }
  
  return output;
}

function tournamentSelection(population: number[][], fitness: number[], tournamentSize: number = 3): number[] {
  let best = Math.floor(Math.random() * population.length);
  
  for (let i = 1; i < tournamentSize; i++) {
    const competitor = Math.floor(Math.random() * population.length);
    if (fitness[competitor] > fitness[best]) {
      best = competitor;
    }
  }
  
  return [...population[best]];
}

function crossover(parent1: number[], parent2: number[], crossoverRate: number): [number[], number[]] {
  if (Math.random() > crossoverRate) {
    return [[...parent1], [...parent2]];
  }
  
  const crossoverPoint = Math.floor(Math.random() * parent1.length);
  const child1 = [...parent1.slice(0, crossoverPoint), ...parent2.slice(crossoverPoint)];
  const child2 = [...parent2.slice(0, crossoverPoint), ...parent1.slice(crossoverPoint)];
  
  return [child1, child2];
}

function mutate(chromosome: number[], mutationRate: number): number[] {
  return chromosome.map(gene => {
    if (Math.random() < mutationRate) {
      return gene + (Math.random() - 0.5) * 0.2;
    }
    return gene;
  });
}

// Utility functions for matrix operations
function transpose(matrix: number[][]): number[][] {
  return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}

function matrixMultiply(a: number[][], b: number[][]): number[][] {
  const result: number[][] = [];
  for (let i = 0; i < a.length; i++) {
    result[i] = [];
    for (let j = 0; j < b[0].length; j++) {
      let sum = 0;
      for (let k = 0; k < b.length; k++) {
        sum += a[i][k] * b[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

function matrixVectorMultiply(matrix: number[][], vector: number[]): number[] {
  return matrix.map(row => 
    row.reduce((sum, val, i) => sum + val * vector[i], 0)
  );
}

function matrixInverse(matrix: number[][]): number[][] {
  const n = matrix.length;
  const identity = Array(n).fill(0).map((_, i) => 
    Array(n).fill(0).map((_, j) => i === j ? 1 : 0)
  );
  
  // Create augmented matrix
  const augmented = matrix.map((row, i) => [...row, ...identity[i]]);
  
  // Gaussian elimination
  for (let i = 0; i < n; i++) {
    // Find pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
        maxRow = k;
      }
    }
    
    // Swap rows
    [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
    
    // Make diagonal element 1
    const pivot = augmented[i][i];
    if (Math.abs(pivot) < 1e-10) {
      // Add small value to diagonal for numerical stability
      augmented[i][i] += 1e-6;
    }
    
    for (let j = 0; j < 2 * n; j++) {
      augmented[i][j] /= augmented[i][i];
    }
    
    // Eliminate column
    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = augmented[k][i];
        for (let j = 0; j < 2 * n; j++) {
          augmented[k][j] -= factor * augmented[i][j];
        }
      }
    }
  }
  
  // Extract inverse matrix
  return augmented.map(row => row.slice(n));
}

function gaussianElimination(A: number[][], b: number[]): number[] {
  const n = A.length;
  const augmented = A.map((row, i) => [...row, b[i]]);
  
  // Forward elimination
  for (let i = 0; i < n; i++) {
    // Find pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
        maxRow = k;
      }
    }
    
    // Swap rows
    [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
    
    // Eliminate
    for (let k = i + 1; k < n; k++) {
      const factor = augmented[k][i] / augmented[i][i];
      for (let j = i; j < n + 1; j++) {
        augmented[k][j] -= factor * augmented[i][j];
      }
    }
  }
  
  // Back substitution
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = augmented[i][n];
    for (let j = i + 1; j < n; j++) {
      x[i] -= augmented[i][j] * x[j];
    }
    x[i] /= augmented[i][i];
  }
  
  return x;
}

// Generate synthetic data as fallback
function generateSyntheticData(): EDMTrainingData[] {
  const data: EDMTrainingData[] = [];
  for (let i = 0; i < 100; i++) {
    const voltage = 20 + Math.random() * 280;
    const current = 1 + Math.random() * 49;
    const pulseOnTime = 0.5 + Math.random() * 99.5;
    const pulseOffTime = 1 + Math.random() * 199;
    const wireSpeed = 10 + Math.random() * 490;
    const dielectricFlow = 0.5 + Math.random() * 19.5;
    
    // Create realistic relationships
    const materialRemovalRate = (voltage * current * pulseOnTime) / (pulseOffTime * 1000) + Math.random() * 0.5;
    const surfaceRoughness = Math.max(0.1, 5 - (voltage / 100) + (pulseOnTime / 20) + Math.random() * 0.5);
    const dimensionalAccuracy = Math.max(1, 10 - (current / 5) - (voltage / 50) + Math.random() * 2);
    const processingTime = Math.max(1, 60 - (current * 0.5) - (voltage * 0.1) + Math.random() * 10);
    
    data.push({
      voltage,
      current,
      pulseOnTime,
      pulseOffTime,
      wireSpeed,
      dielectricFlow,
      materialRemovalRate,
      surfaceRoughness,
      dimensionalAccuracy,
      processingTime
    });
  }
  return data;
}