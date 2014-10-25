package com.davidkwlam.neuralnet;

import java.util.Random;
import java.util.stream.IntStream;

public class NeuralNet {

	private final int _numInputs, _numHidden, _numOutputs;
	private final double BIAS = 1.0, _learningRate, _momentumTerm;
	private final double[] _inputNodes, _hiddenNodes, _outputNodes;
	private final double[][] inputToHiddenWeights, inputToHiddenWeightsPrev, hiddenToOutputWeights, hiddenToOutputWeightsPrev; 

	public NeuralNet(int numInputs, int numHidden, int numOutputs, double learningRate, double momentumTerm) {
		_numInputs = numInputs;
		_numHidden = numHidden;
		_numOutputs = numOutputs;
		_learningRate = learningRate;
		_momentumTerm = momentumTerm;
		
		_inputNodes = new double[numInputs + 1];
		_hiddenNodes = new double[numHidden + 1];
		_outputNodes = new double[numOutputs];
		
		// Set biases
		_inputNodes[numInputs] = BIAS;
		_hiddenNodes[numHidden] = BIAS;
		
		hiddenToOutputWeights = new double[numOutputs][numHidden + 1];
		hiddenToOutputWeightsPrev = new double[numOutputs][numHidden + 1];

		inputToHiddenWeights = new double[numHidden][numInputs + 1];
		inputToHiddenWeightsPrev = new double[numHidden][numInputs + 1];
		
		// Initialize weights to a a random value between -0.5 and 0.5
		Random r = new Random();

		for (int i = 0; i < hiddenToOutputWeights.length; i++) {
			for (int j = 0; j < hiddenToOutputWeights[i].length; j++) {
				hiddenToOutputWeights[i][j] = r.nextDouble() - 0.5;
				hiddenToOutputWeightsPrev[i][j] = hiddenToOutputWeights[i][j];
			}
		}

		for (int i = 0; i < inputToHiddenWeights.length; i++) {
			for (int j = 0; j < inputToHiddenWeights[i].length; j++) {
				inputToHiddenWeights[i][j] = r.nextDouble() - 0.5;
				inputToHiddenWeightsPrev[i][j] = inputToHiddenWeights[i][j];
			}
		}
	}

	public double[] output(double[] inputs) {
		calculate(inputs);
		return _outputNodes;
	}
	
	private void calculate(double[] inputs) {
		if (inputs.length != _numInputs) {
			// Throw a hissy fit
		}

		IntStream.range(0, _numInputs).parallel()
				.forEach(i -> _inputNodes[i] = inputs[i]);
		
		IntStream.range(0, _numHidden).parallel()
				.forEach(i -> _hiddenNodes[i] = sigmoid(dotProduct(_inputNodes, inputToHiddenWeights[i])));
		
		IntStream.range(0, _numOutputs).parallel()
				.forEach(i -> _outputNodes[i] = sigmoid(dotProduct(_hiddenNodes, hiddenToOutputWeights[i])));
	}

	public double[] train(double[] inputs, double values[]) {
		if (values.length != _numOutputs) {
			// Throw a hissy fit
		}
		
		calculate(inputs);

		double[] deltas = new double[_numOutputs];
		
		// 1. Train weights from hidden layer to output neuron
		for (int i = 0; i < _numOutputs; i++) {
			deltas[i] = fPrime(_outputNodes[i]) * (values[i] - _outputNodes[i]);
			updateWeights(hiddenToOutputWeights[i], hiddenToOutputWeightsPrev[i], _hiddenNodes, deltas[i]);
		}
				
		// 2. Train weights from input layer to hidden layer
		for (int i = 0; i < _numHidden; i++) {			
			// Get weights from this hidden node to all output nodes
			double[] outputWeights = new double[_numOutputs];
			for (int j = 0; j < hiddenToOutputWeights.length; j++) {
				outputWeights[j] = hiddenToOutputWeights[j][i];
			}

			double delta = fPrime(_hiddenNodes[i]) * dotProduct(deltas, outputWeights);
			updateWeights(inputToHiddenWeights[i], inputToHiddenWeightsPrev[i], _inputNodes, delta);
		}

		// 3. Return the errors
		double[] errors = new double[_numOutputs];
		for (int i = 0; i < _numOutputs; i++) {
			errors[i] = Math.abs(_outputNodes[i] - values[i]);
		}
		return errors;
	}

	private void updateWeights(double[] currWeights, double[] prevWeights, double[] inputs, double delta) {
		for (int i = 0; i < currWeights.length; i++) {
			double currWeight = currWeights[i];
			double prevWeight = prevWeights[i];
			currWeights[i] = currWeight + _momentumTerm * (currWeight - prevWeight) + _learningRate * delta * inputs[i];
			prevWeights[i] = currWeight;
		}
	}

	private double fPrime(double x) {
		return 0.5 * (1 - Math.pow(x, 2));
	}

	private double sigmoid(double x) {
		return 2 / (1 + Math.exp(-x)) - 1;
	}
	
	private double dotProduct(double[] a, double[] b) {
		return IntStream.range(0, a.length).mapToDouble( i -> a[i] * b[i] ).reduce(0, Double::sum);
	}
}