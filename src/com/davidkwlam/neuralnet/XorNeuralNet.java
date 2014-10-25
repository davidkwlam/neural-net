package com.davidkwlam.neuralnet;

import java.util.stream.IntStream;

/*
 * Training an XOR neural net
 */
public class XorNeuralNet {

	public static void main(String[] args) {

		double[][] inputs = new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		double[][] values = new double[][] {   { -1 },    { 1 },    { 1 },   { -1 } };

		NeuralNet nn = new NeuralNet(2, 4, 1, 0.2, 0);

		double totalError = Double.POSITIVE_INFINITY;
		int epochs = 0;
		
		while (totalError > 0.05) {
			IntStream.range(0, inputs.length).forEach(i -> nn.train(inputs[i], values[i]));

			epochs++;
			
			totalError = IntStream
					.range(0, inputs.length)
					.mapToDouble(i -> 0.5 * Math.pow(values[i][0] - nn.output(inputs[i])[0], 2))
					.reduce(0, Double::sum);
			
			System.out.println(epochs + ":" + totalError);
		}

		// Output the results
		// Note: The results will be close to, but not exactly, -1 or 1. 
		//		 For this neural net, -1 == <-0.8 and 1 = >0.8
		System.out.println(nn.output(inputs[0])[0]);
		System.out.println(nn.output(inputs[1])[0]);
		System.out.println(nn.output(inputs[2])[0]);
		System.out.println(nn.output(inputs[3])[0]);
	}
	
}
