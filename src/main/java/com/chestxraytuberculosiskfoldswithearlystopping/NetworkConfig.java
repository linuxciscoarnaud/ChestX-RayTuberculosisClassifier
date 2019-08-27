/**
 * 
 */
package com.chestxraytuberculosiskfoldswithearlystopping;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author Arnaud
 *
 */

public class NetworkConfig {

	Params params = new Params();
	
	public ComputationGraph getNetworkConfig() {
		
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(params.getSeed())
				
				//.weightInit(WeightInit.DISTRIBUTION)
                //.dist(new GaussianDistribution(0, 0.005))
				
				.optimizationAlgo(params.getOptimizationAlgorithm())
				.weightInit(params.getWeightInit())
				.activation(params.getActivation())
				.updater(params.getUpdater())
				.convolutionMode(ConvolutionMode.Same)
				.trainingWorkspaceMode(params.getWorkspaceMode())
                .inferenceWorkspaceMode(params.getWorkspaceMode())
                .cacheMode(params.getCacheMode())
                .l2(5 * 1e-4)
				.miniBatch(true)
				.graphBuilder()
				.addInputs("in")
                .setInputTypes(InputType.convolutional(params.getHeight(), params.getWidth(), params.getChannels()))
                
             // block 1
                .addLayer("conv1_1", new ConvolutionLayer.Builder()
                		.convolutionMode(ConvolutionMode.Truncate)
                		.kernelSize(3, 3)
                		.stride(2, 2)
                		.padding(1, 1)
                		.nIn(params.getChannels())
                		.nOut(16)
                		.build(), "in")
                .addLayer("batchn1_1", new BatchNormalization.Builder()
                		.build(), "conv1_1")
                .addLayer("conv1_2", new ConvolutionLayer.Builder()
                		.convolutionMode(ConvolutionMode.Truncate)
                		.kernelSize(3, 3)
                		.stride(2, 2)
                		.padding(1, 1)
                		.nOut(16)
                		.build(), "batchn1_1")
                .addLayer("batchn1_2", new BatchNormalization.Builder()
                		.build(), "conv1_2")
                .addLayer("conv1", new ConvolutionLayer.Builder()
                		.convolutionMode(ConvolutionMode.Truncate)
                		.kernelSize(1, 1)
                		.stride(4, 4)
                		.padding(0, 0)
                		.nIn(params.getChannels())
                		.nOut(16)
                		.build(), "in")
                .addLayer("batchn1", new BatchNormalization.Builder()
                		.build(), "conv1")
                .addVertex("bloc1_merge", new ElementWiseVertex(ElementWiseVertex.Op.Add), "batchn1_2", "batchn1")
                .addLayer("maxpool1", new SubsamplingLayer.Builder()
                		.poolingType(SubsamplingLayer.PoolingType.MAX)
                		.kernelSize(3, 3)
                		.stride(2, 2)
                		.build(), "bloc1_merge")
                
                 // block 2
                .addLayer("conv2_1", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(32)
                		.build(), "maxpool1")
                .addLayer("batchn2_1", new BatchNormalization.Builder()
                		.build(), "conv2_1")
                .addLayer("conv2_2", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(32)
                		.build(), "batchn2_1")
                .addLayer("batchn2_2", new BatchNormalization.Builder()
                		.build(), "conv2_2")
                .addLayer("conv2", new ConvolutionLayer.Builder()
                		.kernelSize(1, 1)
                		.stride(1, 1)
                		.padding(0, 0)
                		.nOut(32)
                		.build(), "maxpool1")
                .addLayer("batchn2", new BatchNormalization.Builder()
                		.build(), "conv2")
                .addVertex("bloc2_merge", new ElementWiseVertex(ElementWiseVertex.Op.Add), "batchn2_2", "batchn2")
                .addLayer("maxpool2", new SubsamplingLayer.Builder()
                		.poolingType(SubsamplingLayer.PoolingType.MAX)
                		.kernelSize(3, 3)
                		.stride(2, 2)
                		.build(), "bloc2_merge")
                
                 // block 3
                .addLayer("conv3_1", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(48)
                		.build(), "maxpool2")
                .addLayer("batchn3_1", new BatchNormalization.Builder()
                		.build(), "conv3_1")
                .addLayer("conv3_2", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(48)
                		.build(), "batchn3_1")
                .addLayer("batchn3_2", new BatchNormalization.Builder()
                		.build(), "conv3_2")
                .addLayer("conv3", new ConvolutionLayer.Builder()
                		.kernelSize(1, 1)
                		.stride(1, 1)
                		.padding(0, 0)
                		.nOut(48)
                		.build(), "maxpool2")
                .addLayer("batchn3", new BatchNormalization.Builder()
                		.build(), "conv3")
                .addVertex("bloc3_merge", new ElementWiseVertex(ElementWiseVertex.Op.Add), "batchn3_2", "batchn3")
                .addLayer("maxpool3", new SubsamplingLayer.Builder()
                		.poolingType(SubsamplingLayer.PoolingType.MAX)
                		.kernelSize(3, 3)
                		.stride(2, 2)
                		.build(), "bloc3_merge")
                
                 // block 4
                .addLayer("conv4_1", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(64)
                		.build(), "maxpool3")
                .addLayer("batchn4_1", new BatchNormalization.Builder()
                		.build(), "conv4_1")
                .addLayer("conv4_2", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(64)
                		.build(), "batchn4_1")
                .addLayer("batchn4_2", new BatchNormalization.Builder()
                		.build(), "conv4_2")
                .addLayer("conv4", new ConvolutionLayer.Builder()
                		.kernelSize(1, 1)
                		.stride(1, 1)
                		.padding(0, 0)
                		.nOut(64)
                		.build(), "maxpool3")
                .addLayer("batchn4", new BatchNormalization.Builder()
                		.build(), "conv4")
                .addVertex("bloc4_merge", new ElementWiseVertex(ElementWiseVertex.Op.Add), "batchn4_2", "batchn4")
                .addLayer("maxpool4", new SubsamplingLayer.Builder()
                		.poolingType(SubsamplingLayer.PoolingType.MAX)
                		.kernelSize(3, 3)
                		.stride(2, 2)
                		.build(), "bloc4_merge")
                
                // block 5
                .addLayer("conv5_1", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(80)
                		.build(), "maxpool4")
                .addLayer("batchn5_1", new BatchNormalization.Builder()
                		.build(), "conv5_1")
                .addLayer("conv5_2", new ConvolutionLayer.Builder()
                		.kernelSize(3, 3)
                		.stride(1, 1)
                		.padding(1, 1)
                		.nOut(80)
                		.build(), "batchn5_1")
                .addLayer("batchn5_2", new BatchNormalization.Builder()
                		.build(), "conv5_2")
                .addLayer("conv5", new ConvolutionLayer.Builder()
                		.kernelSize(1, 1)
                		.stride(1, 1)
                		.padding(0, 0)
                		.nOut(80)
                		.build(), "maxpool4")
                .addLayer("batchn5", new BatchNormalization.Builder()
                		.build(), "conv5")
                .addVertex("bloc5_merge", new ElementWiseVertex(ElementWiseVertex.Op.Add), "batchn5_2", "batchn5")
                .addLayer("maxpool5", new SubsamplingLayer.Builder()
                		.poolingType(SubsamplingLayer.PoolingType.MAX)
                		.kernelSize(3, 3)
                		.stride(2, 2)
                		.build(), "bloc5_merge")
                
                //Global average pooling layer
                .addLayer("GAP", new GlobalPoolingLayer.Builder()
                		.poolingType(PoolingType.AVG)
                		.build(), "maxpool5")
                
                // Fully-connected sofmax layer with two output
                .addLayer("fc", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                		.dist(new GaussianDistribution(0, 1.76e-5))
                        //.weightInit(WeightInit.RELU)
                		.activation(Activation.SOFTMAX)                		
                		.nOut(ChestXRayTuberculosis.numClasses)
                		.build(), "GAP")
				
				.setOutputs("fc")                
				.build();
				
		return new ComputationGraph(conf);
	}
}
