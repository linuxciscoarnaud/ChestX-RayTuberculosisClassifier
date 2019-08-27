/**
 * 
 */
package com.chestxraytuberculosiskfoldswithearlystopping;

import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;

/**
 * @author Arnaud
 *
 */

public class Params {

	// Parameters for network configuration
	private OptimizationAlgorithm optimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
	private Activation activation = Activation.RELU;
	private WeightInit weightInit = WeightInit.RELU; // This is known as He normal initialization
	private IUpdater updater = new Adam(1e-4, 1e-8, 0.9, 0.999); // I changed the learning rate from 8e-5 to 1e-4
	private CacheMode cacheMode = CacheMode.NONE;
	private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
	
	// Parameters for input data
	private int height = 512;
	private int width = 512;
	private int channels = 1; // We are dealing with gray scale images
    
    // Parameters for the training phase
    protected static int miniBatchSize = 4;
    protected static int epochs = 400;
    protected static int maxTimeIterTerminationCondition = 48; // 48 hours
    
    protected static long seed = 123; // Integer for reproducibility of a random number generator
    protected static Random rng = new Random(seed);
    
    // Getters
	
 	public OptimizationAlgorithm getOptimizationAlgorithm() {
 		return optimizationAlgorithm;
 	}
 	
 	public Activation getActivation() {
 		return activation;
 	}
 	
 	public WeightInit getWeightInit() {
 		return weightInit;
 	}
 	
 	public IUpdater getUpdater() {
 		return updater;
 	}
 	
 	public CacheMode getCacheMode() {
 		return cacheMode;
 	}
 	
 	public WorkspaceMode getWorkspaceMode() {
 		return workspaceMode;
 	}
 	
 	public int getEpochs() {
 		return epochs;
 	}
 	
 	public int getMaxTimeIterTerminationCondition() {
 		return maxTimeIterTerminationCondition;
 	}
 	
 	public int getMiniBatchSize() {
 		return miniBatchSize;
 	}
 	
 	public int getHeight() {
 		return height;
 	}
 	
 	public int getWidth( ) {
 		return width;
 	}
 	
 	public int getChannels() {
 		return channels;
 	}
 	
 	public long getSeed() {
 		return seed;
 	}
 	
 	public Random getRng() {
 		return rng;
 	}    
}
