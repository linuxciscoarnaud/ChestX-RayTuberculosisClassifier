/**
 * 
 */
package com.chestxraytuberculosiskfoldswithearlystopping;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static java.lang.Math.toIntExact;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
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
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Arnaud
 *
 */
public class ChestXRayTuberculosis {

    protected static final Logger log = LoggerFactory.getLogger(ChestXRayTuberculosis.class);
	
	protected static int height = 512;
    protected static int width = 512;
    protected static int channels = 1; // We are dealing with gray scale images
    protected static int miniBatchSize = 4;
    
    protected static long seed = 123; // Integer for reproducibility of a random number generator
    protected static Random rng = new Random(seed);
    protected static int epochs = 400;
    protected static int maxTimeIterTerminationCondition = 24; // 24 hours
    
    private int numClasses;
 		    
 	protected static boolean save = true;
 	 	
 	public void execute(String[] args) throws Exception {
 		
 		/**
         * Loading the data
        **/
		
		//System.out.println("Loading data....");
		log.info("Loading data....");
		
		// Returns as label the base name of the parent directory of the data.
    	ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();    	
    	// Gets the main path
        File mainTrainPath = new File(System.getProperty("user.dir"), "/src/main/resources/6-FoldsData/run3/data/");
        File maintTestPath = new File(System.getProperty("user.dir"), "/src/main/resources/6-FoldsData/run3/Validation3/");
        // Split up a root directory in to files
        FileSplit trainFileSplit = new FileSplit(mainTrainPath, NativeImageLoader.ALLOWED_FORMATS, rng); 
        FileSplit testFileSplit = new FileSplit(maintTestPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        // Get the total number of images
        int numExamples = toIntExact(trainFileSplit.length());
        int numTest = toIntExact(testFileSplit.length());
        // Gets the total number of classes
        // This only works if the root directory is clean, meaning it contains only label sub directories.
        numClasses = trainFileSplit.getRootDir().listFiles(File::isDirectory).length; 
        // Randomizes the order of paths in an array and removes paths randomly to have the same number of paths for each label.
        //BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numClasses, maxPathsPerLabel);       
        // Randomizes the order of paths of all the images in an array. (There is no attempt to have the same number of paths for each label, so there is no random paths removal).
        RandomPathFilter trainPathFilter = new RandomPathFilter(rng, null, numExamples); 
        RandomPathFilter testPathFilter = new RandomPathFilter(rng, null, numTest); 
        // Gets the list of loadable locations exposed as an iterator.
        //InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest); 
        InputSplit[] trainInputSplit = trainFileSplit.sample(trainPathFilter);
        InputSplit[] testInputSplit = testFileSplit.sample(testPathFilter);
        // Gets the training data
        InputSplit trainData = trainInputSplit[0];
        InputSplit testData = testInputSplit[0];
        
        // Data transformation
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
        boolean shuffle = true;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(new Pair<>(warpTransform,0.8));
        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);
        
        //Data normalization. Puts all the data in the same scale.
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        
        // This will read a local file system and parse images of a given height and width.
        // All images are rescaled and converted to the given height, width, and number of channels.
        ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
        ImageRecordReader testRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
        
        // Train data will be initially transformed
        trainRecordReader.initialize(trainData, transform);        
        // Iterator which Traverses through the dataset and preparing the data for the network.
        DataSetIterator trainDataIter = new RecordReaderDataSetIterator(trainRecordReader, miniBatchSize, 1, numClasses);
        scaler.fit(trainDataIter);
        trainDataIter.setPreProcessor(scaler);
        
        // Test data will not be initially transformed
        testRecordReader.initialize(testData);
        // Iterator which Traverses through the test set and preparing the data for the network.
        DataSetIterator testDataIter = new RecordReaderDataSetIterator(testRecordReader, miniBatchSize, 1, numClasses);
        scaler.fit(testDataIter);
        testDataIter.setPreProcessor(scaler);
        
        /**
         * Building model...
        **/
        
        log.info("Building model....");
        
        ComputationGraph network;
		String modelFilename = "chestXRayTuberculosis6Folk3Model.zip";
		
		if (new File(modelFilename).exists()) {
			log.info("Loading existing model...");
			network = ModelSerializer.restoreComputationGraph(modelFilename);
		} else {
			network = chestXRayTuberculosisConfig();
			network.init();
			//System.out.println(network.summary(InputType.convolutional(height, width, channels)));
			log.info(network.summary(InputType.convolutional(height, width, channels)));
			
			/**
             * Enabling the UI...
            **/
        	
        	// Initialize the user interface backend.
            UIServer uiServer = UIServer.getInstance();
            // Configure where the information is to be stored. Here, we store in the memory.
            // It can also be saved in file using: StatsStorage statsStorage = new FileStatsStorage(File), if we intend to load and use it later.
            StatsStorage statsStorage = new InMemoryStatsStorage();
            // Attach the StatsStorage instance to the UI. This allows the contents of the StatsStorage to be visualized.
            uiServer.attach(statsStorage);
            // Add the StatsListener to collect information from the network, as it trains.
        	network.setListeners(new StatsListener(statsStorage));
			
			/**
		     * Configuring early stopping
		    **/
			
			EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
					      .epochTerminationConditions(new MaxEpochsTerminationCondition(epochs))
					      .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(maxTimeIterTerminationCondition, TimeUnit.HOURS))
					      .scoreCalculator(new DataSetLossCalculator(testDataIter, true))
					.evaluateEveryNEpochs(1)
					      .modelSaver(new LocalFileGraphSaver(new File(System.getProperty("user.dir")).toString()))
					      .build();
			
			EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf, network, trainDataIter);
			
			
			/**
             * Conducting early stopping training of the model...
            **/
			
			log.info("Training model....");
			EarlyStoppingResult<ComputationGraph> result = trainer.fit();
			
			/**
	         * Saving the best model...
	        **/
	        
	        log.info("Saving the best model....");
	        //Get the best model:
	        network = result.getBestModel();
	        if (save) {
	        	ModelSerializer.writeModel(network, modelFilename, true);
	        }
	        log.info("Best model saved....");
	        
	        /**
             * Getting the results...
            **/
			
			//Print out the results:
			System.out.println("Termination reason: " + result.getTerminationReason());
			System.out.println("Termination details: " + result.getTerminationDetails());
			System.out.println("Total epochs: " + result.getTotalEpochs());
			System.out.println("Best epoch number: " + result.getBestModelEpoch());
			System.out.println("Score at best epoch: " + result.getBestModelScore());
		}
 	}
 	
 	/**
     * Configuring the network....
    **/
 	
 	public ComputationGraph chestXRayTuberculosisConfig() {
 		
 		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
    			.seed(seed)
    			
    			.weightInit(WeightInit.DISTRIBUTION)
                .dist(new GaussianDistribution(0, 0.005))
                
    			.weightInit(WeightInit.RELU) // This is known as He normal initialization
				.activation(Activation.RELU)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Adam(8e-5, 1e-8, 0.9, 0.999))
				.convolutionMode(ConvolutionMode.Same)
				.trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .cacheMode(CacheMode.NONE)
				.l2(5 * 1e-4)
				.miniBatch(true)
				.graphBuilder()
				.addInputs("in")
                .setInputTypes(InputType.convolutional(height, width, channels))
                
                // block 1
                .addLayer("conv1_1", new ConvolutionLayer.Builder()
                		.convolutionMode(ConvolutionMode.Truncate)
                		.kernelSize(3, 3)
                		.stride(2, 2)
                		.padding(1, 1)
                		.nIn(channels)
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
                		.nIn(channels)
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
                		.nOut(numClasses)
                		.build(), "GAP")
                
                
                
                .setOutputs("fc")                
				.build();
		
		return new ComputationGraph(conf);
 	}
	
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		new ChestXRayTuberculosis().execute(args);
	}
}
