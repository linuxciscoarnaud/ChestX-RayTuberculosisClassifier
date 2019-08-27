/**
 * 
 */
package com.chestxraytuberculosiskfoldswithearlystopping;

import java.io.File;
import java.util.Arrays;
import java.util.List;
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
import org.datavec.image.transform.ScaleImageTransform;
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
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Arnaud
 *
 */
public class ChestXRayTuberculosis {

    protected static final Logger log = LoggerFactory.getLogger(ChestXRayTuberculosis.class);
    
    Params params = new Params();
    private NetworkConfig networkConfig = new NetworkConfig();
    public static int numClasses = 0;
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
        File mainTrainPath = new File(System.getProperty("user.dir"), "/src/main/resources/Montgomery/6-FoldsData/run1/data/");
        File maintTestPath = new File(System.getProperty("user.dir"), "/src/main/resources/Montgomery/6-FoldsData/run1/Validation1/");
        // Split up a root directory in to files
        FileSplit trainFileSplit = new FileSplit(mainTrainPath, NativeImageLoader.ALLOWED_FORMATS, params.getRng()); 
        FileSplit testFileSplit = new FileSplit(maintTestPath, NativeImageLoader.ALLOWED_FORMATS, params.getRng());
        // Get the total number of images
        int numExamples = toIntExact(trainFileSplit.length());
        int numTest = toIntExact(testFileSplit.length());
        log.info("Number of images: " + numExamples);
        log.info("Number of test images: " + numTest);
        // Gets the total number of classes
        // This only works if the root directory is clean, meaning it contains only label sub directories.
        numClasses = trainFileSplit.getRootDir().listFiles(File::isDirectory).length; 
        log.info("Number of classes: " + numClasses);
        // Randomizes the order of paths in an array and removes paths randomly to have the same number of paths for each label.
        //BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numClasses, maxPathsPerLabel);       
        // Randomizes the order of paths of all the images in an array. (There is no attempt to have the same number of paths for each label, so there is no random paths removal).
        RandomPathFilter trainPathFilter = new RandomPathFilter(params.getRng(), null, numExamples); 
        RandomPathFilter testPathFilter = new RandomPathFilter(params.getRng(), null, numTest); 
        // Gets the list of loadable locations exposed as an iterator.
        //InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest); 
        InputSplit[] trainInputSplit = trainFileSplit.sample(trainPathFilter);
        InputSplit[] testInputSplit = testFileSplit.sample(testPathFilter);
        // Gets the training data
        InputSplit trainData = trainInputSplit[0];
        InputSplit testData = testInputSplit[0];
        log.info("Train data: " + trainData.length());
        log.info("Test data: " + testData.length());
        
        // Data transformation
        ImageTransform warpTransform = new WarpImageTransform(params.getRng(), 42);
        ImageTransform scaleTransform = new ScaleImageTransform(5); // added
        boolean shuffle = true;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(new Pair<>(warpTransform,0.9),
        		                                                   new Pair<>(scaleTransform,0.9));
        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);
        
        //Data normalization. Puts all the data in the same scale.
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        
        // This will read a local file system and parse images of a given height and width.
        // All images are rescaled and converted to the given height, width, and number of channels.
        ImageRecordReader trainRecordReader = new ImageRecordReader(params.getHeight(), params.getWidth(), params.getChannels(), labelMaker);
        ImageRecordReader testRecordReader = new ImageRecordReader(params.getHeight(), params.getWidth(), params.getChannels(), labelMaker);
        
        // Train data will be initially transformed
        trainRecordReader.initialize(trainData, transform);        
        // Iterator which Traverses through the dataset and preparing the data for the network.
        DataSetIterator trainDataIter = new RecordReaderDataSetIterator(trainRecordReader, params.getMiniBatchSize(), 1, numClasses);
        scaler.fit(trainDataIter);
        trainDataIter.setPreProcessor(scaler);
        //log.info("SHAPE THE FEATURES: " + trainDataIter.next(1));
        
        // Test data will not be initially transformed
        testRecordReader.initialize(testData);
        // Iterator which Traverses through the test set and preparing the data for the network.
        DataSetIterator testDataIter = new RecordReaderDataSetIterator(testRecordReader, params.getMiniBatchSize(), 1, numClasses);
        scaler.fit(testDataIter);
        testDataIter.setPreProcessor(scaler);
        
        /**
         * Building model...
        **/
        
        log.info("Building model....");
        
        ComputationGraph network;
		String modelFilename = "chestXRayTuberculosis6FolkRun1Model.zip";
		
		if (new File(modelFilename).exists()) {
			log.info("Loading existing model...");
			network = ModelSerializer.restoreComputationGraph(modelFilename);
		} else {
			network = networkConfig.getNetworkConfig();
			network.init();
			//System.out.println(network.summary(InputType.convolutional(height, width, channels)));
			log.info(network.summary(InputType.convolutional(params.getHeight(), params.getWidth(), params.getChannels())));
			
			/**
             * Enabling the UI...
            **/
        	
        	// Initialize the user interface backend.
            //UIServer uiServer = UIServer.getInstance();
            // Configure where the information is to be stored. Here, we store in the memory.
            // It can also be saved in file using: StatsStorage statsStorage = new FileStatsStorage(File), if we intend to load and use it later.
            //StatsStorage statsStorage = new InMemoryStatsStorage();
            // Attach the StatsStorage instance to the UI. This allows the contents of the StatsStorage to be visualized.
            //uiServer.attach(statsStorage);
            // Add the StatsListener to collect information from the network, as it trains.
            //network.setListeners(new StatsListener(statsStorage));
			
	    /**
             * Configuring early stopping
            **/
			
	    EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
			   .epochTerminationConditions(new MaxEpochsTerminationCondition(params.getEpochs()))
			   .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(params.getMaxTimeIterTerminationCondition(), TimeUnit.HOURS))
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
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		new ChestXRayTuberculosis().execute(args);
	}
}

