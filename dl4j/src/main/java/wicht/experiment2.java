package wicht;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//import org.deeplearning4j.nn.conf.LearningRatePolicy;

/**
 * Created by agibsonccc on 9/16/15.
 * Modified by dmichelin on 12/10/2016 to add documentation
 */
public class experiment2 {
    private static final Logger log = LoggerFactory.getLogger(experiment2.class);

    public static void main(String[] args) throws Exception {
        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        int batchSize = 100; // Test batch size
        int numEpochs = 50; // Number of training epochs
        int iterations = 1; // Number of training iterations

        /*
            Create an iterator using the batch size for one iteration
         */
        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,123);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,123);

        /*
            Construct the neural network
         */
        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .iterations(1)
                .regularization(false)
                .learningRate(0.1)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD)
                .momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(8)
                        .activation("sigmoid")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(8)
                        .activation("sigmoid")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("sigmoid")
                        .nOut(150).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                .backprop(true).pretrain(false);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.setListeners(new ScoreIterationListener(600));  //print the score with every iteration
        model.init();

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
        	log.info("Epoch " + i);
            model.fit(mnistTrain);

            // We need the train error after each epoch

            mnistTrain.reset();

            Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
            while(mnistTrain.hasNext()){
                DataSet next = mnistTrain.next();
                INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
                eval.eval(next.getLabels(), output); //check the prediction against the true class
            }

            log.info("Train accuracy:" + eval.accuracy());

            mnistTrain.reset();
        }

        // After training, we need the test error

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
    }
}
