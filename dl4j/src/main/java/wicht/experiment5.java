package wicht;

import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
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
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
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

/**
 * Created by agibsonccc on 9/16/15.
 * Modified by dmichelin on 12/10/2016 to add documentation
 */
public class experiment5 {
    private static final Logger log = LoggerFactory.getLogger(experiment5.class);

    public static void main(String[] args) throws Exception {
        log.info("Build data....");
        DataSetIterator trainIt = new CifarDataSetIterator(100, 50000, true);
        DataSetIterator testIt = new CifarDataSetIterator(100, 10000, false);

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .iterations(1)
                .regularization(false)
                .learningRate(0.001)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(3)
                        .stride(1, 1)
                        .nOut(12)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(24)
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu").nOut(64).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(10)
                        .activation("softmax")
                        .build())
                .setInputType(InputType.convolutionalFlat(32,32,3))
                .backprop(true).pretrain(false);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.setListeners(new ScoreIterationListener(500));
        model.init();

        log.info("Train model....");
        for( int i=0; i<50; i++ ){
            log.info("Epoch " + i);
            model.fit(trainIt);

            // We need the train error after each epoch

            //trainIt.reset();

            //Evaluation eval = new Evaluation(10);
            //while(trainIt.hasNext()){
                //DataSet next = trainIt.next();
                //INDArray output = model.output(next.getFeatureMatrix());
                //eval.eval(next.getLabels(), output);
            //}

            //log.info("Train accuracy:" + eval.accuracy());

            //trainIt.reset();
        }

        // After training, we need the test error

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(10);
        while(testIt.hasNext()){
            DataSet next = testIt.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), output);
        }

        log.info(eval.stats());
    }
}
