package wicht;

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
import org.deeplearning4j.optimize.listeners.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import org.datavec.api.io.labels.*;
import org.datavec.image.recordreader.*;
import org.deeplearning4j.datasets.datavec.*;
import org.datavec.api.split.FileSplit;

public class experiment6 {
    private static final Logger log = LoggerFactory.getLogger(experiment6.class);

    public static void main(String[] args) throws Exception {
        int nChannels = 3; // Number of input channels
        int outputNum = 1000; // The number of possible outcomes
        int batchSize = 64; // Test batch size
        int numEpochs = 10; // Number of training epochs
        int iterations = 1; // Number of training iterations

        /*
            Create an iterator using the batch size for one iteration
         */
        log.info("Load data....");

        File parentDir = new File("/data/datasets/imagenet_resized/train/");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(256,256,nChannels,labelMaker);
        recordReader.initialize(new FileSplit(parentDir));

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,-1,outputNum);

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .iterations(1)
                .regularization(false)
                .learningRate(0.01)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD)
                .momentum(0.9)
                .list()

                .layer(0, new ConvolutionLayer.Builder(3, 3).stride(1, 1).padding(1, 1).nIn(nChannels).nOut(16).activation("relu").build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).stride(2,2).build())

                .layer(2, new ConvolutionLayer.Builder(3, 3).stride(1, 1).padding(1, 1).nOut(16).activation("relu").build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).stride(2,2).build())

                .layer(4, new ConvolutionLayer.Builder(3, 3).stride(1, 1).padding(1, 1).nOut(32).activation("relu").build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).stride(2,2).build())

                .layer(6, new ConvolutionLayer.Builder(3, 3).stride(1, 1).padding(1, 1).nOut(32).activation("relu").build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).stride(2,2).build())

                .layer(8, new ConvolutionLayer.Builder(3, 3).stride(1, 1).padding(1, 1).nOut(32).activation("relu").build())
                .layer(9, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).stride(2,2).build())

                .layer(10, new DenseLayer.Builder().activation("relu").nOut(2048).build())
                .layer(11, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .setInputType(InputType.convolutionalFlat(256,256,nChannels)) //See note below
                .backprop(true).pretrain(false);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.setListeners(new ScoreIterationListener(1), new PerformanceListener(1));
        model.init();

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            log.info("Epoch " + i);
            model.fit(dataIter);

            // We need the train error after each epoch

            dataIter.reset();
        }

        // After training, we need the test error

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while(dataIter.hasNext()){
            DataSet next = dataIter.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
    }
}
