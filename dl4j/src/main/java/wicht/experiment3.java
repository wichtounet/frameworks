package wicht;

import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.api.Layer;
import static org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit;
import static org.deeplearning4j.nn.conf.layers.RBM.HiddenUnit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;

public class experiment3 {
    private static final Logger log = LoggerFactory.getLogger(experiment3.class);

    public static void main(String[] args) throws Exception {
        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(100, 60000, true);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .regularization(false)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list()
            .layer(0, new RBM.Builder()
                    .nIn(784).nOut(500)
                    .weightInit(WeightInit.XAVIER)
                    .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                    .updater(Updater.NESTEROVS)
                    .learningRate(0.1)
                    .momentum(0.9)
                    .k(1)
                    .build())
            .pretrain(true).backprop(false)
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(600));

        {
            while(mnistTrain.hasNext()){
                DataSet next = mnistTrain.next();
                INDArray in = next.getFeatureMatrix();
                INDArray out = model.reconstruct(in, 1);

                log.info("    distance(1):" + in.distance1(out));
                log.info("    distance(2):" + in.distance2(out));
                log.info("square distance:" + in.squaredDistance(out));

                break;
            }

            mnistTrain.reset();
        }

        for(int i = 0; i < 50; i++) {
            model.fit(mnistTrain);

            //while(mnistTrain.hasNext()){
                //DataSet next = mnistTrain.next();
                //INDArray in = next.getFeatureMatrix();
                //INDArray out = model.reconstruct(in, 1);

                //log.info("    distance(1):" + in.distance1(out));
                //log.info("    distance(2):" + in.distance2(out));
                //log.info("square distance:" + in.squaredDistance(out));
            //}

            //mnistTrain.reset();
        }
    }
}
