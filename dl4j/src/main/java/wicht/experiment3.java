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
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
            .miniBatch(true)
            .regularization(false)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .learningRate(0.1)
            .momentum(0.9)
            //.list()
            .layer(new RBM.Builder()
                    .nIn(784).nOut(500)
                    .activation("sigmoid")
                    .weightInit(WeightInit.XAVIER)
                    .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                    .updater(Updater.NESTEROVS)
                    .learningRate(0.1)
                    .momentum(0.9)
                    .k(1)
                    .build())
            //.backprop(false)
            .build();

        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);

        //MultiLayerNetwork model = new MultiLayerNetwork(conf);
        //model.init();
        //model.setListeners(new ScoreIterationListener(600));

        //org.deeplearning4j.nn.layers.feedforward.rbm.RBM rbm = (org.deeplearning4j.nn.layers.feedforward.rbm.RBM) model.getLayer(0);
        org.deeplearning4j.nn.layers.feedforward.rbm.RBM rbm = (org.deeplearning4j.nn.layers.feedforward.rbm.RBM) layer;

        {
            double d1 = 0;

            while(mnistTrain.hasNext()){
                DataSet next = mnistTrain.next();

                INDArray v0 = next.getFeatureMatrix();
                // TODO Restore
                /*Pair<INDArray, INDArray> h0 = rbm.sampleHiddenGivenVisible(v0);
                Pair<INDArray, INDArray> v1 = rbm.sampleVisibleGivenHidden(h0.getFirst());

                v0.subi(v1.getFirst());
                INDArray error = v0.mul(v0);

                d1 += (Double) error.meanNumber();*/
            }

            d1 /= 600.0;

            log.info("Rec. Error: " + d1);

            mnistTrain.reset();
        }

        for(int i = 0; i < 50; i++) {
            while(mnistTrain.hasNext()){
                DataSet next = mnistTrain.next();
                INDArray v0 = next.getFeatureMatrix();

                rbm.fit(v0);
            }

            mnistTrain.reset();

            //model.fit(mnistTrain);

            {
                double d1 = 0;

                while(mnistTrain.hasNext()){
                    DataSet next = mnistTrain.next();

                    // TODO Restore
                    /*INDArray v0 = next.getFeatureMatrix();
                    Pair<INDArray, INDArray> h0 = rbm.sampleHiddenGivenVisible(v0);
                    Pair<INDArray, INDArray> v1 = rbm.sampleVisibleGivenHidden(h0.getFirst());

                    v0.subi(v1.getFirst());
                    INDArray error = v0.mul(v0);

                    d1 += (Double) error.meanNumber();
                    */
                }

                d1 /= 600.0;

                log.info("Rec. Error: " + d1);

                mnistTrain.reset();
            }
        }
    }
}
