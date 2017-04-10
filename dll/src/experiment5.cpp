//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include "dll/neural/conv_layer.hpp"
#include "dll/neural/dense_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "cifar/cifar10_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    auto dataset = cifar::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 3, 32, 32>>();

    for (auto& image : dataset.training_images) {
        for (auto& pixel : image) {
            pixel *= (1.0 / 256.0);
        }
    }

    for (auto& image : dataset.test_images) {
        for (auto& pixel : image) {
            pixel *= (1.0 / 256.0);
        }
    }

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_desc<3, 32, 32, 12, 5, 5, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<12, 28, 28, 1, 2, 2>::layer_t,
            dll::conv_desc<12, 14, 14, 24, 3, 3, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<24, 12, 12, 1, 2, 2>::layer_t,
            dll::dense_desc<24 * 6 * 6, 64, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<64, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::momentum, dll::batch_size<100>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.001;
    dbn->initial_momentum = 0.9;
    dbn->momentum = 0.9;
    dbn->goal = -1.0;

    dbn->display();

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;

    return 0;
}
