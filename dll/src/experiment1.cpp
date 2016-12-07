//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include "dll/neural/dense_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::dyn_vector<float>>();

    mnist::binarize_dataset(dataset);

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 500>::layer_t,
            dll::dense_desc<500, 250>::layer_t,
            dll::dense_desc<250, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::momentum, dll::batch_size<100>, dll::trainer<dll::sgd_trainer>>::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.1;
    dbn->initial_momentum = 0.9;
    dbn->momentum = 0.9;
    dbn->goal = -1.0; // Don't stop

    dbn->display();

    auto ft_error = dbn->fine_tune(dataset.training_images, dataset.training_labels, 50);
    std::cout << "ft_error:" << ft_error << std::endl;

    auto test_error = dll::test_set(dbn, dataset.test_images, dataset.test_labels, dll::predictor());
    std::cout << "test_error:" << test_error << std::endl;

    return 0;
}
