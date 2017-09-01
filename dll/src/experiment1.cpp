//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include "dll/neural/dense_layer.hpp"
#include "dll/test.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = dll::make_mnist_dataset(0, dll::batch_size<100>{}, dll::binarize_pre<30>{});

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::dense_desc<28 * 28, 500>::layer_t,
            dll::dense_desc<500, 250>::layer_t,
            dll::dense_desc<250, 10, dll::activation<dll::function::SOFTMAX>>::layer_t>,
        dll::batch_size<100>,
        dll::updater<dll::updater_type::MOMENTUM>
        >::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.1;
    dbn->initial_momentum = 0.9;
    dbn->momentum = 0.9;
    dbn->goal = -1.0; // Don't stop

    dbn->display();

    dbn->fine_tune(dataset.train(), 50);

    dbn->evaluate(dataset.test());

    return 0;
}
