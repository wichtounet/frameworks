//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include "dll/neural/conv_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/neural/dense_layer.hpp"
#include "dll/test.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    // Load the dataset
    auto dataset = dll::make_mnist_dataset(dll::batch_size<100>{}, dll::binarize_pre<30>{});

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_layer<1, 28, 28, 8, 5, 5>,
            dll::mp_3d_layer<8, 24, 24, 1, 2, 2>,
            dll::conv_layer<8, 12, 12, 8, 5, 5>,
            dll::mp_3d_layer<8, 8, 8, 1, 2, 2>,
            dll::dense_layer<8 * 4 * 4, 150>,
            dll::dense_layer<150, 10, dll::softmax>>,
        dll::updater<dll::updater_type::MOMENTUM>,
        dll::batch_size<100>
        >::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.1;
    dbn->initial_momentum = 0.9;
    dbn->momentum = 0.9;
    dbn->goal = -1.0;

    dbn->display();

    dbn->fine_tune(dataset.train(), 50);

    dbn->evaluate(dataset.test());

    return 0;
}
