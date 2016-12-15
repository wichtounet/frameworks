//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>

#include "dll/rbm/conv_rbm.hpp"
#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    auto dataset = mnist::read_dataset_direct<std::vector, etl::fast_dyn_matrix<float, 1, 28, 28>>();

    mnist::binarize_dataset(dataset);

    dll::conv_rbm_desc_square<
        1, 28, 8, 24,
        dll::batch_size<100>,
        dll::momentum>::layer_t rbm;

    rbm.learning_rate = 0.001;
    rbm.initial_momentum = 0.9;
    rbm.final_momentum = 0.9;

    rbm.train(dataset.training_images, 50);

    return 0;
}
