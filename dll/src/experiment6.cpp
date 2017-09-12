//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/neural/dense_layer.hpp"
#include "dll/neural/conv_same_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/dbn.hpp"

#include "dll/datasets.hpp"
#include "dll/datasets/imagenet.hpp"

int main(int /*argc*/, char* /*argv*/ []) {
    constexpr const char* imagenet_path = "/home/wichtounet/datasets/imagenet_resized/";
    constexpr size_t B = 128;

    // Load the dataset
    auto dataset = dll::make_imagenet_dataset(imagenet_path, 0, dll::batch_size<B>{}, dll::scale_pre<255>{});

    using dbn_t = dll::dyn_network_desc<
            dll::dbn_layers<
                dll::conv_same_layer<3, 256, 256, 16, 3, 3, dll::relu>,
                dll::mp_2d_layer<16, 256, 256, 2, 2>,

                dll::conv_same_layer<16, 128, 128, 16, 3, 3, dll::relu>,
                dll::mp_2d_layer<16, 128, 128, 2, 2>,

                dll::conv_same_layer<16, 64, 64, 32, 3, 3, dll::relu>,
                dll::mp_2d_layer<32, 64, 64, 2, 2>,

                dll::conv_same_layer<32, 32, 32, 32, 3, 3, dll::relu>,
                dll::mp_2d_layer<32, 32, 32, 2, 2>,

                dll::conv_same_layer<32, 16, 16, 32, 3, 3, dll::relu>,
                dll::mp_2d_layer<32, 16, 16, 2, 2>,

                dll::dense_layer<2048, 2048, dll::relu>,
                dll::dense_layer<2048, 1000, dll::relu>
            >,
            dll::batch_size<B>,
            dll::updater<dll::updater_type::MOMENTUM>,
            dll::verbose,
            dll::no_epoch_error
        >::network_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.01;
    dbn->initial_momentum = 0.9;
    dbn->momentum = 0.9;

    dbn->display();
    dataset.display();

    dbn->fine_tune(dataset.train(), 5);
    dbn->evaluate(dataset.train());

    return 0;
}
