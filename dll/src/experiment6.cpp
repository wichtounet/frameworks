//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <queue>
#include <unordered_map>
#include <chrono>
#include <utility>
#include <vector>
#include <algorithm>
#include <dirent.h>

// Only for image loading...
#include <opencv2/opencv.hpp>

#include "dll/neural/conv_layer.hpp"
#include "dll/neural/dense_layer.hpp"
#include "dll/neural/conv_same_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/trainer/stochastic_gradient_descent.hpp"

//static constexpr const char* imagenet_path = "/data_ext/imagenet/resized/";
static constexpr const char* imagenet_path = "/home/wichtounet/datasets/imagenet_resized/";

static constexpr size_t batches_cache = 4;
static constexpr size_t batches_net = 4;
static constexpr size_t batch_size = 64;

static constexpr bool verbose = false;

struct imagenet_image_iterator : std::iterator<
                                 std::input_iterator_tag,
                                 etl::fast_dyn_matrix<float, 3, 256, 256>,
                                 ptrdiff_t,
                                 etl::fast_dyn_matrix<float, 3, 256, 256>*,
                                 etl::fast_dyn_matrix<float, 3, 256, 256>&
                                 > {

    using value_type = etl::fast_dyn_matrix<float, 3, 256, 256>;

    std::vector<std::pair<size_t, size_t>>& files;
    std::unordered_map<size_t, size_t>& labels;

    std::queue<value_type> cache;
    std::queue<size_t> requests;

    size_t index;

    std::mutex main_lock;
    std::condition_variable condition;
    std::condition_variable ready_condition;
    volatile bool stop_flag = false;

    std::thread main_thread;
    bool threaded = false;

    void construct(){
        if(index != files.size()){
            threaded = true;

            for(size_t i = 0; i < batches_cache * batch_size; ++i){
                requests.push(index + i);
            }

            main_thread = std::thread([this]{
                while (true) {
                    size_t index = 0;

                    {
                        std::unique_lock<std::mutex> ulock(main_lock);

                        condition.wait(ulock, [this] {
                            return stop_flag || !requests.empty();
                        });

                        if (stop_flag) {
                            return;
                        }

                        index = requests.front();
                        requests.pop();
                    }

                    auto& image_file = files[index];

                    auto label = std::string("/n") + (image_file.first < 10000000 ? "0" : "") + std::to_string(image_file.first);

                    auto image_path =
                        std::string(imagenet_path) + "/train" + label + label +
                        "_" + std::to_string(image_file.second) + ".JPEG";

                    auto mat = cv::imread(image_path, CV_LOAD_IMAGE_ANYDEPTH);

                    if(!mat.data){
                        std::cerr << "ERROR: Failed to read image: " << image_path << std::endl;
                    }

                    value_type image;

                    for (size_t x = 0; x < 256; ++x) {
                        for (size_t y = 0; y < 256; ++y) {

                            auto pixel = mat.at<cv::Vec3b>(y, x);

                            image(0, x, y) = pixel.val[0];
                            image(1, x, y) = pixel.val[1];
                            image(2, x, y) = pixel.val[2];
                        }
                    }

                    if(verbose){
                        std::cout << "thread: Push image " << image_file.first << ":" << image_file.second << ":" << labels[image_file.first] << std::endl;
                    }

                    {
                        std::unique_lock<std::mutex> ulock(main_lock);

                        cache.emplace(std::move(image));

                        if(verbose){
                            std::cout << " cache " << cache.size() << std::endl;
                        }

                        ready_condition.notify_one();
                    }
                }
            });
        }
    }

    imagenet_image_iterator(std::vector<std::pair<size_t, size_t>>& files, std::unordered_map<size_t, size_t>& labels, size_t index, bool delay = false) :
        files(files), labels(labels), index(index)
    {
        // The delay trick is ugly, but will save some time (if working with DLL)
        if(!delay){
            construct();
        }
    }

    imagenet_image_iterator(const imagenet_image_iterator& rhs) : files(rhs.files), labels(rhs.labels), index(rhs.index) {
        cpp_assert(!rhs.index || index == files.size(), "Only start iterators can be copied");

        std::cout << "Iterators are getting copied" << std::endl;

        construct();
    }

    ~imagenet_image_iterator(){
        if(threaded){
            cpp::with_lock(main_lock, [this] { stop_flag = true; });

            condition.notify_all();

            main_thread.join();
        }
    }

    imagenet_image_iterator& operator=(const imagenet_image_iterator& rhs) = delete;

    imagenet_image_iterator& operator++(){
        ++index;
        return *this;
    }

    // Note: DLL will never call this function because in batch mode, but must
    // still compile
    imagenet_image_iterator operator++(int){
        cpp_unreachable("Should never be called");

        return *this;
    }

    value_type operator*(){
        std::unique_lock<std::mutex> ulock(main_lock);

        if (!cache.empty()) {
            value_type o(std::move(cache.front()));
            cache.pop();

            requests.push(index + batches_cache * batch_size);
            condition.notify_one();

            return o;
        }

        ready_condition.wait(ulock, [this] {
            return !cache.empty();
        });

        value_type o(std::move(cache.front()));
        cache.pop();

        requests.push(index + batches_cache * batch_size);
        condition.notify_one();

        return o;
    }

    bool operator==(const imagenet_image_iterator& rhs) const {
        return index == rhs.index;
    }

    bool operator!=(const imagenet_image_iterator& rhs) const {
        return index != rhs.index;
    }
};

struct imagenet_label_iterator {
    std::vector<std::pair<size_t, size_t>>& files;
    std::unordered_map<size_t, size_t>& labels;

    size_t index;

    imagenet_label_iterator(std::vector<std::pair<size_t, size_t>>& files, std::unordered_map<size_t, size_t>& labels, size_t index) :
        files(files), labels(labels), index(index)
    {
        // Nothing else to init
    }

    imagenet_label_iterator(const imagenet_label_iterator& rhs) = default;
    imagenet_label_iterator& operator=(const imagenet_label_iterator& rhs) = default;

    imagenet_label_iterator& operator++(){
        ++index;
        return *this;
    }

    imagenet_label_iterator operator++(int){
        auto it = *this;
        ++index;
        return it;
    }

    size_t operator*(){
        return labels[files[index].first];
    }

    bool operator==(const imagenet_label_iterator& rhs) const {
        return index == rhs.index;
    }

    bool operator!=(const imagenet_label_iterator& rhs) const {
        return index != rhs.index;
    }
};

void read_files(std::vector<std::pair<size_t, size_t>>& files, std::unordered_map<size_t, size_t>& label_map, const std::string& file_path){
    std::cout << "Read images from '" << file_path << "'" << std::endl;

    files.reserve(1200000);

    struct dirent* entry;
    auto dir = opendir(file_path.c_str());

    while ((entry = readdir(dir))) {
        std::string file_name(entry->d_name);

        if (file_name.find("n") != 0) {
            continue;
        }

        std::string label_name(file_name.begin() + 1, file_name.end());
        size_t label = std::atoi(label_name.c_str());

        auto l = label_map.size();
        label_map[label] = l;

        struct dirent* sub_entry;
        auto sub_dir = opendir((file_path + "/" + file_name).c_str());

        while ((sub_entry = readdir(sub_dir))) {
            std::string image_name(sub_entry->d_name);

            if (image_name.find("n") != 0){
                continue;
            }

            std::string image_number(image_name.begin() + image_name.find('_') + 1, image_name.end() - 5);
            size_t image = std::atoi(image_number.c_str());

            files.emplace_back(label, image);
        }
    }
}

int main(int /*argc*/, char* /*argv*/ []) {
    // 0. Read list of files with the label

    std::vector<std::pair<size_t, size_t>> train_files;
    std::unordered_map<size_t, size_t> label_map;

    read_files(train_files, label_map, std::string(imagenet_path) + "train");

    std::cout << train_files.size() << " images found" << std::endl;
    std::cout << label_map.size() << " labels found" << std::endl;

    std::random_device rd;
    std::default_random_engine engine(rd());
    std::shuffle(train_files.begin(), train_files.end(), engine);

    imagenet_image_iterator iit(train_files, label_map, 0, true);
    imagenet_image_iterator iend(train_files, label_map, train_files.size(), true);

    imagenet_label_iterator lit(train_files, label_map, 0);
    imagenet_label_iterator lend(train_files, label_map, train_files.size());

    using dbn_t = dll::dbn_desc<
        dll::dbn_layers<
            dll::conv_same_desc<3, 256, 256, 16, 3, 3, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<16, 256, 256, 1, 2, 2>::layer_t,

            dll::conv_same_desc<16, 128, 128, 16, 3, 3, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<16, 128, 128, 1, 2, 2>::layer_t,

            dll::conv_same_desc<16, 64, 64, 32, 3, 3, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<32, 64, 64, 1, 2, 2>::layer_t,

            dll::conv_same_desc<32, 32, 32, 32, 3, 3, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<32, 32, 32, 1, 2, 2>::layer_t,

            dll::conv_same_desc<32, 16, 16, 32, 3, 3, dll::activation<dll::function::RELU>>::layer_t,
            dll::mp_layer_3d_desc<32, 16, 16, 1, 2, 2>::layer_t,

            dll::dense_desc<2048, 2048, dll::activation<dll::function::RELU>>::layer_t,
            dll::dense_desc<2048, 1000, dll::activation<dll::function::SOFTMAX>>::layer_t
        >,
        dll::batch_mode, dll::big_batch_size<batches_net>, dll::batch_size<batch_size>,
        dll::momentum, dll::trainer<dll::sgd_trainer>,
        dll::verbose>::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.001;
    dbn->initial_momentum = 0.9;
    dbn->momentum = 0.9;
    dbn->goal = -1.0;

    dbn->display();

    auto ft_error = dbn->fine_tune(iit, iend, lit, lend, 2);
    std::cout << "ft_error:" << ft_error << std::endl;

    //TODO

    return 0;
}
