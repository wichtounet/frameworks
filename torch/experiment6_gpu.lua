require 'torch'
require 'cunn'
require 'optim'
require 'image'

-- sadly only way to get wall time more precise than seconds...
--local posix = require 'posix'

training_images = {}
training_labels = {}

batch_size = 64
files = 0
label_counter = 0

for folder in paths.files('/data/datasets/imagenet_resized/train/') do
    if #folder > 2 then
        label_counter = label_counter + 1

        for file in paths.files('/data/datasets/imagenet_resized/train/' .. folder) do
            if #file > 2 then
                table.insert(training_images, '/data/datasets/imagenet_resized/train/' .. folder .. '/'.. file)
                table.insert(training_labels, label_counter)
            end
        end
    end
end

nice_n = (math.floor(#training_images / batch_size) * batch_size)

print(#training_images)
print(#training_labels)

function swap(array, index1, index2)
    array[index1], array[index2] = array[index2], array[index1]
end

function shuffle(array1, array2)
    local counter = #array1
    while counter > 1 do
        local index = math.random(counter)
        swap(array1, index, counter)
        swap(array2, index, counter)
        counter = counter - 1
    end
end

shuffle(training_images, training_labels)

function get_batch()
    local index = 1

    B = torch.CudaTensor(batch_size,3,256,256)
    L = torch.CudaTensor(batch_size)

    while index <= batch_size do
        local imagetensor = image.load(training_images[current_index])

        if imagetensor:size(1) == 1 then
            B[index][1] = imagetensor
            B[index][2] = imagetensor
            B[index][3] = imagetensor
        else
            B[index] = imagetensor
        end

        L[index] = training_labels[current_index]

        index = index + 1
        current_index = current_index + 1
    end

    collectgarbage()

    return B, L
end

model = nn.Sequential()

--- conv1
model:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

--- conv2
model:add(nn.SpatialConvolution(16, 16, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

--- conv3
model:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

--- conv4
model:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

--- conv5
model:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.Reshape(2048))

--- fc1
model:add(nn.Linear(2048, 2048))
model:add(nn.ReLU())

--- fc2
model:add(nn.Linear(2048, 1000))

sgd_params = {
   learningRate = 0.001,
   learningRateDecay = 0.0,
   weightDecay = 0.0,
   momentum = 0.9
}

model:add(nn.LogSoftMax())
model:cuda()

criterion = nn.ClassNLLCriterion()
criterion:cuda()

x, dl_dx = model:getParameters()

print('<mnist> using model:')
print(model)

run_epoch = function()
    local current_loss = 0
    local count = 0

    current_index = 1

    while current_index + batch_size < #training_images do
        local inputs, targets = get_batch()

        --local start_time = posix.clock_gettime(0)

        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()

            -- perform mini-batch gradient descent
            local loss = criterion:forward(model:forward(inputs), targets)
            model:backward(inputs, criterion:backward(model.output, targets))

            return loss, dl_dx
        end

        _, fs = optim.sgd(feval, x, sgd_params)
        count = count + 1
        current_loss = current_loss + fs[1]

        --local end_time = posix.clock_gettime(0)
        --local elapsed_time = end_time - start_time

        --print(elapsed_time)
        print(string.format("Batches: %d/%d loss: %4f time: ", current_index, #training_images,fs[1]))
    end

    -- normalize loss
    return current_loss / count
end

eval = function()
    local count = 0

    current_index = 1

    while current_index + batch_size < #training_images do
        local inputs, targets = get_batch()
        local outputs = model:forward(inputs)

        for j = 1, batch_size do
            local output = outputs[j]
            local _, index = torch.max(output, 1)
            local label = index[1] - 1
            if label == targets[j] then
                count = count + 1
            end
        end
    end

    return count / nice_n
end

max_iters = 10

print("Start training")

do
    for i = 1,max_iters do
        local loss = run_epoch()
        local accuracy = eval()
        print(string.format('Epoch: %d loss: %4f train acc: %5f', i, loss, accuracy))
    end
end
