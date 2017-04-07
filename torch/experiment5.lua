require 'torch'
require 'nn'
require 'optim'

trsize = 50000
tesize = 10000

-- download dataset
if not paths.dirp('cifar-10-batches-t7') then
   local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
   local tar = paths.basename(www)
   os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
end

-- load dataset
trainset = {
   data = torch.Tensor(50000, 3072),
   label = torch.Tensor(50000),
   size = trsize
}
for i = 0,4 do
   subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainset.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainset.label[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
--trainset.label = trainset.label + 1

subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
testset = {
   data = subset.data:t():double(),
   label = subset.labels[1]:double(),
   size = tesize
}
--testset.label = testset.label + 1

-- resize dataset (if using small version)
trainset.data = trainset.data[{ {1,trsize} }]
trainset.label = trainset.label[{ {1,trsize} }]

testset.data = testset.data[{ {1,tesize} }]
testset.label = testset.label[{ {1,tesize} }]

-- reshape data
trainset.data = trainset.data:reshape(trsize,3,32,32)








model = nn.Sequential()

model:add(nn.Reshape(3,32,32))
model:add(nn.SpatialConvolution(3, 12, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.SpatialConvolution(12, 24, 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.Reshape(24 * 6 * 6))
model:add(nn.Linear(24 * 6 * 6, 64))
model:add(nn.ReLU())
model:add(nn.Linear(64, 10))

sgd_params = {
   learningRate = 0.001,
   learningRateDecay = 0.0,
   weightDecay = 0.0,
   momentum = 0.9
}

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

x, dl_dx = model:getParameters()

print('<mnist> using model:')
print(model)

step = function(batch_size)
    local current_loss = 0
    local count = 0
    local shuffle = torch.randperm(trainset.size)
    batch_size = batch_size or 100

    for t = 1,trainset.size,batch_size do
        -- setup inputs and targets for this mini-batch
        local size = math.min(t + batch_size - 1, trainset.size) - t
        local inputs = torch.Tensor(size, 3, 32, 32)
        local targets = torch.Tensor(size)
        for i = 1,size do
            local input = trainset.data[shuffle[i+t]]
            local target = trainset.label[shuffle[i+t]]
            -- if target == 0 then target = 10 end
            inputs[i] = input:view(3, 32, 32)
            targets[i] = target
        end
        targets:add(1)

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
        -- fs is a table containing value of the loss function
        -- (just 1 value for the SGD optimization)
        count = count + 1
        current_loss = current_loss + fs[1]
    end

    -- normalize loss
    return current_loss / count
end

eval = function(dataset, batch_size)
    local count = 0
    batch_size = batch_size or 100

    for i = 1,dataset.size,batch_size do
        local size = math.min(i + batch_size - 1, dataset.size) - i
        local inputs = dataset.data[{{i,i+size-1}}]
        local targets = dataset.label[{{i,i+size-1}}]:long()
        local outputs = model:forward(inputs)
        local _, indices = torch.max(outputs, 2)
        indices:add(-1)
        local guessed_right = indices:eq(targets):sum()
        count = count + guessed_right
    end

    return count / dataset.size
end

max_iters = 50

trainset.data = trainset.data:double()
testset.data = testset.data:double()

print("Start training")

do
    for i = 1,max_iters do
        local loss = step()
        local accuracy = eval(trainset)
        print(string.format('Epoch: %d loss: %4f train acc: %5f', i, loss, accuracy))
    end
end

local accuracy = eval(testset)
print(string.format('Test acc: %5f', accuracy))
