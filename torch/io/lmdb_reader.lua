require 'lmdb'
require 'string'
require 'torch'
require 'image'

pb = require './pb'

lmdb_reader = {}
lmdb_reader.__index = lmdb_reader

setmetatable(lmdb_reader, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})


function lmdb_reader.new(file_path)
  local self = setmetatable({}, lmdb_reader)
  self.dims = {3, 256, 256}
  self.datum = pb.Datum()
  self.db= lmdb.env{
      Path = file_path,
  }
  self.db:open()
  local reader = self.db:txn(true)
  self.cursor = reader:cursor()

  return self
end

function lmdb_reader:addString(stack, s)
  table.insert(stack, s) --push 's' into the stack
  for i=table.getn(stack)-1, 1,-1 do
    if string.len(stack[i]) > string.len(stack[i+1]) then
      break
    end
    stack[i] = stack[i] .. table.remove(stack)
  end
end


function lmdb_reader:get_data(num_elements)
  dataset = {}
  function dataset:size() return num_elements end -- 100 examples

  for i=1,num_elements do
      dataset[i] = self:get_tensor(datum)
      self.cursor:next()
  end
  return dataset
end

function lmdb_reader:get_training_data(num_elements)
  dataset = {}
  function dataset:size() return num_elements end

  for i=1,num_elements do
      local v = self:get_tensor(datum)
      dataset[i] = {v, v}
      self.cursor:next()
  end
  return dataset
end

function lmdb_reader:get_tensor()
  local data = self.cursor:getData()
  local str = self.tensor_2_string(data)
  self.datum:ParseFromString(str)
  return self.string_2_tensor(self.datum.data, self.dims)
end
--this is the slow part!
function lmdb_reader.tensor_2_string(tensor)

  s = {""} -- an empty stack
  for i=1, tensor:size()[1] do
    lmdb_reader:addString(s, string.char(tensor[i]))
  end
  s = table.concat(s)
  s = tostring(s)
  return s
end

--this is kind of slow too!
function lmdb_reader.string_2_tensor(string, dims)
  local tensor = torch.Tensor(dims[1], dims[2], dims[3])
  local idx = 1
  for c = 1, dims[1] do
    for x = 1, dims[2] do
      for y = 1, dims[3] do
        tensor[c][x][y] = string:byte(idx) / 256
        idx = idx + 1
      end
    end
  end
  return tensor
end
