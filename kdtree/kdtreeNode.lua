local node = torch.class('KDTreeNode')

function node:__init (args)
	self.maxNum = args.maxNum or 100
	self.split = false
	self.splitDim = 0
	self.splitPoint = nil
	self.splitPointValue = nil
	self.dataNum = 0
	self.buf = {}
	self.left = nil
	self.right = nil
end

function node:add (key, value)
	if not self.split then
		self.buf[#self.buf + 1] = {key=key, value=value}

		if #self.buf > self.maxNum then
			self:chooseAndSplit()
		end
	else
		if self:cmp(key) then
			self.left:add(key, value)
		else
			self.right:add(key, value)
		end
	end

	self.dataNum = self.dataNum + 1
end

function node:cmp (key)
	return key[self.splitDim] < self.splitPoint[self.splitDim]
end

function node:abs (key)
	return math.abs(key[self.splitDim] - self.splitPoint[self.splitDim])
end

function node:chooseAndSplit ()
	local keyDim = self.buf[1].key:size(1)

	local tmp = torch.Tensor(#self.buf, keyDim)
	for i = 1, #self.buf do
		tmp[i] = self.buf[i].key:clone()
	end

	local _, dim = tmp:std(1):max(2)
	dim = dim[1][1]

	local _, sort = tmp:sort(1)
	local k = sort[math.floor(#self.buf/2 + 0.5)][dim]

	local point = tmp[k]
	self:doSplit(dim, point)
end

function node:doSplit (splitDim, splitPoint)
	self.split = true
	self.splitDim = splitDim
	self.splitPoint = splitPoint
	self.left = KDTreeNode({maxNum=self.maxNum})
	self.right = KDTreeNode({maxNum=self.maxNum})

	for i = 1, #self.buf do
		local key = self.buf[i].key
		local value = self.buf[i].value

		if not key:equal(splitPoint) then
			if self:cmp(key) then
				self.left:add(key, value)
			else
				self.right:add(key, value)
			end
		else
			self.splitPointValue = value
		end
	end

	self.buf = {{key=self.splitPoint, value=self.splitPointValue}}
end

function node:find (key)
	if self.split then
		if key:equal(self.splitPoint) then
			return {key=self.splitPoint, value=self.splitPointValue}
		end

		if self:cmp(key) then
			return self.left:find(key)
		else
			return self.right:find(key)
		end
	else
		for i = 1, #self.buf do
			if key:equal(self.buf[i].key) then
				return self.buf[i]
			end
		end
		return nil
	end
end

function node:kNearest (key, k, ansBuf)
	for i = 1, #self.buf do
		local dist = math.sqrt(key:clone():csub(self.buf[i].key):pow(2):sum())

		ansBuf:add(self.buf[i], dist, k)
	end
end

function node:kNearestDFS (key, k, ansBuf)
	if self.split then
		local tmpNode
		if self:cmp(key) then
			self.left:kNearestDFS(key, k, ansBuf)
			tmpNode = self.right
		else
			self.right:kNearestDFS(key, k, ansBuf)
			tmpNode = self.left
		end

		self:kNearest(key, k, ansBuf)

		if #ansBuf < k or self:abs(key) < ansBuf.maxDist then
			tmpNode:kNearestDFS(key, k, ansBuf)
		end
	else
		self:kNearest(key, k, ansBuf)
	end
end

function node:print (path)
	print("###### " .. path .. " #########")
	if self.split then
		print("splitDim:" .. self.splitDim)
		print("splitPoint" .. key2str(self.splitPoint))
		self.left:print(path .. "/left")
		self.right:print(path .. "/right")
	else
		for i = 1, #self.buf do
			print("key:" .. key2str(self.buf[i].key))
		end
	end
end
