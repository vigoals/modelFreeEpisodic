require "kdtree/kdtreeNode"
require "kdtree/kNearestBuf"

function key2str (key)
	local str = "("
	local n = key:size(1)
	for i = 1, n do
		if i ~= 1 then str = str .. " " end
		str = str .. string.format("%.3f", key[i])
		if i ~= n then str = str .. "," end
	end
	str = str .. ")"
	return str
end

local kdtree = torch.class('KDTree')

function kdtree:__init (args)
	self.maxNum = args.maxNum or 100
	self.tree = KDTreeNode({maxNum=self.maxNum})
end

function kdtree:add (key, value)
	assert(key:dim() == 1)
	self.tree:add(key, value)
end

function kdtree:find (key)
	assert(key:dim() == 1)
	return self.tree:find(key)
end

function kdtree:kNearest (key, k)
	assert(key:dim() == 1)
	local ansBuf = KNearestBuf()
	self.tree:kNearestDFS(key, k, ansBuf)
	return ansBuf:getData()
end

function kdtree:print ()
	self.tree:print("root")
end

function kdtree:__len ()
	return self.tree.dataNum
end
