local knb = torch.class("KNearestBuf")

function knb:__init ()
	self.buf = {}
	self.maxDist = nil
end

function knb:add (data, dist, k)
	local i = #self.buf
	while i > 0 and self.buf[i].dist > dist do
		i = i - 1
	end

	if i < k then
		table.insert(self.buf, i + 1, {data=data, dist=dist})
	end

	if #self.buf > k then
		self.buf[#self.buf] = nil
	end

	if #self.buf == k then
		self.maxDist = self.buf[k].dist
	end
end

function knb:getData ()
	local buf = {}
	for i = 1, #self.buf do
		buf[i] = self.buf[i].data
	end

	return buf
end

function knb:__len ()
	return #self.buf
end
